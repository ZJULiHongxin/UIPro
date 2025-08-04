import json, os, re, time
from playwright.async_api import async_playwright
from typing import Any
import asyncio

IN_VIEWPORT_RATIO_THRESHOLD = 0.6
INVALID_NODE_ROLES = ["generic", "img", "list", "strong", "paragraph", "banner", "navigation", "Section", "LabelText", "Legend", "listitem", "alert", "superscript", "LineBreak", "Canvas"]
IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "disabled",
    "describedby"
)
SNAPSHOTS_DIR = os.path.abspath("task")
DATA_DIR = "/data0/jingran/workspace/hongxin_li/Mind2Web/Mind2Web_data/data/train_each_task"

tasks = sorted(os.listdir(SNAPSHOTS_DIR))

html_files = []
for task_id in tasks:
    task_dir = os.path.join(SNAPSHOTS_DIR, task_id, "processed/snapshots")
    
    # Instantiate a Playwright page of this html
    for html_file in os.listdir(task_dir):
        html_files.append(os.path.join(task_dir, html_file).replace('\\', '/'))

print(f"Need to process {len(html_files)} files!")

def prune_accessibility_tree_wo_bound(
    accessibility_tree,
) -> tuple[str, dict[str, Any]]:
    """Parse the accessibility tree into a string text"""
    
    def remove_node_in_graph(node) -> None:
        # update the node information in the accessibility tree
        nodeid = node["nodeId"]
        parent_nodeid = node["parentId"]
        children_nodeids = node["childIds"]
        # update the children of the parent node
        assert (
            accessibility_tree[parent_nodeid].get("parentId", "Root")
            is not None
        )
        # remove the nodeid from parent's childIds
        index = accessibility_tree[parent_nodeid]["childIds"].index(
            nodeid
        )
        accessibility_tree[parent_nodeid]["childIds"].pop(index)
        # Insert children_nodeids in the same location
        for child_nodeid in children_nodeids:
            accessibility_tree[parent_nodeid]["childIds"].insert(
                index, child_nodeid
            )
            index += 1
        # update children node's parent
        for child_nodeid in children_nodeids:
            accessibility_tree[child_nodeid][
                "parentId"
            ] = parent_nodeid
        # mark as removed
        accessibility_tree[nodeid]["parentId"] = "[REMOVED]"
    
    for obs_node_id, node in accessibility_tree.items():
        valid_node = True
        try:
            role = node["role"]["value"]
            name = node["name"]["value"]

            node_str = f"[{obs_node_id}] {role} {repr(name)}"
            properties = []
            
            if role == 'textbox':
                for x in node["name"]['sources']:
                    if x['type'] == 'placeholder' and 'value' in x.keys():
                        properties.append(f"placeholder: [{x['value']['value']}]")
            
            for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    properties.append(
                        f'{property["name"]}: {property["value"]["value"]}'
                    )
                except KeyError:
                    pass
            
            

            if properties:
                node_str += " " + " ".join(properties)

            # check valid
            if not node_str.strip():
                valid_node = False

            # empty generic node
            if not name.strip():
                if not properties:
                    if role in INVALID_NODE_ROLES:
                        valid_node = False
                elif role in ["listitem"]:
                    valid_node = False

            if not valid_node:
                remove_node_in_graph(node)
                continue
        except Exception as e:
            valid_node = False
            remove_node_in_graph(node)
    
    for nodeId in list(accessibility_tree.keys()):
        if accessibility_tree[nodeId].get("parentId", "-1") == "[REMOVED]":
            del accessibility_tree[nodeId]

    return accessibility_tree

async def get_bounding_client_rect_async(
    client, backend_node_id: str
) -> dict[str, Any]:
    try:
        remote_object = await client.send(
            "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
        )
        remote_object_id = remote_object["object"]["objectId"]
        response = await client.send(
            "Runtime.callFunctionOn",
            {
                "objectId": remote_object_id,
                "functionDeclaration": """
                    function() {
                        if (this.nodeType == 3) {
                            var range = document.createRange();
                            range.selectNode(this);
                            var rect = range.getBoundingClientRect().toJSON();
                            range.detach();
                            return rect;
                        } else {
                            return this.getBoundingClientRect().toJSON();
                        }
                    }
                """,
                "returnByValue": True,
            },
        )
        return response
    except Exception as e:
        return {"result": {"subtype": "error"}}
    
async def parse_accessibility_tree(accessibility_tree, start_node_id: str = '1', numbering_start: int = 1) -> tuple[str, dict[str, Any]]:
    """Parse the accessibility tree into a string text."""
    obs_nodes_info = {}
    reorder = {}  # map numbering orders to real identifiers
    node_ids = set(accessibility_tree.keys())

    def is_valid_node(node, role, name, properties):
        if not name.strip() and not properties and role in INVALID_NODE_ROLES:
            return False
        if role == "listitem" and not properties:
            return False
        return True

    # Find the root node
    for start_node_id, v in accessibility_tree.items():
        if v['role']['value'] == 'RootWebArea': break
    else: raise Exception("Invalid webpage without a root node!")
    
    stack = [(start_node_id, 0)]
    tree_lines = []

    while stack:
        obs_node_id, depth = stack.pop()
        if obs_node_id not in node_ids:
            continue

        node = accessibility_tree[obs_node_id]
        indent = "\t" * depth
        
        role = node["role"].get("value", None)
        if role is None: continue
        
        name = node["name"].get("value", None)
        if name is None: continue
        
        if node.get("backendDOMNodeId", None) is None: continue
        
        reorder[str(numbering_start)] = obs_node_id

        properties = []
        
        if role == 'textbox':
            for x in node["name"]['sources']:
                if x['type'] == 'placeholder' and 'value' in x.keys():
                    properties.append(f"placeholder: [{x['value']['value']}]")
                    break
                        
        for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    properties.append(
                        f'{property["name"]}: {property["value"]["value"]}'
                    )
                except KeyError:
                    pass
        
        node_str = f"[{numbering_start}] {role} {repr(name)}" if numbering_start != -1 else f"[{numbering_start}] {role} {repr(name)}"
        
        if properties:
            node_str += " " + " ".join(properties)

        numbering_start += 1
        valid_node = is_valid_node(node, role, name, properties)

        if valid_node:
            tree_lines.append(f"{indent}{node_str}")
            obs_nodes_info[obs_node_id] = {
                "nodeId": obs_node_id,
                "backend_id": node["backendDOMNodeId"],
                "union_bound": node["union_bound"],
                "text": node_str,
                "name": name,
                "role": role,
                'parentId': node.get('parentId', ''),
                'childIds': node['childIds']
            }

        for child_node_id in reversed(node["childIds"]):
            child_depth = depth + 1 if valid_node else depth
            stack.append((child_node_id, child_depth))

    tree_str = "\n".join(tree_lines)
    if len(obs_nodes_info) == 0:
            print("Empty AXTree")
    # update child IDs
    return tree_str, obs_nodes_info, reorder

async def clean_accesibility_tree(tree_str: str) -> str:
    """further clean accesibility tree"""
    clean_lines: list[str] = []
    for line in tree_str.split("\n"):
        # remove statictext if the content already appears in the previous line
        if "statictext" in line.lower():
            prev_lines = clean_lines[-3:]

            pattern = r"\[\d+\] StaticText (.+)"

            match = re.search(pattern, line, re.DOTALL)
            if match:
                static_text = match.group(1)[1:-1]  # remove the quotes
                if static_text and all(
                    static_text not in prev_line
                    for prev_line in prev_lines
                ):
                    clean_lines.append(line)
        else:
            clean_lines.append(line)

    return "\n".join(clean_lines)

async def process_node(
        client,
        node,
):
    backend_node_id = str(node["backendDOMNodeId"])
    if node["role"]["value"] == "RootWebArea":
        # always inside the viewport
        node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
    else:
        response = await get_bounding_client_rect_async(
            client, backend_node_id
        )
        if response.get("result", {}).get("subtype", "") == "error":
            node["union_bound"] = None
        else:
            x = response["result"]["value"]["x"]
            y = response["result"]["value"]["y"]
            width = response["result"]["value"]["width"]
            height = response["result"]["value"]["height"]
            node["union_bound"] = [x, y, width, height]

async def extract_axtree(html_file, cnt, t):
    # html_file: '/data0/jingran/workspace/hongxin_li/globus/raw_dump/task/000ada18-5007-4fd4-8a12-8987ba543d31/processed/snapshots/0daf1895-493d-4b9a-ba8a-ba6a65c23a21_after.mhtml'
    task_id = html_file[html_file.find("task/")+5:html_file.find("/proc")]
    action_id = html_file[html_file.find("shots/")+6:html_file.rfind("_")]

    with open(os.path.join(DATA_DIR, f"{task_id}.json"), "r") as f:
        task_data = json.load(f)
        for action_data in task_data["actions"]:
            if action_data["action_uid"] == action_id:
                break
        candidate = action_data["pos_candidates"][0]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        assert os.path.exists(html_file)
        
        # await page.goto("http://36.111.143.215:9999/")
        await page.goto(f'file://{html_file}')

        # get the axtree
        client = await page.context.new_cdp_session(page)
        await client.send("Accessibility.enable")
        await page.bring_to_front()
        
        response = await client.send(
            "Accessibility.getFullAXTree", {}
        )
        axtree = response["nodes"]
        _accessibility_tree = {}
        for node in axtree:
            _accessibility_tree[node['nodeId']] = node
        axtree = _accessibility_tree
        
        axtree = prune_accessibility_tree_wo_bound(axtree)
        
        tasks = []
        for node in axtree.values():
            # usually because the node is not visible etc
            if "backendDOMNodeId" not in node:
                node["union_bound"] = None
                continue
            task = asyncio.create_task(process_node(client, node))
            tasks.append(task)

        await asyncio.gather(*tasks)
        
        content, obs_nodes_info, reorder = await parse_accessibility_tree(
                axtree
            )
        
        content = await clean_accesibility_tree(content)
        
        # Find the correspinding AXTree node via bbox matching
        backend_id, bbox, elem_tag, elem_attrs = candidate["backend_node_id"], candidate["bounding_box_rect"], candidate["tag"], candidate["attributes"]

        # get the axtree_json
        split = '/' if '/' in html_file else r'\\'
        save_dir = '/'.join(html_file.split(split)[:-3]) + "/axtree"

        os.makedirs(save_dir, exist_ok=True)
        
        task_action_id = os.path.basename(html_file)[:-6]
        with open(f"{save_dir}/{task_action_id}_raw.json", "w") as f:
            json.dump(axtree, f, indent=2)
        with open(f"{save_dir}/{task_action_id}_clean.json", "w") as f:
            json.dump({'content': content, 'obs_nodes_info': obs_nodes_info, 'reorder': reorder}, f, indent=2)
        
        # close the browser
        await browser.close()
        
        if cnt % 10 == 0:
            print(f"Extracted axtree for {cnt} tasks in {time.time()-t:.2f}s")
        

selected_tasks = []
t = time.time()
skip_cnt = 0
for html_file in html_files:
    task_action_id = os.path.basename(html_file)
    saved_file_root = ('/'.join(html_file.split('/')[:-3])) + f"/axtree/{task_action_id[:-6]}_"
    
    if True: #not os.path.exists(saved_file_root + 'raw.json') or not os.path.exists(saved_file_root + 'clean.json'):
        selected_tasks.append(html_file)
    else: skip_cnt += 1

print(f"SKip {skip_cnt} files")
cnt = 0
for html_file in selected_tasks:
    cnt += 1
    asyncio.run(extract_axtree(html_file, cnt, t))

# Run the asyncio event loop
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())