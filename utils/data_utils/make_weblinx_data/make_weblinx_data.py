from pathlib import Path
import weblinx as wl
import os, glob, magic, json, lxml
from tqdm import tqdm
import cv2, random, re
import numpy as np
from utils.data_utils.misc import add_text, is_pure_color, generate_negative_action_plans_for_web, keep_unique_actions
from weblinx_tools.processing import represent_element_as_dict, convert_elem_dict_to_str_legacy
from weblinx_tools.dom import remove_uid_when_not_candidate, sanitize_elem_attributes
from utils.data_utils.task_prompt_lib import *
from typing import  List
import weblinx.utils.html as wh

def clean_text(text):
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple tabs with a single tab
    text = re.sub(r'\t+', '\t', text)
    
    # Optionally, if you want to remove all tabs and newlines entirely:
    # text = re.sub(r'[\n\t]+', '', text)

    return text.strip()


def format_relevant_elements_for_single_turn(
    turn, uid_key="data-webtasks-id"
) -> List[dict]:
    bboxes_filt = wh.filter_bboxes(
        turn.bboxes,
        viewport_height=turn.viewport_height,
        viewport_width=turn.viewport_width,
    )
    root = lxml.html.fromstring(turn.html)
    root_tree = root.getroottree()
    sanitize_elem_attributes(root_tree, remove_data_attrs=True, remove_underscore_attrs=True, remove_angular_attrs=True, remove_alpine_attrs=True, remove_xml_attrs=True, remove_google_attrs=True, remove_id_attrs=True, uid_key="data-webtasks-id")
    
    elements = root.xpath(f"//*[@{uid_key}]")
    elements_filt = [p for p in elements if p.attrib[uid_key] in bboxes_filt]

    records = []

    for elem in elements_filt:
        bbox = turn.bboxes[elem.attrib[uid_key]]
        elem_dict = represent_element_as_dict(elem, bbox, root_tree)
        elem_str = convert_elem_dict_to_str_legacy(elem_dict)

        record = {
            "doc": elem_str,
            "uid": elem.attrib[uid_key],
            "turn_index": turn.index,
            "elem_dict": elem_dict,
        }
        records.append(record)

    return records

wl_dir = "/mnt/vdb1/hongxin_li/WebLINX/wl_data"
base_dir = wl_dir + "/demonstrations"
split_path = wl_dir + "/splits.json"

# Load the name of the demonstrations in the training split
demo_names = wl.utils.load_demo_names_in_split(split_path, split='train')
# You can use: train, valid, test_iid, test_vis, test_cat, test_geo, test_web
# Or you can specify the demo_names yourself, such as the ones we just fetched

demo_names = [x for x in glob.glob(base_dir + '/*') if os.path.isdir(x) and not os.path.basename(x).startswith('.')]  # 3 random demo from valid

DEBUG = False

DATASET_NAME = "WebLINX"
SCALE=1000
ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"
SAVE_ROOT = os.path.join(ROOT, f"{DATASET_NAME}_processed")
os.makedirs(SAVE_ROOT,exist_ok=True)
DRAW = False

REWARDMODEL_EVAL = False

USE_ACTION_REFEXP = True
DEVICE_TAG = 'Web'

# Load the demonstrations
demos = [wl.Demonstration(name, base_dir=base_dir) for name in demo_names]

action_types = set() # {'tabswitch', 'scroll', 'paste', 'change', 'submit', 'copy', None, 'tabcreate', 'load', 'textInput', 'tabremove', 'hover', 'click'}

# Select a demo to work with
samples, rewardmodel_eval_samples = [], []
action_stats = {}
num_iterated = num_collected = reward_model_eval_cnt = 0
invalid = []
num_eps = valid_eps = 0

if DEBUG: demos = random.sample(demos, 100)

for demo_id, demo in tqdm(enumerate(demos), total=len(demos)):
    # Load the Replay object, which contains the turns of the demonstration
    num_eps += 1
    if not demo.is_valid(): continue
    valid_eps += 1

    replay = wl.Replay.from_demonstration(demo)

    task = demo.form['description']
    
    if not task.isascii():
        continue
    # 提取出所有有效步骤
    all_turns = list(replay)
    
    # 提取出agent所有动作
    new_actions, actions, action_formats, intents, elem_infos, attrs, imgs_sizes, tids = [], [], [], [], [], [], [], []
    
    all_img_paths = demo.list_all_screenshots(return_str=True)
    for tid, t in enumerate(all_turns):
        if t.type == 'chat': continue

        # load img info
        if t.has_screenshot():
            img_path = t.get_screenshot_path()
            W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_path)).groups(1)))
            
            try:
                elems = format_relevant_elements_for_single_turn(t)
                elem_box_text_record = []
                for elem in elems:
                    elem_text = elem['elem_dict']['text'].strip()
                    if not elem_text: continue
                    
                    if 'subnav' in elem['elem_dict']['attributes']: continue

                    x1,y1,w,h = list(map(lambda p: float(p.split('=')[-1]), elem['elem_dict']['bbox'].split()))
                    
                    if y1+h/2 > H: continue

                    x2,y2 = x1+w, y1+h

                    elem_box_text_record.append({'bbox': list(map(round, [x1,y1,x2,y2])), 'text': elem_text, 'tag': elem['elem_dict']['tag']})
            except: elem_box_text_record = []

            if False:
                img = cv2.imread(img_path)
                for elem_info in elem_box_text_record:
                    x1,y1,x2,y2 = elem_info['bbox']
                    cv2.rectangle(img, [x1,y1], [x2,y2], color=(0,0,255), thickness=3)
                cv2.imwrite('test.png', img)

            # Get the next observation
            # next_img_path = all_img_paths[min(len(all_img_paths)-1, all_img_paths.index(img_path) + 1)]
            if tid < len(all_turns) - 1: # WebLinx两个动作之间的帧会丢失，这里排除这些样本
                try:
                    next_img_path = all_turns[tid+1].get_screenshot_path()
                except:
                    next_img_path = None
            else: next_img_path = img_path

            imgs_sizes.append([img_path, next_img_path, elem_box_text_record, [int(W),int(H)]])
        else:
            W = H = -1
            imgs_sizes.append([None, None, None, [W,H]])
        elem_info = t.element

        attr = {}
        if elem_info is not None and 'bbox' in elem_info:
            x1, y1 = round(elem_info['bbox']['x']), round(elem_info['bbox']['y'])
            x2, y2 = round(elem_info['bbox']['right']), round(elem_info['bbox']['bottom'])
            attr['unnormalized_box'] = [x1,y1,x2,y2]

        act = t.intent
        raw_intent = t.format_text()

        node_desc = ''
        if elem_info is not None:
            node_text = elem_info['textContent'].strip()
            if node_text:
                node_desc = node_text
            
            if not node_desc and 'id(' in elem_info['xpath']:
                node_desc = elem_info['xpath'].split('")')[0][4:].strip(' _./')
            
            if len(node_desc) > 50:
                node_desc = node_desc[:50] + '...'

        node_desc = ' '.join(x.strip() for x in node_desc.replace('\t', '\n').split('\n') if len(x.strip()))
        node_desc = node_desc.strip(' :/')

        if act == 'load':
            short_url = t.url.split('&')[0].split('://')[1]
            attr['url'] = short_url
            intent = f"load the website {t.url}"
            new_actions.append('go_to')
        elif act in ['click', 'hover']:
            assert 'unnormalized_box'  in attr
            raw_intent
            node_tag = elem_info['tagName'].lower()

            if node_tag in ['img', 'svg']:
                match = re.search(r'alt="([^"]*)"', elem_info['outerHTML'])
                if match:
                    imgalt = match.group(1).strip()
                    if len(imgalt):
                        node_desc = imgalt

            assert 'div[2]/div[2]' not in node_desc
            intent = f'{"click on" if act == "click" else "hover over"} [{node_tag}]{" " + node_desc}'
            new_actions.append(act)
        elif act == 'submit':
            intent = 'press enter to submit the completed form'
            new_actions.append('enter')
        elif act == 'scroll':
            intent = 'scroll the wegpage'
            new_actions.append('scroll')
        elif act == 'tabswitch':
            intent = 'switch to another tab'
            new_actions.append('switch_tab')
        elif act == 'tabcreate':
            intent = 'create a new tab'
            new_actions.append('new_tab')
        elif act == 'tabremove':
            intent = 'close a tab'
            new_actions.append('close_tab')
        elif act == 'copy': # ignore
            intent = ''
            new_actions.append('copy')
        elif act in ['textInput']: # 跳过'paste'，因为paste之后经常紧接着输入同样文本的input_text
            assert 'unnormalized_box'  in attr
            textfield_tag = elem_info['tagName'].lower()
            place_holder = elem_info['textContent'].strip(' "')
            text = clean_text(raw_intent[raw_intent.find(':', raw_intent.find('->'))+1:]).replace("\n", "\\n").replace('"', '\\"')

            # holder_str = f' with place holder "{place_holder}"' if len(place_holder) > 0 else ''
            
            if len(text) > 50:
                text_repr = text[:text.find(' ', 50)] + '...'
            else: text_repr = text

            if len(text) > 50:
                text_repr = text[:text.find(' ', 50)] + '...'
            else: text_repr = text

            intent = f'type "{text}" into [{textfield_tag}]{" " + node_desc}'
            attr['text'] = text
            new_actions.append('input_text')
        elif act == 'change': # ignore
            menu_tag = elem_info['tagName'].lower()
            menu_name = elem_info['xpath'].split('")')[0][4:]
            item = clean_text(raw_intent.split('CHANGE:')[1]).replace("\n", "\\n").replace('"', '\\"')
            intent = f'select {item} in [{menu_tag}] {menu_name}'
            attr['value'] = item
            new_actions.append('select')
        else:
            new_actions.append(act)
            intent = ''
        
        action_formats.append(t.format_text())
        tids.append(t.index)
        actions.append(act)
        intents.append(intent)
        elem_infos.append(elem_info)
        attrs.append(attr)

        if DRAW and t.has_screenshot():
            img_path = t.get_screenshot_path()
            img = cv2.imread(img_path)
            
            if elem_info is not None and 'bbox' in elem_info:
                x1, y1 = round(elem_info['bbox']['x']), round(elem_info['bbox']['y'])
                x2, y2 = round(elem_info['bbox']['right']), round(elem_info['bbox']['bottom'])
                cv2.rectangle(img, [x1,y1], [x2,y2], color=[random.randint(150,250) for _ in range(3)], thickness=3)
            try:
                after = all_turns[tid+1].get_screenshot_path()
                after_img = cv2.imread(after)
                merge = np.concatenate([img, after_img], axis=1)
            except: merge = img 
            merge = add_text(merge, intent)
            
            cv2.imwrite('test.png', merge)
            1+1
    
    assert len(set([len(actions), len(new_actions), len(intents), len(attrs)])) == 1

    # 跳过输入文本末尾是...的paste样本
    for step_idx, (tid, new_action, attr, img_size, elem_info, action_format) in enumerate(zip(tids, new_actions, attrs, imgs_sizes, elem_infos, action_formats)):
        img_path, next_img_path, elem_box_text_record, size = img_size
        
        num_iterated += 1
        
        if img_path is None: 
            invalid.append([img_path, 'non-existent image'])
            continue

        img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        W, H = size

        if elem_info is not None and elem_info.get('outerHTML', '').count('</') >= 8:
            invalid.append([img_path, 'element too complex'])
            continue

        if 'unnormalized_box' in attr:
            x1,y1,x2,y2 = attr['unnormalized_box']

            if not (0<=x1<=W and 0<=x2<=W and 0<=y1<=H and 0<=y2<=H) or x1 >= x2 or y1 >= y2:
                invalid.append([img_path, 'invalid box coordinates'])
                continue

            box_w, box_h = x2-x1, y2-y1
            #  detect extreme aspect ratio
            if box_w / box_h < 0.07 or box_w / box_h > 14:
                invalid.append([img_path, 'extreme aspect ratio'])
                continue

            if box_w / W <= 0.005 or box_h / H <= 0.005:
                invalid.append([img_path, 'too small elemnent'])
                continue

            if box_w * box_h / (W*H) >= 0.65:
                invalid.append([img_path, 'oversize elemnent'])
                continue
        # html = load_html_from_file(html_path)
        # wl.processing.dom.get_tree_repr_simple(html, keep_html_brackets=False, copy=True, postfix="\nAbove are the pruned HTML contents of the page.")
        boxes = [x['bbox'] for x in elem_box_text_record]

        if new_action in ['click', 'hover']:
            if 'unnormalized_box' not in attr:
                invalid.append([img_path, 'no box attached'])
                continue

            unnorm_box = attr['unnormalized_box']

            if is_pure_color(img, unnorm_box):
                invalid.append([img_path, 'element not displayed'])
                continue
            
            unnorm_center_x, unorm_center_y = (unnorm_box[0]+unnorm_box[2])/2, (unnorm_box[1]+unnorm_box[3])/2
            center_x, center_y = max(0, min(SCALE-1, round(unnorm_center_x/W*SCALE))), max(0, min(SCALE-1, round(unorm_center_y/H*SCALE)))
            
            template = CLICK_TEMPLATE if new_action == 'click' else HOVER_TEMPLATE
            action_str = template.format(target_x=center_x, target_y=center_y)

            action_refexp = f"{random.choice(ACTION_PREFIXES[new_action]['specific'])} the {cvt_elem_tag_to_friendly_name(elem_info['tagName'])} {intents[step_idx].split('] ')[-1]}"

            if False:
                cv2.rectangle(img, unnorm_box[:2], unnorm_box[2:], color=
                (0,0,250), thickness=3)
                print(action_format, ' | ', intents[step_idx])
                cv2.imwrite('test.png', img)
                1+1
            if REWARDMODEL_EVAL:
                neg_actions = generate_negative_action_plans_for_web(gt_act_type=new_action, W=W, H=H, scale=SCALE, gt_box=unnorm_box, boxes=boxes)
        elif new_action == 'input_text':
            unnorm_box = attr['unnormalized_box']

            if len(attr['text']) == 0: 
                invalid.append([img_path, 'text invalid'])
                continue

            if len(attr['text']) > 120: 
                invalid.append([img_path, 'text too long'])
                continue
            # if is_pure_color(img, unnorm_box):
            #     invalid.append([img_path, 'element not displayed'])
            #     continue
            center_x, center_y = max(0, min(SCALE-1, round((unnorm_box[0]+unnorm_box[2])/2/W*SCALE))), max(0, min(SCALE-1, round((unnorm_box[1]+unnorm_box[3])/2/H*SCALE)))
            action_str = INPUT_TARGET_TEMPLATE.format(target_x=center_x, target_y=center_y, text=attr['text'])
            
            try:
                textbox_name = elem_info['attributes']['id'].replace('_', ' ')
            except:
                textbox_name = 'the text box'
            action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=attr['text'], target=textbox_name)

            if False:
                cv2.rectangle(img, unnorm_box[:2], unnorm_box[2:], color=
                (0,0,250), thickness=3)
                print(action_format, ' | ', intents[step_idx])
                cv2.imwrite('test.png', img)
                1+1
            if REWARDMODEL_EVAL:
                neg_actions = generate_negative_action_plans_for_web(gt_act_type='input_text', W=W, H=H, scale=SCALE, gt_box=unnorm_box, boxes=boxes, text=attr['text'])

        elif new_action == 'enter':
            action_str == PRESSKEY_TEMPLATE.format(key='Enter')
            action_refexp = f"{random.choice(PRESSKEY_PREFIXES['Enter'])} {node_desc}"
        elif new_action == 'select':
            unnorm_box = attr['unnormalized_box']
            if len(attr['value']) > 120: 
                invalid.append([img_path, 'text too long'])
                continue
            if is_pure_color(img, unnorm_box):
                invalid.append([img_path, 'element not displayed'])
                continue
            center_x, center_y = max(0, min(SCALE-1, round((unnorm_box[0]+unnorm_box[2])/2/W*SCALE))), max(0, min(SCALE-1, round((unnorm_box[1]+unnorm_box[3])/2/H*SCALE)))
            action_str = SELECT_TEMPLATE.format(target_x=center_x, target_y=center_y, value=attr['value'])

            action_refexp = f"{random.choice(SELECT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=attr['value'], target=node_desc)}"

            if REWARDMODEL_EVAL:
                neg_actions = generate_negative_action_plans_for_web(gt_act_type='input_text', W=W, H=H, scale=SCALE, gt_box=unnorm_box, boxes=boxes, text=attr['value'])
        else:
            invalid.append([img_path, 'action not supported'])
            continue

        action_stats[new_action] = action_stats.get(new_action,0) + 1
        
        step_instructions = []
        cur_id = step_idx
        while cur_id >= 0:
            temp = intents[cur_id].strip()
            if len(temp):
                step_instructions.insert(0, temp)
            
            cur_id -= 1

        # ignore the actions before the final webpage loading action
        last_load_idx = len(step_instructions) - 1
        while last_load_idx >= 0 and 'load the website' not in step_instructions[last_load_idx].strip():
            last_load_idx -= 1
        
        clean_step_instructions = keep_unique_actions(step_instructions[max(0,last_load_idx):])
        history = clean_step_instructions[max(0,step_idx-MAX_PREV_ACT):step_idx]
        history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(history, start=max(1,step_idx-MAX_PREV_ACT+1))) if len(history) > 0 else 'None'

        if USE_ACTION_REFEXP:
            action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"
    
        sample = make_actionplanning_sample_web(
            task_id=f'autogui_WebLINX_planning_{demo.form["shortcode"]}-{tid}', 
            global_task=task, 
            history=history_str, 
            gt_action='Action: '+action_str,
            with_cot=False,
            use_action_refexp=USE_ACTION_REFEXP,
            device_tag=DEVICE_TAG
            )
        
        short_img_path =  img_path[img_path.find(DATASET_NAME):]
        short_nextimg_path = next_img_path[next_img_path.find(DATASET_NAME):] if next_img_path is not None else next_img_path

        sample['image'], sample['next_image'], sample['task'], sample['task_attr'], sample['step_instruction'], sample['action_refexp'], sample['history'], sample["action_type"], sample['wxh'] = short_img_path, short_nextimg_path, task, attr, step_instructions[-1], action_refexp, step_instructions[:-1], new_action, f'{W}x{H}'
        
        samples.append(sample)

        if REWARDMODEL_EVAL:
            rmeval_sample = {'id': f'autogui_{DATASET_NAME}_rewardmodeleval_{len(rewardmodel_eval_samples)}', "image": short_img_path, "next_image": short_nextimg_path, 'ep_id': demo.form["shortcode"], 'step_id': step_idx, 'turn_idx': tid, 'task': task, "step_instruction": step_instructions[-1], "action_type": new_action, "history": step_instructions[:-1], 'gt_action': action_str, 'neg_actions': neg_actions, "task_attr": attr, "wxh": f"{W}x{H}"}

            if DRAW:
                if next_img_path is not None:
                    before, after = cv2.imread(img_path), cv2.imread(next_img_path)
                    act = eval(action_str)
                    
                    for box in boxes:
                        cv2.rectangle(before, box[:2], box[2:], color=(0,0,255), thickness=3)
                    if 'target' in act:
                        H,W = before.shape[:2]
                        X,Y = round(act['target'][0]/SCALE*W), round(act['target'][1]/SCALE*H)
                        cv2.circle(before, [X,Y], color=(0,0,255), radius=6, thickness=2)
                    merged = np.concatenate([before, after], axis=1)
                    cv2.imwrite("test.png", merged)

            rewardmodel_eval_samples.append(rmeval_sample)
iterated_elem_cnt = len(samples) + len(invalid)

unique_img_cnt = len(set(x['image'] for x in samples))

report = f"#Unique images: {unique_img_cnt} | Valid eps / Total eps: {valid_eps} / {num_eps} = {valid_eps/num_eps:.2f} | Valid Steps / Total steps: {len(samples)} / {iterated_elem_cnt} = {len(samples) / iterated_elem_cnt:.2f}"

print(report)

save_to_file = os.path.join(SAVE_ROOT, f"{DATASET_NAME}{'_wActRef' if USE_ACTION_REFEXP else ''}_s{SCALE}_{len(samples)}.json")

with open(save_to_file.replace(".json","_stats.json"), "w") as f:
    json.dump({"report": report, "valid_ep_cnt": valid_eps, 
               "total_ep_cnt": num_eps,
               "valid_step_cnt": len(samples),
               "total_step_cnt": iterated_elem_cnt,
               'action_stats': action_stats}, f, indent=2)

with open(save_to_file.replace(".json","_sample.json"), "w") as f:
    json.dump(random.sample(samples,160), f, indent=2)

with open(save_to_file, "w") as f:
    json.dump(samples, f)

if REWARDMODEL_EVAL:
    save_file = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_rmeval_s{SCALE}_{len(rewardmodel_eval_samples)}.json")
    with open(save_file.replace(".json", "_sample.json"), "w") as f:
        json.dump(random.sample(rewardmodel_eval_samples,128), f, indent=2)
    
    with open(save_file, "w") as f:
        json.dump(rewardmodel_eval_samples, f)