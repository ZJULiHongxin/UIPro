import os, json, random, shutil, glob, gzip
import cv2, traceback
import numpy as np
from tqdm import tqdm
from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import prune_accessibility_tree_wo_bound
from collections import defaultdict
from utils.data_utils.misc import resize_image, is_pure_color, is_valid_string, contain_network_errors
from collections import defaultdict

DEBUG = False   
DRAW = False

TEXTLOC = False
OCR = False
ICONGND = True
ICONREF = True
ELEMGND = False
ELEMREF = False
ELEMCLASS = True
INTENTGND = True
WIDGETLIST = True

USE_ACTION_PROMPT = False
SKIP_CHECKING = False

PROB_BOX = 0.3
SCALE=1000
LONGEST = 1344

RATIO = {
    'iPhone-13 Pro': 2532 / 1170,
}

FORCE_RESIZE = {
    'iPhone-13 Pro': 3,
    'iPad-Pro': 2
}
ICON_ROLES = [
    "button",
    "image",
    "link",
    "toggle",
    "menuitem",
    "menubutton",
    "tab",
    "tabitem",
    "checkbox",
    "radio",
    "svg",
    "textbox",
    "combobox",
    "searchbox",
    "listbox",
    "treeitem",
    "slider",
    "spinbutton",
    "progressbar",
    "scrollbar",
    "separator",
    "dialog",
    "tooltip",
    "notification",
    "alert",
    "calendar",
    "colorwheel",
    "dateeditor",
    "menu",
    "menubar",
    "spinbutton",
    "treeview",
    "window"
]

DATASET_NAME = 'WebUI'

webui_dir = f"/mnt/vdb1/hongxin_li/{DATASET_NAME}/dh2"
webui_processed_img_dir = f"/mnt/vdb1/hongxin_li/{DATASET_NAME}/WebUI_screenshots"
ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"
SAVE_ROOT = os.path.join(ROOT, f"{DATASET_NAME}_processed")
os.makedirs(SAVE_ROOT, exist_ok=True)
# if os.path.exists(webui_processed_img_dir):
#     shutil.rmtree(webui_processed_img_dir)
os.makedirs(webui_processed_img_dir, exist_ok=True)

def make_webui_data():
    processed_img_cnt = valid_img_cnt = iterated_elem_cnt = annotated_node_cnt = 0
    text_loc_cnt = ocr_cnt = icongnd_cnt = iconref_cnt = elemgnd_cnt = elemref_cnt = intentgnd_cnt = widgetlist_cnt = elemclass_cnt = 0

    elemclass_stats = defaultdict(list)

    # record invalid samples
    invalid_elem_record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    if os.path.exists(invalid_elem_record_file):
        invalid_elems = json.load(open(invalid_elem_record_file))
        for k, v in invalid_elems.items():
            invalid_elems[k] = set(v)
    else:
        invalid_elems = {INVALID_TEXT_LANGUAGE: set(), TOO_SMALL_ELEMENT: set(), INVALID_ELEM_BOX: set(), INVALID_ELEM_CONTENT: set(), BLANK_ELEM: set(), EMPTY_ELEM_TEXT: set(), OVERLY_LENGTHY_ELEM_TEXT: set(), DUPLICATE_ELEMEMNT: set(), INCORRECT_TEXT_ANNO: set(), OVERSIZED_ELEMENT: set(), EXTREME_ASPECT_RATIO: set()}

    sample_dirs = sorted(os.listdir(webui_dir))

    if DEBUG: sample_dirs = sample_dirs[:1000]

    samples = []
    unique_elems = {}
    errors = []

    for sample_idx, sample_dir in tqdm(enumerate(sample_dirs), total=len(sample_dirs)):
        # 找到所有型号的axtree
        axtree_paths = glob.glob(os.path.join(webui_dir, sample_dir, '*axtree.json.gz'))
        
        for axtree_path in axtree_paths:
            try:
                device_type = os.path.basename(axtree_path).split('-axtree')[0]
                if device_type != 'iPhone-13 Pro': continue
                if any(res in axtree_path for res in ['1536-864', '1920-1080', '1366-768']):
                    continue
                
                processed_img_cnt += 1

                bbox_file = axtree_path.replace('axtree.json.gz', 'bb.json.gz')
                if not os.path.exists(bbox_file):
                    continue
                
                img_file = axtree_path.replace('axtree.json.gz', 'screenshot-full.webp')
                if not os.path.exists(img_file):
                    continue

                img = cv2.imread(img_file)
                
                if img is None: continue

                if img_file not in unique_elems: unique_elems[img_file] = []
                
                with gzip.open(bbox_file, 'rt', encoding='utf-8') as f:
                    bboxes = json.load(f)

                with gzip.open(axtree_path, 'rt', encoding='utf-8') as f:
                    axtree_raw = f.read()

                if contain_network_errors(axtree_raw):
                    continue
                axtree = json.loads(axtree_raw)['nodes']
                
                raw_axtree = {node['nodeId']: node for node in axtree}
                axtree = prune_accessibility_tree_wo_bound(raw_axtree)

                # 仅保留叶节点与图标结点
                axtree = [node for node in axtree.values() if len(node['childIds']) == 0]

                # if True:
                    
                #     for node in axtree:
                #         if 'backendDOMNodeId' not in node:
                #             continue
                #         bbox = bboxes[str(node['backendDOMNodeId'])]
                #         if bbox is None:
                #             continue
                #         x,y,w,h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                #         x1,y1,x2,y2 = x,y,x+w,y+h
                #         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                    
                #     cv2.imwrite("test.png", img)
                #     1+1

                # 将整张网页划分成多份，选图标含量最高的一份
                H, W = img.shape[:2]
                
                ratio = 2532 / 1170 if 'iPhone' in img_file else 720/1280

                block_h = int(W * ratio)
                num_blocks = H // block_h
                block_hs = np.arange(0, H, block_h)
                num_icons_each_block = {block_idx: {'nodes':[], 'icon_nodes':[],'num_icons':0} for block_idx in range(num_blocks)}
                
                for node in axtree:
                    if 'backendDOMNodeId' not in node:
                        continue
                    
                    nodebkid = str(node['backendDOMNodeId'])
                    if nodebkid not in bboxes:
                        continue
                    bbox = bboxes[nodebkid]
                    if bbox is None:
                        continue
                    
                    bbox['x'] *= FORCE_RESIZE.get(device_type, 1)
                    bbox['y'] *= FORCE_RESIZE.get(device_type, 1)
                    bbox['width'] *= FORCE_RESIZE.get(device_type, 1)
                    bbox['height'] *= FORCE_RESIZE.get(device_type, 1)
                    x1, y1, x2, y2 = bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']

                    block_idx = np.digitize(y1+bbox['height']//2, block_hs) - 1
                    
                    if block_idx in [-1, num_blocks]: continue

                    node['box'] = list(map(round, [x1, y1, x2, y2]))
                    if node['role']['value'] in ICON_ROLES:
                        num_icons_each_block[block_idx]['num_icons'] += 1

                        num_icons_each_block[block_idx]['icon_nodes'].append(node)

                    num_icons_each_block[block_idx]['nodes'].append(node)

                block_idx_with_most_icons, max_icon_cnt = -1, 0
                for block_idx, block_info in num_icons_each_block.items():
                    if block_info['num_icons'] > max_icon_cnt:
                        block_idx_with_most_icons = block_idx
                        max_icon_cnt = block_info['num_icons']
                
                if max_icon_cnt == 0:
                    continue

                h_start = block_h * block_idx_with_most_icons
                
                screenshot, resizing_ratio = resize_image(img[h_start:h_start+block_h], max_size=LONGEST)
                
                traj_dir = os.path.join(webui_processed_img_dir, sample_dir)
                os.makedirs(traj_dir, exist_ok=True)
                screenshot_file = os.path.join(traj_dir, f"{device_type}.png")
                short_screenshot_name = f"{DATASET_NAME}/{sample_dir}/{device_type}.png"
                
                if not os.path.exists(screenshot_file):
                    cv2.imwrite(screenshot_file, screenshot)
                sc_h, sc_w = screenshot.shape[:2]

                
                # 剔除：①中心点超出边界；② 过小元素；③ 长或宽过大元素；④ 无色元素；⑤坐标为负数的元素；⑥ 非中英文UI
                block_nodes = num_icons_each_block[block_idx_with_most_icons]['nodes']

                # Detect illegal language types (except English and Chinese)
                nodes_containing_valid_characters_cnt = []
                sample_identifiers = []

                axtree_anme = axtree_path.split('dh2/')[1][:-8]
                for node in block_nodes:
                    text = node['name']['value']
                    sample_identifiers.append(axtree_anme + f"|{node['nodeId']}")
                    if text is None: continue
                    nodes_containing_valid_characters_cnt.append(is_valid_string(text))

                if not SKIP_CHECKING and (sum(nodes_containing_valid_characters_cnt) / len(nodes_containing_valid_characters_cnt) <= 0.4):
                    invalid_elems[INVALID_TEXT_LANGUAGE].extend(sample_identifiers)
                    iterated_elem_cnt += len(block_nodes)
                    continue

                valid_nodes_in_block = []; used_boxes = []
                for node, sample_identifier in zip(block_nodes, sample_identifiers):
                    
                    iterated_elem_cnt += 1

                    if node['box'] not in unique_elems[img_file]:
                        unique_elems[img_file].append(node['box'])

                    x1,y1,x2,y2 = node['box']
                    x1,y1,x2,y2 = list(map(lambda x: round(x * resizing_ratio), [x1,y1-h_start,x2,y2-h_start]))
                    node['box'] = [x1,y1,x2,y2]
                    
                    if DRAW:
                        loc = f'{x1/sc_w:.2f}, {y1/sc_h:.2f} {x2/sc_w:.2f}, {y2/sc_h:.2f}'
                        print(loc, node['role']['value'], node['name']['value'])
                        cv2.rectangle(screenshot, (x1,y1), (x2,y2), (0,0,255) if node['name']['value'].strip() == '' else(0,255,0) , 2)
                    
                        cv2.imwrite("test.png", screenshot)
                        1+1
                    
                    if not SKIP_CHECKING:
                        # remove duplicates
                        if node['box'] in used_boxes:
                            invalid_elems[DUPLICATE_ELEMEMNT].add(sample_identifier)
                            continue
                        
                        # remove nodes outside of the screenshot
                        if x1 < 0 or y1 < 0 or x2 >= sc_w or y2 >= sc_h:
                            invalid_elems[INVALID_ELEM_BOX].add(sample_identifier)
                            continue

                        node_w, node_h = x2 - x1, y2 - y1

                        # remove tiny elements
                        if node_w / sc_w <= 0.005 or node_h / sc_h <= 0.005:
                            invalid_elems[TOO_SMALL_ELEMENT].add(sample_identifier)
                            continue

                        if node_w * node_h / (sc_w*sc_h) >= 0.65:
                            invalid_elems[OVERSIZED_ELEMENT].add(sample_identifier)
                            continue

                        #  detect extreme aspect ratio
                        if node_w / node_h < 0.05 or node_w / node_h > 20:
                            invalid_elems[EXTREME_ASPECT_RATIO].add(sample_identifier)
                            continue
                    
                        # if len(node['name']['value'].strip()) == 0:
                        #     node_invalid_types['no meaningful textual or resource annotation'] += 1
                        #     continue
                        
                        # detect unsuccessfully displayed nodes
                        if is_pure_color(screenshot, [x1,y1,x2,y2]):
                            invalid_elems[BLANK_ELEM].add(sample_identifier)
                            continue
                            if False:
                                cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.imwrite("test.png", screenshot)
                                1+1

                    used_boxes.append(node['box'])
                    valid_nodes_in_block.append(node)
                
                if len(valid_nodes_in_block) == 0:
                    continue

                # 筛选出留在视窗内的元素
                if DRAW:
                    icons_in_block = num_icons_each_block[block_idx_with_most_icons]['icon_nodes']            
                    for node in valid_nodes_in_block:
                        if 'backendDOMNodeId' not in node:
                            continue

                        bbox = bboxes[str(node['backendDOMNodeId'])]
                        if bbox is None:
                            continue

                        x1,y1,x2,y2 = node['box']
                        x1,y1,x2,y2 = list(map(lambda x: round(x * resizing_ratio), [x1,y1,x2,y2]))
                        y1 -= h_start; y2 -= h_start

                        loc = f'{x1/sc_w:.2f}, {y1/sc_h:.2f} {x2/sc_w:.2f}, {y2/sc_h:.2f}'
                        print(loc, node['role']['value'], node['name']['value'])
                        cv2.rectangle(screenshot, (x1,y1), (x2,y2), (0,0,255) if node['name']['value'].strip() == '' else(0,200,255) , 3)
                    
                        cv2.imwrite("test.png", screenshot)
                        1+1

                        
                # start collecting data
                all_node_names = [node['name']['value'].strip() for node in valid_nodes_in_block]
                all_boxes, all_node_box_strs, all_node_roles = [], [], []

                for node_name, node in zip(all_node_names, valid_nodes_in_block):
                    x1,y1, x2,y2 = node['box']
                    all_boxes.append(node['box'])
                    node_role = node['role']['value']
                    all_node_roles.append(node_role)
                    node_area = (x2-x1) * (y2-y1)
                    
                    normalized_box = [round(x1/sc_w*SCALE),round(y1/sc_h*SCALE),round(x2/sc_w*SCALE),round(y2/sc_h*SCALE)]

                    normalized_box = list(map(lambda p: max(0, min(SCALE-1, p)), normalized_box))

                    center_x, center_y = (x1+x2)/2, (y1+y2)/2
                    normalized_center = (round(center_x/sc_w*SCALE), round(center_y/sc_h*SCALE))
                    normalized_center = list(map(lambda p: max(0, min(SCALE-1, p)), normalized_center))
                    
                    with_box = random.random() <= PROB_BOX
                    box_str = f'({normalized_box[0]},{normalized_box[1]},{normalized_box[2]},{normalized_box[3]})'
                    all_node_box_strs.append(box_str)

                    loc = box_str if with_box else f'({normalized_center[0]},{normalized_center[1]})'
                    
                    is_unique = all_node_names.count(node_name) <= 1

                    if node_role != 'StaticText' and len(node_name.strip()):
                        if (ICONGND or ICONREF) and node_area / (sc_h*sc_w) <= 0.01:
                            if ICONGND and is_unique:
                                sample = make_icongnd_sample(task_id=f'autogui_webui_icongnd_{icongnd_cnt}', icon_desc=node_name, loc=loc, with_box=with_box)
                                sample['unnormalized_box'], sample['task_attr'], sample['image'], sample['wxh'] = node['box'], node_name, short_screenshot_name, f"{sc_w}x{sc_h}"

                                samples.append(sample); icongnd_cnt += 1
                        
                            if ICONREF:
                                sample = make_iconref_sample(task_id=f'autogui_webui_iconref_{iconref_cnt}', icon_desc=node_name, loc=loc, with_box=with_box)
                                sample['unnormalized_box'], sample['task_attr'], sample['image'], sample['wxh'] = node['box'], loc, short_screenshot_name, f"{sc_w}x{sc_h}"

                                samples.append(sample); iconref_cnt += 1
                        else:
                            if ELEMGND and is_unique:
                                if USE_ACTION_PROMPT:
                                    action = CLICK_TEMPLATE.format(target_x=normalized_center[0],target_y=normalized_center[1])
                                    sample = make_actionplanning_sample(task_id=f'autogui_{DATASET_NAME}_elemgnd_{len(samples)}', global_task=TURN_GND_INTO_PLANNING_PROMPT.format(instruc=node_name), gt_action=action, history='None', prompt_format_type='aguvis')
                                else:
                                    sample = make_elemgnd_sample(task_id=f'autogui_webui_elemgnd_{elemgnd_cnt}', elem_desc=node_name, loc=loc, with_box=with_box)

                                sample['unnormalized_box'], sample['task_attr'], sample['image'], sample['wxh'] = node['box'], node_name, short_screenshot_name, f"{sc_w}x{sc_h}"
                                samples.append(sample); elemgnd_cnt += 1
                            
                            if ELEMREF:
                                sample = make_elemref_sample(task_id=f'autogui_webui_elemref_{elemref_cnt}', elem_desc=node_name, loc=loc, with_box=with_box)
                                sample['unnormalized_box'], sample['task_attr'], sample['image'], sample['wxh'] = node['box'], loc, short_screenshot_name, f"{sc_w}x{sc_h}"
                                samples.append(sample); elemref_cnt += 1
                        
                        if INTENTGND and is_unique:
                            intent = gen_naive_action_gnd_anno(node_name, node_role, normalized_center, scale=SCALE)
                            sample = make_intentgnd_sample(task_id=f'autogui_webui_intentgnd_{intentgnd_cnt}', intent=intent, loc=loc, with_box=with_box)
                            sample['unnormalized_box'], sample['task_attr'], sample['image'], sample['wxh'] = node['box'], intent, short_screenshot_name, f"{sc_w}x{sc_h}"
                            samples.append(sample); intentgnd_cnt += 1

                    if ELEMCLASS:
                        sample = make_elemclass_sample(task_id='autogui_webui_elemclass_', elem_cls=node_role, loc=loc, with_box=with_box)
                        sample['unnormalized_box'], sample['task_attr'], sample['image'], sample['wxh'] = node['box'], node_role, short_screenshot_name, f"{sc_w}x{sc_h}"
                        elemclass_stats[node_role].append(sample)
                    
                if WIDGETLIST and 2 <= len(all_node_names) <= 80:
                    node_texts_boxes = []
                    for node_role, node_name, unnormalized_box, box_str in zip(all_node_roles, all_node_names, all_boxes, all_node_box_strs):
                        normalized_box = [round(unnormalized_box[0] / W * SCALE), round(unnormalized_box[1] / H * SCALE), round(unnormalized_box[2] / W * SCALE), round(unnormalized_box[3] / H * SCALE)]

                        node_texts_boxes.append([f"{node_role} '{node_name}' {box_str}", unnormalized_box])


                    node_texts_boxes.sort(key=lambda x: (x[1][1]+x[1][3], x[1][0]+x[1][2]))
                    elem_list_str = '\n'.join(f"{i} {x[0]}" for i, x in enumerate(node_texts_boxes))
                    
                    sample = make_widgetlist_sample(task_id=f'autogui_webui_widgetlist_{widgetlist_cnt}', elem_list=elem_list_str)
                    sample['unnormalized_box'], sample['image'], sample['wxh'] = [x[1] for x in node_texts_boxes], short_screenshot_name, f"{sc_w}x{sc_h}"
                    samples.append(sample); widgetlist_cnt += 1

                valid_img_cnt += 1

            except Exception as e:
                errors.append([axtree_path, traceback.format_exc()])

        if sample_idx % 10000 == 0 or sample_idx == len(sample_dirs) - 1:
            with open(invalid_elem_record_file, 'w') as f:
                json.dump({k:list(v) for k,v in invalid_elems.items()}, f, indent=2)
                    
    # 平衡elem分类任务的类别
    if ELEMCLASS:
        final_elemcls_cnts = [len(v) for v in elemclass_stats.values()]
        num_sample_each_cls = round(np.percentile(final_elemcls_cnts, 75))
        num_valid_imgs = len([k for k,v in unique_elems.items() if len(v)])

        new_elemcls_samples, stats_after_rebal = [], defaultdict(int)
        for elemcls, samples_eachcls in elemclass_stats.items():
            if len(samples_eachcls) > num_sample_each_cls:
                new_elemcls_samples.extend(random.sample(samples_eachcls, num_sample_each_cls))
            else:
                samples_eachcls = samples_eachcls * int(num_sample_each_cls // len(samples_eachcls)) + random.sample(samples_eachcls, num_sample_each_cls % len(samples_eachcls))
                new_elemcls_samples.extend(samples_eachcls)
            
        for i, sample in enumerate(new_elemcls_samples):
            sample['id'] += str(i)
            stats_after_rebal[sample['conversations'][1]['value']] += 1
        
        elemclass_cnt = len(new_elemcls_samples)
        samples.extend(new_elemcls_samples)
    else: stats_after_rebal = []

    # 最后统计数据
    num_invalid_elem = sum(len(v) for v in invalid_elems.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k,v in unique_elems.items() if len(v)])

    report = f"#Samples: {len(samples)}\n#Unique elements: {num_unique_elems}\n#Valid unique image: {num_valid_imgs}\n#All unique images {len(unique_elems)}\nInvalid elem ratio: {num_invalid_elem} / {iterated_elem_cnt} = {(num_invalid_elem/iterated_elem_cnt if iterated_elem_cnt > 0 else 0.0):.2f}\ntext_loc_cnt: {text_loc_cnt} | ocr_cnt: {ocr_cnt} | icongnd_cnt: {icongnd_cnt} | iconref_cnt: {iconref_cnt} | elemgnd_cnt: {elemgnd_cnt} | elemref_cnt: {elemref_cnt} | elemclass_cnt: {elemclass_cnt} | intentgnd_cnt: {intentgnd_cnt} | widgetlist_cnt: {widgetlist_cnt}"
    print(report)
    
    save_file = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_scale{SCALE}_{len(samples)//1000}k{'_actformat' if USE_ACTION_PROMPT else ''}.json")

    with open(save_file.replace('.json', '_info.json'), "w") as f:
        json.dump({'num_samples': len(samples), '#valid_unique_images': num_valid_imgs, '#all_unique_images': len(unique_elems), '#num_unique_elems': num_unique_elems, '#invalid_elems': num_invalid_elem, '#all_elems': iterated_elem_cnt, '#invalid_elem_ratio': num_invalid_elem/iterated_elem_cnt, 'text_loc_cnt': text_loc_cnt, 'ocr_cnt': ocr_cnt, 'icongnd_cnt': icongnd_cnt, 'iconref_cnt': iconref_cnt, 'elemclass_cnt': elemclass_cnt, 'intentgnd_cnt': intentgnd_cnt, 'widgetlist_cnt': widgetlist_cnt, 'report': report, 'elem_class_stats_before_rebalance': {k:len(v) for k,v in elemclass_stats.items()}, 'num_sample_each_elem_class_after_rebalance': stats_after_rebal, 'errors': errors}, f, indent=2)
    
    with open(save_file.replace('.json', '_sample.json'), "w") as f:
        json.dump(random.sample(samples, 160), f, indent=2)

    with open(save_file, "w") as f:
        json.dump(samples, f)
     
make_webui_data()