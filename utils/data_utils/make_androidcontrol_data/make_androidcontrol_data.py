"""
AndroidControl dataset conversion and sample generation utilities.

This module reads AndroidControl TFRecord episodes, decodes screenshots and
accessibility trees, filters/validates UI elements, and generates model-ready
samples for multiple tasks, including:

- Action planning (high-level and high+low-level variants)
- Intent grounding (optional)

The code favors correctness and traceability: each step keeps track of the
current UI, action metadata, and lightweight statistics for reporting.

Note: The original codebase includes domain-specific templates and utilities
imported from `task_prompt_lib` and `misc`. Those symbols are treated as
black-box utilities here and should remain unchanged.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import json, os, random, cv2, re, tensorflow as tf
from tqdm import tqdm
from android_env.proto.a11y import android_accessibility_forest_pb2
import numpy as np
import ast
import xml.etree.ElementTree as ET

from collections import defaultdict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *
from misc import (
    parse_axtrees_proto,
    resize_image,
    find_smallest_box_containing_point,
    is_pure_color,
    box2center,
    random_substring,
    revise_swipe_action,
    keep_unique_actions,
    lower_first_letter,
)

# Treat very short windows as status bars; skip strict validation for them
MAX_STATUSBAR_HEIGHT = 130

def generate_negative_action_plans(
    gt_act_type: str,
    W: int,
    H: int,
    scale: int,
    gt_center: Optional[Sequence[float]] = None,
    neg_boxes: Optional[Sequence[Sequence[int]]] = None,
    direction: str = '',
    text: str = '',
    goal_status: str = ''
) -> List[List[List[str]]]:
    """Generate negative action plans for contrastive/evaluation purposes.

    Arguments:
        gt_act_type: Ground-truth action type, e.g., 'click', 'scroll', etc.
        W, H: Width and height of the current UI image.
        scale: Normalization scale for coordinates.
        gt_center: Optional ground-truth (x, y) center in original pixel space.
        neg_boxes: Optional sequence of negative candidate boxes.
        direction: Scroll direction when the ground-truth action is 'scroll'.
        text: Text content associated with input/open-app actions.
        goal_status: Goal status string for 'status' actions.

    Returns:
        A list of negative action plan lists. Each inner list contains pairs of
        [reason, action_json_like_string].
    """

    neg_action_plans: List[List[List[str]]] = []

    # Normalize optional inputs to avoid None checks downstream
    if neg_boxes is None:
        neg_boxes = []

    neg_actions_other_types: List[List[str]] = []

    # If the GT action is a touch action, generate negatives using other element boxes
    if len(neg_boxes) >= 1:
        for act_type, template in zip(['click', 'long_press'], [CLICK_TEMPLATE, LONG_PRESS_TEMPLATE]):
            if gt_act_type == act_type:
                # Generate negative targets by sampling other boxes
                neg_cands: List[List[str]] = []
                selected_idxs = random.sample(list(range(len(neg_boxes))), min(len(neg_boxes), 9))
                for idx in selected_idxs:
                    box = neg_boxes[idx]
                    normalized_center = box2center(box, W, H, scale)
                    neg_click = template.format(target_x=normalized_center[0], target_y=normalized_center[1])
                    neg_cands.append([INCORRECT_CLICK_TARGET.format(action=act_type), neg_click])

                random.shuffle(neg_cands)
                neg_action_plans.append(neg_cands)
            else:
                # Keep the target location but switch the touch mode (click vs long_press)
                if gt_center is not None:
                    normalized_center = [
                        max(0, min(scale, round(gt_center[0] / W * scale))),
                        max(0, min(scale, round(gt_center[1] / H * scale)))
                    ]
                else:
                    normalized_center = box2center(random.choice(neg_boxes), W, H, scale)

                neg_act = template.format(target_x=normalized_center[0], target_y=normalized_center[1])
                if gt_act_type in ['click', 'long_press']:
                    incorr_reason = INCORRECT_TOUCH_MODE + f". Should {gt_act_type} instead of {act_type}"
                else:
                    incorr_reason = INCORRECT_ACTION

                neg_actions_other_types.append([incorr_reason, neg_act])

    # For scroll, generate negatives for directions different from the GT direction
    if gt_act_type == 'scroll':
        neg_cands = []
        for cand_dir in ['up', 'down', 'left', 'right']:
            if direction == cand_dir:
                continue
            swipe_dir, start, end = format_swiping_dual_points(cand_dir, scale=scale, scroll2swipe=False)
            neg_swipe = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=swipe_dir, distance="medium")
            neg_cands.append([INCORRECT_SWIPE_DIRECTION + f'. Should swipe {swipe_dir} instead of {cand_dir}', neg_swipe])

        random.shuffle(neg_cands)
        neg_action_plans.append(neg_cands)
    else:
        # If GT is not scroll, mix in a random swipe negative
        rand_dir = random.choice(['up', 'down', 'left', 'right'])
        swipe_dir, start, end = format_swiping_dual_points(rand_dir, scale=scale, scroll2swipe=False)
        neg_swipe = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=swipe_dir, distance="medium")
        neg_actions_other_types.append([INCORRECT_ACTION, neg_swipe])

    # If GT is input_text, generate character-level variant negatives
    if gt_act_type == 'input_text':
        neg_cands = []
        used = [text]
        for _ in range(9):
            rand_text = random_substring(text)
            if rand_text in used:
                continue
            used.append(rand_text)
            neg_type = INPUT_TEMPLATE.format(text=rand_text)
            neg_cands.append([INCORRECT_INPUT_TEXT + f'. Should be "{text}" instead of "{rand_text}"', neg_type])

    # Add an open_app negative candidate
    app = random.choice([
        "Facebook", "Instagram", "WhatsApp", "TikTok", "Snapchat",
        "YouTube", "Twitter", "Spotify", "Netflix", "Zoom",
        "Google Maps", "Amazon", "Gmail", "Pinterest", "Reddit",
        "LinkedIn", "Telegram", "Discord", "Uber", "Airbnb"
    ])
    neg_open_app = OPEN_APP_TEMPLATE.format(app_name=app)
    incorr_reason = INCORRECT_OPEN_APP + f'. Should be {text} instead of {app}.' if gt_act_type == 'open_app' else INCORRECT_ACTION
    neg_actions_other_types.append([incorr_reason, neg_open_app])

    # Navigation negatives
    if gt_act_type != 'navigate_back':
        neg_back = NAVIGATE_BACK_TEMPLATE
        incorr_reason = INCORRECT_NAVIGATION_ACTION + '. Should navigate home instead of navigate back.' if 'home' in gt_act_type else INCORRECT_ACTION
        neg_actions_other_types.append([incorr_reason, neg_back])

    if gt_act_type != 'navigate_home':
        neg_home = NAVIGATE_HOME_TEMPLATE
        incorr_reason = INCORRECT_NAVIGATION_ACTION + '. Should navigate back instead of navigate home.' if 'back' in gt_act_type else INCORRECT_ACTION
        neg_actions_other_types.append([incorr_reason, neg_home])

    # Status negatives
    if gt_act_type == 'status':
        if goal_status == 'successful':
            neg_status = 'infeasible'
        elif goal_status == 'infeasible':
            neg_status = 'successful'
        else:
            neg_status = random.choice(['successful', 'infeasible'])
        incorr_reason = INCORRECT_STATUS
    else:
        neg_status = random.choice(['successful', 'infeasible'])
        incorr_reason = INCORRECT_ACTION

    neg_status = STATUS_TEMPLATE.format(goal_status=neg_status, answer='')
    neg_actions_other_types.append([incorr_reason, neg_status])
    random.shuffle(neg_actions_other_types)
    neg_action_plans.append(neg_actions_other_types)

    return neg_action_plans


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

SPLIT = ['train', 'test'][0]

RESUME = True

DEBUG = False    

PLANNING = True; ADD_DUMMY_COMPLETE = True
INTENTGND = False; POINT_FORMAT = ['plain', 'qwen'][1]

SCALE = 1000
LONGEST = 1008

RETAIN_STATUSBAR_IN_XML = True

DISCARD_OPENAPP = False

USE_ACTION_REFEXP = False # <|object_ref_start|>Right mouse click on the element that displays detailed information about the 'TORALD' product, allowing users to view its specifications, read reviews, and perform actions like sharing or adding the product to their favorites list or bag<|object_ref_end|>\n{\"action_type\": \"right_click\", \"target\": (501,647)}

DEVICE_TAG = "Android"

DRAW = False
planning_cnt = 0

ANDROIDCONTROL_ROOT = ["/mnt/vdb1/hongxin_li/AndroidControl/raw", "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/AndroidControl_raw"][0]

with open(os.path.join(ANDROIDCONTROL_ROOT, "splits.json"), "r") as f:
    splits = json.load(f)

filenames = tf.io.gfile.glob(os.path.join(ANDROIDCONTROL_ROOT, 'android_control*'))
dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

apps = {k:set() for k in splits.keys()}

DATASET_NAME = "AndroidControl" + ("_debug" if DEBUG else "") + ("_test" if SPLIT == 'test' else "")

SAVE_DIR = f"/mnt/jfs/copilot/lhx/ui_data/{DATASET_NAME}"
print(f"save imgs to {SAVE_DIR}")

save_invalidcache_to = os.path.join(SAVE_DIR, "invalid_cache")
#if os.path.exists(save_invalidcache_to): shutil.rmtree(save_invalidcache_to)
os.makedirs(save_invalidcache_to, exist_ok=True)

samples = []
open_app_indices  = set()

avg_iou_list = []
invalid_UIs = []

total_ep_cnt, valid_ep_cnt = 0, 0
total_step_cnt, valid_step_cnt = 0, 0
valid_elem_cnt = 0
node_invalid_types = defaultdict(int)

intentgnd_cnt = 0

for idx, d in tqdm(enumerate(dataset), total=15283, desc=f'fetching androidcontrol...'):
    if DEBUG and idx % 500 >= 1: continue
    ex = tf.train.Example.FromString(d)
    ep_id = ex.features.feature['episode_id'].int64_list.value[0]
    
    if ep_id not in splits[SPLIT]:
        continue
    total_ep_cnt += 1

    # task
    task = ex.features.feature['goal'].bytes_list.value[0].decode().replace(" .", ".").replace(" ,", ",")
    # Extract AXTrees and decode screenshots
    raw_AXTrees = ex.features.feature['accessibility_trees'].bytes_list.value
    screenshots = ex.features.feature['screenshots'].bytes_list.value
    actions = [x.decode() for x in ex.features.feature['actions'].bytes_list.value]
    step_instructions = [x.decode() for x in ex.features.feature['step_instructions'].bytes_list.value]
    total_step_cnt += len(step_instructions)
    # As opening-app actions may be repeated, we need to dedup
    new_actions, new_step_instructions, remained_indices = [], [], []
    is_openapp = False

    for step_idx, (act, inst) in enumerate(zip(actions, step_instructions)):
        if 'open_app' not in act or not is_openapp:
            new_actions.append(act)
            new_step_instructions.append(inst)
            remained_indices.append(step_idx)
            is_openapp = 'open_app' in act

    save_img_to = None

    # Collect indices of open_app actions for optional analytics
    for step_idx, act in enumerate(actions):
        if 'open_app' in act:
            open_app_indices.add(step_idx)

    action_meta = [] # 保存动作轨迹，用以标注功能
    valid_sc_list = []

    # Cache a random waiting instruction to keep this episode consistent
    WAITING_STEP_INSTRUC = random.choice(WAIT_INSTRUC)

    for step_idx, (raw_AXTree, screenshot) in enumerate(zip(raw_AXTrees, screenshots)):
        # 解析Proto格式的AXtree，并提取App名称
        forest = android_accessibility_forest_pb2.AndroidAccessibilityForest().FromString(raw_AXTree)
        total_step_cnt += 1
        action_meta.append({"action_type":"meaningless action"})

        if step_idx == 0:
            app_name = list(forest.windows[0].tree.nodes)[0].package_name
            save_img_to = os.path.join(SAVE_DIR, 'images', app_name, str(ep_id))
            os.makedirs(save_img_to, exist_ok=True)

        # 解析UI图像
        img = cv2.imdecode(np.frombuffer(screenshot, dtype=np.uint8), cv2.IMREAD_COLOR)
        ORIG_H, ORIG_W = img.shape[:2]
        
        img, ratio = resize_image(img, LONGEST)
        new_img_path = os.path.join(save_img_to, f'{step_idx}.png')
        short_img_path = new_img_path[new_img_path.find(DATASET_NAME):]
        valid_sc_list.append(short_img_path)
        if not (RESUME and os.path.exists(new_img_path)):
            cv2.imwrite(new_img_path, img)

        H, W = img.shape[:2]

        # 解析AXtree成XML格式，并提取出结点关键状态
        try:
            tree_str_lst, node_key_attrs_lst = parse_axtrees_proto(forest.windows, ratio) # axtree_str 为 xml格式，node_key_attrs 包括  [{'class': node_cls, 'box': [x1, y1, x2, y2], 'node_desc': node_desc, 'is_leaf': is_leaf, 'is_interactable': is_interactable}]
        except Exception as e:
            print(e)
            invalid_UIs.append([ep_id, step_idx, new_img_path, str(e)])
            continue

        # 剔除越界或过大过小元素
        # windows有如下属性['bounds_in_screen', 'display_id', 'id', 'layer', 'title', 'window_type', 'is_accessibility_focused', 'is_active', 'is_focused', 'is_in_picture_in_picture_mode', 'tree']
        new_node_key_attrs_lst = []; checked_cnt = 0; invalid_elem_cnt = 0
        for win_idx, node_key_attrs in enumerate(node_key_attrs_lst):
            window_H, window_W = forest.windows[win_idx].bounds_in_screen.bottom - forest.windows[win_idx].bounds_in_screen.top, forest.windows[win_idx].bounds_in_screen.right - forest.windows[win_idx].bounds_in_screen.left
            window_H = round(window_H * ratio); window_W = round(window_W * ratio)
            # Skip strict checks for status bar windows
            if window_H <= MAX_STATUSBAR_HEIGHT:
                new_node_key_attrs_lst.append(node_key_attrs)
                continue

            new_node_key_attrs, used_boxes = [], []

            checked_cnt += len(node_key_attrs)
            for node_key_attr in node_key_attrs:
                is_invalid = False
                if node_key_attr['box'] in used_boxes:
                    node_invalid_types['duplicate box'] += 1; is_invalid = True
                    continue

                x1, y1, x2, y2 = node_key_attr['box']
                if x1 < 0 or x2 > W or y1 < 0 or y2 > H or x1 >= x2 or y1 >= y2:
                    node_invalid_types['invalid box coordinates'] += 1; is_invalid = True
                    continue
                if (x2 - x1) / W <= 0.005 or (y2 - y1) / H <= 0.005:
                    node_invalid_types['too small element'] += 1; is_invalid = True
                    continue
                if (x2-x1) * (y2-y1) / (window_H*window_W) >= 0.65:
                    node_invalid_types['oversize element'] += 1; is_invalid = True
                    continue
                
                if len(node_key_attr['node_desc']) == 0 and len(node_key_attr['resource_id']) == 0:
                    node_invalid_types['no meaningful textual or resource annotation'] += 1; is_invalid = True
                    continue

                # detect unsuccessfully displayed nodes
                if is_pure_color(img, [x1,y1,x2,y2]):
                    node_invalid_types['element not displayed'] += 1; is_invalid = True; invalid_elem_cnt+=1
                    continue

                used_boxes.append(node_key_attr['box'])
                if is_invalid: node_key_attr['invalid'] = True

                new_node_key_attrs.append(node_key_attr)
            new_node_key_attrs_lst.append(new_node_key_attrs)

        valid_elem_cnt_this_step = sum(len(v) for v in new_node_key_attrs_lst)

        # node_key_attrs_lst = new_node_key_attrs_lst

        # 如果无效元素过多，直接放弃标注当前UI界面
        if invalid_elem_cnt >= 10:
            node_invalid_types['others'] += checked_cnt - invalid_elem_cnt
            invalid_UIs.append([ep_id, step_idx, new_img_path, 'Discard the traj due to too many invalid elements', f'invalid_elem_cnt:{invalid_elem_cnt}'])
            #print(f"{invalid_UIs} invalid UIs")

            for node_info in new_node_key_attrs_lst[-1]:
                inter_color, noninter_color = (0,random.randint(150,250), random.randint(150,250)), (random.randint(150,250), random.randint(150,250), 0)
                elem_cls, elem_box, elem_desc, is_leaf, is_interactable = node_info['class'], node_info['box'], node_info['node_desc'], node_info['is_leaf'], node_info['is_interactable']
                # if len(node_info['node_desc']) == 0 and len(node_info['resource_id']) == 0: continue
                if is_interactable:
                    cv2.rectangle(img, (round(elem_box[0]), round(elem_box[1])), (round(elem_box[2]), round(elem_box[3])), inter_color, 2)
                else:
                    cv2.rectangle(img, (elem_box[0], elem_box[1]), (elem_box[2], elem_box[3]), noninter_color, 2)
                # print(node_info)
                
            # cv2.imwrite(os.path.join(save_invalidcache_to, f'{len(os.listdir(save_invalidcache_to)):04d}' + '-'.join(new_img_path.split('/')[-3:])), img)
            # 1+1
            continue

        # 检查元素是否覆盖率过高
        # all_boxes = np.array([x['box'] for x in node_key_attrs_lst[0]])
        # avg_iou = average_iou(all_boxes)
        # avg_iou_list.append([avg_iou, new_img_path])

        # 这里只保存UI主界面AXTree，丢弃头部状态栏。用于测试axtree能否被lxml解析: dom_tree = ET.fromstring(tree_str)
        # 由于一个UI有可能有超过2个windows，这里不可默认第二个window是打开的App
        save_xml_to = os.path.join(save_img_to, f'{step_idx}.xml')
        if not (RESUME and os.path.exists(save_xml_to)):
            tree_str_list_wo_statusbar = []
            for win_idx, tree_str in enumerate(tree_str_lst):
                if (not RETAIN_STATUSBAR_IN_XML and forest.windows[win_idx].window_type != 1) or len(tree_str.strip()) == 0:
                    continue
                tree_str_list_wo_statusbar.append(tree_str)

            if len(tree_str_list_wo_statusbar) == 0:
                node_invalid_types['UIs without main app windows'] += valid_elem_cnt_this_step
                continue
            
            windows_tree_str = []
            for win_idx, tree in enumerate(tree_str_list_wo_statusbar, start=1):
                window_tag = f'window_{win_idx}' + ('_statusbar' if forest.windows[win_idx-1].window_type != 1 else '_main-app')
                
                windows_tree_str.append(f'<{window_tag}>\n{tree}\n</{window_tag}>')

            tree_str = f"<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>\n" + '<hierarchy>\n' + '\n'.join(windows_tree_str) + '</hierarchy>'

            with open(save_xml_to, 'w') as f:
                f.write(tree_str)


        # 检查两棵axtree是否是状态栏和主页面
        if False:
            # tree 1
            # for node_info in node_key_attrs_lst[0]:
            #     inter_color, noninter_color = (random.randint(0,150),0, random.randint(0,150)), (random.randint(0,150), random.randint(0,150), 0)
            #     elem_cls, elem_box, elem_desc, is_leaf, is_interactable, text_key = node_info['class'], node_info['box'], node_info['node_desc'], node_info['is_leaf'], node_info['is_interactable'], node_info['text_key']
            #     if len(node_info['node_desc']) == 0: continue
            #     if is_interactable:
            #         cv2.rectangle(screenshot, (elem_box[0], elem_box[1]), (elem_box[2], elem_box[3]), inter_color, 2)
            #     else:
            #         cv2.rectangle(screenshot, (elem_box[0], elem_box[1]), (elem_box[2], elem_box[3]), noninter_color, 2)
            #     print(node_info)
            #     cv2.imwrite("test.png", screenshot)
            #     1+1
            # tree 2
            
            img = cv2.imread(new_img_path)
            for node_info in new_node_key_attrs_lst[-1]:
                elem_cls, elem_box, elem_desc, is_leaf, is_interactable = node_info['class'], node_info['box'], node_info['node_desc'], node_info['is_leaf'], node_info['is_interactable']
                #if len(node_info['node_desc']) == 0 and len(node_info['resource_id']) == 0: continue
                if 'is_invalid' in node_info: color = (0,0,255)
                else: color = (0,random.randint(150,250), random.randint(150,250))  if is_interactable else (random.randint(150,250), random.randint(150,250), 0) 

                cv2.rectangle(img, (round(elem_box[0]), round(elem_box[1])), (round(elem_box[2]), round(elem_box[3])), color, 2)

                print(node_info)
                cv2.imwrite("test.png", img)
                1+1

        # 跳过历史步骤涉及open_app的样本
        # opena_app出现位置举例
        # 0 = '{"action_type":"wait"}'
        # 1 = '{"action_type":"open_app","app_name":"Google Play Books"}'

        # 0 = '{"action_type":"navigate_home"}'
        # 1 = '{"action_type":"open_app","app_name":"BudgetBytes"}'

        # 连续尝试打开apps
        # open expedia app
        # open expedia app
        # open expedia app
        # click on the first result
        #
        # 再次打开app
        # Open the digital timer app.
        # Tap on the timer icon present in the bottom-right corner of the screen.
        # Tap on the timer icon present in the bottom-right corner of the screen.
        # Tap on the timer icon present in the bottom-right corner of the screen.
        # Open the digital timer app.
        # Tap on the timer icon present in the bottom-right corner of the screen.
        # Tap on the timer icon present in the bottom-right corner of the screen.

        # 跨app操作
        # Click on the ok button.
        # Click on the three dots button.
        # Click on the share button.
        # Click on the copy link.
        # Now open the gmail button. <-
        # Click on the compose button.
        # Enter the email of the friend.
        # Select the given option.
        # Click on the send button.

        valid_elem_cnt += valid_elem_cnt_this_step

        click_action_added = False
        if (PLANNING or INTENTGND) and step_idx in remained_indices and valid_elem_cnt_this_step:
            cur_action = ast.literal_eval(actions[step_idx])
            neg_actions = None

            if SPLIT == 'train' and DISCARD_OPENAPP and 'open_app' in actions[step_idx]:
                action_meta[-1] = cur_action
            else:
                action_type = cur_action['action_type']
                
                if INTENTGND and action_type not in ['click', 'long_press']:
                    continue

                main_app_elems = []
                for window_elems_lst in new_node_key_attrs_lst:
                    main_app_elems.extend(window_elems_lst)

                # Get bboxes
                bboxes = [x['box'] for x in main_app_elems]

                if action_type in ['click', 'long_press']: # Example: '{"action_type":"click","x":313,"y":708}'
                    click_x, click_y = int(cur_action["x"]) * ratio, int(cur_action["y"]) * ratio
                    if SCALE == -1: # unnormalized
                        normalized_center = [round(click_x), round(click_y)]
                    else:
                        normalized_center = [max(0, min(SCALE-1, round(click_x / W * SCALE))), max(0, min(SCALE-1, round(click_y / H * SCALE)))]

                    target = f'({normalized_center[0]},{normalized_center[1]})'

                    if action_type == 'click':
                        action_str = CLICK_TEMPLATE.format(target_x=normalized_center[0],target_y=normalized_center[1]) # f'{{"action_type":"{action_type}", "target":{target}}}'
                    else:
                        action_str = LONG_PRESS_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1]) # f'{{"action_type":"{action_type}", "target":{target}}}'
                    # 由于标注信息里没有给交互元素，这里需要找到包围交互点的最小元素

                    interacted_box, index = find_smallest_box_containing_point(np.array([click_x, click_y]), boxes=np.array(bboxes))

                    if index is None:
                        print(f"Bad bbox: cannot locate the bbox")
                        continue
                    elif not (interacted_box[0] <= click_x <= interacted_box[2] and interacted_box[1] <= click_y <= interacted_box[3]):
                        print(f"Bad bbox: [{click_x}, {click_y}] not in bbox {interacted_box}")
                        continue
                    else:
                        interacted_node = main_app_elems[index]

                        if DRAW:
                            print(step_instructions[step_idx], cur_action)
                            cv2.circle(img, (round(click_x), round(click_y)), 5, (0, 255, 0), 2)
                            cv2.circle(img, (round(click_x), round(click_y)), 3, (0, 255, 0), -3)
                            cv2.rectangle(img, (interacted_box[0], interacted_box[1]), (interacted_box[2], interacted_box[3]), (0, 0, 255), 3)
                            cv2.imwrite("test.png", img)
                            1+1
                        
                        if not ('Open' in step_instructions[step_idx] and 'app' in step_instructions[step_idx]) and not INTENTGND:
                            step_instructions[step_idx] = f'Click on "{interacted_node["node_desc"]}" {interacted_node["class"]}.'

                        # generate action refexp
                        action_refexp = random.choice(ACTION_PREFIXES[action_type]['specific']) + f' {interacted_node["class"]} "{interacted_node["node_desc"].strip()}"'

                        assert 0<=interacted_box[0]<=W+10 and 0<=interacted_box[2]<=W+10
                        action_meta[-1] = {"action_type": "DualPoint", "touch_point": [click_x, click_y], "lift_point": [click_x, click_y], "typed_text": f"TAP:{interacted_node['class']} Box:[{interacted_box[0]},{interacted_box[1]},{interacted_box[2]},{interacted_box[3]}]", "action_refexp": action_refexp}
                        click_action_added = True

                    # intentgnd
                    if INTENTGND and SPLIT != 'test':
                        intent = lower_first_letter(action_refexp if USE_ACTION_REFEXP else step_instructions[step_idx])
                        sample = make_intentgnd_sample(task_id=f"autogui_{DATASET_NAME}_intentgnd_{ep_id}-{step_idx}", intent=intent, loc=normalized_center, output_tag=WITHPOINT_TAG_LONG, point_format=POINT_FORMAT)
                        sample['task_attr'], sample['image'] = intent, short_img_path
                        samples.append(sample); intentgnd_cnt += 1   
                elif action_type == 'scroll': # Example: '{"action_type":"scroll","direction":"down"}' # scroll down 即 swipe up （浏览视窗以下的内容）
                    direction = cur_action['direction']
                    distance = random.choice(["medium","short","long"])

                    direction, start, end = format_swiping_dual_points(direction, scale=SCALE, scroll2swipe=True, distance=distance)
                    
                    action_str = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=direction, distance=distance) # f'{{"action_type":"swipe", "start":"({start[0]},{start[1]})", "direction":"{direction}", "distance":"{distance}"}}'
                    action_meta[-1] = ast.literal_eval(action_str)
                    step_instructions[step_idx] = revise_swipe_action(step_instructions[step_idx], f"Swipe {direction}")

                    action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction)


                elif action_type == 'navigate_back':
                    action_str = NAVIGATE_BACK_TEMPLATE
                    action_refexp = random.choice(NAVIGATE_BACK_PREFIXES)
                elif action_type == 'navigate_home':
                    action_str = NAVIGATE_HOME_TEMPLATE
                    action_refexp = random.choice(NAVIGATE_HOME_PREFIXES)
                elif action_type == 'wait':
                    # the step instruction for the waiting aciton is the same as this makes cleaning redundant previous steps easier.
                    step_instructions[step_idx] = WAITING_STEP_INSTRUC
                    action_str = WAIT_TEMPLATE
                    action_refexp = random.choice(WAIT_INSTRUC)
                elif action_type == 'input_text':
                    text = cur_action['text'].strip(' \\').replace("\n", "\\n").replace('"', '\\"')
                    action_str = INPUT_TEMPLATE.format(text=text)
                    action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")

                # No termination is provided in Android Control dataset. So we need to generate termination action for the last step.
                elif action_type == 'status':
                    action_str = STATUS_TEMPLATE.format(goal_status=cur_action['goal_status'], answer='')
                    action_refexp = random.choice(TASK_STATUS_SENTENCES[cur_action['goal_status']])

                elif action_type == 'open_app':
                    action_str = OPEN_APP_TEMPLATE.format(app_name=cur_action['app_name'])
                    action_refexp = random.choice(OPEN_APP_PREFIXES).format(app_name=cur_action['app_name'])

                else:
                    # '{"action_type":"input_text","text":"Paramedic news"}' | '{"action_type":"wait"}' | '{"action_type":"navigate_home"}' | '{"action_type":"navigate_back"}'
                    action_str = actions[step_idx]
                    action_meta[-1] = cur_action

                # 合并历史
                retained_idxs, retained_history = keep_unique_actions(step_instructions[:step_idx])
                history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(retained_history, start=max(1,len(retained_history) - MAX_PREV_ACT+1))) if len(retained_history) > 0 else 'None'

                action_type = ast.literal_eval(action_str)['action_type']
                
                task_attr = ast.literal_eval(actions[step_idx])
                if click_action_added:
                    task_attr['x'] *= ratio; task_attr['y'] *= ratio; 
                    task_attr['bbox'] = interacted_box #.tolist()

                if USE_ACTION_REFEXP:
                    action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"

                # sample_only_highlevel_task
                if PLANNING:
                    sample = make_actionplanning_sample(
                        task_id=f"autogui_androidcontrol_planning_{ep_id}-{step_idx}-H",
                        global_task=task,
                        history=history_str,
                        gt_action='Action: ' + action_str,
                        device_tag=DEVICE_TAG,
                    use_action_refexp=USE_ACTION_REFEXP
                    )
                    sample['ep_id'], sample['step_id'], sample['task'], sample['step_instruction'], sample['action_refexp'], sample["action_type"], sample["history"], sample["image"], sample["task_attr"], sample["wxh"], sample['orig_wxh'], sample['bboxes'], sample['#total_steps'] = ep_id, step_idx, task, step_instructions[step_idx], action_refexp, action_type, retained_history, short_img_path, task_attr, f"{W}x{H}", f"{ORIG_W}x{ORIG_H}", [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bboxes], len(raw_AXTrees) + ADD_DUMMY_COMPLETE

                    samples.append(sample)
                    planning_cnt += 1

                    # Debug the bboxes
                    if False:
                        img = cv2.imread(os.path.join(os.path.dirname(SAVE_DIR), sample["image"]))
                        for bbox in sample['bboxes']:
                            cv2.rectangle(img, (int(bbox[0] * W), int(bbox[1] * H)), (int(bbox[2] * W), int(bbox[3] * H)), (0, 0, 255), 2)
                        cv2.imwrite("test.png", img)

                    assert os.path.exists(os.path.join(os.path.dirname(SAVE_DIR), sample["image"]))
                

                    if step_idx < min(len(raw_AXTrees), len(screenshots)) - 1:
                        next_img_path = os.path.join(save_img_to, f'{step_idx+1}.png')
                    else: next_img_path = new_img_path
                    short_nextimg_path = next_img_path[next_img_path.find(DATASET_NAME):]

                    # sample_highlowlevel_task
                    # 由于step instruction不准，这里暂时不纳入prompt里包含step instruction的样本。后面用gpt-4o重新生成下

                    if not (step_idx > 1 and step_instructions[step_idx] == step_instructions[step_idx-1]):
                        sample = make_actionplanning_sample(task_id=f"autogui_androidcontrol_planning_{ep_id}-{step_idx}-HL", global_task=task, history=history_str, gt_action='Action: ' + action_str, step_instruction=f"The next step instruction: {step_instructions[step_idx]}\n", device_tag=DEVICE_TAG, use_action_refexp=USE_ACTION_REFEXP)
                        
                        sample['ep_id'], sample['step_id'], sample['task'], sample['step_instruction'], sample['action_refexp'], sample["action_type"], sample["history"], sample["image"], sample['next_image'], sample["task_attr"], sample["wxh"], sample['bboxes'], sample['#total_steps'] = ep_id, step_idx, task, step_instructions[step_idx], action_refexp, action_type, retained_history, short_img_path, short_nextimg_path, task_attr, f"{W}x{H}", [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bboxes], len(raw_AXTrees) + ADD_DUMMY_COMPLETE
                        samples.append(sample)
                        planning_cnt += 1
                    
                    valid_step_cnt += 1

    else:
        if PLANNING and step_idx == len(raw_AXTrees) - 1:
            # HL
            step_instruc_complete = "Task completed"
            last_step_history = retained_history + ([step_instructions[-1]] if len(step_instructions) else [])
            history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(last_step_history, start=max(1,len(last_step_history) - MAX_PREV_ACT+1))) if len(last_step_history) > 0 else 'None'
            action_str = STATUS_TEMPLATE.format(goal_status='successful', answer='')
            task_attr = eval(action_str)

            sample = make_actionplanning_sample(task_id=f"autogui_androidcontrol_planning_{ep_id}-{step_idx+1}-HL", global_task=task, history=history_str, gt_action=f'Action: {action_str}', step_instruction=f"The next step instruction: {step_instruc_complete}\n", device_tag=DEVICE_TAG, use_action_refexp=USE_ACTION_REFEXP)
            sample['ep_id'], sample['step_id'], sample['task'], sample['step_instruction'], sample['action_refexp'], sample["action_type"], sample["history"], sample["image"], sample['next_image'], sample["task_attr"], sample["wxh"], sample['bboxes'], sample['#total_steps'] = ep_id, step_idx+1, task, step_instruc_complete, action_refexp, 'status', last_step_history, short_img_path, short_nextimg_path, task_attr, f"{W}x{H}", [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bboxes], len(raw_AXTrees) + ADD_DUMMY_COMPLETE
            samples.append(sample)
            planning_cnt += 1

            # H
            sample = make_actionplanning_sample(task_id=f"autogui_androidcontrol_planning_{ep_id}-{step_idx+1}-H", global_task=task, history=history_str, gt_action=f'Action: {action_str}', device_tag=DEVICE_TAG, use_action_refexp=USE_ACTION_REFEXP)
            sample['ep_id'], sample['step_id'], sample['task'], sample['step_instruction'], sample['action_refexp'], sample["action_type"], sample["history"], sample["image"], sample['next_image'], sample["task_attr"], sample["wxh"], sample['bboxes'], sample['#total_steps'] = ep_id, step_idx+1, task, step_instruc_complete, action_refexp, 'status', last_step_history, short_img_path, short_nextimg_path, task_attr, f"{W}x{H}", [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bboxes], len(raw_AXTrees) + ADD_DUMMY_COMPLETE
            samples.append(sample)
            planning_cnt += 1

                    
    assert len(action_meta) == len(screenshots)

    meta_file = os.path.join(save_img_to, 'meta.json')
    if not (os.path.exists(meta_file) and RESUME):
        with open(os.path.join(save_img_to, 'meta.json'), 'w') as f:
            json.dump(action_meta, f) # 最后一个动作是冗余的，但是在annotate_android_func时会忽略掉，所以这里不做处理

    valid_ep_cnt += 1

iterated_elem_cnt = valid_elem_cnt + sum(v for v in node_invalid_types.values())

unique_img_cnt = len(set(x['image'] for x in samples))

report = f"#Unique images: {unique_img_cnt} | Valid eps / Total eps: {valid_ep_cnt} / {total_ep_cnt} = {valid_ep_cnt/total_ep_cnt:.2f} | Valid Steps / Total steps: {valid_step_cnt} / {total_step_cnt} = {valid_step_cnt / total_step_cnt:.2f} | Valid elems / Total iterated elems: {valid_elem_cnt} / {iterated_elem_cnt} = {valid_elem_cnt / iterated_elem_cnt:.2f}"

# 统计动作分布
act_stats = defaultdict(int)
for x in samples:
    if 'planning' in x['id']:
        act_stats[re.search(r'"action_type":\s*"([^"]+)"', x['conversations'][1]['value']).group(1)] += 1

scale_str = f"_s{SCALE}" if SCALE != -1 else "_unnorm"
save_file = os.path.join(SAVE_DIR, f"{DATASET_NAME}{'_wActRef' if USE_ACTION_REFEXP else ''}{'_IntengGnd' if INTENTGND else ''}{scale_str}_{len(samples)}.json")

with open(os.path.join(SAVE_DIR, save_file.replace(".json", "_stats.json")), "w") as f:
    json.dump({"report": report, "valid_ep_cnt": valid_ep_cnt, 
               "total_ep_cnt": total_ep_cnt,
               "valid_step_cnt": valid_step_cnt,
               "total_step_cnt": total_step_cnt,
               "valid_elem_cnt": valid_elem_cnt,
               "iterated_elem_cnt": iterated_elem_cnt,
               "planning_samples_cnt": planning_cnt,
               'action_stats':act_stats}, f, indent=2)

print(f'save {len(samples)} samples to {save_file.replace(".json", "_sample.json")}')
with open(save_file.replace(".json", "_sample.json"), "w") as f:
    json.dump(random.sample(samples,min(160,len(samples))), f, indent=2)

print(f"save {len(samples)} samples to {save_file}")
with open(save_file, "w") as f:
    json.dump(samples, f, indent=2)