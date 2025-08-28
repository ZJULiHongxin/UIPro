import magic, os, json, random, glob, re, ast

import cv2, numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *
import numpy as np

from collections import defaultdict

from misc import resize_image, generate_negative_action_plans

DATASET_NAME = 'GUIOdyssey'

random.seed(666)
SPLIT = ['train', 'test'][0]
DEBUG = False

PLATFORM_IDX = 1
DATA_ROOT = [f"/mnt/vdb1/hongxin_li/{DATASET_NAME}",
             f"/mnt/jfs/copilot/lhx/ui_data/{DATASET_NAME}"][PLATFORM_IDX]
INSTRUC_DIR = os.path.join(DATA_ROOT + '_raw', 'annotations')
IMAGE_DIR = os.path.join(DATA_ROOT + '_raw', 'screenshots')

SAVE_ROOT = ["/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp",
             "/data/hongxin_li/scaling_exp"][PLATFORM_IDX]

SAVE_IMAGE_DIR = DATA_ROOT
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)

RESUME = True
PLANNING = True
REWARDMODEL_EVAL = True

DRAW = False

LONGEST = -1

SCALE = 1000
BOX_PROB = 0.3

ALL_DATA = True

def judge_text_selection(step_id, all_actions):
    # check continuous scroll
    
    cur_id = step_id
    while cur_id >= 0 and all_actions[cur_id] == 'SCROLL':
        cur_id -= 1
    
    is_text_selection = True if all_actions[cur_id] == 'LONG_PRESS' else False
    
    return is_text_selection, cur_id

def make_guiodyssey_data():
    if  ALL_DATA:
        eps = glob.glob(os.path.join(INSTRUC_DIR, '*.json'))
    else:
        split_data = []
        for split in os.listdir(os.path.join(DATA_ROOT, 'splits')):
            split_data.append([split, set(json.load(open(os.path.join(DATA_ROOT, 'splits', split), 'r'))['train'])])

            eps = split_data[0][1]
            for s in split_data[1:]:
                eps = eps.intersection(s[1])

        print(' | '.join(f"{split} inter rate: {len(eps) / len(split_samples):.2f}" for split, split_samples in split_data))

    # 如果取四个划分方法的训练集的交集，那么这个交集的样本展每个划分方法训练样本的比例为
    # app_split.json inter rate: 0.54 | device_split.json inter rate: 0.55 | random_split.json inter rate: 0.61 | task_split.json inter rate: 0.53
    samples, rewardmodel_eval_samples = [], []
    planning_cnt = reward_model_eval_cnt = 0

    # 动作列表：'CLICK', 'TEXT', 'INCOMPLETE', 'COMPLETE', 'LONG_PRESS', 'SCROLL'

    if DEBUG: eps = random.sample(eps, 100)

    for ep_idx, ep_json_name in tqdm(enumerate(eps), total=len(eps)):
        #if ep_idx <= 50: continue
        with open(os.path.join(INSTRUC_DIR, ep_json_name), 'r') as f:
            ep_meta = json.load(f)

        all_actions = [x['action'] for x in ep_meta['steps']]

        for step_idx, step in enumerate(ep_meta['steps']):
            action_type = step['action']

            last_action = ep_meta['steps'][step_idx-1]['action'] if step_idx > 0 else None

            img_file = os.path.join(IMAGE_DIR, step["screenshot"])

            if LONGEST != -1:
                save_img_to = os.path.join(SAVE_IMAGE_DIR, step["screenshot"])
                if not (RESUME and os.path.exists(save_img_to)):
                    img = cv2.imread(img_file)
                    img, ratio = resize_image(img, LONGEST)
                    H, W = img.shape[:2]
                    cv2.imwrite(save_img_to, img)
                else:
                    ORIG_W, ORIG_H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_file)).groups(1)))
                    W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(save_img_to)).groups(1)))
                    ratio = H / ORIG_H
            else:
                ratio = 1.0
                W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_file)).groups(1)))
                save_img_to = img_file

            short_img_path = save_img_to[save_img_to.find(DATASET_NAME):]

            history = 'None'

            if action_type in ['CLICK', 'LONG_PRESS']:
                if step["info"] == 'KEY_HOME':
                    action_str = NAVIGATE_HOME_TEMPLATE
                elif step["info"] == 'KEY_BACK':
                    action_str = NAVIGATE_BACK_TEMPLATE
                elif step["info"] == 'KEY_APPSELECT':
                    action_str = NAVIGATE_RECENT_TEMPLATE
                    
                    if DRAW:
                        print(step)
                        print(action_str + '\n')

                        after = cv2.imread(img_file.replace(f"{step_idx}.png", f"{step_idx+1}.png"))
                        if LONGEST != -1:
                            after, _ = resize_image(after, max_size=LONGEST)
                        merge = np.concatenate([img, after], axis=1)
                        cv2.imwrite("test.png", merge)
                        1+1
                else:
                    target_x, target_y = step["info"][0]
                    
                    norm_target_x, norm_target_y = max(0, min(SCALE-1, round(target_x / 1000 * SCALE))), max(0, min(SCALE-1, round(target_y / 1000 * SCALE)))
                    
                    if action_type == 'CLICK':
                        template = CLICK_TEMPLATE
                        if last_action == 'TEXT':
                            history = 'Step 1. Type "{}" into a text field.'.format(ep_meta['steps'][step_idx-1]['info'])
                        else:
                            history = 'Step 1. Go to a page to find the necessary element I should click on.'
                    else:
                        template = LONG_PRESS_TEMPLATE
                        history = 'Step 1. go to a page to find the necessary element I should long-press.'

                    action_str = template.format(target_x=norm_target_x, target_y=norm_target_y)

                    if REWARDMODEL_EVAL:
                        # 由于GUIOdyssey没有给所有元素的标注，所以这里不生成点击操作负样本
                        neg_actions = generate_negative_action_plans(gt_act_type='click', W=W, H=H, scale=SCALE, boxes=[])

            elif action_type == "SCROLL":
                norm_from_x, norm_from_y = step["info"][0][0], step["info"][0][1]
                norm_to_x, norm_to_y = step["info"][1][0], step["info"][1][1]
                
                is_text_selection, longpress_id = judge_text_selection(step_idx, all_actions)
                
                if is_text_selection:
                    action_str = DRAG_TEMPLATE.format(start_x=max(0, min(SCALE-1, round(norm_from_x/1000*SCALE))), start_y=max(0, min(SCALE-1, round(norm_from_y/1000*SCALE))), end_x=max(0, min(SCALE-1, round(norm_to_x/1000*SCALE))), end_y=max(0, min(SCALE-1, round(norm_to_y/1000*SCALE))))
                    history = 'Step 1. long-press the target to select it.'
                    
                    text_selection_imgs = []
                    for drag_id in range(longpress_id+1,step_idx):
                        history += f" Step {drag_id-longpress_id+1}. drag the selection handle to adject the selection range."
                        if DRAW:
                            text_selection_imgs.append(cv2.imread(img_file.replace(f"{step_idx}.png", f"{drag_id}.png")))
                    
                    if DRAW:
                        text_selection_imgs.insert(0, cv2.imread(img_file.replace(f"{step_idx}.png", f"{longpress_id}.png")))
                        text_selection_imgs.append(img)
                        merge = np.concatenate(text_selection_imgs, axis=1)
                        cv2.imwrite("test.png", merge)
                        1+1

                    if REWARDMODEL_EVAL:
                        neg_actions = generate_negative_action_plans(gt_act_type='drag', W=W, H=H, scale=SCALE, boxes=[], drag_start=[norm_from_x/1000*W, norm_from_y/1000*H], drag_end=[norm_to_x/1000*W, norm_to_y/1000*H])
                else:
                    vertical_shift, horizontal_shift = norm_to_y - norm_from_y, norm_to_x - norm_from_x

                    # judged the scrolling direction
                    if abs(vertical_shift) > abs(horizontal_shift):
                        direction = 'down' if vertical_shift > 0 else 'up'
                        distance = discretize_dist(abs(vertical_shift)/1000)
                    else:
                        direction = 'right' if horizontal_shift > 0 else 'left'
                        distance = discretize_dist(abs(horizontal_shift)/1000)
                    
                    action_str = SWIPE_TEMPLATE.format(start_x=max(0, min(SCALE-1, round(norm_from_x/1000*SCALE))), start_y=max(0, min(SCALE-1, round(norm_from_y/1000*SCALE))), direction=direction, distance=distance)
                    history = 'Step 1. I have navigated to a page which I should explore to find the target content required by the task.'

                    if REWARDMODEL_EVAL:
                        neg_actions = generate_negative_action_plans(gt_act_type='swipe', W=W, H=H, scale=SCALE, boxes=[], direction=direction)

                if DRAW:
                    print(step)
                    print(action_str + '\n')
                    real_from = (round(norm_from_x/1000*W), round(norm_from_y/1000*H))
                    real_to = (round(norm_to_x/1000*W), round(norm_to_y/1000*H))
                    cv2.circle(img, real_from, 5, (0, 0, 255), -1)
                    cv2.circle(img, real_to, 5, (0, 255, 0), -1)
                    cv2.arrowedLine(img, real_from, real_to, (255, 0, 0), 2)
                    after = cv2.imread(img_file.replace(f"{step_idx}.png", f"{step_idx+1}.png"))
                    if LONGEST != -1:
                        after, _ = resize_image(after, max_size=LONGEST)
                    merge = np.concatenate([img, after], axis=1)
                    cv2.imwrite("test.png", merge)
            
            elif action_type == "TEXT":
                text = step["info"].strip(' \\').replace("\n", "\\n").replace('"', '\\"')
                if not text:
                    continue

                action_str = INPUT_TEMPLATE.format(text=text)
                history = 'Step 1. focus on the text field to input texts.' # 目前只能用这一句概括性的描述

                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type='input_text', W=W, H=H, scale=SCALE, boxes=[], text=text)

            elif action_type == "COMPLETE":
                action_str = STATUS_TEMPLATE.format(goal_status="successful", answer='')
                history = "Step 1. I have reached the destination and found the content required by the user's task."
                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type='status', W=W, H=H, scale=SCALE, boxes=[], goal_status="successful")
            elif action_type == 'INCOMPLETE':
                action_str = STATUS_TEMPLATE.format(goal_status='infeasible', answer='')
                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type='status', W=W, H=H, scale=SCALE, boxes=[], goal_status="infeasible")
            else:
                1+1
            
            action_type = eval(action_str)['action_type']
            sample = make_actionplanning_sample(task_id=f'autogui_{DATASET_NAME}_planning_{ep_meta["episode_id"]}-{step_idx}-task', global_task=ep_meta['task_info']['task'], gt_action=f"Action: {action_str}", history=history)
            
            step['task'], step['instruction'] = ep_meta['task_info']['task'], ep_meta['task_info']['instruction']

            sample["task"], sample["history"], sample["step_info"], sample["image"], sample["wxh"] = step['task'], history, step, short_img_path, f"{W}x{H}"
            samples.append(sample)
            planning_cnt += 1
            
            sample = make_actionplanning_sample(task_id=f'autogui_{DATASET_NAME}_planning_{ep_meta["episode_id"]}-{step_idx}-instruction', global_task=ep_meta['task_info']['instruction'], gt_action=f"Action: {action_str}", history=history)
            
            prev_actions = re.findall(r"Step \d+\.\s*([^\.]+)\.", history)
            sample["task"], sample["history"], sample["step_info"], sample["image"], sample["wxh"] = step['task'], prev_actions, step, short_img_path, f"{W}x{H}"
            samples.append(sample)
            planning_cnt += 1

            if REWARDMODEL_EVAL:
                if neg_actions is None:
                    neg_actions = generate_negative_action_plans(gt_act_type=action_type, W=W, H=H, scale=SCALE, boxes=[])

                neg_sample = {'id': f'autogui_{DATASET_NAME}_rewardmodeleval_{reward_model_eval_cnt}',
                              "image": short_img_path,
                              'ep_id': ep_meta['episode_id'], 'step_id': step_idx,
                              'task': ep_meta['task_info']['task'],
                              'histroy': prev_actions,
                              "action_type": action_type,
                              'gt_action': action_str,
                              'neg_actions': neg_actions,
                              "wxh": f"{W}x{H}"
                              }
                rewardmodel_eval_samples.append(neg_sample); reward_model_eval_cnt += 1

    print(f"Generate {planning_cnt} planning samples")
    
    save_to_root = os.path.join(SAVE_ROOT, f'{DATASET_NAME}_processed')
    os.makedirs(save_to_root, exist_ok=True)
    save_to = os.path.join(save_to_root, f"{DATASET_NAME}_{len(samples)}.json")

    # 统计动作分布
    act_stats = defaultdict(int)
    for x in samples:
        if 'planning' in x['id']:
            act_stats[ast.literal_eval(x['conversations'][1]['value'][7:].replace('\n', ' '))['action_type']] += 1

    with open(save_to.replace('.json', '_stats.json'), "w") as f:
        json.dump({'num_samples': len(samples), 'planning_cnt': planning_cnt,'action_stats':act_stats}, f, indent=2)

    print(f"Save {len(samples)} to {save_to}")
    with open(save_to.replace('.json', '_sample.json'), "w") as f:
        json.dump(random.sample(samples, 256), f, indent=2)
    
    with open(save_to, "w") as f:
        json.dump(samples, f)

    if REWARDMODEL_EVAL:
        save_file = os.path.join(save_to_root, f"{DATASET_NAME}-{SPLIT}_rmeval_s{SCALE}_{len(rewardmodel_eval_samples)}.json")
        with open(save_file.replace(".json", "_sample.json"), "w") as f:
            json.dump(random.sample(rewardmodel_eval_samples,128), f, indent=2)
        
        with open(save_file, "w") as f:
            json.dump(rewardmodel_eval_samples, f)

make_guiodyssey_data()
    