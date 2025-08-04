# 给AIW生成以下类型任务：1. 给定图像、任务、历史动作，预测下一动作的决策过程；2. 给定当前click意图，预测gnd坐标
import json, os, re, random, cv2, magic, numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *
from collections import defaultdict
from datasets import Dataset
import ast

from misc import generate_negative_action_plans, keep_unique_actions, lower_first_letter

DATASET_NAME = 'AITW'

ROOT = "/mnt/shared-storage/groups/pretrain2/lhx/ui_data/AITW"
SCALE=1000
SPLIT = ['train', 'val', 'trainval', 'test'][2]

APPS = ['all', 'general', 'install', 'googleapps', 'single', 'webshopping'][0]
ONLY_ACTION = 'click'# 'enter'
PUSH2HUB = False
PLANNING = False
INTENTGND = True; POINT_FORMAT = ['plain', 'qwen2', 'florence'][0]

REWARDMODEL_EVAL = False

USE_ACTION_REFEXP = False

DEVICE_TAG = 'Android'

aitw_imgs_dir =  os.path.join(ROOT, "aitw_images")

split_name = f"aitw_data_{SPLIT}"

def make_aitw_data():
    if SPLIT == 'trainval':
        aitw_data = []
        for v in json.load(open(f'{ROOT}/aitw_data_train.json', 'r')).values(): aitw_data.extend(v)
        for v in json.load(open(f'{ROOT}/aitw_data_val.json', 'r')).values(): aitw_data.extend(v)
    else:
        aitw_data = json.load(open(f'{ROOT}/{split_name}.json', 'r'))

        aitw_data = aitw_data["general"] + aitw_data["single"] + aitw_data["webshopping"] + \
                  aitw_data["install"] + aitw_data["googleapps"]

    planning_cnt = intentgnd_cnt = reward_model_eval_cnt = 0
    samples, rewardmodel_eval_samples, invalid_samples = [], [], []

    for ep_id, episode in tqdm(enumerate(aitw_data), total=len(aitw_data), desc=f'{ROOT}/{split_name}.json'):
        step_instructions = [x['action_addition'].replace("scroll down", "swipe up").replace("scroll up", "swipe down") .replace("scroll left", "swipe right").replace("scroll right", "swipe left") for x in episode]

        # 把存在多个重复动作的轨迹去掉
        if SPLIT != 'test':
            same_cnt = 0; last_action = ''; is_invalid_ep = False
            for step in episode:
                if step['action_addition'] == last_action:
                    same_cnt += 1
                else:
                    same_cnt = 0
                if same_cnt >= 2:
                    is_invalid_ep = True
                    break
                last_action = step['action_addition']
            
            if is_invalid_ep:
                invalid_samples.extend([f"{x['ep_id']}-{x['step']}" for x in episode])
                continue
        
        for step_idx, step_info in enumerate(episode):
            img_filename = step_info["img_filename"] + '.png'
            img_path = os.path.join(aitw_imgs_dir, img_filename)
            
            W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_path)).groups(1)))

            if not os.path.exists(img_path):
                print('image not found')
                continue
            # if len(img_filename) > 100:     # several image with long filename lead to error in linux, just jump it
            short_img_path = f'{DATASET_NAME}/' + img_path.split('aitw_images/')[1]

            if step_idx < len(episode)-1:
                nextimg_path = os.path.join(aitw_imgs_dir, episode[step_idx+1]["img_filename"] + '.png')
                short_nextimg_path = f'{DATASET_NAME}/' + nextimg_path.split('aitw_images/')[1]
            else: short_nextimg_path = short_img_path

            action_type = step_info['action_type_text']
            from_point =  step_info['touch']
            to_point =  step_info['lift']

            neg_actions = None
            boxes = []
            for anno_i in range(0,len(step_info['annot_position']),4):
                anno_y, anno_x, anno_h, anno_w = step_info['annot_position'][anno_i:anno_i+4]
                anno_x1, anno_y1 = round(anno_x*W), round(anno_y*H)
                anno_x2, anno_y2 = round((anno_x+anno_w)*W), round((anno_y+anno_h)*H)
                boxes.append([anno_x1, anno_y1, anno_x2, anno_y2])

            if action_type == 'click':
                start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), from_point))

                action_str = CLICK_TEMPLATE.format(target_x=start_x, target_y=start_y)

                action_intent = step_info['action_addition'].strip(' .')
                click_target = action_intent[action_intent.find(' ')+1:]
                action_refexp = random.choice(ACTION_PREFIXES[action_type]['specific']) + f' the element "{click_target}"'

                if False:
                    before = cv2.imread(img_path)
                    H,W = before.shape[:2]
                    center_x, center_y = round(from_point[0]*W), round(from_point[1]*H)
                    print(step_info['action_addition'], from_point)
                    cv2.circle(before, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    if True:
                        for anno_i in range(0,len(step_info['annot_position']),4):
                            anno_y, anno_x, anno_h, anno_w = step_info['annot_position'][anno_i:anno_i+4]
                            anno_x1, anno_y1 = round(anno_x*W), round(anno_y*H)
                            anno_x2, anno_y2 = round((anno_x+anno_w)*W), round((anno_y+anno_h)*H)
                            cv2.rectangle(before, (anno_x1, anno_y1), (anno_x2, anno_y2), (0, 255, 0), 2)
                    cv2.imwrite("test.png", before)
                    1+1
                # intentgnd
                if INTENTGND and SPLIT != 'test':
                    intent = lower_first_letter(action_refexp if USE_ACTION_REFEXP else action_intent)
                    sample = make_intentgnd_sample(task_id=f"autogui_{DATASET_NAME}_intentgnd_{step_info['ep_id']}-{step_info['step']}", intent=intent, loc=(start_x, start_y), output_tag=WITHPOINT_TAG_LONG, point_format=POINT_FORMAT)
                    sample['step_info'], sample['task_attr'], sample['image'] = step_info, intent, short_img_path
                    samples.append(sample); intentgnd_cnt += 1

                if False:
                    img = cv2.imread(img_path)
                    for box in boxes:
                        cv2.rectangle(img, box[:2], box[2:], color=(0,255,0), thickness=3)
                    cv2.circle(img, [round(from_point[0]*W),round(from_point[1]*H)], color=(0,255,0), radius=8, thickness=3)
                    cv2.imwrite('test.png', img)
                        
                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type='click', W=W, H=H, scale=SCALE, gt_center=[from_point[0]*W, from_point[1]*H], boxes=boxes)

            elif 'scroll' in action_type:
                start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), from_point))
                scroll_direction = action_type.split()[-1]
                
                if scroll_direction in ['down', 'up']:
                    shift = to_point[1] - from_point[1]
                    shift_abs = abs(shift)
                    swipe_direction = 'down' if shift > 0 else 'up'
                else:
                    shift = to_point[0] - from_point[0]
                    shift_abs = abs(shift)
                    swipe_direction = 'right' if shift > 0 else 'left'
                
                step_info['action_addition'] = f"swipe {swipe_direction}"
                distance = discretize_dist(shift_abs)

                action_str = SWIPE_TEMPLATE.format(start_x=start_x, start_y=start_y, direction=swipe_direction, distance=distance)
                action_refexp = random.choice(SWIPE_PHRASES).format(direction=swipe_direction)

                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type='swipe', W=W, H=H, scale=SCALE, gt_center=[round(from_point[0]*W), round(from_point[1]*H)],  boxes=boxes, direction=swipe_direction)
                        
                if False:
                    cur_step_id = img_filename.split('_')[-1][:-4]
                    before = cv2.imread(img_path)
                    after = cv2.imread(os.path.join(aitw_imgs_dir, img_filename.replace(f"{cur_step_id}.png", f"{int(cur_step_id)+1}.png")))
                    if after is not None:
                        print(step_info['action_addition'])
                        print(from_point, to_point)
                        print(action_str+'\n')
                        cv2.imwrite("test.png", np.concatenate([before, after], axis=1))
            elif action_type == 'type':
                text = step_info['type_text']
                if text.count('"') % 2 != 0: text = text.strip('"')
                if text.count("'") % 2 != 0: text = text.strip("'")
                text = step_info['type_text'].strip(' \\').replace("\n", "\\n").replace('"', '\\"')
                
                if not text:
                    continue

                action_str = INPUT_TEMPLATE.format(text=text)
                action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")

                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type='input_text', W=W, H=H, scale=SCALE, boxes=boxes, text=text)

            elif action_type == 'status task complete':
                action_str = STATUS_TEMPLATE.format(goal_status='successful', answer='')
                action_refexp = random.choice(TASK_STATUS_SENTENCES['successful'])
                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type='status', W=W, H=H, scale=SCALE, boxes=boxes, goal_status='successful')
            elif action_type == 'status task impossible':
                action_str = STATUS_TEMPLATE.format(goal_status='infeasible', answer='')
                action_refexp = random.choice(TASK_STATUS_SENTENCES['infeasible'])
                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type='status', W=W, H=H, scale=SCALE, boxes=boxes, goal_status='infeasible')
            elif action_type == 'press back':
                action_str = NAVIGATE_BACK_TEMPLATE
                action_refexp = random.choice(NAVIGATE_BACK_PREFIXES)
            elif action_type == 'press home':
                action_str = NAVIGATE_HOME_TEMPLATE
                action_refexp = random.choice(NAVIGATE_HOME_PREFIXES)
            elif action_type == 'press enter':
                action_str = PRESSKEY_TEMPLATE.format(key='Enter')
                action_refexp = random.choice(PRESSKEY_PREFIXES['Enter'])
            else:
                raise ValueError(f"Unknown action type: {action_type}")

            if len(ONLY_ACTION) > 0 and ONLY_ACTION not in action_str: continue

            # old: history_str = 'None' if step_idx <= 1 else ' '.join(f'Step {i}. {step.strip(" .")}.' for i, step in enumerate(step_instructions[max(0,step_idx-MAX_PREV_ACT):step_idx], start=1))

            clean_prev_step_instructions = keep_unique_actions(step_instructions[:step_idx])
            retained_history = clean_prev_step_instructions[-MAX_PREV_ACT:]
            history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(retained_history, start=max(1,len(clean_prev_step_instructions) - MAX_PREV_ACT+1))) if len(retained_history) > 0 else 'None'
                
            action_type = ast.literal_eval(action_str)['action_type']

            if USE_ACTION_REFEXP:
                action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"

            if APPS != 'all' and img_filename.split('/')[0] != APPS: continue
            if PUSH2HUB:
                print(len(samples), img_filename)
                samples.append({'image': img_filename, 'step': step_info, 'history': step_instructions[:step_idx], 'step_instruction': step_info['action_addition'], 'action_type': action_type, 'action_refexp': action_refexp})
            else:
                gt_action = "Action: {action}".format(action=action_str)

                if PLANNING:
                    # planning
                    sample = make_actionplanning_sample(
                        task_id=f"autogui_{DATASET_NAME}_planning_{step_info['ep_id']}-{step_info['step']}-H",
                        global_task=step_info['goal'],
                        history=history_str,
                        gt_action=gt_action,
                        with_cot=False,
                        device_tag=DEVICE_TAG,
                        use_action_refexp=USE_ACTION_REFEXP)
                    
                    sample['action_type'], sample['step_info'], sample['image'], sample['next_image'], sample['history'], sample['step_instruction'], sample['action_refexp'] = action_type, step_info, short_img_path, short_nextimg_path, step_instructions[:step_idx], step_info['action_addition'], action_refexp
                    samples.append(sample); planning_cnt += 1

                    sample = make_actionplanning_sample(
                        task_id=f"autogui_{DATASET_NAME}_planning_{step_info['ep_id']}-{step_info['step']}-HL",
                        global_task=step_info['goal'],
                        history=history_str,
                        gt_action=gt_action,
                        step_instruction=f"The next step instruction: {step_info['action_addition']}\n",
                        with_cot=False,
                        device_tag=DEVICE_TAG,
                        use_action_refexp=USE_ACTION_REFEXP)
                    
                    sample['action_type'], sample['step_info'], sample['image'], sample['next_image'], sample['history'], sample['step_instruction'], sample['action_refexp'] = action_type, step_info, short_img_path, short_nextimg_path, step_instructions[:step_idx], step_info['action_addition'], action_refexp

                    samples.append(sample); planning_cnt += 1

            if REWARDMODEL_EVAL:
                if neg_actions is None:
                    neg_actions = generate_negative_action_plans(gt_act_type=action_type, W=W, H=H, scale=SCALE, boxes=boxes)

                neg_sample = {'id': f'autogui_{DATASET_NAME}_rewardmodeleval_{reward_model_eval_cnt}', "image": short_img_path, "next_image": short_nextimg_path, 'ep_id': step_info['ep_id'], 'step_id': step_idx, 'task': step_info['goal'], 'step_instruction': step_info['action_addition'], "action_type": action_type, "history": step_instructions[:step_idx], 'gt_action': action_str, 'neg_actions': neg_actions, "wxh": f"{W}x{H}"}
                rewardmodel_eval_samples.append(neg_sample); reward_model_eval_cnt += 1

    app = '' if APPS == 'all' else f'-{APPS}'
    act_limit = '' if len(ONLY_ACTION) == 0 else f'-{ONLY_ACTION}'
            
    if PUSH2HUB:
        dataset = Dataset.from_list(samples)
        dataset.push_to_hub(f"HongxinLi/AITW_test{app}{act_limit}_v2", private=False, token='', split=SPLIT)
    else:
        action_stats = defaultdict(int)
        for x in samples:
            if 'planning' not in x['id']: continue
            action_stats[re.search(r'"action_type":\s*"([^"]+)"', x['conversations'][1]['value']).group(1)] += 1

        report = f"Total samples: {len(samples)+len(invalid_samples)} Valid samples: {len(samples)} | #Unique imgs: {len(set(x['image'] for x in samples))} | planning: {planning_cnt} | intentgnd: {intentgnd_cnt}"
        print(report)
        save_to_dir = f"/data/hongxin_li/scaling_exp/{DATASET_NAME}_processed"
        os.makedirs(save_to_dir, exist_ok=True)
        save_to_file = os.path.join(save_to_dir, f"{DATASET_NAME}{app}{act_limit}_{SPLIT}{'_wActRef' if PLANNING and USE_ACTION_REFEXP else ''}{'_IntengGnd' if INTENTGND else ''}_s{SCALE}_{len(samples)}.json")

        print(f"save {len(samples)} samples to {save_to_file}")
        with open(save_to_file.replace(".json", "_stats.json"), "w") as f:
            json.dump({"total_sample_cnt": len(samples)+len(invalid_samples), "valid_sample_cnt": len(samples), "planning": planning_cnt, "action_stats": action_stats, "intentgnd": intentgnd_cnt, "invalid_samples": invalid_samples}, f, indent=2)

        print(f'save {len(samples)} samples to {save_to_file.replace(".json", "_sample.json")}')
        with open(save_to_file.replace(".json", "_sample.json"), "w") as f:
            json.dump(random.sample(samples, min(len(samples),160)), f, indent=2)

        with open(save_to_file, "w") as f:
            json.dump(samples, f, indent=2)

    if REWARDMODEL_EVAL:
        save_file = os.path.join(save_to_dir, f"{DATASET_NAME}-{SPLIT}_rmeval_s{SCALE}_{len(rewardmodel_eval_samples)}.json")
        with open(save_file.replace(".json", "_sample.json"), "w") as f:
            json.dump(random.sample(rewardmodel_eval_samples,128), f, indent=2)
        
        with open(save_file, "w") as f:
            json.dump(rewardmodel_eval_samples, f)

make_aitw_data()
