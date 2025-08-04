# 给AITZ生成以下类型任务：1. 给定图像、任务、历史动作，预测下一动作的推理、决策过程；2. 给定图像、任务、历史动作、当前click步骤意图，预测gnd坐标；3. 图像描述

# epid: 17693907265291426996 step 0 有标注错误：coat_action_think里说要swipe down from the top of the screen to access the quick settings panel，但实际上执行的动作是'coat_action_desc' = 'swipe up'

# epid: 9088265727317240175 steps 8-10 标注错误。UI画面显示此时做的操作是通过swipe来移动输入光标，但gpt生成的coat_action_result没有根据正确的意图来总结动作结果。

# epid: 17652164182256576003 step 1。这里的swipe用于拖动滑块，未来的动作空间应该允许这种精细拖动操作

# AITZ的图像分辨率都比较小，不用resize

import json, os, random, cv2, glob, re, magic, numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *
from collections import defaultdict
import ast

from misc import generate_negative_action_plans, keep_unique_actions


DATASET_NAME = 'AITZ'
DRAW=False


ROOT = f"/mnt/shared-storage/groups/pretrain2/lhx/ui_data/{DATASET_NAME}"
SCALE=1000
SPLIT = ['train', 'test'][0]

INTENTGND = True
UICAPTION = True
REWARDMODEL_EVAL = False

USE_ACTION_REFEXP = False

DEVICE_TAG = 'Android'

def filter_aitz_data():
    data_dir = os.path.join(ROOT, SPLIT)
    save_to_dir = f"/data/hongxin_li/scaling_exp/{DATASET_NAME}_processed"
    os.makedirs(save_to_dir, exist_ok=True)

    all_ep_meta_files = sorted(glob.glob(os.path.join(data_dir, "*", "*", "*.json")))
    
    planning_cnt = uicaption_cnt = intentgnd_cnt = reward_model_eval_cnt = 0

    samples, rewardmodel_eval_samples = [], []
    invalid_samples = []

    all_traj_samples = []

    for ep_idx, ep_meta_file in tqdm(enumerate(all_ep_meta_files), total=len(all_ep_meta_files)):
        with open(ep_meta_file, 'r') as f:
            ep_meta = json.load(f)
        
        step_instructions = []

        traj_samples = []

        for step_idx, step_info in enumerate(ep_meta):
            img_path = os.path.join(ROOT, step_info['image_full_path'].split('aitw_with_gpt/')[1])
            W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_path)).groups(1)))

            short_img_path = f'{DATASET_NAME}/' + step_info['image_full_path'].split('aitw_with_gpt/')[1]
            
            if step_idx < len(ep_meta)-1:
                short_nextimg_path = f'{DATASET_NAME}/' + ep_meta[step_idx+1]['image_full_path'].split('aitw_with_gpt/')[1]
            else: short_nextimg_path = short_img_path

            step_info['coat_action_desc'] = step_info['coat_action_desc'].strip(' .')
            
            step_instructions.append(step_info['coat_action_desc'])
            
            # load the elem positions
            boxes = eval(step_info['ui_positions'])
            boxes = [[p[1],p[0],p[1]+p[3],p[0]+p[2]] for p in boxes]

            neg_actions = None

            if step_info['result_action_type'] == 4:
                from_point = list(map(lambda x: float(x), step_info['result_touch_yx'][1:-1].split(',')))
                to_point = list(map(lambda x: float(x), step_info['result_lift_yx'][1:-1].split(',')))

                unnorm_start_y, unnorm_start_x = from_point[1]*H, from_point[0]*W
                start_y, start_x = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), from_point))

                if to_point != from_point:
                    vertical_shift, horizontal_shift = to_point[0] - from_point[0], to_point[1] - from_point[1]
                    vertical_shift_abs, horizontal_shift_abs = abs(vertical_shift), abs(horizontal_shift)
                    shift_ratio = vertical_shift_abs / (horizontal_shift_abs+1e-6)
                    
                    if 2.4 > shift_ratio > 0.39:
                        invalid_samples.append([step_info['image_path'].split('-')[-1][:-4], 'The swiping action is vague'])
                        continue
                    
                    step_info['coat_action_think'] = step_info['coat_action_think'].replace("back down", "down").replace("back up", "up").replace("back left", "left").replace("back right", "right")

                    if any(k in step_info['coat_action_desc'] for k in ['down','up']):
                        direction = 'down' if vertical_shift > 0 else 'up'
                        if "down" in step_info['coat_action_think']:
                            step_info['coat_action_think'] = step_info['coat_action_think'].replace("crolling down", "wiping down").replace("croll down", "wipe down")
                        elif "up" in step_info['coat_action_think']:
                            step_info['coat_action_think'] = step_info['coat_action_think'].replace("crolling up", "wiping up").replace("croll up", "wipe up")

                        if "up" in step_info['coat_action_desc']:
                            step_info['coat_action_desc'] = step_info['coat_action_desc'].replace("crolling up", "wiping up").replace("croll up", "wipe up")
                        elif "down" in step_info['coat_action_desc']:
                            step_info['coat_action_desc'] = step_info['coat_action_desc'].replace("crolling down", "wiping down").replace("croll down", "wipe down")
                    else:
                        direction = 'right' if horizontal_shift > 0 else 'left'
                        
                        if "left" in step_info['coat_action_think']:
                            step_info['coat_action_think'] = step_info['coat_action_think'].replace("crolling left", "wiping left").replace("croll left", "wipe left")
                        elif "right" in step_info['coat_action_think']:
                            step_info['coat_action_think'] = step_info['coat_action_think'].replace("crolling right", "wiping right").replace("croll right", "wipe right")

                        if "left" in step_info['coat_action_desc']:
                            step_info['coat_action_desc'] = step_info['coat_action_desc'].replace("crolling left", "wiping left").replace("croll left", "wipe left")
                        elif "right" in step_info['coat_action_desc']:
                            step_info['coat_action_desc'] = step_info['coat_action_desc'].replace("crolling right", "wiping right").replace("croll right", "wipe right")

                    distance = discretize_dist(vertical_shift_abs)

                    action_str = SWIPE_TEMPLATE.format(start_x=start_x, start_y=start_y, direction=direction, distance=distance)
                    action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction)

                    if all(k not in step_info['coat_action_think'] for k in [step_info['coat_action_desc'][1:], step_info['coat_action_desc'][1:].replace("e ", "ing ")]):
                        invalid_samples.append([step_info['image_path'].split('-')[-1][:-4], 'The action in the thought does not match the action taken'])
                        continue

                    if REWARDMODEL_EVAL:
                        neg_actions = generate_negative_action_plans(gt_act_type='swipe', W=W, H=H, scale=SCALE, gt_center=[round(from_point[1]*W),round(from_point[0]*H)], boxes=boxes, direction=direction)

                    if DRAW:
                        cur_step_id = step_info['image_path'].split('_')[-1][:-4]
                        before = cv2.imread(os.path.join("/mnt/vdb1/hongxin_li/AITZ/train", step_info['image_path']))
                        after = cv2.imread(os.path.join("/mnt/vdb1/hongxin_li/AITZ/train", step_info['image_path'].replace(f"{cur_step_id}.png", f"{int(cur_step_id)+1}.png")))
                        if after is not None:
                            print(step_info['coat_action_desc'])
                            print(from_point, to_point)
                            print(action_str+'\n')
                            cv2.imwrite("test.png", np.concatenate([before, after], axis=1))
                else:
                    action_str = CLICK_TEMPLATE.format(target_x=start_x, target_y=start_y)
                    action_refexp = step_info['coat_action_desc'].strip(' .')

                    # intentgnd
                    if INTENTGND:
                        sample = make_intentgnd_sample(task_id=f"autogui_AITZ_intentgnd_{step_info['episode_id']}-{step_info['step_id']}", intent=action_refexp, loc=f'({start_x},{start_y})', output_tag=WITHPOINT_TAG_LONG)
                        sample['step_info'], sample['task_attr'], sample['image'] = step_info, step_info['coat_action_desc'], short_img_path
                        samples.append(sample); intentgnd_cnt += 1
                    
                    if DRAW:
                        img = cv2.imread(img_path)
                        for box in boxes:
                            cv2.rectangle(img, box[:2], box[2:], color=(0,255,0), thickness=3)
                        cv2.circle(img, [round(from_point[1]*W),round(from_point[0]*H)], color=(0,255,0), radius=8, thickness=3)
                        cv2.imwrite('test.png', img)
                        
                    if REWARDMODEL_EVAL:
                        neg_actions = generate_negative_action_plans(gt_act_type='click', W=W, H=H, scale=SCALE, gt_center=[from_point[1]*W,from_point[0]*H], boxes=boxes)

            elif step_info['result_action_type'] == 3:
                text = step_info['result_action_text'].strip()
                if text.count('"') % 2 != 0: text = text.strip('"')
                if text.count("'") % 2 != 0: text = text.strip("'")
                text = text.strip(' \\').replace("\n", "\\n").replace('"', '\\"')
                
                if not text:
                    continue

                action_str = INPUT_TEMPLATE.format(text=text)
                action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")

                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type='input_text', W=W, H=H, scale=SCALE, boxes=boxes, text=text)
            elif step_info['result_action_type'] == 10:
                action_str = STATUS_TEMPLATE.format(goal_status='successful', answer='')
                action_refexp = random.choice(TASK_STATUS_SENTENCES['successful'])

                if REWARDMODEL_EVAL:
                    neg_actions = generate_negative_action_plans(gt_act_type=action_type, W=W, H=H, scale=SCALE, boxes=boxes, goal_status='successful')
            elif step_info['result_action_type'] == 5:
                action_str = NAVIGATE_BACK_TEMPLATE
                action_refexp = random.choice(NAVIGATE_BACK_PREFIXES)
            elif step_info['result_action_type'] == 6:
                action_str = NAVIGATE_HOME_TEMPLATE
                action_refexp = random.choice(NAVIGATE_HOME_PREFIXES)
            elif step_info['result_action_type'] == 7:
                action_str = PRESSKEY_TEMPLATE.format(key='Enter')
                action_refexp = random.choice(PRESSKEY_PREFIXES['Enter'])
            else:
                raise ValueError(f"Unknown action type: {step_info['result_action_type']}")

            # planning
            action_type = ast.literal_eval(action_str)['action_type']

            # Merge history
            clean_prev_step_instructions = keep_unique_actions(step_instructions[:step_idx])
            retained_history = clean_prev_step_instructions[-MAX_PREV_ACT:]
            history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(retained_history, start=max(1,len(clean_prev_step_instructions) - MAX_PREV_ACT+1))) if len(retained_history) > 0 else 'None'

            if USE_ACTION_REFEXP:
                action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"

            gt_action = "Screen description: {screen_desc}\nThought: {thought}\nAction: {action}".format(screen_desc=step_info['coat_screen_desc'], thought=f"{step_info['coat_action_think']} I will {step_info['coat_action_desc'].strip('')}. {step_info['coat_action_result']}", action=action_str)

            sample = make_actionplanning_sample(
                task_id=f"autogui_AITZ_planning_{step_info['episode_id']}-{step_info['step_id']}-H", 
                global_task=step_info['instruction'], 
                history=history_str, 
                gt_action=gt_action, 
                with_cot=True,
                use_action_refexp=USE_ACTION_REFEXP,
                device_tag=DEVICE_TAG
                )
            
            sample['action_type'], sample['step_info'], sample['step_instruction'], sample['action_refexp'], sample['image'], sample['next_image'], sample['history'] = action_type, step_info, step_info['coat_action_desc'], action_refexp, short_img_path, short_nextimg_path, step_instructions[:step_idx]
            samples.append(sample); planning_cnt += 1
            
            traj_samples.append(sample)

            gt_action_wo_cot = "Action: {action}".format(action=action_str)
            sample = make_actionplanning_sample(
                task_id=f"autogui_AITZ_planning_{step_info['episode_id']}-{step_info['step_id']}-HL", 
                global_task=step_info['instruction'],
                history=history_str,
                gt_action=gt_action_wo_cot,
                step_instruction=f"The next step instruction: {step_info['coat_action_desc']}\n",
                with_cot=False,
                use_action_refexp=USE_ACTION_REFEXP,
                device_tag=DEVICE_TAG
                )
            
            sample['action_type'], sample['step_info'], sample['step_instruction'], sample['action_refexp'], sample['image'], sample['next_image'], sample['history'] = action_type, step_info, step_info['coat_action_desc'], action_refexp, short_img_path, short_nextimg_path, step_instructions[:step_idx]
            samples.append(sample); planning_cnt += 1

            if REWARDMODEL_EVAL:
                if neg_actions is None:
                    neg_actions = generate_negative_action_plans(gt_act_type=action_type, W=W, H=H, scale=SCALE, boxes=boxes)

                neg_sample = {'id': f'autogui_{DATASET_NAME}_rewardmodeleval_{reward_model_eval_cnt}', "image": short_img_path, "next_image": short_nextimg_path, 'ep_id': step_info['episode_id'], 'step_id': step_idx, 'task': step_info['instruction'], 'step_instruction': step_info['coat_action_desc'], "action_type": action_type, "history": step_instructions[:step_idx], 'gt_action': action_str, 'neg_actions': neg_actions, "wxh": f"{W}x{H}"}
                rewardmodel_eval_samples.append(neg_sample); reward_model_eval_cnt += 1

            # ui caption
            if UICAPTION:
                sample = make_uicaption_sample(task_id=f"autogui_AITZ_uicaption_{step_info['episode_id']}-{step_info['step_id']}", ui_caption=step_info['coat_screen_desc'])
                sample['step_info'], sample['image'] = step_info, short_img_path
                samples.append(sample); uicaption_cnt += 1

        if all(x['action_type'] != 'swipe' for x in traj_samples) or len(traj_samples) > 20:
            continue
        
        traj_steps = [
            {
                "step_id": i,
                "model_query": "",
                "action": x['action_type'],
                "step_instruction": x["step_instruction"],
                "history": x['history'],
                "conversations": x["conversations"],
                "image_path": "s3://guidata-lhx/" + x['image'],
                "app_name": "",
                "user_input": ""
            }
            for i, x in enumerate(traj_samples)
        ]

        all_traj_samples.append({
            'task': step_info['instruction'],
            'demonstrations': traj_steps
        })

    action_stats = defaultdict(int)
    for x in samples:
        if 'planning' not in x['id']: continue
        action_stats[re.search(r'"action_type":\s*"([^"]+)"', x['conversations'][1]['value']).group(1)] += 1

    report = f"Total samples: {len(samples)+len(invalid_samples)} Valid samples: {len(samples)} | planning: {planning_cnt} | uicaption: {uicaption_cnt} | intentgnd: {intentgnd_cnt}"
    print(report)
    os.makedirs(save_to_dir, exist_ok=True)
    save_to_file = os.path.join(save_to_dir, f"AITZ_{SPLIT}{'_wActRef' if USE_ACTION_REFEXP else ''}_s{SCALE}_meta_{len(all_traj_samples)}trajs.json")

    with open(save_to_file.replace(".json", "_sample.json"), "w") as f:
        json.dump(random.sample(all_traj_samples, 10), f, indent=2)

    print(f"save {len(all_traj_samples)} trajs to {save_to_file}")
    with open(save_to_file, "w") as f:
        json.dump(all_traj_samples, f, indent=2)

if __name__ == '__main__':
    filter_aitz_data()