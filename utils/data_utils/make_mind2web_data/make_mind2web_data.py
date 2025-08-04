# 给AIW生成以下类型任务：1. 给定图像、任务、历史动作，预测下一动作的决策过程；2. 给定当前click意图，预测gnd坐标
import json, os, random, cv2, re, numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *
from collections import defaultdict
from datasets import Dataset, concatenate_datasets
import ast

from misc import is_pure_color, keep_unique_actions, remove_redundant_spaces

DEBUG = False

DATASET_NAME = 'Mind2Web'

SCALE=1000
SPLIT = ['train', 'test', 'test_domain', 'test_task', 'test_website',][0]

PUSH2HUB = False
INTENTGND = False

MERGE_ACTION = False

DEVICE_TAG = 'Web'
USE_ACTION_REFEXP = True

platform_idx = 1
MIND2WEB_ROOT = ["/mnt/nvme0n1p1/hongxin_li/UI_training_data/Mind2Web", "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/Mind2Web"][platform_idx]
SAVE_ROOT = ["/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/Mind2Web_processed", "/data/hongxin_li/scaling_exp/Mind2Web_processed"][platform_idx]

mind2web_imgs_dir = os.path.join(MIND2WEB_ROOT, "mind2web_images")

split_name = f"mind2web_data_{SPLIT}"
if SPLIT == 'train':
    mind2web_data = json.load(open(f'{MIND2WEB_ROOT}/{split_name}.json', 'r'))
else:
    mind2web_data = json.load(open(f'{MIND2WEB_ROOT}/mind2web_data_test_task.json', 'r')) + json.load(open(f'{MIND2WEB_ROOT}/mind2web_data_test_website.json', 'r')) + json.load(open(f'{MIND2WEB_ROOT}/mind2web_data_test_domain.json', 'r'))

# Load HTML
LOAD_HTML = False
if LOAD_HTML:
    from datasets import load_dataset
    ds = load_dataset("osunlp/Multimodal-Mind2Web")
    html_dict = {}
    if SPLIT == 'train':
        split_data = ds['train']
    elif SPLIT == 'test':
        split_data = concatenate_datasets([ds['test_domain'],ds['test_task'],ds['test_website']])
    else:
        split_data = ds[SPLIT]
    for sample in tqdm(split_data, total=len(split_data), desc="Extracting info..."):
        html_dict[f"{sample['annotation_id']}-{sample['action_uid']}"] = {'cleaned_html': sample['cleaned_html'], 'neg_candidates': sample['neg_candidates']}
else: html_dict = None

def make_mind2web_data():
    planning_cnt = intentgnd_cnt = 0
    samples, rewardmodel_eval_samples, invalid_samples = [], [], []

    for ep_id, episode in tqdm(enumerate(mind2web_data), total=len(mind2web_data), desc=f'{MIND2WEB_ROOT}/{split_name}.json'):
        if DEBUG and ep_id % 20 > 1: continue
        step_instructions = []
        for step_idx, action_repr in enumerate(episode['action_reprs']):
            elem, act = action_repr.split('->')
            act = act.strip()
            elem = remove_redundant_spaces(elem.replace('  ',' ').replace('[', ' ', 1).replace(']', ' ', 1).strip())
            if 'TYPE:' in act:
                split_id = act.find(':')
                act, text = act[:split_id], act[split_id+1:]
                text = text.strip(' \n\\').replace('"', '\\"').replace('\n', '\\n')
                prev_act_str = f"type \"{text}\" into the {elem}"
                episode['actions'][step_idx]['elem_desc'] = elem
            elif act == 'ENTER':
                prev_act_str = f"press enter on {elem}"
                episode['actions'][step_idx]['elem_desc'] = elem
            elif act == 'CLICK':
                prev_act_str = f"click on {elem}"
                episode['actions'][step_idx]['elem_desc'] = elem
            elif act == 'HOVER':
                prev_act_str = f"hover over {elem}"
                episode['actions'][step_idx]['elem_desc'] = elem
            elif 'SELECT:' in act:
                split_id = act.find(':')
                act, value = act[:split_id], act[split_id+1:]
                value = value.strip()
                prev_act_str = f"select {value} in the {elem}"
                episode['actions'][step_idx]['elem_desc'] = elem
            else:
                raise Exception(f"unknown action: {act}")
            step_instructions.append(prev_act_str)

        for step_idx, step_info in enumerate(episode['actions']):
            if DEBUG and step_idx % 2 == 0: continue
            identifier = f"{episode['annotation_id']}-{step_info['action_uid']}"
                
            if "bbox" not in step_info:
                print("action not found")
                continue
        
            img_filename = f"{episode['annotation_id']}-{step_info['action_uid']}.jpg"
            img_path = os.path.join(mind2web_imgs_dir, img_filename)

            action_type = step_info['operation']['original_op']

            if not os.path.exists(img_path):
                print('image not found')
                continue
            # if len(img_filename) > 100:     # several image with long filename lead to error in linux, just jump it
            short_img_path = img_path[img_path.find(DATASET_NAME):]
            img = cv2.imread(img_path)
            H, W = img.shape[:2]
            
            point_x = step_info["bbox"]["x"] + (step_info["bbox"]["width"] / 2)
            point_y = step_info["bbox"]["y"] + (step_info["bbox"]["height"] / 2)
            x1, y1, x2, y2 = step_info["bbox"]["x"], step_info["bbox"]["y"], step_info["bbox"]["x"] + step_info["bbox"]["width"], step_info["bbox"]["y"] + step_info["bbox"]["height"]

            step_info['normalized_bbox'] = [x1 / W, y1 / H, x2 / W, y2 / H]
            
            if (step_info['normalized_bbox'][2]-step_info['normalized_bbox'][0]) * (step_info['normalized_bbox'][3]-step_info['normalized_bbox'][1]) >= 0.65:
                print('invalid bbox')
                continue

            x1, y1, x2, y2 = list(map(round, [x1, y1, x2, y2]))
            
            click_point = [point_x / W, point_y / H]

            if action_type in ['HOVER', 'CLICK', 'ENTER']:
                if MERGE_ACTION:
                    start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), click_point))

                    action_str = CLICK_TEMPLATE.format(target_x=start_x, target_y=start_y)
                    action_refexp = random.choice(ACTION_PREFIXES['click']['specific']) + ' ' + step_info['elem_desc']
                else:
                    if action_type == 'CLICK':
                        start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), click_point))

                        action_str = CLICK_TEMPLATE.format(target_x=start_x, target_y=start_y)
                        action_refexp = random.choice(ACTION_PREFIXES['click']['specific']) + ' ' + step_info['elem_desc']
                    elif action_type == 'HOVER':
                        start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), click_point))

                        action_str = HOVER_TEMPLATE.format(target_x=start_x, target_y=start_y)
                        action_refexp = random.choice(ACTION_PREFIXES['hover']['specific']) + ' ' + step_info['elem_desc']
                    elif action_type == 'ENTER':
                        action_str = PRESSKEY_TEMPLATE.format(key="Enter")
                        action_refexp = random.choice(PRESSKEY_PREFIXES['Enter'])
                        
                
            elif action_type == 'SELECT':
                img = cv2.imread(img_path)
                
                if is_pure_color(img, [x1, y1, x2, y2]):
                    print('blank selection element skipped')
                    continue
                text = step_info['operation']['value'].replace('"', '\\"')
                start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), click_point))
                action_str = SELECT_TEMPLATE.format(target_x=start_x, target_y=start_y, value=text)

                action_refexp = random.choice(SELECT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target=step_info['elem_desc'])
                if False:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imwrite("test.png", img)

            elif action_type == 'TYPE':
                text = step_info["operation"]["value"].strip()
                if text.count('"') % 2 != 0: text = text.strip('"')
                if text.count("'") % 2 != 0: text = text.strip("'")
                text = text.replace('"', '\\"')
                start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), click_point))
                
                action_str = INPUT_TARGET_TEMPLATE.format(target_x=start_x, target_y=start_y, text=text)
                action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target=step_info['elem_desc'])

            _, clean_step_instructions = keep_unique_actions(step_instructions)
            history = clean_step_instructions[max(0,step_idx-MAX_PREV_ACT):step_idx]
            history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(history, start=max(1,step_idx-MAX_PREV_ACT+1))) if len(history) > 0 else 'None'

            action_type = ast.literal_eval(action_str)['action_type']

            if USE_ACTION_REFEXP:
                action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"

            if PUSH2HUB:
                samples.append({'image': img_filename, 'step': step_info, 'task':episode['confirmed_task'], 'history': history_str})
            else:
                gt_action = "Action: {action}".format(action=action_str)

                # planning
                step_info['step_idx'] = step_idx

                sample = make_actionplanning_sample_web(
                    task_id=f"autogui_{DATASET_NAME}_planning_{episode['annotation_id']}-{step_idx}",
                    global_task=episode['confirmed_task'],
                    history=history_str,
                    gt_action=gt_action,
                    with_cot=False,
                    use_action_refexp=USE_ACTION_REFEXP,
                    device_tag=DEVICE_TAG)
                
                sample['action_type'], sample['task'], sample['history'], sample['step_info'], sample['image'], sample['action_refexp'], sample['step_instruction'] = action_type, episode['confirmed_task'], history, step_info, short_img_path, action_refexp, step_instructions[step_idx]
                samples.append(sample); planning_cnt += 1

    if PUSH2HUB:
        dataset = Dataset.from_list(samples)
        dataset.push_to_hub(f"HongxinLi/{DATASET_NAME}_test", private=False, token='', split=SPLIT)
    else:
        action_stats = defaultdict(int)
        for x in samples:
            if 'planning' not in x['id']: continue
            action_stats[re.search(r'"action_type":\s*"([^"]+)"', x['conversations'][1]['value']).group(1)] += 1

        report = f"Total samples: {len(samples)+len(invalid_samples)} Valid samples: {len(samples)} | #Unique imgs: {len(set(x['image'] for x in samples))} | planning: {planning_cnt}"
        print(report)

        os.makedirs(SAVE_ROOT, exist_ok=True)
        save_to_file = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_{SPLIT}{'_wActRef' if USE_ACTION_REFEXP else ''}_s{SCALE}_{len(samples)}.json")

        with open(save_to_file.replace(".json", "_stats.json"), "w") as f:
            json.dump({"total_sample_cnt": len(samples)+len(invalid_samples), "valid_sample_cnt": len(samples), "planning": planning_cnt, "action_stats": action_stats, "intentgnd": intentgnd_cnt, "invalid_samples": invalid_samples}, f, indent=2)

        with open(save_to_file.replace(".json", "_sample.json"), "w") as f:
            json.dump(random.sample(samples, min(len(samples),160)), f, indent=2)

        print(f"save to {save_to_file}")
        with open(save_to_file, "w") as f:
            json.dump(samples, f, indent=2)
        
make_mind2web_data()