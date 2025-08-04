# 给AIW生成以下类型任务：1. 给定图像、任务、历史动作，预测下一动作的决策过程；2. 给定当前click意图，预测gnd坐标
import json, os, random, cv2
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *
from datasets import  concatenate_datasets

from misc import is_pure_color, remove_redundant_spaces

DEBUG = False

DATASET_NAME = 'Mind2Web'

SCALE=1000
SPLIT = ['train', 'test', 'test_domain', 'test_task', 'test_website',][0]

PUSH2HUB = False
INTENTGND = False

MERGE_ACTION = False

DEVICE_TAG = 'Web'
USE_ACTION_REFEXP = True

MIND2WEB_ROOT = "/data0/jingran/workspace/UI_training_data/Mind2Web"
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

            elem_role = elem.split('[')[1].split(']')[0]
            elem_text = elem[elem.find(']')+1:].strip()

            if len(elem_text.strip()) == 0:
                step_instructions.append(None)
                continue

            if random.random() > 0.5:
                # '[link]  NBA '
                if random.random() > 0.5:
                    elem_text = f'"{elem_text}"'
                elem = f"{elem_text} {elem_role}"

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
            if step_instructions[step_idx] is None: continue

            identifier = f"{episode['annotation_id']}-{step_info['action_uid']}"
                
            if "bbox" not in step_info:
                print("action not found")
                continue
        
            img_filename = f"{episode['annotation_id']}-{step_info['action_uid']}.jpg"

            # bad imgs
            if any(k in img_filename for k in ['e48f848d-62b8-441e-aafb-c76aeb2c4f84-5df6d848-d5b7-4202-ac80-1959faf35581']):
                continue

            img_path = os.path.join(mind2web_imgs_dir, img_filename)

            action_type = step_info['operation']['original_op']

            if not os.path.exists(img_path):
                print('image not found')
                continue
            # if len(img_filename) > 100:     # several image with long filename lead to error in linux, just jump it
            short_img_path = img_path[img_path.find(DATASET_NAME):]
            print(img_path)
            img = cv2.imread(img_path)
            H, W = img.shape[:2]
            
            point_x = step_info["bbox"]["x"] + (step_info["bbox"]["width"] / 2)
            point_y = step_info["bbox"]["y"] + (step_info["bbox"]["height"] / 2)
            unnormalized_box = step_info["bbox"]["x"], step_info["bbox"]["y"], step_info["bbox"]["x"] + step_info["bbox"]["width"], step_info["bbox"]["y"] + step_info["bbox"]["height"]
            x1, y1, x2, y2 = unnormalized_box

            step_info['normalized_bbox'] = [x1 / W, y1 / H, x2 / W, y2 / H]
            
            if (step_info['normalized_bbox'][2]-step_info['normalized_bbox'][0]) * (step_info['normalized_bbox'][3]-step_info['normalized_bbox'][1]) >= 0.65:
                print('invalid bbox')
                continue

            x1, y1, x2, y2 = list(map(round, [x1, y1, x2, y2]))
            
            click_point = [point_x / W, point_y / H]
            click_point = list(map(lambda x: max(0, min(999, round(x*1000))), click_point))
            
            if action_type in ['HOVER', 'CLICK', 'ENTER']:
                pass      
            elif action_type == 'SELECT':
                img = cv2.imread(img_path)
                
                if is_pure_color(img, [x1, y1, x2, y2]):
                    print('blank selection element skipped')
                    continue
            elif action_type == 'TYPE':
                pass
            
            sample = make_intentgnd_sample(task_id=f"autogui_Mind2Web_intentgnd_{identifier}", intent=step_instructions[step_idx], loc=[click_point[0], click_point[1]], output_tag='(Output the center coordinates of the target)', point_format='plain')
            sample['image'], sample['unnormalized_box'], sample['task_attr'], sample['ep_id'], sample['step_idx'] = img_path, unnormalized_box, step_instructions[step_idx], episode['annotation_id'], step_idx
            
            samples.append(sample); planning_cnt += 1

    save_to_dir = f"/data0/jingran/workspace/UI_training_data/Ours-Pretrain/scaling_exp/{DATASET_NAME}_processed"
    os.makedirs(save_to_dir, exist_ok=True)
    save_to_file = os.path.join(save_to_dir, f"{DATASET_NAME}_{SPLIT}_IntentGnd_s{SCALE}_{len(samples)}.json")

    with open(save_to_file.replace(".json", "_sample.json"), "w") as f:
        json.dump(random.sample(samples, min(len(samples),160)), f, indent=2)

    with open(save_to_file, "w") as f:
        json.dump(samples, f, indent=2)
        
make_mind2web_data()
