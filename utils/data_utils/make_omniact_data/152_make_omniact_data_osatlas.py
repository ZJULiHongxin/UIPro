import os, json, random, re, magic
import cv2
import numpy as np
from tqdm import tqdm
from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import find_smallest_box_containing_point
from utils.data_utils.misc import resize_image, is_pure_color

SCALE = 1000

DATASET_NAME = 'OmniAct'

SPLIT = ['train', 'test'][0]
DEVICE = ['web', 'desktop', 'all'][1]

"""
    "0": {
        "task": "data/tasks/web/asics/task_1.30.txt",
        "image": "data/data/web/asics/screen_1.png",
        "ocr": "ocr/web/asics/screen_1.json",
        "color": "detact_color/web/asics/screen_1.json",
        "icon": "detact_icon/web/asics/screen_1.json",
        "box": "data/metadata/web/boxes/asics/screen_1.json"
    },

"""
SKIP_CHECKING = False
USE_ACTION_PROMPT = False

ELEMGND = True; OUTPUT_TAG = '' # 'Please output its center coordinates.'

PLANNING = True
ALLOWED_ACTIONS = [] # ['click', 'double_click', 'right_click']

USE_FULL_PROMPT = True

USE_ALL_ACTIONS = True

def extract_coords(text):
    coords = text[text.find('(')+1:text.rfind(')')].strip(' ,')
    if len(coords) == 0:
        x = y = -1
    else:
        x, y = list(map(float, coords.split(',')))
    return [x,y]

def get_scroll_dist(amount):
    if amount <= 100: return 'short'
    elif amount <= 500: return 'short'
    else: return 'long'

def make_omniact_data():
    ROOT = "/mnt/vdb1/hongxin_li/OmniAct"
    SAVE_ROOT = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"

    data = json.load(open(os.path.join(ROOT, f"{SPLIT}.json")))

    samples = []
    unique_elems = {}
    num_iterated_elems = 0

    elemgnd_cnt = planning_cnt = 0

    # record invalid samples
    invalid_elem_record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    
    if os.path.exists(invalid_elem_record_file):
        invalid_elem = json.load(open(invalid_elem_record_file))
        for k, v in invalid_elem.items():
            invalid_elem[k] = set(v)
    else:
        invalid_elem = {TOO_SMALL_ELEMENT: set(), INVALID_ELEM_BOX: set(), INVALID_ELEM_CONTENT: set(), BLANK_ELEM: set(), EMPTY_ELEM_TEXT: set(), OVERLY_LENGTHY_ELEM_TEXT: set(), DUPLICATE_ELEMEMNT: set()}
        
    action_stats = {}
    used_instruc = set()

    for sample_idx, sample in tqdm(enumerate(data.values()), total=len(data)):
        if DEVICE != 'all' and f'{DEVICE}/' not in sample['task']:
            continue
        task_file = os.path.join(ROOT, sample["task"])
        
        if 'web/' in sample["image"]: sample["image"]=sample["image"].replace('screen_','screen')

        img_path = os.path.join(ROOT, sample["image"])
        
        if sample["image"] not in unique_elems:
            unique_elems[sample["image"]] = []

        short_img_name = img_path[img_path.find(DATASET_NAME):]
        
        W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_path)).groups()))

        img = None # cv2.imread(img_path)
        #H, W = img.shape[:2]

        # # load boxes
        with open(os.path.join(ROOT, sample["box"] if 'desktop/' in sample["box"]  else (sample["box"].replace('screen_','screen')).replace('.json','_boxes.json')), 'r') as f:
            boxes = list(json.load(f).values())
        
        if False:
            img = cv2.imread(img_path)
            for box in boxes:
                cv2.rectangle(img, box['top_left'], box['bottom_right'], (0, 255, 0), 2)
                cv2.putText(img, box['label'], box['top_left'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 125), 2)
            
            cv2.imwrite('test.png', img)
            1+1
        
        if ELEMGND:
            num_iterated_elems += len(boxes)
            
            for elem_idx, elem_info in enumerate(boxes):
                sample_identifier = sample["image"] + f'|{elem_idx}'
                
                is_invalid = False
                for v in invalid_elem.values():
                    if sample_identifier in v:
                        is_invalid = True; break
                if is_invalid: continue
                
                unnorm_bbox = elem_info['top_left'] + elem_info['bottom_right']

                if unnorm_bbox not in unique_elems[sample["image"]]:
                    unique_elems[sample["image"]].append(unnorm_bbox)
            
                instruc = elem_info['label']

                bbox = [unnorm_bbox[0]/W, unnorm_bbox[1]/H, unnorm_bbox[2]/W, unnorm_bbox[3]/H]
                
                instruc_box = f'{instruc}|{str(list(map(round, unnorm_bbox)))}'
                if not SKIP_CHECKING:
                    if instruc_box in used_instruc:
                        invalid_elem[DUPLICATE_ELEMEMNT].add(sample_identifier)
                        continue
                    # skip swipe actions
                    if not isinstance(instruc, str):
                        invalid_elem[INVALID_ELEM_CONTENT].add(sample_identifier)
                        continue

                    if abs(bbox[0] - bbox[2]) <= 0.005 or abs(bbox[1] - bbox[3]) <= 0.005:
                        invalid_elem[TOO_SMALL_ELEMENT].add(sample_identifier)
                        continue

                    if not (0<=bbox[0]<=1 and 0<=bbox[1]<=1 and 0<=bbox[2]<=1 and 0<=bbox[3]<=1 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                        invalid_elem[INVALID_ELEM_BOX].add(sample_identifier)
                        continue
                        
                    if len(instruc) == 0:
                        invalid_elem[EMPTY_ELEM_TEXT].add(sample_identifier)
                        continue

                    if img is None:
                        img = cv2.imread(img_path)

                    if is_pure_color(img, unnorm_bbox):
                        invalid_elem[BLANK_ELEM].add(sample_identifier)
                        continue
                
                used_instruc.add(instruc_box)

                norm_center = [max(0, min(SCALE-1, round((bbox[0]+bbox[2])/2*SCALE))), max(0, min(SCALE-1, round((bbox[1]+bbox[3])/2*SCALE)))]
                center_str = f'({norm_center[0]},{norm_center[1]})'
                
                if USE_ACTION_PROMPT:
                    action = CLICK_TEMPLATE.format(target_x=norm_center[0],target_y=norm_center[1])
                    query = TURN_GND_INTO_PLANNING_PROMPT.format(instruc=instruc) if len(instruc.strip().split()) == 1 else instruc
                    sample = make_actionplanning_sample(task_id=f'autogui_{DATASET_NAME}_elemgnd_{len(samples)}', global_task=query, gt_action=action, history='None', prompt_format_type='aguvis')
                else:
                    sample = make_elemgnd_sample(task_id=f'autogui_{DATASET_NAME}_elemgnd_{len(samples)}', text=instruc, loc=center_str, output_tag=OUTPUT_TAG, foramt=None)

                sample['image'], sample['unnormalized_box'], sample['task_attr'] = img_path.split(f'{DATASET_NAME}/')[1], unnorm_bbox, instruc
                samples.append(sample); elemgnd_cnt += 1

        if sample_idx > 0 and sample_idx % 100 == 0 or sample_idx == len(data) - 1:
            with open(invalid_elem_record_file, 'w') as f:
                json.dump({k:list(v) for k,v in invalid_elem.items()}, f, indent=2)
            
        if PLANNING:
            """Task: Record a lap on the stopwatch
            Output Script:
            pyautogui.click(1504.5,1971.5)"""
            
            with open(task_file, 'r') as f:
                raw = f.read()
                task = raw[5:raw.find('\n')].strip()
                plan = raw[raw.find('yautogui.'):].strip()
            
            # use the 1st action
            steps = [step.strip() for step in plan.split("\n") if 'moveto(' not in step]
            
            action_raw = steps[0]
            
            if len(ALLOWED_ACTIONS) > 0 and any(act in action_raw for act in ALLOWED_ACTIONS): continue

            if 'click' in action_raw:
                action_type = 'click'
                x,y = extract_coords(action_raw)

                if SPLIT == 'train' and x == -1: continue
                interacted_box, index = find_smallest_box_containing_point(np.array([x, y]), boxes=np.array([x['top_left'] + x['bottom_right'] for x in boxes]))
                
                elem_label = None if index is None else boxes[index]['label'].replace('_', ' ')
                action_attr = {'action_type': 'click', 'target': [x,y], 'unnormalized_box': interacted_box, 'elem_label': elem_label}
                action_str = f"CLICK <point>[[{max(0, min(SCALE-1, round(x/W*SCALE)))}, {max(0, min(SCALE-1, round(y/H*SCALE)))}]]</point>"
                action_refexp = random.choice(ACTION_PREFIXES[action_type]['specific']) + (f' the element "{elem_label}"') if elem_label is not None else 'the task-relevant element'
            elif 'moveTo' in action_raw:
                action_type = 'hover'
                x,y = extract_coords(action_raw)
                interacted_box, index = find_smallest_box_containing_point(np.array([x, y]), boxes=np.array([x['top_left'] + x['bottom_right'] for x in boxes]))
                
                elem_label = None if index is None else boxes[index]['label'].replace('_', ' ')

                action_attr = {'action_type': 'hover', 'target': [x,y], 'unnormalized_box': interacted_box, 'elem_label': elem_label}
                action_str = f"MOVETO <point>[[{max(0, min(SCALE-1, round(x/W*SCALE)))}, {max(0, min(SCALE-1, round(y/H*SCALE)))}]]</point>"
                action_refexp = random.choice(ACTION_PREFIXES['hover']['specific']) + (f' the element "{elem_label}"') if elem_label is not None else 'the task-relevant element'
            elif 'rightClick' in action_raw:
                action_type = 'right_click'
                x,y = extract_coords(action_raw)
                if SPLIT == 'train' and x == -1: continue
                interacted_box, index = find_smallest_box_containing_point(np.array([x, y]), boxes=np.array([x['top_left'] + x['bottom_right'] for x in boxes]))

                elem_label = None if index is None else boxes[index]['label'].replace('_', ' ')
                action_attr = {'action_type': 'right_click', 'target': [x,y], 'unnormalized_box': interacted_box, 'elem_label': elem_label}
                action_str = f"RIGHTCLICK <point>[[{max(0, min(SCALE-1, round(x/W*SCALE)))}, {max(0, min(SCALE-1, round(y/H*SCALE)))}]]</point>"
                action_refexp = random.choice(ACTION_PREFIXES[action_type]['specific']) + (f' the element "{elem_label}"') if elem_label is not None else 'the task-relevant element'
            elif 'doubleClick' in action_raw:
                action_type = 'double_click'
                x,y = extract_coords(action_raw)
                if SPLIT == 'train' and x == -1: continue
                interacted_box, index = find_smallest_box_containing_point(np.array([x, y]), boxes=np.array([x['top_left'] + x['bottom_right'] for x in boxes]))

                elem_label = None if index is None else boxes[index]['label'].replace('_', ' ')
                action_attr = {'action_type': 'double_click', 'target': [x,y], 'unnormalized_box': interacted_box, 'elem_label': elem_label}
                action_str = f"DOUBLECLICK <point>[[{max(0, min(SCALE-1, round(x/W*SCALE)))}, {max(0, min(SCALE-1, round(y/H*SCALE)))}]]</point>"
                action_refexp = random.choice(ACTION_PREFIXES[action_type]['specific']) + (f' the element "{elem_label}"') if elem_label is not None else 'the task-relevant element'
                1+1
            elif 'dragTo' in action_raw:
                action_type = 'drag'
                x,y = extract_coords(action_raw)
                if SPLIT == 'train' and x == -1: continue
                interacted_box, index = find_smallest_box_containing_point(np.array([x, y]), boxes=np.array([x['top_left'] + x['bottom_right'] for x in boxes]))

                action_attr = {'action_type': 'drag', 'target': [x,y], 'unnormalized_box': interacted_box, 'elem_label': None if index is None else boxes[index]['label']}
                action_str = f"DRAG <point>[[{max(0, min(SCALE-1, round(x/W*SCALE)))}, {max(0, min(SCALE-1, round(y/H*SCALE)))}]]</point>"# DRAG_TEMPLATE.format(target_x=max(0, min(SCALE-1, round(x/W*SCALE))), target_y=max(0, min(SCALE-1, round(y/H*SCALE))))
                action_refexp = random.choice(DRAG_PHRASES['specific']).format(target=f'the element "{elem_label}"') if elem_label is not None else 'the task-relevant element'
            elif 'write' in action_raw:
                action_type = 'input_text'
                text = action_raw[action_raw.find('write("')+7:action_raw.rfind('"')]
                action_attr = {'action_type': 'input_text', 'text': text}
                action_str = f"TYPE [{text}]"# INPUT_TEMPLATE.format(text=text)
                action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")
            elif 'press' in action_raw:
                action_type = 'press_key'
                key = action_raw[action_raw.find('press("')+7:action_raw.rfind('"')].strip(' "')
                action_attr = {'action_type': 'press_key', 'key': key}
                action_str = f"PRESS_{key.upper()}"# PRESSKEY_TEMPLATE.format(key=key)
                action_refexp = random.choice(PRESSKEY_PREFIXES[key.lower()])
            elif 'hotkey' in action_raw or 'hotKey' in action_raw:
                action_type = 'hotkey'
                keys = action_raw[action_raw.find('hotkey("')+8:action_raw.rfind('"')].split(',')
                keys = [key.strip('" ') for key in keys]
                keycomb = '-'.join(keys)
                action_attr = {'action_type': 'hotkey', 'key': keycomb}
                action_str = f"HOTKEY [{keycomb.upper()}]"# KEYCOMB_TEMPLATE.format(key_combination=keycomb)
                action_refexp = random.choice(KEYCOMB_PREFIXES[keycomb.lower()])
            elif '.scroll' in action_raw:
                action_type = 'scroll'
                amount = int(action_raw[action_raw.find('(')+1:action_raw.rfind(')')])
                direction = 'up' if amount > 0 else 'down'
                action_attr = {'action_type': 'scroll', 'direction': direction}
                action_str = f"SCROLL [{direction.upper()}]"# SCROLL_TEMPLATE.format(direction=direction, distance=get_scroll_dist(amount))
                action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction).replace('Swipe', 'Scroll')
            elif '.hscroll' in action_raw:
                action_type = 'scroll'
                amount = int(action_raw[action_raw.find('(')+2:action_raw.rfind(')')-1])
                direction = 'right' if amount > 0 else 'left'
                action_attr = {'action_type': 'scroll', 'direction': direction}
                action_str = f"SCROLL {direction.upper()}"# SCROLL_TEMPLATE.format(direction=direction, distance=get_scroll_dist(amount))
                action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction).replace('Swipe', 'Scroll')
            else:
                raise Exception(f"unknown action: {action_raw}")

            action_stats[action_type] = action_stats.get(action_type, 0) + 1

            action_refexp = action_refexp.strip(' .')

            prompt_template = OSATLAS_OMNIACT_PROMPT if USE_FULL_PROMPT else OSATLAS_OMNIACT_PROMPT_ABLATION
            
            answer = f"Thoughts:\nThe task is to {task.strip(' .')}. I should {action_refexp}\n\nActions:\n{action_str}"
            prompt = prompt_template.format(task=task, action=action_str)

            task_instruc = ATLAS_PROMPT.format(global_task=task, history='None')
            
            sample = {
                'id': f'autogui_{DATASET_NAME}_planning_{len(samples)}',
                'task': task,
                'action_type': action_type,
                'action_refexp': action_refexp,
                'history': 'None',
                'image': short_img_name,
                'task_attr': action_attr,
                'wxh': f"{W}x{H}",
                'conversations': [
                    {
                        "from": "human",
                        "value": f"{prompt}<image>\n{task_instruc}"
                    },
                    {
                        "from": "gpt",
                                "value": answer
                    }]
            }

            # add key usage
            # UIPro用的仅预测第一个动作的prompt
            # sample = make_actionplanning_sample_desktop(task_id=f"autogui_{DATASET_NAME}_planning_{len(samples)}", global_task=task, gt_action=action_str)
            
            action_attr['plan'] = plan
            sample['task'], sample["action_type"], sample['action_refexp'], sample["history"], sample["image"], sample["task_attr"], sample["wxh"] = task, action_type, action_refexp, [], short_img_name, action_attr, f"{W}x{H}"
            
            samples.append(sample); planning_cnt += 1

    # resample press_key and hotkey as these samples are too few
    resampled = [x for x in samples if x['action_type'] in ['press_key', 'hotkey']]
    samples = resampled * 10 + samples

    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k,v in unique_elems.items() if len(v)])

    save_to = os.path.join(SAVE_ROOT, f"{DATASET_NAME}-{DEVICE}-{SPLIT}_OSATLAS{'_FullActSpace' if USE_FULL_PROMPT else '_AblatedActSpace'}_s{SCALE}_{len(samples)}.json")
    os.makedirs(SAVE_ROOT, exist_ok=True)
    with open(save_to.replace(".json", "_sample.json"), 'w') as f:
        json.dump(random.sample(samples,160), f, indent=2)

    with open(save_to, 'w') as f:
        json.dump(samples, f, indent=2)
    
    with open(save_to.replace('.json', '_stats.json'), 'w') as f:
        json.dump({'num_samples': len(samples), '#num_unique_elems': num_unique_elems, '#all_elems': num_iterated_elems, 'num_invalid_elements': num_invalid_elem, '#valid_unique_images': num_valid_imgs, 'all_unique_images': len(unique_elems), 'elemgnd_cnt': elemgnd_cnt, 'planning_cnt': planning_cnt, 'action_stats': action_stats, 'invalid_elem_types': {k:len(v) for k,v in invalid_elem.items()}}, f, indent=2)

make_omniact_data()