import pandas as pd, os, json, random, glob, re, ast
import cv2, pytesseract, numpy as np
from rapidfuzz import fuzz

from PIL import Image
from io import BytesIO
from tqdm import tqdm
from utils.data_utils.task_prompt_lib import *
import numpy as np
from utils.data_utils.misc import is_pure_color, is_valid_string, decode_img_base64, keep_unique_actions

from collections import defaultdict

DATASET_NAME = 'GUICourse'

random.seed(666)
SPLIT = ['train', 'test'][0]

DEVICE_TYPE = ['web', 'smartphone'][1]

DEBUG = False
DATA_ROOT = "/mnt/vdb1/hongxin_li/GUICourse"

SAVE_ROOT = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data//scaling_exp/{DATASET_NAME}_processed"

TEXTLOC = False
OCR = False
INTENTGND = False

USE_ACTION_PROMPT = False
SKIP_CHECKING = False

OUTPUT_TAG = ''# 'Please output its center coordinates.'
LOC_FORMAT = None#'action_json'

SCALE = 1000
BOX_PROB = 0.0

TASK_MAPPING = {
    'text2bbox': 'textloc',
    'bbox2text': 'ocr'
}

USE_ACTION_REFEXP = True

# GUIEnv
if False:
    # record invalid samples
    invalid_elem_record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    if os.path.exists(invalid_elem_record_file):
        invalid_elem = json.load(open(invalid_elem_record_file))
        for k, v in invalid_elem.items():
            invalid_elem[k] = set(v)
    else:
        invalid_elem = {TOO_SMALL_ELEMENT: set(), INVALID_ELEM_BOX: set(), INVALID_ELEM_CONTENT: set(), BLANK_ELEM: set(), EMPTY_ELEM_TEXT: set(), OVERLY_LENGTHY_ELEM_TEXT: set(), DUPLICATE_ELEMEMNT: set(), INCORRECT_TEXT_ANNO: set()}
        
    iterated_elem_cnt = 0

    dataset_path = os.path.join(DATA_ROOT, "GUIEnv")
    parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))

    textloc_cnt = ocr_cnt = 0

    guienv_samples = []
    unique_elems = {}

    for parquet_file in parquet_files:
        if SPLIT not in parquet_file: continue
        print(f"Processing {parquet_file}")
        guienv_data = pd.read_parquet(parquet_file)

        img_save_path = os.path.join(dataset_path, "imgs")
        os.makedirs(img_save_path, exist_ok=True)

        samples_each_image = {}
        with open(parquet_file.replace("images.parquet", "data.json"), "r") as f:
            metadata = json.load(f)
            for x in metadata:
                samples_each_image.setdefault(x['image_id'], []).append(x)

        for x in tqdm(guienv_data.iterrows(), total=len(guienv_data), desc='saving GUIEnv imgs ...'):
            img_id = x[0]
            img_path = os.path.join(img_save_path, f"{img_id}.png")
            if os.path.exists(img_path):continue
            
            image_data = decode_img_base64(x[1][0])
            image = Image.open(BytesIO(image_data))
            image.save(os.path.join(img_save_path, f"{img_id}.png"))
        
        used_instruc = []

        for group_idx, (img_id, samples) in tqdm(enumerate(samples_each_image.items()), total=len(samples_each_image), desc='making llava-format samples...'):
            img_path = os.path.join(img_save_path, f"{img_id}.png")
            img = None

            if img_id not in unique_elems:
                unique_elems[img_id] = []

            for s in samples:
                _, _, _, _, task, idx = s['uid'].split("_")
                
                sample_identifier = s['uid']
                task = TASK_MAPPING[task]

                if task == 'textloc':
                    if not TEXTLOC: continue

                    instruc, boxes = s['question'].strip(), s['answer']['absolute']
                    if len(boxes) > 1:continue
                    iterated_elem_cnt += 1
                    
                    unnormalized_box = list(map(int, boxes[0][5:-6].split(',')))

                    if not SKIP_CHECKING and instruc in used_instruc:
                        invalid_elem[DUPLICATE_ELEMEMNT].add(sample_identifier)
                        continue
                    
                    used_instruc.append(instruc)
                else:
                    if not OCR: continue
                    iterated_elem_cnt += 1
                    instruc = s['answer'].strip()
                    unnormalized_box = list(map(int, s['question']['absolute'][5:-6].split(','))) 

                is_invalid = False
                for v in invalid_elem.values():
                    if sample_identifier in v:
                        is_invalid=True; break
                if is_invalid: continue
                
                x1, y1, x2, y2 = unnormalized_box

                if unnormalized_box not in unique_elems[img_id]:
                    unique_elems[img_id].append(unnormalized_box)

                W, H = s['image_size']['width'], s['image_size']['height']

                if not SKIP_CHECKING:
                    if len(instruc) == 0:
                        invalid_elem[EMPTY_ELEM_TEXT].add(sample_identifier)
                        continue

                    if abs(x1 - x2) / W <= 0.005 or abs(y1 - y2) / H <= 0.005:
                        invalid_elem[TOO_SMALL_ELEMENT].add(sample_identifier)
                        continue

                    if not (0<=x1<=W and 0<=x2<=W and 0<=y1<=H and 0<=y2<=H and x1 < x2 and y1 < y2):
                        invalid_elem[INVALID_ELEM_BOX] += 1
                        continue
                    
                    if img is None: 
                        img = cv2.imread(img_path)

                    if is_pure_color(img, unnormalized_box):
                        invalid_elem[BLANK_ELEM].add(sample_identifier)
                        continue
                    
                    # only for ascii strings
                    if is_valid_string(instruc):
                        tesseract_ocr_result = pytesseract.image_to_string(cv2.cvtColor(img[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)).strip()
                    
                        tesseract_similarity_ratio = fuzz.ratio(tesseract_ocr_result.lower(), instruc.lower())

                        if tesseract_similarity_ratio < 22:
                            invalid_elem[INCORRECT_TEXT_ANNO].add(sample_identifier)
                            continue
                            
                DRAW = False
                if DRAW:
                    img = cv2.imread(img_path)
                    print(f"{s['question']} || {s['answer']}")
                    if x1 == x2:
                        cv2.circle(img, (x1, y1), 6, (0, 0, 255), 2)
                        cv2.circle(img, (x1, y1), 3, (0, 0, 255), -1)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    cv2.imwrite("test.png", img)
                    1+1

                normalized_box = [round(x1 / W * SCALE), round(y1 / H * SCALE),
                                round(x2 / W * SCALE), round(y2 / H * SCALE)]
                
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                center_x, center_y = round(center_x / W * SCALE), round(center_y / H * SCALE)

                if x1 != x2 and y1 != y2 and random.random() < BOX_PROB:
                    loc = '(' + ','.join(map(str, normalized_box)) + ')'
                    with_box = True
                else:
                    loc = f'({center_x},{center_y})'
                    with_box = False

                short_img_path = img_path[img_path.find(DATASET_NAME):]
                if task == 'textloc':
                    # 仅有1个目标的textloc任务
                    if USE_ACTION_PROMPT:
                        action = CLICK_TEMPLATE.format(target_x=center_x,target_y=center_y)
                        sample = make_actionplanning_sample(task_id=f'autogui_{DATASET_NAME}_textloc_{len(samples)}', global_task=instruc, gt_action=action, history='None', prompt_format_type='aguvis')
                    else:
                        sample = make_textloc_sample(task_id=f"autogui_GUIEnv_{task}_{idx}", text=s['question'], loc=loc, output_tag=OUTPUT_TAG, foramt=LOC_FORMAT)
                        sample["image"], sample["task_attr"], sample["wxh"], sample['unnormalized_box'] = short_img_path, s['question'], f"{W}x{H}", unnormalized_box
                    textloc_cnt += 1
                else:
                    sample = make_ocr_sample(task_id=f"autogui_GUIEnv_{task}_{idx}", text=s['answer'], loc=loc, with_box=with_box)
                    sample["image"], sample["task_attr"], sample["wxh"], sample['unnormalized_box'] = short_img_path, loc, f"{W}x{H}", unnormalized_box
                    ocr_cnt += 1

                guienv_samples.append(sample)

            if group_idx > 0 and group_idx % 10000 == 0 or group_idx == len(samples_each_image) - 1:
                with open(invalid_elem_record_file, 'w') as f:
                    json.dump({k:list(v) for k,v in invalid_elem.items()}, f, indent=2)
                
    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k,v in unique_elems.items() if len(v)])

    report = f"#Samples: {len(guienv_samples)}\n#Unique elements: {num_unique_elems}\n#Valid unique images: {num_valid_imgs}\n#All unique images: {len(unique_elems)}\nInvalid elem ratio: {num_invalid_elem} / {iterated_elem_cnt} = {num_invalid_elem/iterated_elem_cnt:.2f}\ntext_loc_cnt: {textloc_cnt} | ocr_cnt: {ocr_cnt}"
    print(report)

    save_to = os.path.join(SAVE_ROOT, f"guienv-{SPLIT}_s{SCALE}_{len(guienv_samples)//1000}k{'_actformat' if USE_ACTION_PROMPT else ''}.json")

    with open(save_to.replace('.json', '_stats.json'), "w") as f:
        json.dump({'num_samples': len(guienv_samples), '#num_unique_elems': num_unique_elems, '#all_elems': iterated_elem_cnt, '#valid_unique_images': num_valid_imgs, '#all_unique_images': len(unique_elems), 'text_loc_cnt': textloc_cnt, 'ocr_cnt': ocr_cnt, 'num_invalid_elements': num_invalid_elem, 'invalid_elem_types': {k:len(v) for k,v in invalid_elem.items()}}, f, indent=2)

    with open(save_to.replace('.json', '_sample.json'), "w") as f:
        json.dump(random.sample(guienv_samples, 256), f, indent=2)
    
    with open(save_to, "w") as f:
        json.dump(guienv_samples, f)


# GUIChat
if False:
    dataset_path = os.path.join(DATA_ROOT, "GUIChat")
    parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))

    uichat_samples = []

    for parquet_file in parquet_files:
        print(f"Processing {parquet_file}")
        guichat_data = pd.read_parquet(parquet_file)

        img_save_path = os.path.join(dataset_path, "imgs")
        os.makedirs(img_save_path, exist_ok=True)

        samples_each_image = {}
        with open(parquet_file.replace("images.parquet", "data.json"), "r") as f:
            metadata = json.load(f)
            for x in metadata:
                samples_each_image.setdefault(x['image_id'], []).append(x)

        # for x in tqdm(guichat_data.iterrows(), total=len(guichat_data), desc='saving GUIChat imgs ...'):
        #     img_id = x[0]
        #     img_path = os.path.join(img_save_path, f"{img_id}.png")
        #     if os.path.exists(img_path):continue

        #     image = Image.open(BytesIO(x[1][0]))
        #     image.save(os.path.join(img_save_path, f"{img_id}.png"))
        
        llava_samples = []
        for img_id, samples in tqdm(samples_each_image.items(), total=len(samples_each_image), desc='making llava-format samples...'):
            img_path = os.path.join(img_save_path, f"{img_id}.png")
            short_img_path = img_path[img_path.find(DATASET_NAME):]
            # GUIChat的回答里可能包含多个bbox
            for s in samples:
                idx = s['uid']
                
                new_convs = []
                for turn in s['text']:
                    if turn['from'] == 'human':
                        user_query = re.sub(r'<image>.*?</image>', '', turn['value']).strip() + ' (with bbox) (with <box></box> tags)'
                        new_convs.append({'from': 'human', 'value': user_query})
                    elif turn['from'] == 'gpt':
                        matches = re.finditer(r'<box>(.*?)</box>', turn['value']) # a list of boxes
                        
                        # Create a list to hold results with coordinates and indices
                        results = [(match.group(1), match.start(), match.end()) for match in matches]
                        
                        new_gpt_answer = ''
                        
                        DRAW = True
                        sc = cv2.imread(img_path)
                        H, W = sc.shape[:2]

                        last = 0
                        unnormalized_boxes = []

                        for result in results:
                            box, start = result[0], result[1]
                            x1, y1, x2, y2 = list(map(int, box.split()))
                            
                            normalized_box = list(map(lambda x: max(0, min(SCALE-1, round(x/1000*SCALE))), [x1, y1, x2, y2]))

                            # unscale
                            x1, y1, x2, y2 = round(x1/1000 * W), round(y1/1000 * H), round(x2/1000 * W), round(y2/1000 * H)
                            unnormalized_boxes.append([x1, y1, x2, y2])

                            new_box_str = "<box>({})</box>".format(','.join(str(p) for p in normalized_box))
                            new_gpt_answer += turn['value'][last:start] + new_box_str
                            
                            last = result[2]
                            if DRAW: cv2.rectangle(sc, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        new_gpt_answer += turn['value'][last:]
                        
                        if DRAW:
                            cv2.imwrite("test.png", sc)
                            1+1
                            
                        new_convs.append({'from': 'gpt', 'value': new_gpt_answer})
                            
                sample = {
                    'id': f"autogui_GUIChat_webqa_{len(uichat_samples)}",
                    'conversations': new_convs,
                    'image': short_img_path,
                    'unnormalized_boxes': unnormalized_boxes,
                    'wxh': f"{W}x{H}"
                }
                uichat_samples.append(sample)
    
    print(f"Generate {len(uichat_samples)} GUI Chat")
    save_to = os.path.join(SAVE_ROOT, f"guichat-{SPLIT}_llava_{len(uichat_samples)//1000}k.json")

    with open(save_to.replace('.json', '_sample.json'), "w") as f:
        json.dump(random.sample(uichat_samples, 256), f, indent=2)
    
    with open(save_to, "w") as f:
        json.dump(uichat_samples, f)


# GUIAct
# ALLOWED_ACTIONS = ['click', 'scroll', 'input', 'enter', 'tap', 'swipe']
DRAW = False

if True:
    dataset_path = os.path.join(DATA_ROOT, "GUIAct")
    parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))

    actionplan_cnt = intentgnd_cnt = 0

    guiact_samples = []
    for parquet_file in parquet_files:
        if SPLIT not in parquet_file: continue
        if DEVICE_TYPE not in parquet_file: continue
        print(f"Processing {parquet_file}")
        guiact_data = pd.read_parquet(parquet_file)

        img_save_path = os.path.join(dataset_path, "imgs")
        os.makedirs(img_save_path, exist_ok=True)

        # 每张图有且只有一个样本
        # samples_each_image = {}
        trajs = defaultdict(list)
        with open(parquet_file.replace("images.parquet", "data.json"), "r") as f:
            metadata = json.load(f)
            for s in metadata:
                if 'web-single' in parquet_file:
                    task_tag = 'web-single'
                    _, _, img_id, task_type, idx = s['uid'].split("_")
                    sample_id = ep_id = img_id # ; step_id = 0
                elif 'web-multi' in parquet_file:
                    task_tag = 'web-multi'
                    _, _, ep_id, _, step_id = s['uid'].split("_")
                    sample_id = f"{ep_id}-{step_id}"
                elif 'smartphone' in parquet_file:
                    task_tag = 'smartphone'
                    _, _, ep_id, _, step_id = s['uid'].split("_")
                    sample_id = f"{ep_id}-{step_id}"

                s['sample_id'] = sample_id

                # Skip the samples with multiple actions for web-single
                if 'web-single' in parquet_file and len(s['actions_label']) > 1: continue

                trajs[ep_id].append(s)
        
        # sort each traj by image_id
        for traj in trajs.values():
            traj.sort(key=lambda x: x['image_id'])

        # for x in tqdm(guiact_data.iterrows(), total=len(guiact_data), desc='saving GUIAct imgs ...'):
        #     img_id = x[0]
        #     img_path = os.path.join(img_save_path, f"{img_id}.png")
        #     if os.path.exists(img_path):continue

        #     image = Image.open(BytesIO(decode_img_base64(x[1][0])))
        #     image.save(os.path.join(img_save_path, f"{img_id}.png"))


        history_dict = defaultdict(list)

        for item_idx, (ep_id, samples) in tqdm(enumerate(trajs.items()), total=len(trajs), desc=f'making llava-format samples for {len(trajs)} trajectories...'):
            if DEBUG and item_idx % 100 != 0 : continue

            # GUIAct里有安卓轨迹，动作空间为{'input', 'answer', 'swipe', 'tap', 'enter'}
            # GUIAct里有Web轨迹，动作空间为{'click', 'hover', 'answer', 'copy', 'select_text', 'enter', 'scroll', 'input'}
            step_instructions = []

            for step_idx, s in enumerate(samples):
                if s['question'] == "What's on Reddit":
                    1+1
                # UGround跳过非点击和多步样本，而我们这里仅保留第一个合法动作
                image_id, sample_id = s['image_id'], s['sample_id']

                img_path = os.path.join(img_save_path, f"{image_id}.png")

                if DEVICE_TYPE == 'smartphone':
                    is_smartphone = True
                    action_info = s['actions_label']
                    DEVICE_TAG = 'Android'
                else:
                    is_smartphone = False
                    if len(s['actions_label']) == 0:
                        continue
                    action_info = s['actions_label'][0]
                    DEVICE_TAG = 'Web'

                action_name = action_info['name'].lower()

                # if action_name != 'select': continue
                # if action_name not in ALLOWED_ACTIONS: continue

                if DRAW:
                    img = cv2.imread(img_path)

                
                W, H = s['image_size']['width'], s['image_size']['height']
                short_img_path = img_path[img_path.find(DATASET_NAME):]
                step_intruct = ''
                last_action = s['actions_history'].split(' ')[-1].strip()

                # parse action
                task_attr = {'original_action_type': action_name}
                if action_name == 'scroll':
                    down, right = float(action_info['scroll']['related']['down']), float(action_info['scroll']['related']['right'])
                    down_abs, right_abs = abs(down), abs(right)
                    # 跳过左右滚动的样本
                    if SPLIT == 'train' and not(down_abs > 0.01 and right_abs <= 0.05 and down_abs > right_abs): continue

                    direction = 'down' if down > 0 else 'up'
                    
                    task_attr['direction'] = direction
                    
                    distance = discretize_dist(abs(down))
                        
                    if DRAW:
                        cv2.imwrite("test.png", img)

                    # '{{"action_type": "scroll", "direction": "{}", "distance": "{}"}}'.format(direction, distance)
                    
                    action_str = SCROLL_TEMPLATE.format(direction=direction, distance=distance)
                    action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction).replace('Swipe', 'Scroll')
                    step_instructions.append(action_refexp)
                elif action_name == 'swipe':
                    dual_points = action_info['dual_point']['related']
                    from_point, to_point = list(map(float, dual_points['from'][7:-8].split(','))), list(map(float, dual_points['to'][7:-8].split(',')))
                    
                    vertical_shift, horizontal_shift = to_point[1] - from_point[1], to_point[0] - from_point[0]
                    vertical_shift_abs, horizontal_shift_abs = abs(vertical_shift), abs(horizontal_shift)
                    
                    shift_ratio = vertical_shift_abs / (horizontal_shift_abs + 1e-6)
                    if 2.4 > shift_ratio > 0.39:
                        continue

                    # judged the scrolling direction
                    if abs(vertical_shift) > abs(horizontal_shift):
                        direction = 'down' if vertical_shift > 0 else 'up'
                        distance = discretize_dist(abs(vertical_shift))
                    else:
                        direction = 'right' if horizontal_shift > 0 else 'left'
                        distance = discretize_dist(abs(horizontal_shift))

                    start = list(map(lambda arg: round(arg * SCALE), from_point))
                    
                    task_attr['direction'] = direction
                    action_str = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=direction, distance=distance)
                    action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction)
                    step_instructions.append(action_refexp)
                     # f'{{"action_type": "swipe", "start": "({start[0]},{start[1]})", "direction": "{direction}", "distance": "{distance}"}}'

                    history = 'Step 1. I have navigated to a page which I should explore to find the target content required by the task.'
                    if DRAW:
                        before = cv2.imread(img_path)
                        after = cv2.imread(img_path.replace(step_id, f'{int(step_id)+1:02d}'))
                        if after is not None:
                            cv2.imwrite('test.png', np.concatenate([before, after], axis=1))
                elif action_name == 'click':
                    x1,y1,x2,y2 = list(map(float, action_info['element']['related'][5:-6].split(',')))
                    task_attr['bbox'] = [x1,y1,x2,y2]
                    normalized_center = [round((x1+x2)/2 * SCALE), round((y1+y2)/2 * SCALE)]
                    if DRAW:
                        
                        img = cv2.imread(img_path)
                        box = list(map(int, action_info['element']['absolute'][5:-6].split(', ')))
                        cv2.rectangle(img, box[:2], box[2:], (0, 255, 0), 2)
                        cv2.imwrite("test.png", img)
                        
                    action_refexp = random.choice(ACTION_PREFIXES['click']['specific']) + f' the task-related element'
                    step_instructions.append(action_refexp)
                    
                    #target = f'({normalized_center[0]},{normalized_center[1]})'
                    if not (normalized_center[0] < SCALE and normalized_center[1] < SCALE): continue
                    action_str = CLICK_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1])# f'{{"action_type": "click", "target": {target}}}'


                    if last_action == 'input':
                        history = 'Step 1. Type texts into a text field.'
                    else:
                        history = 'Step 1. Go to a page to find the necessary element I should click on.'
                    
                    # make extra intentgnd samples
                    if s['thoughts']:
                        step_intruct = s['thoughts']
                        if INTENTGND:
                            if random.random() <= BOX_PROB:
                                with_box = True
                                x1,y1,x2,y2 = list(map(lambda arg: max(0, min(SCALE-1, round(arg * SCALE))), [x1,y1,x2,y2]))
                                target = f'({x1},{y1},{x2},{y2})'
                            else:
                                center_x, center_y = max(0, min(SCALE-1, round((x1+x2)/2 * SCALE))), max(0, min(SCALE-1, round((y1+y2)/2 * SCALE)))
                                target = f'({center_x},{center_y})'
                                with_box = False
                                
                            intentgnd_sample = make_intentgnd_sample(task_id=f"autogui_GUIAct_{task_tag}-intentgnd_{sample_id}", intent=s['thoughts'], loc=target, with_box=with_box)
                            
                            intentgnd_sample["image"], intentgnd_sample["task_attr"], intentgnd_sample["unnormalzied_box"], intentgnd_sample["wxh"] = short_image_path, s['thoughts'], list(map(float, action_info['element']['absolute'][5:-6].split(','))), f"{W}x{H}"
                            guiact_samples.append(intentgnd_sample); intentgnd_cnt += 1
                        
                elif action_name == 'tap':
                    center = list(map(float, action_info['point']['related'][7:-8].split(',')))
                    task_attr['center'] = center
                    normalized_center = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), center))
                    if DRAW:
                        
                        img = cv2.imread(img_path)
                        pt = list(map(int, action_info['point']['absolute'][7:-8].split(', ')))
                        cv2.circle(img, pt, 5, (0, 255, 0), 2)
                        cv2.circle(img, pt, 2, (0, 255, 0), -1)
                        cv2.imwrite("test.png", img)

                    action_str = CLICK_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1]) # f'{{"action_type": "click", "target": ({normalized_center[0]},{normalized_center[1]})}}'
                    action_refexp = random.choice(ACTION_PREFIXES['click']['specific']) + f' a task-related element'
                    step_instructions.append(action_refexp)

                    if last_action == 'input':
                        history = 'Step 1. Type texts into a text field.'
                    else:
                        history = 'Step 1. Go to a page to find the necessary element I should click on.'

                elif action_name == 'input':
                    text = action_info['text'].strip(' \n\\').replace('"', '\\"')
                    task_attr['text'] = text

                    if is_smartphone:
                        action_str = INPUT_TEMPLATE.format(text=text) # f'{{"action_type": "input_text", "text": "{text}"}}'
                        textfield_desc = ''
                    else:
                        textfield_box = list(map(int, action_info['element']['absolute'][5:-6].split(', ')))
                        if DRAW:
                            img = cv2.imread(img_path)
                            if not is_smartphone:
                                
                                cv2.rectangle(img, (textfield_box[0], textfield_box[1]), (textfield_box[2], textfield_box[3]), (0, 255, 0), 2)
                            cv2.imwrite("test.png", img)
                        textfield_center_x, textfield_center_y = max(0, min(SCALE-1, round((textfield_box[0]+textfield_box[2])/2/W*SCALE))), max(0, min(SCALE-1, round((textfield_box[1]+textfield_box[3])/2/H*SCALE)))
                        action_str = INPUT_TARGET_TEMPLATE.format(target_x=textfield_center_x, target_y=textfield_center_y, text=text)
                        elem_path = s['actions_label'][0]['element_path']
                        textfield_desc = elem_path[3:elem_path.find('")')+1] + ' '

                    action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")
                    step_instructions.append(action_refexp)
                    history = f'Step 1: Focus on the {textfield_desc}text field to input texts.' # 目前只能用这一句概括性的描述
                elif action_name == 'hover': # hover的point属性不准，这里从box计算中心点
                    x1,y1,x2,y2 = list(map(float, action_info['element']['related'][5:-6].split(',')))
                    task_attr['bbox'] = [x1,y1,x2,y2]
                    relative_area = (x2-x1) * (y2-y1)
                    if SPLIT == 'train' and relative_area >= 0.65 or relative_area <= 0.001: continue
                    normalized_center = [round((x1+x2)/2 * SCALE), round((y1+y2)/2 * SCALE)]

                    if DRAW:
                        img = cv2.imread(img_path)
                        pt = list(map(int, action_info['point']['absolute'][7:-8].split(', ')))
                        x1,y1,x2,y2 = list(map(int, action_info['element']['absolute'][5:-6].split(',')))
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
                        cv2.circle(img, pt, 5, (0, 255, 0), 2)
                        cv2.circle(img, pt, 2, (0, 255, 0), -1)
                        cv2.imwrite("test.png", img) 
                        1+1

                    action_refexp = random.choice(ACTION_PREFIXES['hover']['specific']) + f' an task-related element'
                    step_instructions.append(action_refexp)

                    if not (normalized_center[0] < SCALE and normalized_center[1] < SCALE): continue
                    action_str = HOVER_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1]) # f'{{"action_type": "click", "target": ({normalized_center[0]},{normalized_center[1]})}}'

                    if last_action == 'input':
                        history = 'Step 1. Type texts into a text field.'
                    else:
                        history = 'Step 1. Go to a page to find the necessary element I should hover over.'
                elif action_name == 'enter':
                    # check the previous two actions. Enter之前的两个动作有如下组合："{'input-swipe', 'tap-input', 'input-enter', 'input', 'input-tap', 'enter-enter', 'swipe-tap', 'swipe-input', 'swipe-swipe', 'tap', 'tap-tap', 'tap-enter', 'enter-input', 'tap-swipe', 'input-input'}"
                    if is_smartphone:
                        last_twp_actions = '-'.join(re.findall(r'step \d+: (.+)', s['actions_history'])[-2:])
                    else:
                        previous_step_start = s['actions_history'].rfind(':')
                        if previous_step_start != -1:
                            last_twp_actions = '-'.join(s['actions_history'][previous_step_start+1:].strip().split(', ')[-2:])

                    # if last_twp_actions in ['tap-input', 'click-input']:
                    #     history = 'Step 1: Click the text field to input text. Step 2: Input a text query into the text field.' # 目前只能用这一句概括性的描述
                    # elif last_twp_actions in ['swipe-input', 'scroll-input']:
                    #     history = 'Step 1: Scroll the screen to find the task-related text field. Step 2: Input a text query into the text field.'
                    # elif last_twp_actions == 'input-input':
                    #     history = 'Step 1: Input a text query into a text field. Step 2: Input a text query into another field.'
                    # elif last_twp_actions == 'copy-input':
                    #     history = 'Step 1: Copy-paste a text query into a text field.'
                    # else:
                    #     history = 'Step 1: Input a text query into the focused text field.'

                    action_str = PRESSKEY_TEMPLATE.format(key='Enter')
                    action_refexp = random.choice(PRESSKEY_PREFIXES['Enter'])
                    step_instructions.append(action_refexp)
                elif action_name == 'answer':
                    task_attr['answer'] = ""
                    if action_info["text"] == 'task complete':
                        history = "Step 1. Navigate to the destination screen specified by the user's task"
                        action_str = STATUS_TEMPLATE.format(goal_status="successful", answer="")
                        task_attr['goal_status'] = 'successful'
                        action_refexp = random.choice(TASK_STATUS_SENTENCES['successful'])

                    elif action_info["text"] == 'task impossible':
                        history = "Step 1. Navigate to the destination screen specified by the user's task"
                        action_str = STATUS_TEMPLATE.format(goal_status="infeasible", answer="")
                        task_attr['goal_status'] = 'infeasible'
                        action_refexp = random.choice(TASK_STATUS_SENTENCES['infeasible'])
                    else:
                        history = "Step 1. Find the content that contains the answer."
                        answer = action_info["text"].replace("\n", "\\n").replace('"', "'")
                        action_str = STATUS_TEMPLATE.format(goal_status="successful", answer=answer)
                        task_attr['goal_status'] = 'successful'
                        action_refexp = random.choice(TASK_STATUS_SENTENCES['successful']) + " Answer based on the image information."
                        task_attr['answer'] = action_info["text"]

                     # '{{"action_type": "status", "goal_status": "successful", "answer": "{}"}}'
                    step_instructions.append(action_refexp)

                    if DRAW:
                        img = cv2.imread(img_path)
                        cv2.imwrite("test.png", img)
                elif action_name == 'select_text':
                    dual_points = action_info['dual_point']['related']
                    from_point, to_point = list(map(float, dual_points['from'][7:-8].split(','))), list(map(float, dual_points['to'][7:-8].split(',')))
                    task_attr['from'], task_attr['to'] = from_point, to_point

                    norm_from_point = list(map(lambda arg: round(arg * SCALE), from_point))
                    norm_to_point = list(map(lambda arg: round(arg * SCALE), to_point))
                    
                    # if action_info["text"] == 'task complete':
                    #     history = "Step 1. Navigate to the destination screen specified by the user's task"
                    #     answer = ""
                    # else:
                    #     history = "Step 1. Find the content that contains the answer."
                    #     answer = action_info["text"].replace("\n", "\\n").replace('"', "'")
                    
                    history = "Step 1. Find the textual content I should select."
                    action_refexp = random.choice(ACTION_PREFIXES['drag']['specific']) + ' the task-related texts'
                    step_instructions.append(action_refexp)
                    
                    if not (norm_from_point[0] < SCALE and norm_from_point[1] < SCALE and norm_to_point[0] < SCALE and norm_to_point[1] < SCALE): continue
                    action_str = DRAG_TEMPLATE.format(start_x=norm_from_point[0], start_y=norm_from_point[1], end_x=norm_to_point[0], end_y=norm_to_point[1]) # '{{"action_type": "status", "goal_status": "successful", "answer": "{}"}}'

                    if DRAW:
                        img = cv2.imread(img_path)
                        H, W = img.shape[:2]
                        real_from_x, real_from_y = round(from_point[0] * W), round(from_point[1] * H)
                        real_to_x, real_to__y = round(to_point[0] * W), round(to_point[1] * H)
                        cv2.rectangle(img, (real_from_x, real_from_y), (real_to_x, real_to__y), (0, 255, 0), 2)
                        cv2.imwrite("test.png", img)
                        1+1
                elif action_name == 'select':
                    x1,y1,x2,y2 = list(map(float, action_info['element']['related'][5:-6].split(',')))
                    normalized_center = [round((x1+x2)/2 * SCALE), round((y1+y2)/2 * SCALE)]
                    
                    history = "Step 1. Navigate to the screen that displays the element I should select."
                    action_str = SELECT_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1], value=action_info['text']) # '{{"action_type": "status", "goal_status": "successful", "answer": "{}"}}'

                    action_refexp = random.choice(SELECT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=action_info['text'].strip("'"), target="the menu")
                    step_instructions.append(action_refexp)

                    if DRAW:
                        before = cv2.imread(img_path)
                        real_x1, real_y1, real_x2, real_y2 = round(x1*W), round(y1*H), round(x2*W), round(y2*H)
                        cv2.rectangle(before, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 0), 2)
                        cv2.imwrite('test.png', before)
                        1+1
                elif action_name == 'copy':
                    history = "Step 1. Find the textual content I should copy."
                    task_attr['key_comb'] = 'ctrl-c'
                    action_str = KEYCOMB_TEMPLATE.format(key_combination='ctrl+c')
                    action_refexp = random.choice(KEYCOMB_PREFIXES['ctrl-c'])
                    step_instructions.append(action_refexp)
                else:
                    raise Exception(f"Unknown action: {action_name}")

                action = ast.literal_eval(action_str)
                action_type = action['action_type']
                
                # Merge history
                _, clean_prev_step_instructions = keep_unique_actions(step_instructions[:step_idx])
                retained_history = clean_prev_step_instructions[-MAX_PREV_ACT:]
                history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(retained_history, start=max(1,len(clean_prev_step_instructions) - MAX_PREV_ACT+1))) if len(retained_history) > 0 else 'None'

                # If the thoughts is a complex sentence, use it as the action refexp
                if 'web-single' in parquet_file and ' and ' not in s['thoughts'] and len(s['thoughts'].split()) > 3:
                    step_instructions[-1] = action_refexp = s['thoughts']
                
                action_refexp = action_refexp.strip(' .')
                if USE_ACTION_REFEXP:
                    action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"

                if is_smartphone:
                    # GUIAct smartphone子集里，任务为疑问句形式的样本，其answer动作标注有问题，主要体现在UI屏幕有时候并未体现出任务已经结束。所以这里任务为疑问句的样本。
                    # if any(s['question'].split()[0].lower().startswith(ques_word) for ques_word in ['what', 'how', 'who', 'where']):
                    #     continue
                    sample = make_actionplanning_sample(
                        task_id=f"autogui_GUIAct-{task_tag}_planning_{sample_id}",
                        global_task=s['question'],
                        history=history_str,
                        gt_action='Action: ' + action_str,
                        with_cot=False,
                        use_action_refexp=USE_ACTION_REFEXP,
                        device_tag=DEVICE_TAG
                        )
                elif 'web' in task_tag:
                    if 'single' in task_tag:
                        sample_id += f'-task{step_idx}'
                    sample = make_actionplanning_sample_web(
                        task_id=f"autogui_GUIAct-{task_tag}_planning_{sample_id}",
                        global_task=s['question'],
                        history=history_str,
                        gt_action='Action: ' + action_str,
                        with_cot=False,
                        use_action_refexp=USE_ACTION_REFEXP,
                        device_tag=DEVICE_TAG
                        )

                action['step_idx'] = 0 if 'single' in task_tag else step_idx
                sample['task'], sample['action_type'], sample['task_attr'], sample['history'], sample['step_instruction'], sample['action_refexp'], sample["image"], sample["step_info"], sample["wxh"], sample["device"] = s['question'], action_type, task_attr, step_instructions[:-1], step_instructions[-1], action_refexp, short_img_path, action, f"{W}x{H}", 'web' if DEVICE_TYPE == 'web' else 'mobile'
                actionplan_cnt += 1

                guiact_samples.append(sample)

    print(f"Generate {actionplan_cnt} action-planning and {intentgnd_cnt} intent gnd")
    save_to = os.path.join(SAVE_ROOT, f"guiact-{DEVICE_TYPE}-{SPLIT}{'_wActRef' if USE_ACTION_REFEXP else ''}_s{SCALE}_{len(guiact_samples)}.json")

    # 统计动作分布
    act_stats = defaultdict(int)
    for x in guiact_samples:
        if 'planning' in x['id']:
            act_stats[re.search(r'"action_type":\s*"([^"]+)"', x['conversations'][1]['value']).group(1)] += 1

    with open(save_to.replace('.json', '_stats.json'), "w") as f:
        json.dump({'num_samples': len(guiact_samples), 'actionplan_cnt': actionplan_cnt, 'intentgnd_cnt': intentgnd_cnt, 'action_stats':act_stats}, f, indent=2)

    print(f"Save {len(guiact_samples)} to {save_to}")
    with open(save_to.replace('.json', '_sample.json'), "w") as f:
        json.dump(random.sample(guiact_samples, min(len(guiact_samples), 256)), f, indent=2, ensure_ascii=False)
    
    with open(save_to, "w") as f:
        json.dump(guiact_samples, f, indent=2, ensure_ascii=False)


# answer动作例子
# {
#   "uid": "uid_record_02472_step_04",
#   "image_id": "uid_record_02472_step_04",
#   "image_size": {
#     "width": 1280,
#     "height": 598
#   },
#   "question": "未成年人网络保护条例的第一条是什么？",
#   "actions_history": "step 0: scroll, click\nstep 1: hover\nstep 2: click\nstep 3: select_text, scroll, copy",
#   "logs": "第一条为了营造有利于未成年人身心健康的网络环境，保障未成年人合法权益，根据《中华人民共和国未成年人保护法》、《中华人民共和国网络安全法》、《中华人民共和国个人信息保护法》等法律，制定本条例。",
#   "thoughts": "",
#   "actions_label": [
#     {
#       "name": "answer",
#       "text": "第一条为了营造有利于未成年人身心健康的网络环境，保障未成年人合法权益，根据《中华人民共和国未成年人保护法》、《中华人民共和国网络安全法》、《中华人民共和国个人信息保护法》等法律，制定本条例。', 'thoughts': '', 'actions_label': [{'name': 'answer', 'text': '未成年人网络保护条例的第一条是为了营造有利于未成年人身心健康的网络环境，保障未成年人合法权益，根据《中华人民共和国未成年人保护法》、《中华人民共和国网络安全法》、《中华人民共和国个人信息保护法》等法律，制定本条例。"
#     }
#   ]
# }

# {
#   "uid": "uid_record_07248_step_09",
#   "image_id": "uid_record_07248_step_09",
#   "image_size": {
#     "width": 1536,
#     "height": 703
#   },
#   "question": "Find me some information about the book To Kill a Mockingbird and its author.",
#   "actions_history": "step 0: click, input\nstep 1: enter, click\nstep 2: click\nstep 3: click\nstep 4: select_text, copy\nstep 5: select_text, copy\nstep 6: scroll\nstep 7: scroll\nstep 8: select_text, copy",
#   "logs": "Harper LeeThe unforgettable novel of a childhood in a sleepy Southern town and the crisis of conscience that rocked it. \"To Kill A Mockingbird\" became both an instant bestseller and a critical success when it was first published in 1960. It went on to win the Pulitzer Prize in 1961 and was later made into an Academy Award-winning film, also a classic.Harper Lee, known as Nelle, was born in the Alabama town of Monroeville, the youngest of four children of Amasa Coleman Lee and Frances Cunningham Finch Lee. Her father, a former newspaper editor and proprietor, was a lawyer who served on the state legislature from 1926 to 1938. As a child, Lee was a tomboy and a precocious reader, and enjoyed the friendship of her schoolmate and neighbor, the young Truman Capote.",
#   "thoughts": "",
#   "actions_label": [
#     {
#       "name": "answer",
#       "text": "This book was  written by Harper Lee. The unforgettable novel of a childhood in a sleepy Southern town and the crisis of conscience that rocked it. \"To Kill A Mockingbird\" became both an instant bestseller and a critical success when it was first published in 1960. It went on to win the Pulitzer Prize in 1961 and was later made into an Academy Award-winning film, also a classic.\nHarper Lee, known as Nelle, was born in the Alabama town of Monroeville."
#     }
#   ]
# }