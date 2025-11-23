#     "clickable_elements": [
        # {
        #     "bbox": [
        #         1310,
        #         133,
        #         1440,
        #         231
        #     ],
        #     "xml_desc": [],
        #     "functionality": ""
        # }]
import magic, os, json, random, glob, re, ast

import cv2, numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *
from misc import resize_image, find_smallest_box_containing_point
import numpy as np

from collections import defaultdict

from misc import generate_negative_action_plans, keep_unique_actions

DATASET_NAME = 'AMEX'

random.seed(666)
SPLIT = ['train', 'test'][0]
DEBUG = False

PLATFORM_INDEX = 1
DATA_ROOT = [
    f"/mnt/vdb1/hongxin_li/{DATASET_NAME}",
    f"/mnt/jfs/copilot/lhx/ui_data/{DATASET_NAME}"
][PLATFORM_INDEX]
INSTRUC_DIR = os.path.join(DATA_ROOT, 'instruction_anno')
ELEM_DIR = os.path.join(DATA_ROOT, 'element_anno')
IMAGE_DIR = os.path.join(DATA_ROOT, 'screenshot')

SAVE_ROOT = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp",
    "/data/hongxin_li/scaling_exp/"
][PLATFORM_INDEX]
save_to_root = os.path.join(SAVE_ROOT, f'{DATASET_NAME}_processed')
os.makedirs(save_to_root, exist_ok=True)

SAVE_IMAGE_DIR = [
    f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}",
    "/mnt/jfs/copilot/lhx/ui_data/AMEX/screenshot_processed"
][PLATFORM_INDEX]

os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)

RESUME = True

PLANNING = True
FUNCGND = False
FUNCREF = False
REWARDMODEL_EVAL = False

DRAW = False

LONGEST = 1260

SCALE = 1000
BOX_PROB = 0.3

CLICK_STR_PLACEHOLDER = "click on the task-related element."

USE_ACTION_REFEXP = True
DEVICE_TAG = 'Android'

def make_amex_data():    
    all_ep_meta_files = sorted(glob.glob(os.path.join(INSTRUC_DIR, "*.json")))

    planning_cnt = funcgnd_cnt = funcref_cnt = reward_model_eval_cnt = 0

    samples, rewardmodel_eval_samples = [], []
    num_iterated_imgs = 0
    invalid_imgs = []
    invalid_annos = defaultdict(int)
    iterated_anno_cnt = invalid_anno_cnt = 0

    # make planning samples
    if PLANNING:
        if DEBUG: all_ep_meta_files = all_ep_meta_files[:10]
        for ep_idx, ep_meta_file in tqdm(enumerate(all_ep_meta_files), total=len(all_ep_meta_files)):
            with open(ep_meta_file, 'r') as f:
                ep_meta = json.load(f)

            prev_actions = []

            # 'image_path' = '2024_3_18_17_19_e8ba0101cbc74242b48af70a57dafdf5-1.png'
            step_instructions = []

            for step_idx, step_info in enumerate(ep_meta['steps']):
                num_iterated_imgs += 1; iterated_anno_cnt += 1
                action_type = step_info['action']
                last_action = ep_meta['steps'][step_idx-1]['action'] if step_idx > 0 else None

                img_file = os.path.join(IMAGE_DIR, step_info["image_path"])

                save_img_to = os.path.join(SAVE_IMAGE_DIR, step_info["image_path"])

                if RESUME and os.path.exists(save_img_to):
                    ORIG_W, ORIG_H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_file)).groups(1)))
                    W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(save_img_to)).groups(1)))
                    ratio = H / ORIG_H
                else:
                    img = cv2.imread(img_file)
                    ORIG_H, ORIG_W = img.shape[:2]
                    if LONGEST != -1:
                        img, ratio = resize_image(img, LONGEST)
                        if not os.path.exists(save_img_to):
                            cv2.imwrite(save_img_to, img)
                    H, W = img.shape[:2]

                anno_file = os.path.join(ELEM_DIR, step_info["image_path"].replace(".png",".json"))
                with open(anno_file, 'r') as f:
                    anno = json.load(f)

                boxes, elem_texts = [], []
                for elem in anno['clickable_elements'] + anno['scrollable_elements']:
                    x1, y1, x2, y2 = list(map(lambda x: round(x* ratio), elem['bbox']))
                    if ((x2-x1)*(y2-y1)) / (H*W) >= 0.65: continue
                    
                    x1 = max(0, min(W-1, x1))
                    y1 = max(0, min(H-1, y1))
                    x2 = max(0, min(W-1, x2))
                    y2 = max(0, min(H-1, y2))
                    boxes.append([x1,y1,x2,y2])
                    elem_texts.append(elem['xml_desc'][0] if len(elem['xml_desc']) else None)

                short_img_path = save_img_to[save_img_to.find(DATASET_NAME):]

                if action_type == 'TAP':
                    target_x, target_y = step_info["touch_coord"][0] * ratio, step_info["touch_coord"][1] * ratio
                    norm_target_x, norm_target_y = max(0, min(SCALE-1, round(target_x / W * SCALE))), max(0, min(SCALE-1, round(target_y / H * SCALE)))
                    
                    action_str = CLICK_TEMPLATE.format(target_x=norm_target_x, target_y=norm_target_y)

                    # if last_action == 'TYPE':
                    #     history = 'Step 1. type "{}" into a text field.'.format(ep_meta['steps'][step_idx-1]['type_text'])
                    # else:
                    #     history = 'Step 1. go to a page to find the necessary element I should click on.'

                    # Find the text of the click target
                    action_refexp = this_action_str = CLICK_STR_PLACEHOLDER
                    if len(boxes):
                        _, box_idx = find_smallest_box_containing_point([target_x, target_y], boxes)
                        if box_idx is not None and (elem_text := elem_texts[box_idx]) is not None:
                            this_action_str = f'click on the "{elem_text}" element'
                            action_refexp = random.choice(ACTION_PREFIXES['click']['specific']) + f' the element "{elem_text}"'
                        else:
                            1+1

                    step_instructions.append(this_action_str)

                    if DRAW:
                        img = cv2.imread(save_img_to)
                        for box in boxes:
                            cv2.rectangle(img, box[:2], box[2:], color=(0,255,0), thickness=3)
                        cv2.circle(img, [round(target_x),round(target_y)], color=(0,0,255), radius=8, thickness=3)
                        cv2.imwrite('test.png', img)

                    if REWARDMODEL_EVAL:
                        neg_actions = generate_negative_action_plans(gt_act_type='click', W=W, H=H, scale=SCALE, gt_center=[round(target_x),round(target_y)], boxes=boxes)
                        
                elif action_type == "SWIPE":
                    from_x, from_y = step_info["touch_coord"][0] * ratio, step_info["touch_coord"][1] * ratio
                    to_x, to_y = step_info["lift_coord"][0] * ratio, step_info["lift_coord"][1] * ratio
                    
                    norm_from_x, norm_from_y = max(0, min(SCALE-1, round(from_x / W * SCALE))), max(0, min(SCALE-1, round(from_y / H * SCALE)))
                    vertical_shift, horizontal_shift = (to_y - from_y) / H, (to_x - from_x) / W

                    # judged the scrolling direction
                    if abs(vertical_shift) > abs(horizontal_shift):
                        direction = 'down' if vertical_shift > 0 else 'up'
                        distance = discretize_dist(abs(vertical_shift))
                    else:
                        direction = 'right' if horizontal_shift > 0 else 'left'
                        distance = discretize_dist(abs(horizontal_shift))
                    
                    action_str = SWIPE_TEMPLATE.format(start_x=norm_from_x, start_y=norm_from_y, direction=direction, distance=distance)
                    action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction)

                    # history = 'Step 1. I have navigated to a page which I should explore to find the target content required by the task.'
                    step_instructions.append(f"swipe {direction} to find task-related content")

                    if REWARDMODEL_EVAL:
                        neg_actions = generate_negative_action_plans(gt_act_type='swipe', W=W, H=H, scale=SCALE, gt_center=[round(from_x),round(from_y)], boxes=boxes, direction=direction)

                    if DRAW:
                        print(step_info)
                        print(action_str)
                        cv2.circle(img, (round(from_x), round(from_y)), 5, (0, 0, 255), -1)
                        cv2.circle(img, (round(to_x), round(to_y)), 5, (0, 255, 0), -1)
                        cv2.arrowedLine(img, (round(from_x), round(from_y)), (round(to_x), round(to_y)), (255, 0, 0), 2)
                        after = cv2.imread(img_file.replace(f"{step_idx+1}.png", f"{step_idx+2}.png"))
                        after, _ = resize_image(after, max_size=LONGEST)
                        merge = np.concatenate([img, after], axis=1)
                        cv2.imwrite("test.png", merge)
                
                elif action_type == "TYPE":
                    text = step_info["type_text"].replace("\n", "\\n").strip()
                    action_str = INPUT_TEMPLATE.format(text=text).replace('"', '\"')
                    action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")

                    if len(step_instructions) and step_instructions[step_idx-1] == CLICK_STR_PLACEHOLDER:
                        step_instructions[step_idx-1] = 'focus on the text field to input texts' # 目前只能用这一句概括性的描述
                        
                    if not text:
                        step_instructions.append(f'type task-related texts into the text field')
                        continue
                    else:
                        step_instructions.append(f'type "{text}" into the text field')

                    if REWARDMODEL_EVAL:
                        neg_actions = generate_negative_action_plans(gt_act_type='input_text', W=W, H=H, scale=SCALE, boxes=boxes, text=text)

                elif action_type == "TASK_COMPLETE":
                    action_str = STATUS_TEMPLATE.format(goal_status="successful", answer='')
                    action_refexp = random.choice(TASK_STATUS_SENTENCES['successful'])

                    # history = "Step 1. I have reached the destination and found the content required by the user's task."
                    if REWARDMODEL_EVAL:
                        neg_actions = generate_negative_action_plans(gt_act_type='status', W=W, H=H, scale=SCALE, boxes=boxes, goal_status="successful")
                    step_instructions.append("Task complete")
                elif action_type == "TASK_IMPOSSIBLE":
                    action_str = STATUS_TEMPLATE.format(goal_status="infeasible", answer='')
                    action_refexp = random.choice(TASK_STATUS_SENTENCES['infeasible'])

                    #history = "Step 1. I have reached the destination and found the content required by the user's task."
                    if REWARDMODEL_EVAL:
                        neg_actions = generate_negative_action_plans(gt_act_type='status', W=W, H=H, scale=SCALE, boxes=boxes, goal_status="infeasible")
                    step_instructions.append("Task infeasible")
                elif action_type == "PRESS_ENTER":
                    action_str = PRESSKEY_TEMPLATE.format(key='Enter')
                    action_refexp = random.choice(PRESSKEY_PREFIXES['Enter'])

                    if step_idx >= 2:
                        last_two_actions = '-'.join(prev_actions[max(0, step_idx-2):step_idx])

                    if last_two_actions in ['click-input_text']:
                        step_instructions[-2] = step_instructions[-2].replace(CLICK_STR_PLACEHOLDER, 'click the text field to input text')
                        #history = 'Step 1. click the text field to input text. Step 2. input a text query into the text field.' # 目前只能用这一句概括性的描述
                    elif last_two_actions in ['swipe-input_text']:
                        history = 'Step 1. swipe the screen to find the task-related text field. Step 2. input a text query into the text field.'
                    elif last_two_actions == 'input_text-input_text':
                        history = 'Step 1. input a text query into a text field. Step 2. Input a text query into another field.'
                    else:
                        history = 'Step 1. input a text query into the focused text field.'
                    
                    step_instructions.append(f'Press Enter')

                elif action_type == 'PRESS_BACK':
                    action_str = NAVIGATE_BACK_TEMPLATE
                    action_refexp = random.choice(NAVIGATE_BACK_PREFIXES)
                    step_instructions.append(f'Navigate back')

                elif action_type == 'PRESS_HOME':
                    action_str = NAVIGATE_HOME_TEMPLATE
                    action_refexp = random.choice(NAVIGATE_HOME_PREFIXES)
                    step_instructions.append(f'Navigate home')
                else:
                    raise ValueError(f"Unknown action type: {action_type}")

                action_type = ast.literal_eval(action_str)['action_type']
                prev_actions.append(action_type)

                # Merge history
                _, clean_prev_step_instructions = keep_unique_actions(step_instructions[:step_idx])
                retained_history = clean_prev_step_instructions[-MAX_PREV_ACT:]
                history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(retained_history, start=max(1,len(clean_prev_step_instructions) - MAX_PREV_ACT+1))) if len(retained_history) > 0 else 'None'

                if USE_ACTION_REFEXP:
                    action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"

                sample = make_actionplanning_sample(
                    task_id=f'autogui_{DATASET_NAME}_planning_{ep_meta["episode_id"]}-{step_idx}',
                    global_task=ep_meta['instruction'],
                    gt_action=f"Action: {action_str}",
                    history=history_str,
                    use_action_refexp=USE_ACTION_REFEXP,
                    device_tag=DEVICE_TAG
                )

                step_info['step_instruction'] = step_instructions[step_idx]
                sample["task"], sample["action_type"], sample["step_info"], sample["action_refexp"], sample['step_instruction'], sample["image"], sample["wxh"], sample["orig_wxh"], sample['history'],  sample["bboxes"] = ep_meta['instruction'], action_type, step_info, action_refexp, step_instructions[step_idx], short_img_path, f"{W}x{H}", f"{ORIG_W}x{ORIG_H}", step_instructions[:step_idx], [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in boxes]

                # Debug the bboxes
                if False:
                    img = cv2.imread(save_img_to)
                    for bbox in sample['bboxes']:
                        cv2.rectangle(img, (int(bbox[0] * W), int(bbox[1] * H)), (int(bbox[2] * W), int(bbox[3] * H)), (0, 0, 255), 2)
                    cv2.imwrite("test.png", img)

                samples.append(sample)
                planning_cnt += 1

                if REWARDMODEL_EVAL:
                    if neg_actions is None:
                        neg_actions = generate_negative_action_plans(gt_act_type=action_type, W=W, H=H, scale=SCALE, boxes=boxes)

                    neg_sample = {'id': f'autogui_{DATASET_NAME}_rewardmodeleval_{reward_model_eval_cnt}', "image": short_img_path, 'ep_id': ep_meta['episode_id'], 'step_id': step_idx, 'task': ep_meta['instruction'], 'step_instruction': step_instructions[step_idx], "action_type": action_type, "history": step_instructions[:step_idx], 'gt_action': action_str, 'neg_actions': neg_actions, "wxh": f"{W}x{H}"}
                    rewardmodel_eval_samples.append(neg_sample); reward_model_eval_cnt += 1

    # make funcgnd samples
    if FUNCGND or FUNCREF:
        anno_files = glob.glob(os.path.join(ELEM_DIR, "*.json"))
        
        anno_files = anno_files[:500] if DEBUG else anno_files
        for anno_file in tqdm(anno_files, total=len(anno_files), desc='make annos'):
            num_iterated_imgs += 1
            
            with open(anno_file, 'r') as f:
                anno = json.load(f)
            
            img_path = os.path.join(IMAGE_DIR, anno['image_path'])
            
            iterated_anno_cnt += len(anno['clickable_elements'] + anno["scrollable_elements"])

            if not os.path.exists(img_path):
                invalid_imgs.append([img_path, 'non-existent image'])
                invalid_annos['non-existent image'] += len(anno['clickable_elements'] + anno["scrollable_elements"])
                continue
            img = cv2.imread(img_path)
            img, ratio = resize_image(img, LONGEST)
                
            save_img_to = os.path.join(SAVE_IMAGE_DIR, anno['image_path'])
            cv2.imwrite(save_img_to, img)
            H, W = img.shape[:2]
            short_img_path = save_img_to[save_img_to.find(DATASET_NAME):]

            for elem_info in anno['clickable_elements'] + anno["scrollable_elements"]:
                func = elem_info.get('functionality', '')
                if not func or 'Non-interactive' in func:
                    invalid_anno_cnt += 1
                    invalid_annos['invalid functionality annotation'] += 1
                    continue

                x1, y1, x2, y2 = list(map(lambda x: x* ratio, elem_info['bbox']))
                
                if x1 >= x2 or y1 >= y2:
                    invalid_anno_cnt += 1
                    invalid_annos['invalid box coordinates'] += 1
                    continue

                if random.random() <= BOX_PROB:
                    normalized_box = max(0, min(SCALE-1, round(x1/W*SCALE))), max(0, min(SCALE-1, round(y1/H*SCALE))), max(0, min(SCALE-1, round(x2/W*SCALE))), max(0, min(SCALE-1, round(y2/H*SCALE)))
                    loc = '({})'.format(','.join([str(x) for x in normalized_box]))
                    with_box = True
                else:
                    center_x, center_y = max(0, min(SCALE-1, round((x1+x2)/2/W*SCALE))), max(0, min(SCALE-1, round((y1+y2)/2/H*SCALE)))
                    loc = '({})'.format(','.join([str(x) for x in [center_x, center_y]]))
                    with_box = False

                sample = make_funcgnd_sample(f"autogui_{DATASET_NAME}_funcgnd_{funcgnd_cnt}", elem_desc=f"This element {func[0].lower() + func[1:]}", loc=loc, with_box=with_box)
                
                sample["image"], sample["task_attr"], sample['unnormalized_box'], sample["wxh"] = short_img_path, func, [x1,y1,x2,y2], f"{W}x{H}"
                samples.append(sample); funcgnd_cnt += 1

                sample = make_funcref_sample(f"autogui_{DATASET_NAME}_funcref_{funcref_cnt}", elem_desc=f"This element {func[0].lower() + func[1:]}", loc=loc, with_box=with_box)
                
                sample["image"], sample["task_attr"], sample['unnormalized_box'], sample["wxh"] = short_img_path, loc, [x1,y1,x2,y2], f"{W}x{H}"
                samples.append(sample); funcref_cnt += 1

    print(f"Generate {planning_cnt} action-planning and {funcgnd_cnt} funcgnd and {funcref_cnt} funcref")

    save_to = os.path.join(save_to_root, f"{DATASET_NAME}-{SPLIT}{'_wActRef' if USE_ACTION_REFEXP else ''}_s{SCALE}_{len(samples)}.json")

    # 统计动作分布
    act_stats = defaultdict(int)
    for x in samples:
        if 'planning' in x['id']:
            act_stats[re.search(r'"action_type":\s*"([^"]+)"', x['conversations'][1]['value']).group(1)] += 1

    with open(save_to.replace('.json', '_stats.json'), "w") as f:
        json.dump({'num_total_imgs': num_iterated_imgs, 'num_invalid_imgs': len(invalid_imgs), 'num_iterated_annos': iterated_anno_cnt, 'invalid_anno_cnt': sum(invalid_annos.values()), 'num_samples': len(samples), 'planning_cnt': planning_cnt, 'funcgnd_cnt': funcgnd_cnt, 'funcref_cnt': funcref_cnt, 'action_stats':act_stats, 'invalid_anno_stats': invalid_annos, 'invalid_anno_files': invalid_imgs}, f, indent=2)

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

make_amex_data()