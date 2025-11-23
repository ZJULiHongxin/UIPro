"""
AITZ dataset preprocessing utilities.

This module prepares training samples for multiple tasks:
1) Given image, task, and history, predict the next action with reasoning.
2) Given image, task, history, and current click intent, predict ground coords.
3) (Optional) UI screen captioning.

Notes from dataset inspection (retained from original script):
- epid: 17693907265291426996 step 0 annotation issue: the thought says "swipe down from the top of the screen to access quick settings," but the action description is actually "swipe up." 
- epid: 9088265727317240175 steps 8-10 annotation issue: the UI shows using a swipe to move the text cursor, but the generated action result summary did not reflect that intent.
- epid: 17652164182256576003 step 1: the swipe is used to drag a slider; future action space should allow fine-grained dragging.
- AITZ images have relatively low resolution; resizing is unnecessary.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
import logging
from collections import defaultdict
from typing import List

import glob
import cv2
import magic
import numpy as np
from tqdm import tqdm

# Ensure sibling imports resolve at runtime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *  # noqa: F403,F401

from misc import generate_negative_action_plans, keep_unique_actions  # noqa: F401


DATASET_NAME: str = 'AITZ'
DRAW: bool = False
ROOT: str = "/mnt/vdb1/hongxin_li/AITZ"
SAVE_DIR: str = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"
SCALE: int = 1000
SPLIT: str = ['train', 'test'][0]

INTENTGND: bool = False
UICAPTION: bool = False

USE_ACTION_REFEXP: bool = True # Set as True when inserting action referring expressions into the ground truth responses. For example, "Action: <|object_ref_start|>click on the "newegg" search result located at the top left<|object_ref_end|>\n{"action_type": "click", "target": (225,176)}"

DEVICE_TAG: str = 'Android'


def _safe_parse_ui_positions(ui_positions_str: str) -> List[List[int]]:
    """Safely parse UI element bounding boxes from string to [x1, y1, x2, y2].

    Expected input is a Python literal list of boxes in [y, x, h, w] format.
    Uses ast.literal_eval first and falls back to eval to preserve legacy
    behavior for non-literal strings.
    """
    try:
        boxes_raw = ast.literal_eval(ui_positions_str)
    except Exception:  # noqa: BLE001 - preserve behavior with broad except
        logging.warning("Falling back to eval() for ui_positions parsing.")
        boxes_raw = eval(ui_positions_str)

    return [[p[1], p[0], p[1] + p[3], p[0] + p[2]] for p in boxes_raw]


def _compute_history_string(step_instructions: List[str], step_idx: int) -> str:
    """Build a compact deduplicated history string for current step."""
    _, clean_prev = keep_unique_actions(step_instructions[:step_idx])
    retained = clean_prev[-MAX_PREV_ACT:]
    if not retained:
        return 'None'
    start_i = max(1, len(clean_prev) - MAX_PREV_ACT + 1)
    return ' '.join(
        f"Step {i}. {instr.strip(' .')}." for i, instr in enumerate(retained, start=start_i)
    )

def make_aitz_data() -> None:
    """Generate AITZ training samples and write them to disk.

    Iterates through all episode metadata files, builds action-planning samples
    (with and without chain-of-thought), and optionally UI caption or intent
    grounding samples. Outputs summary statistics and a small random subset for
    quick inspection.
    """
    data_dir = os.path.join(ROOT, SPLIT)
    save_to_dir = os.path.join(SAVE_DIR, f"{DATASET_NAME}_processed")

    all_ep_meta_files = sorted(glob.glob(os.path.join(data_dir, "*", "*", "*.json")))
    
    planning_cnt = uicaption_cnt = intentgnd_cnt = 0

    samples = []
    invalid_samples = []

    for ep_idx, ep_meta_file in tqdm(enumerate(all_ep_meta_files), total=len(all_ep_meta_files)):
        with open(ep_meta_file, 'r') as f:
            ep_meta = json.load(f)
        
        step_instructions = []

        for step_idx, step_info in enumerate(ep_meta):
            img_path = os.path.join(ROOT, step_info['image_full_path'].split('aitw_with_gpt/')[1])
            W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_path)).groups(1)))

            short_img_path = f'{DATASET_NAME}/' + step_info['image_full_path'].split('aitw_with_gpt/')[1]
            
            if step_idx < len(ep_meta)-1:
                short_nextimg_path = f'{DATASET_NAME}/' + ep_meta[step_idx+1]['image_full_path'].split('aitw_with_gpt/')[1]
            else: short_nextimg_path = short_img_path

            step_info['coat_action_desc'] = step_info['coat_action_desc'].strip(' .')
            
            step_instructions.append(step_info['coat_action_desc'])
            
            # Load UI element positions (used only when DRAW is True)
            boxes = _safe_parse_ui_positions(step_info['ui_positions'])

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
                        sample = make_intentgnd_sample(task_id=f"autogui_AITZ_intentgnd_{step_info['episode_id']}-{step_info['step_id']}", intent=action_refexp, loc=f'({start_x},{start_y})', with_box=False)
                        sample['step_info'], sample['task_attr'], sample['image'] = step_info, step_info['coat_action_desc'], short_img_path
                        samples.append(sample); intentgnd_cnt += 1
                    
                    if DRAW:
                        img = cv2.imread(img_path)
                        for box in boxes:
                            cv2.rectangle(img, box[:2], box[2:], color=(0,255,0), thickness=3)
                        cv2.circle(img, [round(from_point[1]*W),round(from_point[0]*H)], color=(0,255,0), radius=8, thickness=3)
                        cv2.imwrite('test.png', img)

            elif step_info['result_action_type'] == 3:
                text = step_info['result_action_text'].strip()
                if text.count('"') % 2 != 0: text = text.strip('"')
                if text.count("'") % 2 != 0: text = text.strip("'")
                text = text.strip(' \\').replace("\n", "\\n").replace('"', '\\"')
                
                if not text:
                    continue

                action_str = INPUT_TEMPLATE.format(text=text)
                action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")

            elif step_info['result_action_type'] == 10:
                action_str = STATUS_TEMPLATE.format(goal_status='successful', answer='')
                action_refexp = random.choice(TASK_STATUS_SENTENCES['successful'])

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
            history_str = _compute_history_string(step_instructions, step_idx)

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

            # ui caption
            if UICAPTION:
                sample = make_uicaption_sample(task_id=f"autogui_AITZ_uicaption_{step_info['episode_id']}-{step_info['step_id']}", ui_caption=step_info['coat_screen_desc'])
                sample['step_info'], sample['image'] = step_info, short_img_path
                samples.append(sample); uicaption_cnt += 1

    action_stats = defaultdict(int)
    for x in samples:
        if 'planning' not in x['id']: continue
        action_stats[re.search(r'"action_type":\s*"([^"]+)"', x['conversations'][1]['value']).group(1)] += 1

    report = f"Total samples: {len(samples)+len(invalid_samples)} Valid samples: {len(samples)} | planning: {planning_cnt} | uicaption: {uicaption_cnt} | intentgnd: {intentgnd_cnt}"
    print(report)
    os.makedirs(save_to_dir, exist_ok=True)
    save_to_file = os.path.join(save_to_dir, f"AITZ_{SPLIT}{'_wActRef' if USE_ACTION_REFEXP else ''}_s{SCALE}_{len(samples)}.json")

    with open(save_to_file.replace(".json", "_stats.json"), "w") as f:
        json.dump({"total_sample_cnt": len(samples)+len(invalid_samples), "valid_sample_cnt": len(samples), "planning": planning_cnt, "action_stats": action_stats, "uicaption": uicaption_cnt, "intentgnd": intentgnd_cnt, "invalid_samples": invalid_samples}, f, indent=2)

    with open(save_to_file.replace(".json", "_sample.json"), "w") as f:
        json.dump(random.sample(samples, 160), f, indent=2)

    print(f"save {len(samples)} samples to {save_to_file}")
    with open(save_to_file, "w") as f:
        json.dump(samples, f, indent=2)

if __name__ == '__main__':
    make_aitz_data()