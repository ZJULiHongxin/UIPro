"""
OmniAct dataset processing and sample generation utilities.

This module processes the OmniAct dataset to generate model-ready samples for:
- Action planning (high-level tasks to PyAutoGUI actions)
- Element grounding (optional)

It reads task files, image metadata, and annotation boxes, then formats them
into instruction-tuning samples with normalized coordinates and action descriptions.
"""

import os
import json
import random
import glob
import re
import magic
import cv2
import traceback
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import sys

# Add parent directory to path to allow imports if running as script
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from task_prompt_lib import *
from misc import (
    find_smallest_box_containing_point,
    resize_image,
    is_pure_color,
    is_valid_string,
    contain_network_errors
)

# ==============================================================================
# Configuration & Constants
# ==============================================================================

# Dataset Paths
DATASET_NAME = 'OmniAct'
# Note: These paths should be adjusted based on the actual environment
ROOT_DIR = "/mnt/vdb1/hongxin_li/OmniAct"
SAVE_ROOT_DIR = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"

# Processing Settings
SPLIT = 'train'  # Options: 'train', 'test'
DEVICE = 'all'  # Options: 'web', 'desktop', 'all'
SCALE = 1000

# Feature Flags
SKIP_CHECKING = False
USE_ACTION_PROMPT = True
ENABLE_ELEMENT_GROUNDING = True  # Set to True to generate element grounding samples
ENABLE_PLANNING = True            # Set to True to generate action planning samples
USE_ACTION_REFEXP = True          # Use referring expressions for actions

# Element Grounding Settings
OUTPUT_TAG = ''  # e.g., 'Please output its center coordinates.'

# Planning Settings
ALLOWED_ACTIONS = []  # Empty list implies all actions are allowed


# ==============================================================================
# Helper Functions
# ==============================================================================

def extract_coords(text):
    """
    Extracts coordinates from a string formatted like '...(x,y)...'.

    Args:
        text (str): The input string containing coordinates in parentheses.

    Returns:
        list: A list [x, y] as floats. Returns [-1, -1] if parsing fails or empty.
    """
    start_idx = text.find('(')
    end_idx = text.rfind(')')
    
    if start_idx == -1 or end_idx == -1:
        return [-1, -1]
        
    coords_str = text[start_idx+1:end_idx].strip(' ,')
    
    if len(coords_str) == 0:
        return [-1, -1]
    
    try:
        x, y = list(map(float, coords_str.split(',')))
        return [x, y]
    except ValueError:
        return [-1, -1]


def get_scroll_distance_label(amount):
    """
    Categorizes scroll amount into 'short' or 'long'.

    Args:
        amount (int): The scroll amount.

    Returns:
        str: 'short' or 'long'.
    """
    if amount <= 500:
        return 'short'
    else:
        return 'long'


# ==============================================================================
# Main Processing Logic
# ==============================================================================

def process_omniact_dataset():
    """
    Main function to process the OmniAct dataset and generate training samples.
    """
    print(f"Processing {DATASET_NAME} dataset...")
    print(f"Split: {SPLIT}, Device: {DEVICE}, Scale: {SCALE}")
    print(f"Reading data from: {ROOT_DIR}")
    print(f"Saving results to: {SAVE_ROOT_DIR}")

    # Load dataset index
    split_file_path = os.path.join(ROOT_DIR, f"{SPLIT}.json")
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Split file not found: {split_file_path}")
    
    data = json.load(open(split_file_path))

    samples = []
    unique_elems = {}
    num_iterated_elems = 0
    
    elemgnd_count = 0
    planning_count = 0
    action_stats = defaultdict(int)
    used_instructions = set()

    # Load or initialize invalid element records
    invalid_elem_record_file = os.path.join(SAVE_ROOT_DIR, 'invalid_elem_record.json')
    if os.path.exists(invalid_elem_record_file):
        invalid_elem = json.load(open(invalid_elem_record_file))
        for k, v in invalid_elem.items():
            invalid_elem[k] = set(v)
    else:
        invalid_elem = {
            KEY_TOO_SMALL: set(),
            KEY_INVALID_BOX: set(),
            KEY_INVALID_CONTENT: set(),
            KEY_BLANK_ELEM: set(),
            KEY_EMPTY_TEXT: set(),
            KEY_OVERLY_LENGTHY: set(),
            KEY_DUPLICATE: set()
        }

    # Ensure save directory exists
    os.makedirs(SAVE_ROOT_DIR, exist_ok=True)

    # Iterate through dataset samples
    for sample_idx, sample_info in tqdm(enumerate(data.values()), total=len(data), desc="Processing samples"):
        # Filter by device type
        if DEVICE != 'all' and f'{DEVICE}/' not in sample_info['task']:
            continue

        task_file = os.path.join(ROOT_DIR, sample_info["task"])
        
        # Fix image path for web data
        if 'web/' in sample_info["image"]:
            sample_info["image"] = sample_info["image"].replace('screen_', 'screen')

        img_path = os.path.join(ROOT_DIR, sample_info["image"])
        
        # Track unique images
        if sample_info["image"] not in unique_elems:
            unique_elems[sample_info["image"]] = []

        short_img_name = img_path[img_path.find(DATASET_NAME):]
        
        # Get image dimensions using magic to avoid reading the whole file if possible
        try:
            # Using regex to parse 'width x height' from magic output
            file_info = magic.from_file(img_path)
            res = re.search('(\d+) x (\d+)', file_info)
            if res:
                W, H = list(map(int, res.groups()))
            else:
                # Fallback to cv2 if magic fails to parse resolution
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Could not read image: {img_path}")
                H, W = img.shape[:2]
        except Exception as e:
            print(f"Error reading image dimensions for {img_path}: {e}")
            continue

        img = None  # Lazy loading only if needed for checks

        # Load bounding boxes
        box_file_path = sample_info["box"] if 'desktop/' in sample_info["box"] else (sample_info["box"].replace('screen_', 'screen')).replace('.json', '_boxes.json')
        box_full_path = os.path.join(ROOT_DIR, box_file_path)
        
        if not os.path.exists(box_full_path):
            print(f"Warning: Box file not found {box_full_path}")
            continue

        with open(box_full_path, 'r') as f:
            boxes = list(json.load(f).values())
        
        # ----------------------------------------------------------------------
        # Element Grounding Generation
        # ----------------------------------------------------------------------
        if ENABLE_ELEMENT_GROUNDING:
            num_iterated_elems += len(boxes)
            
            for elem_idx, elem_info in enumerate(boxes):
                sample_identifier = sample_info["image"] + f'|{elem_idx}'
                
                # Check if element is already known as invalid
                is_invalid = False
                for v in invalid_elem.values():
                    if sample_identifier in v:
                        is_invalid = True
                        break
                if is_invalid:
                    continue
                
                unnorm_bbox = elem_info['top_left'] + elem_info['bottom_right']

                if unnorm_bbox not in unique_elems[sample_info["image"]]:
                    unique_elems[sample_info["image"]].append(unnorm_bbox)
            
                instruction = elem_info['label']
                bbox = [unnorm_bbox[0]/W, unnorm_bbox[1]/H, unnorm_bbox[2]/W, unnorm_bbox[3]/H]
                
                instruction_box_key = f'{instruction}|{str(list(map(round, unnorm_bbox)))}'
                
                if not SKIP_CHECKING:
                    if instruction_box_key in used_instructions:
                        invalid_elem[KEY_DUPLICATE].add(sample_identifier)
                        continue
                    
                    if not isinstance(instruction, str):
                        invalid_elem[KEY_INVALID_CONTENT].add(sample_identifier)
                        continue

                    # Check for very small elements
                    if abs(bbox[0] - bbox[2]) <= 0.005 or abs(bbox[1] - bbox[3]) <= 0.005:
                        invalid_elem[KEY_TOO_SMALL].add(sample_identifier)
                        continue

                    # Check for valid normalized coordinates
                    if not (0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1 and 0 <= bbox[2] <= 1 and 0 <= bbox[3] <= 1 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                        invalid_elem[KEY_INVALID_BOX].add(sample_identifier)
                        continue
                        
                    if len(instruction) == 0:
                        invalid_elem[KEY_EMPTY_TEXT].add(sample_identifier)
                        continue

                    # Check for blank (pure color) elements
                    if img is None:
                        img = cv2.imread(img_path)

                    if is_pure_color(img, unnorm_bbox):
                        invalid_elem[KEY_BLANK_ELEM].add(sample_identifier)
                        continue
                
                used_instructions.add(instruction_box_key)

                norm_center = [
                    max(0, min(SCALE-1, round((bbox[0]+bbox[2])/2*SCALE))), 
                    max(0, min(SCALE-1, round((bbox[1]+bbox[3])/2*SCALE)))
                ]
                center_str = f'({norm_center[0]},{norm_center[1]})'
                
                if USE_ACTION_PROMPT:
                    action_str = CLICK_TEMPLATE.format(target_x=norm_center[0], target_y=norm_center[1])
                    query = TURN_GND_INTO_PLANNING_PROMPT.format(instruc=instruction) if len(instruction.strip().split()) == 1 else instruction
                    sample = make_actionplanning_sample(
                        task_id=f'autogui_{DATASET_NAME}_elemgnd_{len(samples)}', 
                        global_task=query, 
                        gt_action=action_str, 
                        history='None', 
                        prompt_format_type='aguvis'
                    )
                else:
                    sample = make_elemgnd_sample(
                        task_id=f'autogui_{DATASET_NAME}_elemgnd_{len(samples)}', 
                        text=instruction, 
                        loc=center_str, 
                        output_tag=OUTPUT_TAG, 
                        format=None
                    )

                sample['image'] = img_path.split(f'{DATASET_NAME}/')[1]
                sample['unnormalized_box'] = unnorm_bbox
                sample['task_attr'] = instruction
                samples.append(sample)
                elemgnd_count += 1

        # Save invalid element cache periodically
        if (sample_idx > 0 and sample_idx % 100 == 0) or sample_idx == len(data) - 1:
            with open(invalid_elem_record_file, 'w') as f:
                json.dump({k: list(v) for k, v in invalid_elem.items()}, f, indent=2)
            
        # ----------------------------------------------------------------------
        # Action Planning Generation
        # ----------------------------------------------------------------------
        if ENABLE_PLANNING:
            # Parse task file to get instruction and PyAutoGUI script
            if not os.path.exists(task_file):
                print(f"Warning: Task file not found {task_file}")
                continue

            with open(task_file, 'r') as f:
                raw_content = f.read()
                # Parse Task Description
                # Format: "Task: [Description]\n..."
                task_description = raw_content[5:raw_content.find('\n')].strip()
                # Parse Script content
                # Format: "...pyautogui.[action]..."
                plan_script = raw_content[raw_content.find('yautogui.')-1:].strip()
            
            # Extract steps (ignore moveto unless it's the main action)
            steps = [step.strip() for step in plan_script.split("\n")]

            for step_idx, step in enumerate(steps):
                if 'moveto(' not in step.lower():
                    action_raw = step
                    break
            else:
                continue
            
            # Skip if action is not in allowed list (if list is not empty)
            if len(ALLOWED_ACTIONS) > 0 and any(act in action_raw for act in ALLOWED_ACTIONS):
                continue

            action_type = 'unknown'
            action_str = ''
            action_refexp = ''
            action_attr = {}

            # ------------------------------------------------------------------
            # Parse Specific Actions
            # ------------------------------------------------------------------
            
            # Helper to get interaction element info
            def get_interacted_element(center_x, center_y, boxes_list):
                # Find smallest box containing the point
                # Ensure boxes have top_left and bottom_right
                box_coords = np.array([b['top_left'] + b['bottom_right'] for b in boxes_list])
                interacted_box, idx = find_smallest_box_containing_point(np.array([center_x, center_y]), boxes=box_coords)
                label = None
                if idx is not None:
                    label = boxes_list[idx]['label'].replace('_', ' ')
                return interacted_box, label

            if 'click' in action_raw.lower() and 'right' not in action_raw.lower() and 'double' not in action_raw.lower():
                action_type = 'click'
                x, y = extract_coords(action_raw)
                if SPLIT == 'train' and x == -1: continue
                
                interacted_box, elem_label = get_interacted_element(x, y, boxes)
                
                action_attr = {'action_type': 'click', 'target': [x, y], 'unnormalized_box': interacted_box, 'elem_label': elem_label}
                action_str = CLICK_TEMPLATE.format(target_x=max(0, min(SCALE-1, round(x/W*SCALE))), target_y=max(0, min(SCALE-1, round(y/H*SCALE))))
                
                prefix = random.choice(ACTION_PREFIXES[action_type]['specific'])
                target_desc = f' the element "{elem_label.strip()}"' if elem_label is not None else ' the task-relevant element'
                action_refexp = prefix + target_desc

            elif 'moveto' in action_raw.lower():
                action_type = 'hover'
                x, y = extract_coords(action_raw)
                
                interacted_box, elem_label = get_interacted_element(x, y, boxes)

                action_attr = {'action_type': 'hover', 'target': [x, y], 'unnormalized_box': interacted_box, 'elem_label': elem_label}
                action_str = HOVER_TEMPLATE.format(target_x=max(0, min(SCALE-1, round(x/W*SCALE))), target_y=max(0, min(SCALE-1, round(y/H*SCALE))))
                
                prefix = random.choice(ACTION_PREFIXES['hover']['specific'])
                target_desc = f' the element "{elem_label.strip()}"' if elem_label is not None else ' the task-relevant element'
                action_refexp = prefix + target_desc

            elif 'rightclick' in action_raw.lower():
                action_type = 'right_click'
                x, y = extract_coords(action_raw)
                if SPLIT == 'train' and x == -1: continue
                
                interacted_box, elem_label = get_interacted_element(x, y, boxes)

                action_attr = {'action_type': 'right_click', 'target': [x, y], 'unnormalized_box': interacted_box, 'elem_label': elem_label}
                action_str = RIGHTCLICK_TEMPLATE.format(target_x=max(0, min(SCALE-1, round(x/W*SCALE))), target_y=max(0, min(SCALE-1, round(y/H*SCALE))))
                
                prefix = random.choice(ACTION_PREFIXES[action_type]['specific'])
                target_desc = f' the element "{elem_label.strip()}"' if elem_label is not None else ' the task-relevant element'
                action_refexp = prefix + target_desc

            elif 'doubleclick' in action_raw.lower():
                action_type = 'double_click'
                x, y = extract_coords(action_raw)
                if SPLIT == 'train' and x == -1: continue
                
                interacted_box, elem_label = get_interacted_element(x, y, boxes)

                action_attr = {'action_type': 'double_click', 'target': [x, y], 'unnormalized_box': interacted_box, 'elem_label': elem_label}
                action_str = DOUBLECLICK_TEMPLATE.format(target_x=max(0, min(SCALE-1, round(x/W*SCALE))), target_y=max(0, min(SCALE-1, round(y/H*SCALE))))
                
                prefix = random.choice(ACTION_PREFIXES[action_type]['specific'])
                target_desc = f' the element "{elem_label.strip()}"' if elem_label is not None else ' the task-relevant element'
                action_refexp = prefix + target_desc

            elif 'dragto' in action_raw.lower():
                action_type = 'drag'
                
                last_moveto = steps[step_idx-1] # 'pyautogui.moveTo(1533.0,688.5)\npyautogui.dragTo(2088.0,663.5, button="left")'
                start_x, start_y = extract_coords(last_moveto)
                if action_raw[action_raw.find('('):action_raw.find(')')+1].count(',') > 1:
                    action_raw = action_raw.rsplit(',',1)[0]+')'
                end_x, end_y = extract_coords(action_raw)

                if SPLIT == 'train' and end_x == -1: continue
                
                interacted_box, elem_label = get_interacted_element(start_x, start_y, boxes)

                action_attr = {'action_type': 'drag', 'start': [start_x, start_y], 'end': [end_x, end_y], 'unnormalized_box': interacted_box, 'elem_label': elem_label}
                action_str = DRAG_TEMPLATE.format(start_x=max(0, min(SCALE-1, round(start_x/W*SCALE))), start_y=max(0, min(SCALE-1, round(start_y/H*SCALE))), end_x=max(0, min(SCALE-1, round(end_x/W*SCALE))), end_y=max(0, min(SCALE-1, round(end_y/H*SCALE))))
                
                target_desc = f'the element "{elem_label.strip()}"' if elem_label is not None else 'the task-relevant element'
                action_refexp = random.choice(DRAG_PHRASES['specific']).format(target=target_desc)

            elif 'write' in action_raw.lower():
                action_type = 'input_text'
                # Extract text between write(" and ")
                text_match = re.search(r'write\("(.+?)"\)', action_raw)
                text = text_match.group(1).strip() if text_match else ""
                
                action_attr = {'action_type': 'input_text', 'text': text}
                action_str = INPUT_TEMPLATE.format(text=text)
                
                action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")

            elif 'press' in action_raw.lower():
                action_type = 'press_key'
                # Extract key between press(" and ")
                key_match = re.search(r'press\("(.+?)"\)', action_raw)
                key = key_match.group(1).strip(' "') if key_match else ""
                
                action_attr = {'action_type': 'press_key', 'key': key}
                action_str = PRESSKEY_TEMPLATE.format(key=key)
                
                # Handling key prefixes safely
                if key.lower() in PRESSKEY_PREFIXES:
                    action_refexp = random.choice(PRESSKEY_PREFIXES[key.lower()])
                else:
                    action_refexp = f"Press the {key} key"

            elif 'hotkey' in action_raw.lower():
                action_type = 'hotkey'
                # Extract keys
                keys_match = re.search(r'hot[kK]ey\("(.+?)"\)', action_raw)
                if keys_match:
                    keys = keys_match.group(1).split(',')
                    keys = [k.strip('" ') for k in keys]
                    keycomb = '-'.join(keys)
                else:
                    keycomb = "unknown"

                action_attr = {'action_type': 'hotkey', 'key': keycomb}
                action_str = KEYCOMB_TEMPLATE.format(key_combination=keycomb)
                
                if keycomb.lower() in KEYCOMB_PREFIXES:
                    action_refexp = random.choice(KEYCOMB_PREFIXES[keycomb.lower()])
                else:
                    action_refexp = f"Press {keycomb}"

            elif '.scroll' in action_raw.lower():
                action_type = 'scroll'
                # Extract amount
                amount_match = re.search(r'scroll\((-?\d+)\)', action_raw)
                amount = int(amount_match.group(1)) if amount_match else 0
                
                direction = 'up' if amount > 0 else 'down'
                action_attr = {'action_type': 'scroll', 'direction': direction}
                action_str = SCROLL_TEMPLATE.format(direction=direction, distance=get_scroll_distance_label(amount))
                
                action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction).replace('Swipe', 'Scroll')

            elif '.hscroll' in action_raw.lower():
                action_type = 'scroll'
                # Extract amount
                amount_match = re.search(r'hscroll\((-?\d+)\)', action_raw)
                amount = int(amount_match.group(1)) if amount_match else 0

                direction = 'right' if amount > 0 else 'left'
                action_attr = {'action_type': 'scroll', 'direction': direction}
                action_str = SCROLL_TEMPLATE.format(direction=direction, distance=get_scroll_distance_label(amount))
                
                action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction).replace('Swipe', 'Scroll')

            else:
                print(f"Unknown action encountered: {action_raw}")
                continue

            # Update stats
            action_stats[action_type] += 1

            # Format Reference Expression
            action_refexp = action_refexp.strip(' .')
            if USE_ACTION_REFEXP:
                action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"

            # Create Sample
            sample = make_actionplanning_sample(
                task_id=f'autogui_{DATASET_NAME}_planning_{len(samples)}',
                global_task=task_description,
                gt_action='Action: ' + action_str,
                history='None',
                prompt_format_type='simple',
                with_cot=False,
                use_action_refexp=USE_ACTION_REFEXP,
            )

            # Enriched metadata
            sample['task'] = task_description
            sample["action_type"] = action_type
            sample['action_refexp'] = action_refexp
            sample["history"] = []
            sample["image"] = short_img_name
            sample["task_attr"] = action_attr
            sample["wxh"] = f"{W}x{H}"
            sample['plan'] = plan_script
            
            samples.append(sample)
            planning_count += 1

    # --------------------------------------------------------------------------
    # Post-processing and Saving
    # --------------------------------------------------------------------------

    # Resample press_key and hotkey actions if they are underrepresented
    if ENABLE_PLANNING:
        rare_actions = [x for x in samples if x['action_type'] in ['press_key', 'hotkey']]
        if rare_actions:
            samples = rare_actions * 10 + samples

    # Calculate statistics
    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k, v in unique_elems.items() if len(v)])
    
    # Construct filename
    ref_tag = '_wActRef' if USE_ACTION_REFEXP else ''
    save_filename = f"{DATASET_NAME}-{DEVICE}-{SPLIT}{ref_tag}_s{SCALE}_{len(samples)}.json"
    save_path = os.path.join(SAVE_ROOT_DIR, save_filename)
    
    # Save a small subset for manual inspection
    sample_preview_path = save_path.replace(".json", "_sample.json")
    print(f"Saving {len(samples)} samples to {sample_preview_path} (preview)")
    with open(sample_preview_path, 'w') as f:
        # Use min to avoid error if samples < 160
        json.dump(random.sample(samples, min(len(samples), 160)), f, indent=2)

    # Save full dataset
    print(f"Saving full dataset to {save_path}")
    with open(save_path, 'w') as f:
        json.dump(samples, f, indent=2)

    # Save statistics
    stats_path = save_path.replace('.json', '_stats.json')
    stats = {
        'num_samples': len(samples),
        'num_unique_elems': num_unique_elems,
        'num_total_elems_iterated': num_iterated_elems,
        'num_invalid_elements': num_invalid_elem,
        'num_valid_unique_images': num_valid_imgs,
        'num_all_unique_images': len(unique_elems),
        'elemgnd_count': elemgnd_count,
        'planning_count': planning_count,
        'action_stats': action_stats,
        'invalid_elem_types': {k: len(v) for k, v in invalid_elem.items()}
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    process_omniact_dataset()
