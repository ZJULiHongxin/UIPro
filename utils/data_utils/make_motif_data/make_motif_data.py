"""
MOTIF dataset conversion and sample generation utilities.

This module processes the MOTIF dataset (from HuggingFace) to generate model-ready
samples for element grounding and action planning tasks. It handles:
- Downloading/loading the dataset.
- Validating UI elements (size, visibility, bounds).
- Formatting samples into instruction-tuning format.
"""

import os
import json
import cv2
import re
import magic
import random
import sys
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

# Add parent directory to path to allow imports if running as script
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from task_prompt_lib import *
from misc import (
    is_pure_color,
    resize_image
)

# ==============================================================================
# Configuration & Constants
# ==============================================================================

DATASET_NAME = 'MOTIF'
SCALE = 1000

# Paths
# Note: Adjust paths as necessary for your environment
IMG_DIR = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/images/motif/"
SAVE_ROOT_DIR = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"

# Processing Flags
DEBUG = False
SKIP_CHECKING = False
USE_ACTION_PROMPT = False # If True, format as action planning; else element grounding
OUTPUT_TAG = '' # Tag for element grounding output, e.g., 'Please output its center coordinates.'


# ==============================================================================
# Main Processing Logic
# ==============================================================================

def process_motif_dataset():
    """
    Main function to process the MOTIF dataset.
    """
    print(f"Processing {DATASET_NAME} dataset...")
    print(f"Reading images from: {IMG_DIR}")
    print(f"Saving results to: {SAVE_ROOT_DIR}")

    # Ensure save directory exists
    os.makedirs(SAVE_ROOT_DIR, exist_ok=True)

    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace...")
    try:
        data = load_dataset("HongxinLi/MOTIF_automation_preprocess", split='train_au_tu')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Debug mode: sample a subset
    if DEBUG:
        print("Debug mode enabled: processing first 200 samples.")
        # If data is a Dataset object, select the first 200 indices
        if hasattr(data, 'select'):
            data = data.select(range(min(len(data), 200)))
        else:
            data = data[:200]

    # Initialize trackers
    unique_elems = defaultdict(list)
    samples = []
    
    # Load or initialize invalid element records
    invalid_elem_record_file = os.path.join(SAVE_ROOT_DIR, 'invalid_elem_record.json')
    if os.path.exists(invalid_elem_record_file):
        invalid_elem = json.load(open(invalid_elem_record_file))
        for k, v in invalid_elem.items():
            invalid_elem[k] = set(v)
    else:
        invalid_elem = {
            TOO_SMALL_ELEMENT: set(),
            INVALID_ELEM_BOX: set(),
            INVALID_ELEM_CONTENT: set(),
            BLANK_ELEM: set(),
            EMPTY_ELEM_TEXT: set(),
            OVERLY_LENGTHY_ELEM_TEXT: set(),
            DUPLICATE_ELEMEMNT: set()
        }

    # Iterate over the dataset
    for group_idx, item in tqdm(enumerate(data), total=len(data), desc=f"Processing {DATASET_NAME}"):
        image_id = item['image_id']
        img_path = os.path.join(IMG_DIR, image_id + '.jpg')
        
        # Check if image exists
        if not os.path.exists(img_path):
            # In a real pipeline, we might want to log this
            continue

        # Get image dimensions
        try:
            # Attempt to use magic to read header only for speed
            file_info = magic.from_file(img_path)
            # Regex to find resolution, e.g., "precision 8, 1080x1920"
            res = re.search(r'precision 8, (\d+)x(\d+)', file_info)
            if res:
                W, H = list(map(int, res.groups()))
            else:
                # Fallback to opencv
                img = cv2.imread(img_path)
                if img is None:
                    continue
                H, W = img.shape[:2]
        except Exception:
            # Fallback if magic fails or file is corrupt
            continue

        img = None # Lazy load only if needed for pixel checks

        # Check bounding boxes
        if len(item['bbox_mc']) == 0:
            continue

        # Current dataset structure seems to imply one main instruction/box per entry in this split
        bbox = item['bbox_mc'][0]
        instruction = item['instr'].strip()
        
        sample_identifier = f"{image_id}|{instruction}"

        # Check if already marked as invalid
        is_known_invalid = False
        for v in invalid_elem.values():
            if sample_identifier in v:
                is_known_invalid = True
                break
        if is_known_invalid:
            continue

        # Unnormalized box [x1, y1, x2, y2]
        unnorm_box = list(map(round, [bbox[0]*W, bbox[1]*H, bbox[2]*W, bbox[3]*H]))
        
        if unnorm_box not in unique_elems[image_id]:
            unique_elems[image_id].append(unnorm_box)
            
        if not SKIP_CHECKING:
            # Filter out specific actions (e.g. swipes/flips) if they are not supported
            if any(k in instruction for k in ['flip', 'swipe']):
                invalid_elem[INVALID_ELEM_CONTENT].add(sample_identifier)
                continue

            # Check if element is too small (threshold 0.5% of dimension)
            if abs(bbox[0] - bbox[2]) <= 0.005 or abs(bbox[1] - bbox[3]) <= 0.005:
                invalid_elem[TOO_SMALL_ELEMENT].add(sample_identifier)
                continue

            # Check for valid normalized coordinates (0-1 range and x1 < x2, y1 < y2)
            if not (0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1 and 0 <= bbox[2] <= 1 and 0 <= bbox[3] <= 1 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                invalid_elem[INVALID_ELEM_BOX].add(sample_identifier)
                continue
                
            if len(instruction) == 0:
                invalid_elem[EMPTY_ELEM_TEXT].add(sample_identifier)
                continue

            # Check for pure color (blank) elements
            if img is None:
                img = cv2.imread(img_path)
            
            if img is not None and is_pure_color(img, unnorm_box):
                invalid_elem[BLANK_ELEM].add(sample_identifier)
                continue
        
        # Calculate center for action/grounding
        norm_center = [
            max(0, min(SCALE-1, round((bbox[0]+bbox[2])/2*SCALE))),
            max(0, min(SCALE-1, round((bbox[1]+bbox[3])/2*SCALE)))
        ]
        center_str = f'({norm_center[0]},{norm_center[1]})'
        
        # Create Sample
        if USE_ACTION_PROMPT:
            # Format as an action planning task (Intent -> Action)
            action_str = CLICK_TEMPLATE.format(target_x=norm_center[0], target_y=norm_center[1])
            sample = make_actionplanning_sample(
                task_id=f'autogui_{DATASET_NAME}_intentgnd_{len(samples)}',
                global_task=instruction,
                gt_action=action_str,
                history='None',
                prompt_format_type='aguvis'
            )
        else:
            # Format as an element grounding task (Text -> Location)
            sample = make_intentgnd_sample(
                task_id=f'autogui_{DATASET_NAME}_intentgnd_{len(samples)}',
                intent=instruction,
                loc=center_str,
                output_tag=OUTPUT_TAG
            )

        # Enriched metadata
        # Ensure path compatibility
        sample['image'] = f'motif/{image_id}.jpg'
        sample['unnormalized_box'] = unnorm_box
        sample['task_attr'] = instruction
        
        samples.append(sample)

        # Save invalid element record periodically
        if (group_idx > 0 and group_idx % 10000 == 0) or group_idx == len(data) - 1:
            with open(invalid_elem_record_file, 'w') as f:
                json.dump({k: list(v) for k, v in invalid_elem.items()}, f, indent=2)

    # --------------------------------------------------------------------------
    # Statistics and Saving
    # --------------------------------------------------------------------------
    
    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k, v in unique_elems.items() if len(v)])
    
    report = (
        f"#Samples: {len(samples)}\n"
        f"#Unique elements: {num_unique_elems}\n"
        f"#Valid unique images: {num_valid_imgs}\n"
        f"#All unique images: {len(unique_elems)}\n"
        f"Invalid elem ratio: {num_invalid_elem} / {len(data)} = {num_invalid_elem/len(data) if len(data) > 0 else 0:.2f}"
    )
    print("Processing Complete.")
    print(report)

    # Construct filenames
    suffix_debug = '_debug' if DEBUG else ''
    suffix_format = '_actformat' if USE_ACTION_PROMPT else ''
    # Naming convention: {DATASET}_{COUNT}k{FLAGS}.json
    base_filename = f"{DATASET_NAME}_{len(samples)//1000}k{suffix_debug}{suffix_format}.json"
    save_path = os.path.join(SAVE_ROOT_DIR, base_filename)
    
    print(f"Saving to {save_path}")

    # Save Statistics
    stats_info = {
        'num_samples': len(samples),
        'num_unique_elems': num_unique_elems,
        'num_all_elems': len(data),
        'num_valid_unique_images': num_valid_imgs,
        'num_all_unique_images': len(unique_elems),
        'text_loc_count': len(samples), # Legacy key
        'ocr_count': 0,
        'elemclass_count': 0,
        'intentgnd_count': 0,
        'widgetlist_count': 0,
        'num_invalid_elements': num_invalid_elem
    }
    with open(save_path.replace('.json', '_info.json'), "w") as f:
        json.dump(stats_info, f, indent=2)

    # Save Sample Preview
    with open(save_path.replace(".json", "_sample.json"), 'w') as f:
        json.dump(random.sample(samples, min(len(samples), 128)), f, indent=2)
        
    # Save Full Dataset
    with open(save_path, 'w') as f:
        json.dump(samples, f, indent=2)

if __name__ == "__main__":
    process_motif_dataset()
