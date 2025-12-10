"""
RefExp dataset preprocessing utilities.

This module processes the RefExp (Reference Expression) dataset to generate training samples
for UI element grounding tasks. The dataset contains UI screenshots with referring expressions
that describe UI elements, and the goal is to localize elements based on natural language descriptions.

The code generates two types of tasks:
1) Element Grounding (elemgnd): Localize elements based on descriptive text (no action verbs).
2) Intent Grounding (intentgnd): Localize elements based on action-oriented instructions.

The preprocessing pipeline includes:
- Loading RefExp dataset from HuggingFace
- Validating bounding boxes and element visibility
- Filtering invalid samples (too small, blank, invalid content, etc.)
- Normalizing coordinates to a fixed scale (default: 1000x1000)
- Generating task-specific training samples with proper formatting
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Set, Tuple

import cv2
import magic
from datasets import load_dataset
from tqdm import tqdm

from utils.data_utils.misc import VerbExtactor, is_pure_color
from utils.data_utils.task_prompt_lib import *  # noqa: F403,F401

# ============================================================================
# Configuration Constants
# ============================================================================

# Dataset configuration
DATASET_NAME: str = 'RefExp'
SCALE: int = 1000  # Coordinate normalization scale (output coordinates in range [0, SCALE-1])

# Directory paths
IMG_DIR: str = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/images/rico/"
SAVE_ROOT: str = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"

# Task generation flags
USE_ACTION_PROMPT: bool = False  # If True, use action planning format instead of intent grounding
DEBUG: bool = False  # If True, only process first 200 samples for testing
OUTPUT_TAG: str = ''  # Optional output format tag (e.g., 'Please output its center coordinates.')

# Validation flags
SKIP_CHECKING: bool = False  # If True, skip all validation checks (not recommended)

# ============================================================================
# Initialize Dataset and Directories
# ============================================================================

# Load RefExp dataset from HuggingFace
ds = load_dataset("ivelin/ui_refexp_saved", split='train')

# Create output directories
os.makedirs(SAVE_ROOT, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def _load_invalid_records(record_file: str) -> Dict[str, Set[str]]:
    """Load previously recorded invalid sample identifiers from disk.
    
    Args:
        record_file: Path to the JSON file containing invalid sample records.
        
    Returns:
        Dictionary mapping invalid type names to sets of sample identifiers.
    """
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            invalid_elem = json.load(f)
        # Convert lists back to sets for efficient lookup
        for k, v in invalid_elem.items():
            invalid_elem[k] = set(v)
        return invalid_elem
    else:
        # Initialize with all invalid types from task_prompt_lib
        return {
            TOO_SMALL_ELEMENT: set(),
            INVALID_ELEM_BOX: set(),
            INVALID_ELEM_CONTENT: set(),
            BLANK_ELEM: set(),
            EMPTY_ELEM_TEXT: set(),
            OVERLY_LENGTHY_ELEM_TEXT: set(),
            DUPLICATE_ELEMEMNT: set()
        }


def _save_invalid_records(record_file: str, invalid_elem: Dict[str, Set[str]]) -> None:
    """Save invalid sample records to disk.
    
    Args:
        record_file: Path to save the JSON file.
        invalid_elem: Dictionary mapping invalid type names to sets of sample identifiers.
    """
    with open(record_file, 'w') as f:
        # Convert sets to lists for JSON serialization
        json.dump({k: list(v) for k, v in invalid_elem.items()}, f, indent=2)


def _load_instruction_types(type_file: str) -> Dict[str, str | None]:
    """Load instruction type classification cache from disk.
    
    The cache maps instruction text to the first verb found in it (or None if no verb).
    This helps distinguish action-based instructions (with verbs) from descriptive labels.
    
    Args:
        type_file: Path to the JSON file containing instruction type mappings.
        
    Returns:
        Dictionary mapping instruction strings to verb strings (or None).
    """
    if os.path.exists(type_file):
        with open(type_file, 'r') as f:
            return json.load(f)
    else:
        return {}


def _save_instruction_types(type_file: str, instruc_type_dict: Dict[str, str | None]) -> None:
    """Save instruction type classification cache to disk.
    
    Args:
        type_file: Path to save the JSON file.
        instruc_type_dict: Dictionary mapping instruction strings to verb strings.
    """
    with open(type_file, 'w') as f:
        json.dump(instruc_type_dict, f, indent=2)


def _get_image_dimensions(img_path: str) -> Tuple[int, int]:
    """Extract image dimensions from file without loading the full image.
    
    Uses the 'magic' library to read file metadata efficiently.
    
    Args:
        img_path: Path to the image file.
        
    Returns:
        Tuple of (width, height) in pixels.
    """
    file_info = magic.from_file(img_path)
    match = re.search(r'precision 8, (\d+)x(\d+)', file_info)
    if match:
        width, height = map(int, match.groups())
        return width, height
    else:
        raise ValueError(f"Could not extract dimensions from {img_path}")


def _normalize_bbox_to_center(bbox: Tuple[float, float, float, float], 
                               width: int, 
                               height: int) -> Tuple[List[int], str]:
    """Normalize bounding box to center point coordinates.
    
    Converts normalized bbox coordinates [0,1] to center point in SCALE coordinate space.
    
    Args:
        bbox: Normalized bounding box (xmin, ymin, xmax, ymax) in range [0, 1].
        width: Image width in pixels.
        height: Image height in pixels.
        
    Returns:
        Tuple of (normalized_center_coords, center_string) where:
        - normalized_center_coords: [x, y] in range [0, SCALE-1]
        - center_string: Formatted as "(x,y)"
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center in normalized coordinates
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    # Scale to SCALE coordinate system and clamp to valid range
    norm_center = [
        max(0, min(SCALE - 1, round(center_x * SCALE))),
        max(0, min(SCALE - 1, round(center_y * SCALE)))
    ]
    
    center_str = f'({norm_center[0]},{norm_center[1]})'
    return norm_center, center_str


def _validate_sample(sample_data: Dict, 
                     invalid_elem: Dict[str, Set[str]],
                     sample_identifier: str,
                     img_path: str,
                     width: int,
                     height: int) -> Tuple[bool, str]:
    """Validate a single sample and determine if it should be included.
    
    Performs multiple validation checks:
    - Skip if already marked as invalid
    - Check for swipe/flip actions (not supported)
    - Check for too small elements
    - Check for invalid bounding box coordinates
    - Check for empty instruction text
    - Check if element is actually visible (not pure color)
    
    Args:
        sample_data: Dictionary containing sample information (instruction, bbox, etc.).
        invalid_elem: Dictionary of invalid sample sets by type.
        sample_identifier: Unique identifier for this sample.
        img_path: Path to the screenshot image.
        width: Image width in pixels.
        height: Image height in pixels.
        
    Returns:
        Tuple of (is_valid, reason) where:
        - is_valid: True if sample passes all checks
        - reason: String describing why sample is invalid (empty if valid)
    """
    # Check if already marked as invalid
    for invalid_set in invalid_elem.values():
        if sample_identifier in invalid_set:
            return False, "previously marked as invalid"
    
    instruc = sample_data['instruction']
    bbox = sample_data['bbox']
    
    # Skip swipe/flip actions (not supported for grounding tasks)
    if any(keyword in instruc for keyword in ['flip', 'swipe']):
        invalid_elem[INVALID_ELEM_CONTENT].add(sample_identifier)
        return False, "contains unsupported action (flip/swipe)"
    
    # Check for too small elements (width or height < 0.5% of screen)
    if abs(bbox[0] - bbox[2]) <= 0.005 or abs(bbox[1] - bbox[3]) <= 0.005:
        invalid_elem[TOO_SMALL_ELEMENT].add(sample_identifier)
        return False, "element too small"
    
    # Validate bounding box coordinates
    if not (0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1 and 
            0 <= bbox[2] <= 1 and 0 <= bbox[3] <= 1 and 
            bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        invalid_elem[INVALID_ELEM_BOX].add(sample_identifier)
        return False, "invalid bounding box coordinates"
    
    # Check for empty instruction
    if len(instruc) == 0:
        invalid_elem[EMPTY_ELEM_TEXT].add(sample_identifier)
        return False, "empty instruction text"
    
    # Check if element is actually visible (not pure color/blank region)
    unnorm_boxes = [round(bbox[0] * width), round(bbox[1] * height), 
                    round(bbox[2] * width), round(bbox[3] * height)]
    img = cv2.imread(img_path)
    if img is not None and is_pure_color(img, unnorm_boxes):
        invalid_elem[BLANK_ELEM].add(sample_identifier)
        return False, "element region is blank/pure color"
    
    return True, ""


def _create_sample(sample_data: Dict,
                  instruc_type_dict: Dict[str, str | None],
                  verb_extractor: VerbExtactor,
                  center_str: str,
                  norm_center: List[int],
                  sample_idx: int) -> Tuple[Dict, str]:
    """Create a training sample from validated data.
    
    Determines whether to create an element grounding or intent grounding sample
    based on whether the instruction contains an action verb.
    
    Args:
        sample_data: Dictionary containing instruction and bbox.
        instruc_type_dict: Cache of instruction -> verb mappings.
        verb_extractor: VerbExtactor instance for detecting verbs.
        center_str: Formatted center coordinate string.
        norm_center: Normalized center coordinates [x, y].
        sample_idx: Current sample index for generating unique task IDs.
        
    Returns:
        Tuple of (sample_dict, task_type) where:
        - sample_dict: Training sample with all required fields
        - task_type: Either 'elemgnd' or 'intentgnd'
    """
    instruc = sample_data['instruction']
    
    # Classify instruction type (detect if it contains an action verb)
    if instruc not in instruc_type_dict:
        verb, verb_idx = verb_extractor.find_first_verb(instruc)
        instruc_type_dict[instruc] = verb
    
    # No verb found -> element grounding task (descriptive label)
    if instruc_type_dict[instruc] is None:
        sample = make_elemgnd_sample(
            task_id=f'autogui_{DATASET_NAME}_elemgnd_{sample_idx}',
            text=instruc,
            loc=center_str,
            output_tag=OUTPUT_TAG
        )
        task_type = 'elemgnd'
    else:
        # Verb found -> intent grounding task (action-based instruction)
        if USE_ACTION_PROMPT:
            # Alternative format: action planning with explicit click action
            action = CLICK_TEMPLATE.format(target_x=norm_center[0], target_y=norm_center[1])
            query = (TURN_GND_INTO_PLANNING_PROMPT.format(instruc=instruc) 
                    if len(instruc.strip().split()) == 1 
                    else instruc)
            sample = make_actionplanning_sample(
                task_id=f'autogui_{DATASET_NAME}_intentgnd_{sample_idx}',
                global_task=query,
                gt_action=action,
                history='None',
                prompt_format_type='aguvis'
            )
        else:
            # Standard format: intent grounding
            sample = make_intentgnd_sample(
                task_id=f'autogui_{DATASET_NAME}_intentgnd_{sample_idx}',
                intent=instruc,
                loc=center_str,
                output_tag=OUTPUT_TAG
            )
        task_type = 'intentgnd'
    
    return sample, task_type


# ============================================================================
# Main Processing Function
# ============================================================================

def make_refexp_data() -> None:
    """Generate RefExp training samples and write them to disk.
    
    Main processing pipeline:
    1. Load or initialize invalid sample records and instruction type cache
    2. Shuffle dataset for better distribution
    3. For each sample:
       - Extract image dimensions and bounding box
       - Validate sample (skip if invalid)
       - Normalize coordinates
       - Classify instruction type (element vs intent grounding)
       - Create training sample with metadata
    4. Save results, statistics, and sample subset
    
    Output files:
    - {DATASET_NAME}_s{SCALE}_{num_samples}k.json: Full dataset
    - {DATASET_NAME}_s{SCALE}_{num_samples}k_sample.json: Random 128-sample subset
    - {DATASET_NAME}_s{SCALE}_{num_samples}k_info.json: Statistics and metadata
    - invalid_elem_record.json: Invalid sample records (updated incrementally)
    - instruc_type_record.json: Instruction type cache (updated incrementally)
    """
    # ========================================================================
    # Step 1: Initialize data structures and load cached records
    # ========================================================================
    
    # Track unique elements per image (for deduplication)
    unique_elems: Dict[str, List[List[int]]] = {}
    
    # Store all generated samples
    samples: List[Dict] = []
    
    # Task counters
    elemgnd_cnt: int = 0
    intentgnd_cnt: int = 0
    
    # Initialize verb extractor for instruction classification
    verb_extractor = VerbExtactor()
    
    # Load or initialize invalid sample records
    invalid_elem_record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    invalid_elem = _load_invalid_records(invalid_elem_record_file)
    
    # Load or initialize instruction type classification cache
    instruc_type_file = os.path.join(SAVE_ROOT, 'instruc_type_record.json')
    instruc_type_dict = _load_instruction_types(instruc_type_file)

    # ========================================================================
    # Step 2: Shuffle dataset for better sample distribution
    # ========================================================================
    
    ds.shuffle()
    
    # ========================================================================
    # Step 3: Process each sample in the dataset
    # ========================================================================
    
    for sample_idx, raw_sample in tqdm(enumerate(ds), total=len(ds), desc=f"Processing {DATASET_NAME}"):
        # Debug mode: only process first 200 samples
        if DEBUG and sample_idx >= 200:
            break
        
        # Extract sample information
        image_id = raw_sample['image_id']
        instruc = raw_sample['prompt'].strip()
        bbox_raw = eval(raw_sample['target_bounding_box'])
        bbox = (bbox_raw['xmin'], bbox_raw['ymin'], bbox_raw['xmax'], bbox_raw['ymax'])
        
        # Initialize unique elements tracker for this image
        if image_id not in unique_elems:
            unique_elems[image_id] = []
        
        # Create unique identifier for this sample
        sample_identifier = f"{image_id}|{raw_sample['prompt']}"
        img_path = os.path.join(IMG_DIR, f'{image_id}.jpg')
        
        # Get image dimensions efficiently (without loading full image)
        try:
            W, H = _get_image_dimensions(img_path)
        except (ValueError, AttributeError) as e:
            # Skip if image dimensions cannot be extracted
            continue
        
        # Convert normalized bbox to pixel coordinates
        unnorm_boxes = [round(bbox[0] * W), round(bbox[1] * H), 
                       round(bbox[2] * W), round(bbox[3] * H)]
        
        # Track unique elements (avoid processing duplicate boxes in same image)
        if unnorm_boxes not in unique_elems[image_id]:
            unique_elems[image_id].append(unnorm_boxes)
        
        # ====================================================================
        # Step 3.1: Validate sample (skip if invalid)
        # ====================================================================
        
        if not SKIP_CHECKING:
            sample_data = {'instruction': instruc, 'bbox': bbox}
            is_valid, reason = _validate_sample(
                sample_data=sample_data,
                invalid_elem=invalid_elem,
                sample_identifier=sample_identifier,
                img_path=img_path,
                width=W,
                height=H
            )
            
            if not is_valid:
                continue  # Skip this sample
        
        # ====================================================================
        # Step 3.2: Normalize coordinates to SCALE coordinate system
        # ====================================================================
        
        norm_center, center_str = _normalize_bbox_to_center(bbox, W, H)
        
        # ====================================================================
        # Step 3.3: Create training sample
        # ====================================================================
        
        sample, task_type = _create_sample(
            sample_data={'instruction': instruc, 'bbox': bbox},
            instruc_type_dict=instruc_type_dict,
            verb_extractor=verb_extractor,
            center_str=center_str,
            norm_center=norm_center,
            sample_idx=len(samples)
        )
        
        # Add metadata
        sample['image'] = f'rico/{image_id}.jpg'
        sample['unnormalized_box'] = unnorm_boxes
        sample['task_attr'] = instruc
        
        samples.append(sample)
        
        # Update task counters
        if task_type == 'elemgnd':
            elemgnd_cnt += 1
        else:
            intentgnd_cnt += 1
        
        # ====================================================================
        # Step 3.4: Periodically save progress (every 10k samples or at end)
        # ====================================================================
        
        if sample_idx > 0 and (sample_idx % 10000 == 0 or sample_idx == len(ds) - 1):
            _save_invalid_records(invalid_elem_record_file, invalid_elem)
            _save_instruction_types(instruc_type_file, instruc_type_dict)

    # ========================================================================
    # Step 4: Generate statistics and summary report
    # ========================================================================
    
    # Calculate statistics
    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k, v in unique_elems.items() if len(v) > 0])
    num_total_imgs = len(unique_elems)
    
    invalid_ratio = num_invalid_elem / len(ds) if len(ds) > 0 else 0
    
    # Print summary report
    report = (
        f"\n{'='*60}\n"
        f"RefExp Dataset Processing Summary\n"
        f"{'='*60}\n"
        f"Total samples processed: {len(samples)}\n"
        f"Unique elements: {num_unique_elems}\n"
        f"Valid unique images: {num_valid_imgs}\n"
        f"Total unique images: {num_total_imgs}\n"
        f"Invalid elements: {num_invalid_elem} / {len(ds)} = {invalid_ratio:.2%}\n"
        f"\nTask Distribution:\n"
        f"  - Element Grounding (elemgnd): {elemgnd_cnt}\n"
        f"  - Intent Grounding (intentgnd): {intentgnd_cnt}\n"
        f"  - Widget List: 0\n"
        f"{'='*60}"
    )
    print(report)
    
    # ========================================================================
    # Step 5: Save results to disk
    # ========================================================================
    
    # Generate output file name
    file_name_parts = [
        f"{DATASET_NAME}_s{SCALE}",
        f"{len(samples) // 1000}k"
    ]
    if DEBUG:
        file_name_parts.append("debug")
    if USE_ACTION_PROMPT:
        file_name_parts.append("actformat")
    
    file_name = os.path.join(SAVE_ROOT, "_".join(file_name_parts) + ".json")
    print(f"\nSaving results to: {file_name}")
    
    # Save metadata and statistics
    info_file = file_name.replace('.json', '_info.json')
    with open(info_file, "w") as f:
        json.dump({
            'num_samples': len(samples),
            'num_unique_elems': num_unique_elems,
            'all_elems': len(ds),
            'valid_unique_images': num_valid_imgs,
            'all_unique_images': num_total_imgs,
            'elemgnd_cnt': elemgnd_cnt,
            'intentgnd_cnt': intentgnd_cnt,
            'widgetlist_cnt': 0,
            'num_invalid_elements': num_invalid_elem,
            'invalid_elem_ratio': invalid_ratio,
            'invalid_elem_type': {k: len(v) for k, v in invalid_elem.items()},
            'config': {
                'SCALE': SCALE,
                'USE_ACTION_PROMPT': USE_ACTION_PROMPT,
                'DEBUG': DEBUG,
                'SKIP_CHECKING': SKIP_CHECKING,
            }
        }, f, indent=2)
    print(f"  - Saved metadata to: {info_file}")
    
    # Save random sample subset for quick inspection
    sample_file = file_name.replace(".json", "_sample.json")
    with open(sample_file, 'w') as f:
        sample_subset = random.sample(samples, min(len(samples), 128))
        json.dump(sample_subset, f, indent=2)
    print(f"  - Saved sample subset (128 samples) to: {sample_file}")
    
    # Save full dataset
    with open(file_name, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"  - Saved full dataset ({len(samples)} samples) to: {file_name}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    make_refexp_data()