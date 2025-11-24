"""
WebUI Dataset Preprocessing Utilities.

This module is responsible for processing the WebUI dataset, which consists of screenshots and 
accessibility trees (axtrees) from various websites. The goal is to generate a clean, normalized, 
and structured dataset for training multi-modal UI understanding models.

Key functionalities:
1.  **Data Loading & Cleaning**: Parses compressed accessibility trees and bounding boxes, 
    filtering out corrupted files or network errors.
2.  **Adaptive Image Splitting**: Web pages are often very long. This script intelligently 
    splits them into manageable blocks (simulating mobile viewports) and selects the most 
    informative regions (highest icon density).
3.  **Robust Validation**: Implements a rigorous validation pipeline to discard elements that are:
    -   Too small or too large.
    -   Off-screen or overlapping.
    -   Pure color (invisible) or having invalid aspect ratios.
    -   Non-English/Chinese text (to maintain language quality).
4.  **Multi-Task Sample Generation**: Creates training samples for five distinct downstream tasks:
    -   **Icon Grounding (IconGnd)**: "Where is the [search icon]?" -> [bbox]
    -   **Element Grounding (ElemGnd)**: "Where is the [Submit button]?" -> [bbox]
    -   **Intent Grounding (IntentGnd)**: "I want to [log in]" -> [bbox of login button]
    -   **Element Classification (ElemClass)**: [bbox] -> "This is a [checkbox]"
    -   **Widget List Generation**: Serializes the screen into a textual list of elements 
        (crucial for LLM-based UI agents).
5.  **Class Balancing**: For classification tasks, it automatically rebalances the dataset 
    to prevent dominant classes (like 'links') from overwhelming rare ones (like 'sliders').

Usage:
    Run this script directly to process the data:
    $ python 141_make_webui_data.py
"""

import gzip
import glob
import json
import os
import random
import shutil
import traceback
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Sibling imports (assuming the environment is set up with correct PYTHONPATH)
from utils.data_utils.task_prompt_lib import (
    make_icongnd_sample,
    make_iconref_sample,
    make_elemgnd_sample,
    make_elemref_sample,
    make_intentgnd_sample,
    make_elemclass_sample,
    make_widgetlist_sample,
    make_actionplanning_sample,
    gen_naive_action_gnd_anno,
    CLICK_TEMPLATE,
    TURN_GND_INTO_PLANNING_PROMPT,
    INVALID_TEXT_LANGUAGE,
    TOO_SMALL_ELEMENT,
    INVALID_ELEM_BOX,
    INVALID_ELEM_CONTENT,
    BLANK_ELEM,
    EMPTY_ELEM_TEXT,
    OVERLY_LENGTHY_ELEM_TEXT,
    DUPLICATE_ELEMEMNT,
    INCORRECT_TEXT_ANNO,
    OVERSIZED_ELEMENT,
    EXTREME_ASPECT_RATIO,
)
from utils.data_utils.misc import (
    prune_accessibility_tree_wo_bound,
    resize_image,
    is_pure_color,
    is_valid_string,
    contain_network_errors,
)

# =============================================================================
# Configuration & Constants
# =============================================================================

DATASET_NAME = 'WebUI'
DEBUG = True       # If True, processes only a small subset of data for testing.
DRAW = False        # If True, draws bounding boxes on images for visual verification.

# --- Task Switches ---
# Toggle these flags to control which types of training samples are generated.
TEXTLOC = False     # Text Localization (not implemented for WebUI)
OCR = False         # Optical Character Recognition (not implemented for WebUI)
ICONGND = True      # Icon Grounding: Text -> Icon Location
ICONREF = True      # Icon Reference: Location -> Icon Text/Desc
ELEMGND = False     # Element Grounding: Text -> Element Location
ELEMREF = False     # Element Reference: Location -> Element Text
ELEMCLASS = True    # Element Classification: Location -> Element Role
INTENTGND = True    # Intent Grounding: User Intent -> Element Location
WIDGETLIST = True   # Widget List: Full screen element serialization

# --- Prompt Engineering Settings ---
USE_ACTION_PROMPT = False  # If True, formats output as an action (e.g., "click(x,y)")
SKIP_CHECKING = False      # If True, bypasses rigorous element validation (use with caution)

# --- Hyperparameters ---
PROB_BOX = 0.3      # Probability of using a bounding box input instead of a point
SCALE = 1000        # Normalization scale for coordinates (0-1000)
LONGEST = 1344      # Target longer side length for image resizing

# --- Device Specifics ---
# Ratios used to split long web screenshots into mobile-sized chunks.
RATIO = {
    'iPhone-13 Pro': 2532 / 1170,
}

# Resize factors to align raw bounding box coordinates with the image resolution.
# iPhone-13 Pro usually renders at 3x scale (Retina display).
FORCE_RESIZE = {
    'iPhone-13 Pro': 3,
    'iPad-Pro': 2,
}

# List of roles considered as "Icons" for the Icon Grounding task.
ICON_ROLES = [
    "button", "image", "link", "toggle", "menuitem", "menubutton",
    "tab", "tabitem", "checkbox", "radio", "svg", "textbox",
    "combobox", "searchbox", "listbox", "treeitem", "slider",
    "spinbutton", "progressbar", "scrollbar", "separator", "dialog",
    "tooltip", "notification", "alert", "calendar", "colorwheel",
    "dateeditor", "menu", "menubar", "treeview", "window"
]

# --- Paths ---
WEBUI_DIR = f"/mnt/vdb1/hongxin_li/{DATASET_NAME}/dh2"
WEBUI_PROCESSED_IMG_DIR = f"/mnt/vdb1/hongxin_li/{DATASET_NAME}/WebUI_screenshots"
ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"
SAVE_ROOT = os.path.join(ROOT, f"{DATASET_NAME}_processed")

# Ensure output directories exist
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(WEBUI_PROCESSED_IMG_DIR, exist_ok=True)


# =============================================================================
# Helper Functions
# =============================================================================

def _load_invalid_elems_record() -> Dict[str, Set[str]]:
    """
    Loads or initializes the record of invalid elements.
    
    This maintains a persistent record of why elements were rejected (e.g., "too small", 
    "invalid language"), which is useful for dataset auditing and debugging.
    
    Returns:
        A dictionary mapping error types to sets of sample identifiers.
    """
    record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    if os.path.exists(record_file):
        try:
            data = json.load(open(record_file))
            # JSON lists are converted back to Python sets for O(1) lookups
            return {k: set(v) for k, v in data.items()}
        except Exception:
            pass
    
    # Initialize empty sets for each failure mode
    return {
        INVALID_TEXT_LANGUAGE: set(),
        TOO_SMALL_ELEMENT: set(),
        INVALID_ELEM_BOX: set(),
        INVALID_ELEM_CONTENT: set(),
        BLANK_ELEM: set(),
        EMPTY_ELEM_TEXT: set(),
        OVERLY_LENGTHY_ELEM_TEXT: set(),
        DUPLICATE_ELEMEMNT: set(),
        INCORRECT_TEXT_ANNO: set(),
        OVERSIZED_ELEMENT: set(),
        EXTREME_ASPECT_RATIO: set(),
    }


def _save_invalid_elems_record(invalid_elems: Dict[str, Set[str]]) -> None:
    """
    Saves the record of invalid elements to disk.
    
    Args:
        invalid_elems: The dictionary of invalid elements to save.
    """
    record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    with open(record_file, 'w') as f:
        # Convert sets to lists for JSON serialization
        json.dump({k: list(v) for k, v in invalid_elems.items()}, f, indent=2)


def _process_axtree_and_image(
    axtree_path: str,
    unique_elems: Dict[str, List[List[int]]],
) -> Tuple[Optional[Any], Optional[np.ndarray], Optional[Dict], Optional[str]]:
    """
    Loads and pre-processes the accessibility tree and corresponding screenshot.

    This function handles:
    1. Validating file existence (axtree, bbox, screenshot).
    2. Filtering out non-standard device types (currently only 'iPhone-13 Pro').
    3. Reading compressed JSON files.
    4. Pruning the accessibility tree to remove irrelevant nodes.

    Args:
        axtree_path: Full path to the compressed accessibility tree file (*.json.gz).
        unique_elems: Shared dictionary to track unique elements per image for statistics.

    Returns:
        A tuple containing:
        - axtree_nodes: List of relevant leaf nodes.
        - img: The raw OpenCV image (numpy array).
        - bboxes: The dictionary of bounding boxes.
        - device_type: String identifier of the device (e.g., 'iPhone-13 Pro').
        
        Returns (None, None, None, None) if any step fails or validation is not met.
    """
    device_type = os.path.basename(axtree_path).split('-axtree')[0]
    
    # STRICT FILTERING: Only process iPhone-13 Pro data for consistent mobile UI aspect ratios.
    if device_type != 'iPhone-13 Pro':
        return None, None, None, None
    
    # Filter out specific resolutions that are known to be problematic or inconsistent
    if any(res in axtree_path for res in ['1536-864', '1920-1080', '1366-768']):
        return None, None, None, None

    # Construct paths for sibling files (bounding boxes and screenshot)
    bbox_file = axtree_path.replace('axtree.json.gz', 'bb.json.gz')
    img_file = axtree_path.replace('axtree.json.gz', 'screenshot-full.webp')
    
    if not os.path.exists(bbox_file) or not os.path.exists(img_file):
        return None, None, None, None

    img = cv2.imread(img_file)
    if img is None:
        return None, None, None, None

    # Initialize uniqueness tracking for this image if new
    if img_file not in unique_elems:
        unique_elems[img_file] = []

    try:
        # Load bounding boxes
        with gzip.open(bbox_file, 'rt', encoding='utf-8') as f:
            bboxes = json.load(f)

        # Load accessibility tree
        with gzip.open(axtree_path, 'rt', encoding='utf-8') as f:
            axtree_raw = f.read()

        # Check for network error artifacts in the HTML/Tree dump
        if contain_network_errors(axtree_raw):
            return None, None, None, None
            
        axtree_data = json.loads(axtree_raw)['nodes']
        raw_axtree = {node['nodeId']: node for node in axtree_data}
        
        # Prune tree: remove invisible nodes, generic containers, etc.
        axtree = prune_accessibility_tree_wo_bound(raw_axtree)

        # IMPORTANT: We only care about LEAF nodes (atomic elements) for grounding tasks.
        # Intermediate nodes (like wrappers) usually don't represent interactable targets.
        axtree_nodes = [node for node in axtree.values() if len(node['childIds']) == 0]
        
        return axtree_nodes, img, bboxes, device_type

    except Exception:
        # Silent failure is preferred here to keep the main loop running smoothly
        return None, None, None, None


def _split_image_into_blocks(
    img: np.ndarray,
    axtree: List[Dict],
    bboxes: Dict,
    device_type: str,
    img_file: str
) -> Tuple[int, Dict[int, Dict]]:
    """
    Intelligently splits a long webpage screenshot into mobile-sized blocks.

    Webpages are often tall scrolls. To make them suitable for training VLM agents 
    (which often operate on single-screen views), we cut the full page into 
    multiple blocks. We then select the "best" block—defined as the one with 
    the highest density of interactive icons.

    Args:
        img: The full webpage screenshot.
        axtree: List of valid leaf nodes.
        bboxes: Bounding box lookup dictionary.
        device_type: Device identifier.
        img_file: Path used to determine aspect ratio logic.

    Returns:
        A tuple of:
        - block_idx_with_most_icons: Index of the chosen block (0-based).
        - num_icons_each_block: Detailed statistics for each block.
    """
    H, W = img.shape[:2]
    
    # Determine the height of a single "screen" block based on device aspect ratio
    # For iPhone: 2532/1170 ~= 2.16 aspect ratio
    ratio = 2532 / 1170 if 'iPhone' in img_file else 720/1280
    block_h = int(W * ratio)
    
    if block_h == 0: 
        return -1, {}
    
    num_blocks = H // block_h
    block_hs = np.arange(0, H, block_h)
    
    # Initialize storage for block statistics
    num_icons_each_block = {
        block_idx: {'nodes': [], 'icon_nodes': [], 'num_icons': 0}
        for block_idx in range(num_blocks)
    }

    # Scaling factor to map CSS pixels to Physical pixels (Retina display logic)
    resize_factor = FORCE_RESIZE.get(device_type, 1)

    for node in axtree:
        if 'backendDOMNodeId' not in node:
            continue
        
        nodebkid = str(node['backendDOMNodeId'])
        if nodebkid not in bboxes:
            continue
        
        bbox = bboxes[nodebkid]
        if bbox is None:
            continue
        
        # Apply scale factor to get actual pixel coordinates
        x = bbox['x'] * resize_factor
        y = bbox['y'] * resize_factor
        w = bbox['width'] * resize_factor
        h = bbox['height'] * resize_factor
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Assign node to a block based on its vertical center
        center_y = y1 + h // 2
        block_idx = np.digitize(center_y, block_hs) - 1
        
        # Skip nodes that fall outside valid blocks (e.g., in the partial footer area)
        if block_idx in [-1, num_blocks]:
            continue

        node['box'] = list(map(round, [x1, y1, x2, y2]))
        
        # Check if this node is an "icon" to contribute to the density score
        if node['role']['value'] in ICON_ROLES:
            num_icons_each_block[block_idx]['num_icons'] += 1
            num_icons_each_block[block_idx]['icon_nodes'].append(node)

        num_icons_each_block[block_idx]['nodes'].append(node)

    # Selection Strategy: Choose the block with the MAXIMUM number of icons.
    # This ensures we train on information-rich screens rather than empty spacers.
    block_idx_with_most_icons = -1
    max_icon_cnt = 0
    for block_idx, block_info in num_icons_each_block.items():
        if block_info['num_icons'] > max_icon_cnt:
            block_idx_with_most_icons = block_idx
            max_icon_cnt = block_info['num_icons']
            
    return block_idx_with_most_icons, num_icons_each_block


def _validate_block_nodes(
    block_nodes: List[Dict],
    axtree_path: str,
    invalid_elems: Dict[str, Set[str]],
    img_file: str,
    unique_elems: Dict[str, List[List[int]]],
    resizing_ratio: float,
    h_start: int,
    sc_w: int,
    sc_h: int,
    screenshot: np.ndarray,
) -> List[Tuple[Dict, str]]:
    """
    Performs rigorous validation on all nodes within the selected block.

    This function acts as a quality gate. It projects the global coordinates 
    to the local block coordinates and filters out low-quality elements.

    Filter Criteria:
    1.  **Language**: Rejects blocks where valid text ratio < 40% (removes non-English/Chinese).
    2.  **Duplicates**: Removes elements with identical bounding boxes.
    3.  **Geometry**: 
        - Must be within screen bounds.
        - Must not be tiny (< 0.5% of screen).
        - Must not be huge (> 65% of screen).
        - Aspect ratio must be reasonable (0.05 < w/h < 20).
    4.  **Visibility**: Checks pixel variance to ensure the element is not just a solid color block.

    Args:
        block_nodes: Nodes belonging to the current block.
        axtree_path: Path for ID generation.
        invalid_elems: Global record for rejection stats.
        img_file: Image identifier.
        unique_elems: Global uniqueness tracker.
        resizing_ratio: Scale factor applied during image resizing (to LONGEST edge).
        h_start: The vertical offset of the block in the full image.
        sc_w, sc_h: Width and height of the *processed* (resized) screenshot.
        screenshot: The actual image data for pixel-level checks.

    Returns:
        A list of valid tuples: (node_dict, unique_sample_identifier).
        Returns empty list if the block itself is rejected (e.g., bad language).
    """
    axtree_name = axtree_path.split('dh2/')[1][:-8]
    
    # --- Block-Level Validation: Language Check ---
    nodes_containing_valid_characters_cnt = []
    sample_identifiers = []
    
    for node in block_nodes:
        text = node['name']['value']
        sample_identifiers.append(f"{axtree_name}|{node['nodeId']}")
        if text is None: 
            continue
        # Check if text contains valid characters (mostly English/Chinese/Numbers)
        nodes_containing_valid_characters_cnt.append(is_valid_string(text))

    # If too much gibberish or unsupported language, discard the whole block.
    if not SKIP_CHECKING and len(nodes_containing_valid_characters_cnt) > 0:
        valid_ratio = sum(nodes_containing_valid_characters_cnt) / len(nodes_containing_valid_characters_cnt)
        if valid_ratio <= 0.4:
            invalid_elems[INVALID_TEXT_LANGUAGE].update(sample_identifiers)
            return []

    valid_nodes_in_block = []
    used_boxes = []

    for node, sample_identifier in zip(block_nodes, sample_identifiers):
        # Record element existence for global statistics
        if node['box'] not in unique_elems[img_file]:
            unique_elems[img_file].append(node['box'])

        x1, y1, x2, y2 = node['box']
        
        # --- Coordinate Transformation ---
        # 1. Translate: y - h_start (move to local block coordinates)
        # 2. Scale: * resizing_ratio (adjust to resized image dimensions)
        x1, y1, x2, y2 = list(map(lambda x: round(x * resizing_ratio), [x1, y1 - h_start, x2, y2 - h_start]))
        node['box'] = [x1, y1, x2, y2]

        # Debug Visualization
        if DRAW:
            loc = f'{x1/sc_w:.2f}, {y1/sc_h:.2f} {x2/sc_w:.2f}, {y2/sc_h:.2f}'
            print(loc, node['role']['value'], node['name']['value'])
            cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255) if node['name']['value'].strip() == '' else (0, 255, 0), 2)

        if not SKIP_CHECKING:
            # 1. Duplicate Check
            if node['box'] in used_boxes:
                invalid_elems[DUPLICATE_ELEMEMNT].add(sample_identifier)
                continue
            
            # 2. Boundary Check
            if x1 < 0 or y1 < 0 or x2 >= sc_w or y2 >= sc_h:
                invalid_elems[INVALID_ELEM_BOX].add(sample_identifier)
                continue

            node_w, node_h = x2 - x1, y2 - y1

            # 3. Tiny Element Check (likely noise or invisible tap targets)
            if node_w / sc_w <= 0.005 or node_h / sc_h <= 0.005:
                invalid_elems[TOO_SMALL_ELEMENT].add(sample_identifier)
                continue

            # 4. Oversized Element Check (backgrounds, full-screen overlays)
            if node_w * node_h / (sc_w * sc_h) >= 0.65:
                invalid_elems[OVERSIZED_ELEMENT].add(sample_identifier)
                continue

            # 5. Extreme Aspect Ratio Check (dividers, weird artifacts)
            if node_h == 0 or node_w / node_h < 0.05 or node_w / node_h > 20:
                invalid_elems[EXTREME_ASPECT_RATIO].add(sample_identifier)
                continue

            # 6. Content Check (is it just a blank colored box?)
            if is_pure_color(screenshot, [x1, y1, x2, y2]):
                invalid_elems[BLANK_ELEM].add(sample_identifier)
                continue

        used_boxes.append(node['box'])
        valid_nodes_in_block.append((node, sample_identifier))
        
    return valid_nodes_in_block


def _create_samples_from_block(
    valid_nodes: List[Dict],
    sc_w: int,
    sc_h: int,
    short_screenshot_name: str,
    counters: Dict[str, int],
    elemclass_stats: Dict[str, List],
    samples: List[Dict]
) -> None:
    """
    Generates multi-task training samples from the validated nodes.

    Iterates through all valid nodes and creates JSON samples for enabled tasks 
    (IconGnd, ElemGnd, IntentGnd, etc.). It handles coordinate normalization 
    (0-1000 scale) and output formatting.

    Args:
        valid_nodes: The list of cleaned, validated UI nodes.
        sc_w, sc_h: Dimensions of the processed screenshot.
        short_screenshot_name: Relative path to image (for dataset portability).
        counters: Dictionary to generate unique Task IDs.
        elemclass_stats: Accumulator for element classification samples (for later rebalancing).
        samples: The main list where generated samples are appended.
    """
    all_node_names = [node['name']['value'].strip() for node in valid_nodes]
    all_boxes = []
    all_node_box_strs = []
    all_node_roles = []

    for node_name, node in zip(all_node_names, valid_nodes):
        x1, y1, x2, y2 = node['box']
        all_boxes.append(node['box'])
        node_role = node['role']['value']
        all_node_roles.append(node_role)
        node_area = (x2 - x1) * (y2 - y1)
        
        # Normalize coordinates to [0, SCALE] (default 1000)
        normalized_box = [
            round(x1 / sc_w * SCALE),
            round(y1 / sc_h * SCALE),
            round(x2 / sc_w * SCALE),
            round(y2 / sc_h * SCALE)
        ]
        # Clamp to ensure [0, SCALE-1] range
        normalized_box = list(map(lambda p: max(0, min(SCALE - 1, p)), normalized_box))

        # Calculate center point
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        normalized_center = (round(center_x / sc_w * SCALE), round(center_y / sc_h * SCALE))
        normalized_center = list(map(lambda p: max(0, min(SCALE - 1, p)), normalized_center))
        
        # Decide whether to represent location as a BOX or a POINT (probabilistic)
        with_box = random.random() <= PROB_BOX
        box_str = f'({normalized_box[0]},{normalized_box[1]},{normalized_box[2]},{normalized_box[3]})'
        all_node_box_strs.append(box_str)

        loc = box_str if with_box else f'({normalized_center[0]},{normalized_center[1]})'
        
        # Uniqueness Check: Ambiguous prompts confuse models.
        # If two buttons are named "Save", we skip grounding tasks for them.
        is_unique = all_node_names.count(node_name) <= 1

        # =====================================================================
        # Task Generation Logic
        # =====================================================================

        # Check if the node is a valid candidate for UI tasks
        if node_role != 'StaticText' and len(node_name.strip()):
            
            # Logic: "Icons" are typically small elements. 
            # Threshold: Area <= 1% of screen surface.
            is_small_icon = node_area / (sc_h * sc_w) <= 0.01
            
            if (ICONGND or ICONREF) and is_small_icon:
                # --- Task: Icon Grounding (Name -> Loc) ---
                if ICONGND and is_unique:
                    sample = make_icongnd_sample(
                        task_id=f'autogui_webui_icongnd_{counters["icongnd_cnt"]}',
                        icon_desc=node_name,
                        loc=loc,
                        with_box=with_box
                    )
                    sample.update({
                        'unnormalized_box': node['box'],
                        'task_attr': node_name,
                        'image': short_screenshot_name,
                        'wxh': f"{sc_w}x{sc_h}"
                    })
                    samples.append(sample)
                    counters["icongnd_cnt"] += 1
            
                # --- Task: Icon Reference (Loc -> Name) ---
                if ICONREF:
                    sample = make_iconref_sample(
                        task_id=f'autogui_webui_iconref_{counters["iconref_cnt"]}',
                        icon_desc=node_name,
                        loc=loc,
                        with_box=with_box
                    )
                    sample.update({
                        'unnormalized_box': node['box'],
                        'task_attr': loc,
                        'image': short_screenshot_name,
                        'wxh': f"{sc_w}x{sc_h}"
                    })
                    samples.append(sample)
                    counters["iconref_cnt"] += 1
            else:
                # If it's not a small icon, treat it as a general Element.
                
                # --- Task: Element Grounding (Name -> Loc) ---
                if ELEMGND and is_unique:
                    if USE_ACTION_PROMPT:
                        # Advanced: format as an agent action "click(x,y)"
                        action = CLICK_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1])
                        sample = make_actionplanning_sample(
                            task_id=f'autogui_{DATASET_NAME}_elemgnd_{len(samples)}',
                            global_task=TURN_GND_INTO_PLANNING_PROMPT.format(instruc=node_name),
                            gt_action=action,
                            history='None',
                            prompt_format_type='aguvis'
                        )
                    else:
                        # Standard: format as coordinate tuple
                        sample = make_elemgnd_sample(
                            task_id=f'autogui_webui_elemgnd_{counters["elemgnd_cnt"]}',
                            elem_desc=node_name,
                            loc=loc,
                            with_box=with_box
                        )

                    sample.update({
                        'unnormalized_box': node['box'],
                        'task_attr': node_name,
                        'image': short_screenshot_name,
                        'wxh': f"{sc_w}x{sc_h}"
                    })
                    samples.append(sample)
                    counters["elemgnd_cnt"] += 1
                
                # --- Task: Element Reference (Loc -> Name) ---
                if ELEMREF:
                    sample = make_elemref_sample(
                        task_id=f'autogui_webui_elemref_{counters["elemref_cnt"]}',
                        elem_desc=node_name,
                        loc=loc,
                        with_box=with_box
                    )
                    sample.update({
                        'unnormalized_box': node['box'],
                        'task_attr': loc,
                        'image': short_screenshot_name,
                        'wxh': f"{sc_w}x{sc_h}"
                    })
                    samples.append(sample)
                    counters["elemref_cnt"] += 1
            
            # --- Task: Intent Grounding (User Intent -> Loc) ---
            # Synthetic Intent: "I want to [click] the [Save] [button]"
            if INTENTGND and is_unique:
                intent = gen_naive_action_gnd_anno(node_name, node_role, normalized_center, scale=SCALE)
                sample = make_intentgnd_sample(
                    task_id=f'autogui_webui_intentgnd_{counters["intentgnd_cnt"]}',
                    intent=intent,
                    loc=loc,
                    with_box=with_box
                )
                sample.update({
                    'unnormalized_box': node['box'],
                    'task_attr': intent,
                    'image': short_screenshot_name,
                    'wxh': f"{sc_w}x{sc_h}"
                })
                samples.append(sample)
                counters["intentgnd_cnt"] += 1

        # --- Task: Element Classification (Loc -> Type) ---
        # We don't add these to 'samples' yet; we collect them to rebalance classes later.
        if ELEMCLASS:
            sample = make_elemclass_sample(
                task_id='autogui_webui_elemclass_',  # Suffix will be added after rebalancing
                elem_cls=node_role,
                loc=loc,
                with_box=with_box
            )
            sample.update({
                'unnormalized_box': node['box'],
                'task_attr': node_role,
                'image': short_screenshot_name,
                'wxh': f"{sc_w}x{sc_h}"
            })
            elemclass_stats[node_role].append(sample)
            
    # --- Task: Widget List (Screen Serialization) ---
    # Generates a textual representation of the UI: "0 button 'Save' (x1,y1,x2,y2)\n..."
    # Only valid if screen has a reasonable number of elements (2-80) to fit context window.
    if WIDGETLIST and 2 <= len(all_node_names) <= 80:
        node_texts_boxes = []
        for node_role, node_name, unnormalized_box, box_str in zip(all_node_roles, all_node_names, all_boxes, all_node_box_strs):
            # Note: The unnormalized_box here is already relative to the *cut* screenshot 
            # (from _validate_block_nodes), so we normalize it against the screenshot dims.
            
            # Redundant calculation for clarity, used for sorting logic below
            # normalized_box_wl = [ ... ] 
            
            node_texts_boxes.append([f"{node_role} '{node_name}' {box_str}", unnormalized_box])

        # Sort elements spatially: Top-to-bottom, then Left-to-right (reading order)
        node_texts_boxes.sort(key=lambda x: (x[1][1] + x[1][3], x[1][0] + x[1][2]))
        
        # Join into a single prompt string
        elem_list_str = '\n'.join(f"{i} {x[0]}" for i, x in enumerate(node_texts_boxes))
        
        sample = make_widgetlist_sample(
            task_id=f'autogui_webui_widgetlist_{counters["widgetlist_cnt"]}',
            elem_list=elem_list_str
        )
        sample.update({
            'unnormalized_box': [x[1] for x in node_texts_boxes],
            'image': short_screenshot_name,
            'wxh': f"{sc_w}x{sc_h}"
        })
        samples.append(sample)
        counters["widgetlist_cnt"] += 1


def make_webui_data():
    """
    Main driver function to process the WebUI dataset.
    
    Pipeline Steps:
    1. Iterate through all sample directories.
    2. Load axtree and image.
    3. Split image into blocks and select the best one.
    4. Validate elements within the block.
    5. Generate initial samples.
    6. Rebalance element classification samples.
    7. Save dataset statistics and final JSON outputs.
    """
    counters = defaultdict(int)
    elemclass_stats = defaultdict(list)
    
    # Load persistent record of invalid elements to avoid re-processing known bad data
    invalid_elems = _load_invalid_elems_record()
    
    sample_dirs = sorted(os.listdir(WEBUI_DIR))
    if DEBUG:
        print("DEBUG MODE: Processing first 1000 samples only.")
        sample_dirs = sample_dirs[:1000]

    samples = []
    unique_elems = {}
    errors = []

    processed_img_cnt = 0
    valid_img_cnt = 0
    iterated_elem_cnt = 0

    print(f"Processing {len(sample_dirs)} directories...")
    
    for sample_idx, sample_dir in tqdm(enumerate(sample_dirs), total=len(sample_dirs)):
        # Each directory might contain multiple axtree files
        axtree_paths = glob.glob(os.path.join(WEBUI_DIR, sample_dir, '*axtree.json.gz'))
        
        for axtree_path in axtree_paths:
            try:
                # Step 1: Load raw data
                axtree_nodes, img, bboxes, device_type = _process_axtree_and_image(axtree_path, unique_elems)
                if axtree_nodes is None:
                    continue
                
                processed_img_cnt += 1

                # Step 2: Intelligent Splitting
                block_idx_with_most_icons, num_icons_each_block = _split_image_into_blocks(
                    img, axtree_nodes, bboxes, device_type, axtree_path
                )
                
                # Skip if no suitable block found or block is empty of icons
                if block_idx_with_most_icons == -1 or num_icons_each_block[block_idx_with_most_icons]['num_icons'] == 0:
                    continue

                # Step 3: Crop Image
                H, W = img.shape[:2]
                # Calculate block height
                ratio = 2532 / 1170 if 'iPhone' in axtree_path else 720/1280
                block_h = int(W * ratio)
                h_start = block_h * block_idx_with_most_icons
                
                cropped_img = img[h_start:h_start + block_h]
                # Resize to standard training resolution (LONGEST side)
                screenshot, resizing_ratio = resize_image(cropped_img, max_size=LONGEST)
                sc_h, sc_w = screenshot.shape[:2]
                
                # Save the processed image for the dataset
                traj_dir = os.path.join(WEBUI_PROCESSED_IMG_DIR, sample_dir)
                os.makedirs(traj_dir, exist_ok=True)
                screenshot_file = os.path.join(traj_dir, f"{device_type}.png")
                # Store relative path for JSON
                short_screenshot_name = f"{DATASET_NAME}/{sample_dir}/{device_type}.png"
                
                if not os.path.exists(screenshot_file):
                    cv2.imwrite(screenshot_file, screenshot)

                # Step 4: Validate Nodes
                block_nodes_raw = num_icons_each_block[block_idx_with_most_icons]['nodes']
                
                valid_node_entries = _validate_block_nodes(
                    block_nodes=block_nodes_raw,
                    axtree_path=axtree_path,
                    invalid_elems=invalid_elems,
                    img_file=axtree_path.replace('axtree.json.gz', 'screenshot-full.webp'), # Key for unique_elems
                    unique_elems=unique_elems,
                    resizing_ratio=resizing_ratio,
                    h_start=h_start,
                    sc_w=sc_w,
                    sc_h=sc_h,
                    screenshot=screenshot
                )
                
                iterated_elem_cnt += len(block_nodes_raw)
                
                if not valid_node_entries:
                    continue

                valid_nodes = [entry[0] for entry in valid_node_entries]
                
                # Step 5: Generate Samples
                _create_samples_from_block(
                    valid_nodes=valid_nodes,
                    sc_w=sc_w,
                    sc_h=sc_h,
                    short_screenshot_name=short_screenshot_name,
                    counters=counters,
                    elemclass_stats=elemclass_stats,
                    samples=samples
                )

                valid_img_cnt += 1

            except Exception:
                # Capture stack trace but continue processing other files
                errors.append([axtree_path, traceback.format_exc()])

        # Periodic Checkpoint: Save invalid element records
        if sample_idx % 10000 == 0 or sample_idx == len(sample_dirs) - 1:
            _save_invalid_elems_record(invalid_elems)

    # Step 6: Post-Processing - Class Rebalancing
    # Some UI elements (like 'links') are extremely common, while others ('sliders') are rare.
    # We downsample common classes and upsample rare ones to target the 75th percentile count.
    if ELEMCLASS:
        print("Balancing Element Classification classes...")
        final_elemcls_cnts = [len(v) for v in elemclass_stats.values()]
        if final_elemcls_cnts:
            # Target count is the 75th percentile of class frequencies
            num_sample_each_cls = round(np.percentile(final_elemcls_cnts, 75))
            
            new_elemcls_samples = []
            stats_after_rebal = defaultdict(int)
            
            for elemcls, samples_eachcls in elemclass_stats.items():
                if not samples_eachcls: continue
                
                if len(samples_eachcls) > num_sample_each_cls:
                    # Downsample
                    new_elemcls_samples.extend(random.sample(samples_eachcls, num_sample_each_cls))
                else:
                    # Upsample (Repeat + Random remainder)
                    multiplier = int(num_sample_each_cls // len(samples_eachcls))
                    remainder = num_sample_each_cls % len(samples_eachcls)
                    
                    extended_samples = samples_eachcls * multiplier + random.sample(samples_eachcls, remainder)
                    new_elemcls_samples.extend(extended_samples)
            
            # Add unique IDs to generated samples
            for i, sample in enumerate(new_elemcls_samples):
                sample['id'] += str(i)
                stats_after_rebal[sample['conversations'][1]['value']] += 1
            
            counters['elemclass_cnt'] = len(new_elemcls_samples)
            samples.extend(new_elemcls_samples)
        else:
            stats_after_rebal = {}
    else:
        stats_after_rebal = {}

    # Step 7: Reporting & Saving
    num_invalid_elem = sum(len(v) for v in invalid_elems.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k, v in unique_elems.items() if len(v)])

    report = (
        f"========================================\n"
        f"Processing Report:\n"
        f"#Samples Generated: {len(samples)}\n"
        f"#Unique Elements: {num_unique_elems}\n"
        f"#Valid Unique Images: {num_valid_imgs}\n"
        f"#All Unique Images Scanned: {len(unique_elems)}\n"
        f"Invalid Element Ratio: {num_invalid_elem} / {iterated_elem_cnt} = "
        f"{(num_invalid_elem / iterated_elem_cnt if iterated_elem_cnt > 0 else 0.0):.2f}\n"
        f"----------------------------------------\n"
        f"Task Counters:\n"
        f"  TextLoc   : {counters['text_loc_cnt']}\n"
        f"  OCR       : {counters['ocr_cnt']}\n"
        f"  IconGnd   : {counters['icongnd_cnt']}\n"
        f"  IconRef   : {counters['iconref_cnt']}\n"
        f"  ElemGnd   : {counters['elemgnd_cnt']}\n"
        f"  ElemRef   : {counters['elemref_cnt']}\n"
        f"  ElemClass : {counters['elemclass_cnt']}\n"
        f"  IntentGnd : {counters['intentgnd_cnt']}\n"
        f"  WidgetList: {counters['widgetlist_cnt']}\n"
        f"========================================"
    )
    print(report)
    
    save_file = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_scale{SCALE}_{len(samples)//1000}k{'_actformat' if USE_ACTION_PROMPT else ''}.json")

    # Save Detailed Info JSON
    info_data = {
        'num_samples': len(samples),
        '#valid_unique_images': num_valid_imgs,
        '#all_unique_images': len(unique_elems),
        '#num_unique_elems': num_unique_elems,
        '#invalid_elems': num_invalid_elem,
        '#all_elems': iterated_elem_cnt,
        '#invalid_elem_ratio': num_invalid_elem / iterated_elem_cnt if iterated_elem_cnt else 0,
        'counters': dict(counters),
        'report': report,
        'elem_class_stats_before_rebalance': {k: len(v) for k, v in elemclass_stats.items()},
        'num_sample_each_elem_class_after_rebalance': stats_after_rebal,
        'errors': errors
    }
    
    with open(save_file.replace('.json', '_info.json'), "w") as f:
        json.dump(info_data, f, indent=2)
    
    # Save Small Sample for Debugging
    if samples:
        with open(save_file.replace('.json', '_sample.json'), "w") as f:
            json.dump(random.sample(samples, min(160, len(samples))), f, indent=2)

    # Save Full Dataset
    print(f"Saving full dataset to {save_file}...")
    with open(save_file, "w") as f:
        json.dump(samples, f)
    print("Done.")


if __name__ == "__main__":
    make_webui_data()
