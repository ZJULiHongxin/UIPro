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

import copy
import gzip
import glob
import json
import multiprocessing as mp
import os
import random
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Sibling imports (assuming the environment is set up with correct PYTHONPATH)
from utils.data_utils.task_prompt_lib import (
    WITHBOX_TAG,
    WITHPOINT_TAG,
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
from utils.data_utils.run_info import (
    print_error_summary,
    renumber_sample_ids,
    save_run_info,
    summarize_errors,
)

# =============================================================================
# Configuration & Constants
# =============================================================================

DATASET_NAME = 'WebUI'
DEBUG = False       # If True, processes only a small subset of data for testing.
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
LONGEST = -1      # Target longer side length for image resizing. LONGEST = 1344 is a proper choice. LONGEST = -1 means no resizing.

# --- Parallelism ---
# Number of worker processes. Set to 1 (or use DRAW=True) to force the
# sequential code path, which is required when visually debugging.
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1)
# How often the cumulative `invalid_elem_record.json` is checkpointed to disk
# (counted in completed sample-directories). Trades durability against I/O.
CHECKPOINT_EVERY = 10000

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
WEBUI_DIR = f"/volume/pt-coder/users/gji/projects/UIPro/data/{DATASET_NAME}/dh2"
WEBUI_PROCESSED_IMG_DIR = f"/volume/pt-coder/users/gji/projects/UIPro/data/{DATASET_NAME}/WebUI_screenshots"
ROOT = "/volume/pt-coder/users/gji/projects/UIPro/data"
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


# ---------------------------------------------------------------------------
# Cumulative-stats persistence
# ---------------------------------------------------------------------------
# `invalid_elem_record.json` is already cumulative (set semantics dedupe across
# runs). To make the reported ratios coherent we keep matching cumulative
# tallies for iterated / processed / valid in a sibling JSON file. Delete
# `cumulative_stats.json` to reset.
_CUMULATIVE_STATS_FILE = os.path.join(SAVE_ROOT, 'cumulative_stats.json')
_CUMULATIVE_KEYS = ('iterated_elem_cnt', 'processed_img_cnt', 'valid_img_cnt')


def _load_cumulative_stats() -> Dict[str, int]:
    """Load cumulative iteration tallies from disk (defaults to all-zero)."""
    if os.path.exists(_CUMULATIVE_STATS_FILE):
        try:
            data = json.load(open(_CUMULATIVE_STATS_FILE))
            return {k: int(data.get(k, 0)) for k in _CUMULATIVE_KEYS}
        except Exception:
            pass
    return {k: 0 for k in _CUMULATIVE_KEYS}


def _save_cumulative_stats(stats: Dict[str, int]) -> None:
    """Persist cumulative iteration tallies to disk."""
    with open(_CUMULATIVE_STATS_FILE, 'w') as f:
        json.dump({k: int(stats.get(k, 0)) for k in _CUMULATIVE_KEYS}, f, indent=2)


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

        if with_box:
            output_tag = WITHBOX_TAG
            loc = box_str
        else:
            output_tag = WITHPOINT_TAG
            loc = f'({normalized_center[0]},{normalized_center[1]})'
        
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
                        output_tag=output_tag
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
                        output_tag=output_tag
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
                            output_tag=output_tag
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
                        output_tag=output_tag
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
                    output_tag=output_tag
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
                output_tag=output_tag
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


# ---------------------------------------------------------------------------
# Parallel-processing worker
# ---------------------------------------------------------------------------
# All buckets that _validate_block_nodes may write to. Workers seed a fresh
# defaultdict-like structure so missing buckets don't crash; main aggregates.
_INVALID_BUCKETS: Tuple[str, ...] = (
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

def _process_one_sample_dir(sample_dir: str) -> Dict[str, Any]:
    """Worker: process every axtree under ``sample_dir`` and return deltas.

    The function is intentionally side-effect-free with respect to the parent
    process — all mutation happens against local containers and only the
    aggregated deltas are returned. This keeps the ``multiprocessing.Pool``
    contract simple and dead-lock free (no shared state, no locks, no nested
    pools).

    Returned dict keys:
      - ``samples``: list of generated samples (with worker-local task IDs).
      - ``elemclass_stats``: ``{role: [class samples]}`` to be merged into the
        global rebalance pool.
      - ``counters``: per-task local counts.
      - ``invalid_updates``: ``{bucket: [sample_id, ...]}`` to union into the
        cumulative invalid-elem record.
      - ``unique_elem_entries``: per-image box lists (image keys are unique
        across dirs, so the parent merges with a plain ``dict.update``).
      - ``iterated_elem_cnt`` / ``processed_img_cnt`` / ``valid_img_cnt``:
        per-dir tallies.
      - ``errors``: ``[[axtree_path, traceback_str], ...]``
    """
    samples: List[Dict] = []
    elemclass_stats: Dict[str, List[Dict]] = defaultdict(list)
    counters: Dict[str, int] = defaultdict(int)
    invalid_updates: Dict[str, Set[str]] = {b: set() for b in _INVALID_BUCKETS}
    unique_elems_local: Dict[str, List[List[int]]] = {}
    errors: List[List[str]] = []
    iterated_elem_cnt = 0
    processed_img_cnt = 0
    valid_img_cnt = 0

    axtree_paths = glob.glob(os.path.join(WEBUI_DIR, sample_dir, '*axtree.json.gz'))
    for axtree_path in axtree_paths:
        try:
            axtree_nodes, img, bboxes, device_type = _process_axtree_and_image(
                axtree_path, unique_elems_local
            )
            if axtree_nodes is None:
                continue

            processed_img_cnt += 1

            block_idx_with_most_icons, num_icons_each_block = _split_image_into_blocks(
                img, axtree_nodes, bboxes, device_type, axtree_path
            )
            if (
                block_idx_with_most_icons == -1
                or num_icons_each_block[block_idx_with_most_icons]['num_icons'] == 0
            ):
                continue

            _, W = img.shape[:2]
            ratio = 2532 / 1170 if 'iPhone' in axtree_path else 720 / 1280
            block_h = int(W * ratio)
            h_start = block_h * block_idx_with_most_icons

            cropped_img = img[h_start:h_start + block_h]
            screenshot, resizing_ratio = resize_image(cropped_img, max_size=LONGEST)
            sc_h, sc_w = screenshot.shape[:2]

            traj_dir = os.path.join(WEBUI_PROCESSED_IMG_DIR, sample_dir)
            os.makedirs(traj_dir, exist_ok=True)
            screenshot_file = os.path.join(traj_dir, f"{device_type}.png")
            short_screenshot_name = (
                f"{DATASET_NAME}/WebUI_screenshots/{sample_dir}/{device_type}.png"
            )
            if not os.path.exists(screenshot_file):
                cv2.imwrite(screenshot_file, screenshot)

            block_nodes_raw = num_icons_each_block[block_idx_with_most_icons]['nodes']
            valid_node_entries = _validate_block_nodes(
                block_nodes=block_nodes_raw,
                axtree_path=axtree_path,
                invalid_elems=invalid_updates,
                img_file=axtree_path.replace(
                    'axtree.json.gz', 'screenshot-full.webp'
                ),
                unique_elems=unique_elems_local,
                resizing_ratio=resizing_ratio,
                h_start=h_start,
                sc_w=sc_w,
                sc_h=sc_h,
                screenshot=screenshot,
            )
            iterated_elem_cnt += len(block_nodes_raw)
            if not valid_node_entries:
                continue

            valid_nodes = [entry[0] for entry in valid_node_entries]
            samples_before = len(samples)
            elemclass_before = sum(len(v) for v in elemclass_stats.values())
            _create_samples_from_block(
                valid_nodes=valid_nodes,
                sc_w=sc_w,
                sc_h=sc_h,
                short_screenshot_name=short_screenshot_name,
                counters=counters,
                elemclass_stats=elemclass_stats,
                samples=samples,
            )
            produced = (
                len(samples) > samples_before
                or sum(len(v) for v in elemclass_stats.values()) > elemclass_before
            )
            if produced:
                valid_img_cnt += 1

        except Exception:
            errors.append([axtree_path, traceback.format_exc()])

    return {
        'samples': samples,
        'elemclass_stats': dict(elemclass_stats),
        'counters': dict(counters),
        'invalid_updates': {k: list(v) for k, v in invalid_updates.items() if v},
        'unique_elem_entries': unique_elems_local,
        'iterated_elem_cnt': iterated_elem_cnt,
        'processed_img_cnt': processed_img_cnt,
        'valid_img_cnt': valid_img_cnt,
        'errors': errors,
    }


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
    
    # Load persistent records. Both `invalid_elem_record.json` (set-based) and
    # `cumulative_stats.json` (count-based) accumulate across runs so the
    # reported ratios stay coherent across the dataset's lifetime.
    invalid_elems = _load_invalid_elems_record()
    cumulative_stats = _load_cumulative_stats()
    
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

    # ------------------------------------------------------------------
    # Streaming aggregator: pulled by both the sequential and parallel
    # branches below so the bookkeeping logic lives in exactly one place.
    # Returns True when this result yielded any output (used only for the
    # `valid_img_cnt` running tally — kept here as a closure to avoid
    # leaking the many locals it touches).
    # ------------------------------------------------------------------
    def _aggregate(result: Dict[str, Any]) -> None:
        nonlocal iterated_elem_cnt, processed_img_cnt, valid_img_cnt
        samples.extend(result['samples'])
        for role, eclass in result['elemclass_stats'].items():
            elemclass_stats[role].extend(eclass)
        for k, v in result['counters'].items():
            counters[k] += v
        for bucket, ids in result['invalid_updates'].items():
            invalid_elems.setdefault(bucket, set()).update(ids)
        # Image keys are unique per sample_dir, so `update` is safe and O(1)/entry.
        unique_elems.update(result['unique_elem_entries'])
        errors.extend(result['errors'])
        iterated_elem_cnt += result['iterated_elem_cnt']
        processed_img_cnt += result['processed_img_cnt']
        valid_img_cnt += result['valid_img_cnt']

    # DRAW mutates pixels of a shared image for human inspection and is only
    # used during debugging; force sequential to keep its side effects visible.
    use_parallel = NUM_WORKERS > 1 and not DRAW
    print(
        f"Processing {len(sample_dirs)} directories "
        f"{'in parallel across ' + str(NUM_WORKERS) + ' workers' if use_parallel else 'sequentially'}..."
    )

    if use_parallel:
        # chunksize ~ N / (workers * 4): big enough to amortize IPC, small
        # enough that imap_unordered keeps progress flowing per result.
        # We don't use a manual queue or Manager — imap_unordered drains
        # results as they're produced, so the only ways to deadlock would be
        # nesting another pool inside the worker or letting an exception
        # bubble inside the worker (we catch and return them instead).
        chunksize = max(1, min(64, len(sample_dirs) // max(1, NUM_WORKERS * 4)))
        completed = 0
        with mp.Pool(processes=NUM_WORKERS) as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    _process_one_sample_dir, sample_dirs, chunksize=chunksize
                ),
                total=len(sample_dirs),
                desc=f"WebUI ×{NUM_WORKERS}",
                smoothing=0.05,
            )
            for result in pbar:
                _aggregate(result)
                completed += 1
                if completed % CHECKPOINT_EVERY == 0:
                    _save_invalid_elems_record(invalid_elems)
    else:
        for sample_dir in tqdm(sample_dirs, total=len(sample_dirs), desc="WebUI"):
            _aggregate(_process_one_sample_dir(sample_dir))

    # Final flush of the cumulative invalid-element record.
    _save_invalid_elems_record(invalid_elems)

    # Surface mass-failures loudly: a regression that breaks every worker call
    # (e.g., an UnboundLocalError in `_create_samples_from_block`) otherwise
    # only manifests as a smaller sample count buried in info.json.
    print_error_summary(errors, total_tasks=len(sample_dirs))

    # After aggregation, each worker's per-task local counters started at 0,
    # so sample IDs collide globally. Renumber by extracting the `<prefix>`
    # from each id and re-striping the trailing integer.
    renumber_sample_ids(samples)

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
            
            for samples_eachcls in elemclass_stats.values():
                if not samples_eachcls: continue
                
                if len(samples_eachcls) > num_sample_each_cls:
                    # Downsample
                    chosen = random.sample(samples_eachcls, num_sample_each_cls)
                else:
                    # Upsample (Repeat + Random remainder)
                    multiplier = int(num_sample_each_cls // len(samples_eachcls))
                    remainder = num_sample_each_cls % len(samples_eachcls)
                    chosen = samples_eachcls * multiplier + random.sample(samples_eachcls, remainder)

                # Deep-copy so repeated references become independent dicts —
                # otherwise mutating `id` below would alias across duplicates.
                new_elemcls_samples.extend(copy.deepcopy(s) for s in chosen)

            # Add unique IDs to generated samples
            for i, sample in enumerate(new_elemcls_samples):
                sample['id'] = f'autogui_webui_elemclass_{i}'
                stats_after_rebal[sample['conversations'][1]['value']] += 1
            
            counters['elemclass_cnt'] = len(new_elemcls_samples)
            samples.extend(new_elemcls_samples)
        else:
            stats_after_rebal = {}
    else:
        stats_after_rebal = {}

    # Step 7: Reporting & Saving
    # Roll this run's tallies into the cumulative-across-runs counters so the
    # reported ratios line up with the cumulative `invalid_elems` record.
    cumulative_stats['iterated_elem_cnt'] += iterated_elem_cnt
    cumulative_stats['processed_img_cnt'] += processed_img_cnt
    cumulative_stats['valid_img_cnt'] += valid_img_cnt
    _save_cumulative_stats(cumulative_stats)

    num_invalid_elem = sum(len(v) for v in invalid_elems.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())

    save_file = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_scale{SCALE}_{len(samples)//1000}k{'_actformat' if USE_ACTION_PROMPT else ''}.json")

    save_run_info(
        dataset=DATASET_NAME,
        output_path=save_file,
        samples=samples,
        counters=counters,
        config={
            'debug': DEBUG,
            'scale': SCALE,
            'prob_box': PROB_BOX,
            'longest': LONGEST,
            'use_action_prompt': USE_ACTION_PROMPT,
            'skip_checking': SKIP_CHECKING,
        },
        images={
            # Cumulative aggregated counts (invalid_elem_record.json is also
            # cumulative). Per-run figures are kept in `extra_sections.run_stats`.
            'processed': cumulative_stats['processed_img_cnt'],
            'valid': cumulative_stats['valid_img_cnt'],
        },
        elements={
            'iterated': cumulative_stats['iterated_elem_cnt'],
            'invalid': num_invalid_elem,
            'unique': num_unique_elems,
            'invalid_breakdown': {k: len(v) for k, v in invalid_elems.items()},
        },
        tasks_enabled={
            'TextLoc': TEXTLOC,
            'OCR': OCR,
            'IconGnd': ICONGND,
            'IconRef': ICONREF,
            'ElemGnd': ELEMGND,
            'ElemRef': ELEMREF,
            'ElemClass': ELEMCLASS,
            'IntentGnd': INTENTGND,
            'WidgetList': WIDGETLIST,
        },
        extra_sections={
            'run_stats': {
                'iterated_elem_cnt': iterated_elem_cnt,
                'processed_img_cnt': processed_img_cnt,
                'valid_img_cnt': valid_img_cnt,
                'unique_imgs_seen_this_run': len(unique_elems),
            },
            'elem_class_stats_before_rebalance': {
                k: len(v) for k, v in elemclass_stats.items()
            },
            'num_sample_each_elem_class_after_rebalance': stats_after_rebal,
            # Keep info.json compact even when mass-failure occurs: group
            # tracebacks by their last line and store up to 5 example paths
            # per group.
            'errors_summary': summarize_errors(errors),
            'errors_total': len(errors),
        },
        preview_size=160,
    )

    # Save Full Dataset
    print(f"Saving full dataset to {save_file}...")
    with open(save_file, "w") as f:
        json.dump(samples, f)
    print("Done.")


if __name__ == "__main__":
    make_webui_data()
