"""
MobileViews dataset preprocessing utilities.
This module processes MobileViews dataset screenshots and view hierarchies to generate
training samples for multiple tasks:
1) Text localization: Given text content, predict its location (box or center point).
2) OCR: Extract text from UI elements using OCR.
3) Intent grounding: Given an intent description, predict the target element location.
4) Widget list: Generate a structured list of all UI elements on the screen.
The code filters invalid UI elements, validates bounding boxes, handles overlapping
elements, and generates normalized coordinates for training.
"""
from __future__ import annotations
import json
import multiprocessing as mp
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import pytesseract
from rapidfuzz import fuzz
from tqdm import tqdm

# Ensure sibling imports resolve at runtime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *  # noqa: F403,F401
from misc import (  # noqa: F401
    classify_node,
    is_box_overlapping_np,
    is_pure_color,
    is_valid_string,
)
from run_info import (  # noqa: E402
    TASK_DISPLAY_NAMES,
    print_error_summary,
    renumber_sample_ids,
    save_run_info,
    summarize_errors,
)

# Configuration constants
DATASET_NAME: str = 'MobileViews'
DEBUG: bool = False # If True, only process the first 3 samples for each dataset part.
SCALE: int = 1000 # The scale of coordinates.
PROB_BOX: float = 0.3 # The probability of generating a box sample.

# Parallelism: 1 forces the sequential code path (also implied when pytesseract
# is unavailable or for visual debugging). Each worker shells out to the
# `tesseract` binary in _filter_overlapping_elements, so high NUM_WORKERS
# multiplies subprocess load — tune down (e.g. 16-32) if the system feels
# thrashed.
NUM_WORKERS: int = max(1, (os.cpu_count() or 1) - 1)

# Task flags - enable/disable specific task generation
TEXTLOC: bool = True # Whether to generate text localization samples.
OCR: bool = True # Whether to generate OCR samples.
ELEMCLASS: bool = False # Whether to generate element classification samples.
INTENTGND: bool = True # Whether to generate intent grounding samples.
WIDGETLIST: bool = True # Whether to generate widget list samples.

# Paths
MOBILEVIEWS_DIR: str = "/volume/pt-coder/users/gji/projects/UIPro/data/MobileViews/" # The directory of MobileViews dataset.
ROOT: str = "/volume/pt-coder/users/gji/projects/UIPro/data"

# Invalid sample IDs to skip (noise samples)
INVALID_SAMPLE_IDS: List[int] = [100547, 100625]

# Dataset parts to process
DATASET_PARTS: List[str] = [
    "MobileViews_0-150000",
    "MobileViews_150001-291197",
    "MobileViews_300000-400000",
    "MobileViews_400001-522301",
]



def _load_screenshot_vh_mapping() -> Dict[str, str]:
    """Load mapping from screenshot files to view hierarchy files.

    Returns:
        Dictionary mapping screenshot file paths to view hierarchy file paths.
    """
    mapping = {}
    for part in DATASET_PARTS:
        csv_path = os.path.join(MOBILEVIEWS_DIR, f'{part.replace("MobileViews", "MobileViews_index")}.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for screenshot_name, vh_name in zip(df.iloc[:, 0], df.iloc[:, 1]):
            mapping[f'{part}/{screenshot_name}'] = f'{part}/{vh_name}'
    return mapping


def _normalize_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Tuple[List[int], str]:
    """Normalize bounding box coordinates to SCALE range.

    Args:
        x1, y1: Top-left corner coordinates.
        x2, y2: Bottom-right corner coordinates.
        width: Image width.
        height: Image height.

    Returns:
        Tuple of (normalized_box, box_str) where normalized_box is [x1, y1, x2, y2]
        in SCALE coordinates and box_str is the string representation.
    """
    box = [
        round(x1 / width * SCALE),
        round(y1 / height * SCALE),
        round(x2 / width * SCALE),
        round(y2 / height * SCALE),
    ]
    box = [min(max(0, i), SCALE - 1) for i in box]
    box_str = f'({box[0]},{box[1]},{box[2]},{box[3]})'
    return box, box_str


def _normalize_center(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Tuple[List[int], str]:
    """Normalize center point coordinates to SCALE range.

    Args:
        x1, y1: Top-left corner coordinates.
        x2, y2: Bottom-right corner coordinates.
        width: Image width.
        height: Image height.

    Returns:
        Tuple of (normalized_center, center_str) where normalized_center is [x, y]
        in SCALE coordinates and center_str is the string representation.
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    normalized_center = [
        min(max(0, round(center_x / width * SCALE)), SCALE - 1),
        min(max(0, round(center_y / height * SCALE)), SCALE - 1),
    ]
    center_str = f"({normalized_center[0]},{normalized_center[1]})"
    return normalized_center, center_str


def _validate_node_bounds(
    node: Dict, width: int, height: int, all_valid_nodes: List[Dict]
) -> bool:
    """Validate a node's bounding box and check for duplicates.

    Args:
        node: UI node dictionary with 'bounds' field.
        width: Image width.
        height: Image height.
        all_valid_nodes: List of already validated nodes.

    Returns:
        True if node passes validation, False otherwise.
    """
    (x1, y1), (x2, y2) = node["bounds"]

    # Check for duplicate boxes
    for existing_node in all_valid_nodes:
        (x3, y3), (x4, y4) = existing_node["bounds"]
        if x1 == x3 and y1 == y3 and x2 == x4 and y2 == y4:
            return False

    # Check for invalid coordinates
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
        return False

    # Check for oversize elements (covering more than 65% of screen)
    if (x2 - x1) * (y2 - y1) / (height * width) >= 0.65:
        return False

    # Check for too small elements (less than 0.5% of screen dimension)
    if (y2 - y1) / height <= 0.005 or (x2 - x1) / width <= 0.005:
        return False

    return True


def _get_node_description(node: Dict) -> str | None:
    """Extract the best available description for a UI node.

    Priority order:
    1. content_description (if more detailed than text)
    2. text
    3. resource_id

    Note: If content_description is more detailed than the text, both are combined
    to provide sufficient information for element localization. For example, in
    seekbar controls, the text might be "4.0" or "5.0" while content_description
    is "media volume" or "call volume", which is more informative.

    Args:
        node: UI node dictionary.

    Returns:
        Node description string, or None if no description available.
    """
    elem_text = node.get('text', None)
    node_desc = None

    # First try text
    if elem_text:
        node_desc = elem_text

    # Then try content_description (often more detailed)
    content_desc = node.get('content_description', None)
    if content_desc is not None:
        content_desc = content_desc.strip()
        if content_desc:
            # If content-desc is more detailed, combine it with text to avoid
            # cases where text alone doesn't provide enough information
            if elem_text is not None and len(elem_text) > 0 and len(content_desc) > len(elem_text):
                node_desc = f"{content_desc}, {elem_text}"
            else:
                node_desc = content_desc

    # Fallback to resource_id
    if node_desc is None:
        resource_id = node.get('resource_id', None)
        if resource_id is not None:
            raw_node_text = resource_id.split('/')[-1].strip()
            if raw_node_text:
                node_desc = raw_node_text

    return node_desc


def _filter_overlapping_elements(
    all_valid_nodes: List[Dict], img: np.ndarray
) -> List[Dict]:
    """Filter out overlapping elements by comparing OCR results with node text.

    For overlapping Icon and Text elements, we use OCR to verify if the element
    text matches what's actually displayed. Elements with low OCR similarity
    (< 22%) are considered invalid and removed.

    Args:
        all_valid_nodes: List of validated UI nodes.
        img: Screenshot image.

    Returns:
        Filtered list of nodes with overlapping invalid elements removed.
    """
    if len(all_valid_nodes) <= 1:
        return all_valid_nodes

    filtered_nodes = []
    for node_idx, node in enumerate(all_valid_nodes):
        (x1, y1), (x2, y2) = node["bounds"]

        # Check if this node overlaps with others
        other_boxes = [
            [x['bounds'][0][0], x['bounds'][0][1], x['bounds'][1][0], x['bounds'][1][1]]
            for cur_idx, x in enumerate(all_valid_nodes)
            if cur_idx != node_idx
        ]
        is_overlap = is_box_overlapping_np(
            target_box=[x1, y1, x2, y2],
            other_boxes=other_boxes,
            threshold=0.01
        )

        if is_overlap:
            elem_type = classify_node(node)
            if elem_type in ['Icon', 'Text']:
                elem_text = node.get('text', None)

                if elem_text is not None:
                    elem_text = elem_text.strip()
                    lower_node_text = elem_text.lower()
                    # Skip if text is a description (e.g., "icon", "button", "back")
                    elem_text_is_description = any(
                        k in lower_node_text for k in ['icon', 'button', 'back']
                    )

                    if not elem_text_is_description:
                        # Perform OCR on the element region
                        roi = img[y1:y2, x1:x2]
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        ocr_result = pytesseract.image_to_string(gray_roi).strip()

                        # Compare OCR result with node text
                        similarity_ratio = fuzz.ratio(ocr_result, elem_text)
                        node['ocr'] = [ocr_result, similarity_ratio]

                        # Remove if similarity is too low (likely invalid element)
                        if similarity_ratio < 22:
                            continue

        filtered_nodes.append(node)

    return filtered_nodes


def _process_node_for_textloc(
    node: Dict,
    elem_text: str,
    all_node_texts: List[str],
    box_str: str,
    center_str: str,
    unnormalized_box: List[int],
    short_img_path: str,
    sample_id: int,
    package: str,
    samples: List[Dict],
    text_loc_cnt: int,
) -> Tuple[int, int]:
    """Generate text localization samples for a node.

    To avoid ambiguity, a text localization task is created only if the element
    text is unique within the screen.

    Args:
        node: UI node dictionary.
        elem_text: Text content of the element.
        all_node_texts: List of all text contents in the current screen.
        box_str: Normalized box string representation.
        center_str: Normalized center point string representation.
        unnormalized_box: Original bounding box [x1, y1, x2, y2].
        short_img_path: Relative path to the image.
        sample_id: Sample identifier.
        package: App package name.
        samples: List to append samples to.
        text_loc_cnt: Current text localization counter.

    Returns:
        Tuple of (updated_counter, samples_added).
    """
    if not TEXTLOC:
        return text_loc_cnt, 0

    # Only create task if text is unique (to avoid ambiguity)
    if all_node_texts.count(elem_text) > 1:
        return text_loc_cnt, 0

    with_box = random.random() < PROB_BOX
    task_id = f'autogui_mobileviews_textloc_{text_loc_cnt}'

    sample = make_textloc_sample(
        task_id=task_id,
        text=elem_text,
        loc=box_str if with_box else center_str,
        output_tag=WITHBOX_TAG if with_box else WITHPOINT_TAG
    )
    sample['task_attr'] = elem_text
    sample['unnormalized_box'] = unnormalized_box
    sample['image'] = short_img_path
    sample['sample_id'] = sample_id
    sample['package'] = package

    samples.append(sample)
    return text_loc_cnt + 1, 1


def _process_node_for_ocr(
    node: Dict,
    elem_text: str,
    box_str: str,
    center_str: str,
    unnormalized_box: List[int],
    short_img_path: str,
    sample_id: int,
    package: str,
    img: np.ndarray,
    center: List[int],
    box_w: int,
    box_h: int,
    samples: List[Dict],
    ocr_cnt: int,
) -> Tuple[int, int]:
    """Generate OCR samples for a node.

    If the element text does not populate the whole box, we use the box as the
    reference since center reference is not accurate in such cases.

    Args:
        node: UI node dictionary.
        elem_text: Text content of the element.
        box_str: Normalized box string.
        center_str: Normalized center point string.
        unnormalized_box: Original bounding box [x1, y1, x2, y2].
        short_img_path: Relative path to the image.
        sample_id: Sample identifier.
        package: App package name.
        img: Screenshot image.
        center: Center coordinates [x, y].
        box_w: Box width.
        box_h: Box height.
        samples: List to append samples to.
        ocr_cnt: Current OCR counter.

    Returns:
        Tuple of (updated_counter, samples_added).
    """
    if not OCR:
        return ocr_cnt, 0

    with_box = random.random() < PROB_BOX

    # If text doesn't populate the whole box, use box reference
    x1 = max(0, center[0] - box_w // 10)
    y1 = max(0, center[1] - box_h // 10)
    x2 = center[0] + box_w // 10
    y2 = center[1] + box_h // 10
    center_roi = [x1, y1, x2, y2]
    if is_pure_color(img, center_roi):
        with_box = True

    loc = box_str if with_box else center_str
    task_id = f'autogui_mobileviews_ocr_{ocr_cnt}'

    sample = make_ocr_sample(
        task_id=task_id,
        text=elem_text,
        loc=loc,
        with_box=with_box
    )
    sample['task_attr'] = loc
    sample['unnormalized_box'] = unnormalized_box
    sample['image'] = short_img_path
    sample['sample_id'] = sample_id
    sample['package'] = package

    samples.append(sample)
    return ocr_cnt + 1, 1


def _process_node_for_intentgnd(
    node: Dict,
    node_desc: str,
    elem_text: str,
    all_node_texts: List[str],
    box_str: str,
    center_str: str,
    normalized_center: List[int],
    unnormalized_box: List[int],
    short_img_path: str,
    sample_id: int,
    package: str,
    used_node_descs: List[str],
    samples: List[Dict],
    intentgnd_cnt: int,
) -> Tuple[int, int]:
    """Generate intent grounding samples for a node.

    Args:
        node: UI node dictionary.
        node_desc: Node description string.
        elem_text: Element text content.
        all_node_texts: List of all text contents in the current screen.
        box_str: Normalized box string representation.
        center_str: Normalized center point string representation.
        normalized_center: Normalized center coordinates [x, y].
        unnormalized_box: Original bounding box [x1, y1, x2, y2].
        short_img_path: Relative path to the image.
        sample_id: Sample identifier.
        package: App package name.
        used_node_descs: List of already used node descriptions (to avoid duplicates).
        samples: List to append samples to.
        intentgnd_cnt: Current intent grounding counter.

    Returns:
        Tuple of (updated_counter, samples_added).
    """
    if not INTENTGND:
        return intentgnd_cnt, 0

    if not node_desc or node_desc in used_node_descs or len(node_desc) > 200:
        return intentgnd_cnt, 0

    # The element content contains no distinguishable content, so we skip it.
    if node_desc.strip().lower() in HTML_TAG_TO_FRIENDLY_NAME.values():
        return intentgnd_cnt, 0

    used_node_descs.append(node_desc)

    # Use Android class tag (even though node might have HTML tag, we prefer Android tag)
    tag = node['class'].split('.')[-1]
    with_box = random.random() < PROB_BOX

    # Only create if element text is unique
    if elem_text not in [None, 'None', 'none'] and all_node_texts.count(elem_text) <= 1:
        intent = gen_naive_action_gnd_anno(
            node_desc.strip(' ,.'),
            tag,
            normalized_center,
            scale=SCALE
        )

        task_id = f'autogui_mobileviews_intentgnd_{intentgnd_cnt}'

        sample = make_intentgnd_sample(
            task_id=task_id,
            intent=intent,
            loc=box_str if with_box else center_str,
            output_tag=''
        )
        sample['task_attr'] = intent
        sample['unnormalized_box'] = unnormalized_box
        sample['image'] = short_img_path
        sample['sample_id'] = sample_id
        sample['package'] = package

        samples.append(sample)
        return intentgnd_cnt + 1, 1

    return intentgnd_cnt, 0


def _process_screen_for_widgetlist(
    all_valid_nodes: List[Dict],
    width: int,
    height: int,
    short_img_path: str,
    sample_id: int,
    package: str,
    samples: List[Dict],
    widgetlist_cnt: int,
) -> Tuple[int, int]:
    """Generate widget list sample for a screen.

    Creates a structured list of all UI elements sorted by position (top-to-bottom,
    left-to-right).

    Args:
        all_valid_nodes: List of validated UI nodes.
        width: Image width.
        height: Image height.
        short_img_path: Relative path to the image.
        sample_id: Sample identifier.
        package: App package name.
        samples: List to append samples to.
        widgetlist_cnt: Current widget list counter.

    Returns:
        Tuple of (updated_counter, samples_added).
    """
    if not WIDGETLIST or len(all_valid_nodes) < 2:
        return widgetlist_cnt, 0

    node_texts_boxes = []
    for node in all_valid_nodes:
        node_desc = '' if node.get('node_desc') is None else node['node_desc']
        (x1, y1), (x2, y2) = node["bounds"]

        unnormalized_box = [int(x1), int(y1), int(x2), int(y2)]
        normalized_box, _ = _normalize_box(
            unnormalized_box[0], unnormalized_box[1],
            unnormalized_box[2], unnormalized_box[3],
            width, height
        )

        # Clean node description (remove newlines, extra spaces)
        clean_desc = ' '.join(
            x for x in node_desc.strip(' ,.').split('\n') if x.strip()
        )

        node_class = node['class'].split('.')[-1]
        node_texts_boxes.append((
            node_class,
            clean_desc,
            unnormalized_box,
            normalized_box
        ))

    # Sort by position: top-to-bottom, left-to-right
    node_texts_boxes.sort(key=lambda x: (x[2][1] + x[2][3], x[2][0] + x[2][2]))

    # Format element list string
    elem_list_str = '\n'.join(
        f"{i} {nodeclass} '{nodetext}' "
        f"({normalized_box[0]},{normalized_box[1]},{normalized_box[2]},{normalized_box[3]})"
        for i, (nodeclass, nodetext, _, normalized_box) in enumerate(node_texts_boxes)
    )

    task_id = f'autogui_mobileviews_widgetlist_{widgetlist_cnt}'

    sample = make_widgetlist_sample(task_id=task_id, elem_list=elem_list_str)
    sample['task_attr'] = None
    sample['image'] = short_img_path
    sample['sample_id'] = sample_id
    sample['package'] = package
    sample['unnormalized_box'] = [x[2] for x in node_texts_boxes]

    samples.append(sample)
    return widgetlist_cnt + 1, 1


# ---------------------------------------------------------------------------
# Parallel-processing worker
# ---------------------------------------------------------------------------
def _process_one_screenshot(item: Tuple[str, str]) -> Dict[str, Any]:
    """Worker: process one (screenshot, view-hierarchy) pair and return deltas.

    The function is side-effect free with respect to the parent process —
    all mutation happens against local containers and only the aggregated
    deltas are returned. Any exception raised during processing is caught
    locally and emitted in ``errors`` so a poisonous input never crashes
    the pool (which would otherwise hang ``imap_unordered``).

    Returned dict keys:
      - ``samples``: list of generated samples (with worker-local task IDs).
      - ``counters``: per-task local counts (canonical ``<task>_cnt`` keys).
      - ``elem_type_cnt`` / ``package_cnt``: aux ``{key: int}`` tallies.
      - ``node_invalid_types``: ``{reason: int}`` for the invalid-elements report.
      - ``per_part_stats``: ``{part_name: {images_seen, images_valid, samples}}``.
      - ``processed`` / ``valid`` / ``iterated``: per-item integer tallies.
      - ``errors``: ``[[path, traceback_str], ...]`` (empty on success).
    """
    screenshot_rel_path, vh_rel_path = item
    samples: List[Dict] = []
    counters: Dict[str, int] = defaultdict(int)
    elem_type_cnt: Dict[str, int] = defaultdict(int)
    package_cnt: Dict[str, int] = defaultdict(int)
    node_invalid_types: Dict[str, int] = defaultdict(int)
    per_part_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'images_seen': 0, 'images_valid': 0, 'samples': 0}
    )
    errors: List[List[str]] = []
    processed = 0
    valid = 0
    iterated = 0

    try:
        sample_id = int(os.path.basename(screenshot_rel_path).split('.')[0])
        if sample_id in INVALID_SAMPLE_IDS:
            return _bundle_screenshot_result(
                samples, counters, elem_type_cnt, package_cnt,
                node_invalid_types, per_part_stats,
                processed, valid, iterated, errors,
            )

        screenshot_file = os.path.join(MOBILEVIEWS_DIR, screenshot_rel_path)
        vh_file = os.path.join(MOBILEVIEWS_DIR, vh_rel_path)
        if not os.path.exists(screenshot_file) or not os.path.exists(vh_file):
            return _bundle_screenshot_result(
                samples, counters, elem_type_cnt, package_cnt,
                node_invalid_types, per_part_stats,
                processed, valid, iterated, errors,
            )

        processed = 1
        part_name = screenshot_rel_path.split('/', 1)[0]
        per_part_stats[part_name]['images_seen'] += 1

        with open(vh_file, 'r') as f:
            vh = json.load(f)
        nodes = vh["views"]

        img = cv2.imread(screenshot_file)
        if img is None:
            return _bundle_screenshot_result(
                samples, counters, elem_type_cnt, package_cnt,
                node_invalid_types, per_part_stats,
                processed, valid, iterated, errors,
            )

        if img.shape[0] > 1920:
            img = img[:1920]
            cv2.imwrite(screenshot_file, img)

        height, width = img.shape[:2]

        # Step 1: filter and validate nodes
        all_valid_nodes: List[Dict] = []
        for node in nodes:
            if len(node.get('children', [])) > 0 or not node.get('visible', False):
                continue
            iterated += 1

            if not _validate_node_bounds(node, width, height, all_valid_nodes):
                (x1, y1), (x2, y2) = node["bounds"]
                is_duplicate = False
                for existing_node in all_valid_nodes:
                    (x3, y3), (x4, y4) = existing_node["bounds"]
                    if x1 == x3 and y1 == y3 and x2 == x4 and y2 == y4:
                        node_invalid_types['duplicate box'] += 1
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
                    node_invalid_types['invalid box coordinates'] += 1
                    continue
                if (x2 - x1) * (y2 - y1) / (height * width) >= 0.65:
                    node_invalid_types['oversize element'] += 1
                    continue
                if (y2 - y1) / height <= 0.005 or (x2 - x1) / width <= 0.005:
                    node_invalid_types['too small element'] += 1
                    continue

            (x1, y1), (x2, y2) = node["bounds"]
            if is_pure_color(img, [x1, y1, x2, y2]):
                node_invalid_types['element not displayed'] += 1
                continue

            all_valid_nodes.append(node)

        # Step 2: overlap filtering via OCR (subprocess to tesseract)
        all_valid_nodes = _filter_overlapping_elements(all_valid_nodes, img)
        if not all_valid_nodes:
            return _bundle_screenshot_result(
                samples, counters, elem_type_cnt, package_cnt,
                node_invalid_types, per_part_stats,
                processed, valid, iterated, errors,
            )

        # Step 3: language check on the screen's texts
        all_node_texts: List[str] = []
        for node in all_valid_nodes:
            raw = node.get('text', None)
            if raw is None:
                continue
            txt = raw.strip()
            if not txt:
                continue
            all_node_texts.append(txt)

        contain_invalid_characters = False
        for text in all_node_texts:
            if text is None:
                continue
            if not is_valid_string(text):
                contain_invalid_characters = True
                break
        if contain_invalid_characters:
            node_invalid_types['invalid text language'] += len(all_valid_nodes)
            return _bundle_screenshot_result(
                samples, counters, elem_type_cnt, package_cnt,
                node_invalid_types, per_part_stats,
                processed, valid, iterated, errors,
            )

        short_img_path = screenshot_file[screenshot_file.find(DATASET_NAME):]

        # Step 4: per-node sample generation
        used_boxes: List[str] = []
        used_node_descs: List[str] = []
        # `package` is reused outside the loop as the screen-level attribution.
        package = ''

        for node in all_valid_nodes:
            (x1, y1), (x2, y2) = node["bounds"]
            bound_box_str = f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"
            if bound_box_str in used_boxes:
                node_invalid_types['overlapping element'] += 1
                continue
            used_boxes.append(bound_box_str)

            _, box_str = _normalize_box(x1, y1, x2, y2, width, height)
            normalized_center, center_str = _normalize_center(
                x1, y1, x2, y2, width, height
            )
            unnormalized_box = [int(x1), int(y1), int(x2), int(y2)]
            box_w, box_h = x2 - x1, y2 - y1
            center = [(x1 + x2) // 2, (y1 + y2) // 2]

            elem_type = classify_node(node)
            elem_text = node.get('text', None)
            package = node.get('package', '')
            elem_type_cnt[elem_type] += 1
            if package:
                package_cnt[package] += 1

            node_desc = _get_node_description(node)
            node['node_desc'] = node_desc

            if elem_text is not None and 0 < len(elem_text) <= 200:
                elem_text = elem_text.strip()
                counters['text_loc_cnt'], _ = _process_node_for_textloc(
                    node=node, elem_text=elem_text,
                    all_node_texts=all_node_texts, box_str=box_str,
                    center_str=center_str, unnormalized_box=unnormalized_box,
                    short_img_path=short_img_path, sample_id=sample_id,
                    package=package, samples=samples,
                    text_loc_cnt=counters['text_loc_cnt'],
                )
                counters['ocr_cnt'], _ = _process_node_for_ocr(
                    node=node, elem_text=elem_text, box_str=box_str,
                    center_str=center_str, unnormalized_box=unnormalized_box,
                    short_img_path=short_img_path, sample_id=sample_id,
                    package=package, img=img, center=center,
                    box_w=int(box_w), box_h=int(box_h), samples=samples,
                    ocr_cnt=counters['ocr_cnt'],
                )

            if node_desc:
                counters['intentgnd_cnt'], _ = _process_node_for_intentgnd(
                    node=node, node_desc=node_desc,
                    elem_text=elem_text or '',
                    all_node_texts=all_node_texts, box_str=box_str,
                    center_str=center_str,
                    normalized_center=normalized_center,
                    unnormalized_box=unnormalized_box,
                    short_img_path=short_img_path, sample_id=sample_id,
                    package=package, used_node_descs=used_node_descs,
                    samples=samples,
                    intentgnd_cnt=counters['intentgnd_cnt'],
                )

        # Step 5: one widget-list sample per screen
        counters['widgetlist_cnt'], _ = _process_screen_for_widgetlist(
            all_valid_nodes=all_valid_nodes,
            width=width, height=height,
            short_img_path=short_img_path, sample_id=sample_id,
            package=package, samples=samples,
            widgetlist_cnt=counters['widgetlist_cnt'],
        )

        if samples:
            valid = 1
            per_part_stats[part_name]['images_valid'] += 1
        per_part_stats[part_name]['samples'] += len(samples)

    except Exception:
        errors.append([screenshot_rel_path, traceback.format_exc()])

    return _bundle_screenshot_result(
        samples, counters, elem_type_cnt, package_cnt,
        node_invalid_types, per_part_stats,
        processed, valid, iterated, errors,
    )


def _bundle_screenshot_result(
    samples: List[Dict],
    counters: Dict[str, int],
    elem_type_cnt: Dict[str, int],
    package_cnt: Dict[str, int],
    node_invalid_types: Dict[str, int],
    per_part_stats: Dict[str, Dict[str, int]],
    processed: int,
    valid: int,
    iterated: int,
    errors: List[List[str]],
) -> Dict[str, Any]:
    """Pack a worker's locals into a small pickleable dict."""
    return {
        'samples': samples,
        'counters': dict(counters),
        'elem_type_cnt': dict(elem_type_cnt),
        'package_cnt': dict(package_cnt),
        'node_invalid_types': dict(node_invalid_types),
        'per_part_stats': {k: dict(v) for k, v in per_part_stats.items()},
        'processed': processed,
        'valid': valid,
        'iterated': iterated,
        'errors': errors,
    }


def make_mobileviews_data() -> None:
    """Generate MobileViews training samples and write them to disk.

    Iterates through all screenshot/view-hierarchy pairs, validates UI elements,
    filters invalid nodes, and generates samples for enabled tasks. Outputs summary
    statistics and a small random subset for quick inspection.
    """
    # Load screenshot to view hierarchy mapping
    mv_screenshot_vh_mapping = _load_screenshot_vh_mapping()

    # Initialize counters and storage
    samples = []
    node_invalid_types = defaultdict(int)

    processed_img_cnt = 0
    valid_img_cnt = 0
    iterated_elem_cnt = 0

    # Canonical per-task counters: defaultdict matches the convention in
    # make_webui_data.py. Keys are `<task>_cnt`; CamelCase display names
    # live in TASK_DISPLAY_NAMES above.
    task_counters: Dict[str, int] = defaultdict(int)

    # Auxiliary stats surfaced in info.json (kept lightweight on purpose).
    elem_type_cnt: Counter = Counter()
    package_cnt: Counter = Counter()
    per_part_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'images_seen': 0, 'images_valid': 0, 'samples': 0}
    )

    errors: List[List[str]] = []

    # Build the work list. In DEBUG mode the original used `sample_idx % 5000`
    # to keep ~4 of every 5000 items — preserve that selection here so the
    # parallel and sequential paths submit the same set.
    items: List[Tuple[str, str]] = list(mv_screenshot_vh_mapping.items())
    if DEBUG:
        items = [it for i, it in enumerate(items) if i % 5000 <= 3]

    # ------------------------------------------------------------------
    # Streaming aggregator: shared by parallel + sequential branches so the
    # bookkeeping logic lives in exactly one place.
    # ------------------------------------------------------------------
    def _aggregate(result: Dict[str, Any]) -> None:
        nonlocal processed_img_cnt, valid_img_cnt, iterated_elem_cnt
        samples.extend(result['samples'])
        for k, v in result['counters'].items():
            task_counters[k] += v
        for k, v in result['elem_type_cnt'].items():
            elem_type_cnt[k] += v
        for k, v in result['package_cnt'].items():
            package_cnt[k] += v
        for k, v in result['node_invalid_types'].items():
            node_invalid_types[k] += v
        for part, stats in result['per_part_stats'].items():
            for k, v in stats.items():
                per_part_stats[part][k] += v
        processed_img_cnt += result['processed']
        valid_img_cnt += result['valid']
        iterated_elem_cnt += result['iterated']
        errors.extend(result['errors'])

    use_parallel = NUM_WORKERS > 1
    print(
        f"Processing {len(items)} screenshots "
        f"{'in parallel across ' + str(NUM_WORKERS) + ' workers' if use_parallel else 'sequentially'}..."
    )

    if use_parallel:
        # imap_unordered streams results back as workers finish — keeps the IPC
        # queue from filling and avoids the classic deadlock where workers
        # block on a full output queue. Exceptions are caught inside the
        # worker and emitted as ``errors`` entries, so a poisonous input
        # cannot hang the pool either.
        chunksize = max(1, min(64, len(items) // max(1, NUM_WORKERS * 4)))
        with mp.Pool(processes=NUM_WORKERS) as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    _process_one_screenshot, items, chunksize=chunksize
                ),
                total=len(items),
                desc=f"MobileViews ×{NUM_WORKERS}",
                smoothing=0.05,
            )
            for result in pbar:
                _aggregate(result)
    else:
        for item in tqdm(items, total=len(items), desc="MobileViews"):
            _aggregate(_process_one_screenshot(item))

    # Surface mass-failures loudly (e.g. tesseract missing, or a typo that
    # crashes every worker call) — same pattern as make_webui_data.
    print_error_summary(errors, total_tasks=len(items))

    # After parallel aggregation each worker's per-task counters started at 0,
    # so sample IDs collide globally. Renumber by extracting the `<prefix>`
    # from each id and re-striping the trailing integer.
    renumber_sample_ids(samples)

    # ------------------------------------------------------------------
    # Output filename + unified reporting via run_info.save_run_info().
    # ------------------------------------------------------------------
    save_to_dir = os.path.join(ROOT, f"{DATASET_NAME}_processed")
    os.makedirs(save_to_dir, exist_ok=True)
    file_name = os.path.join(
        save_to_dir,
        f"{DATASET_NAME}_{'TextLoc_' if TEXTLOC else ''}{'OCR_' if OCR else ''}"
        f"{'ElemClass_' if ELEMCLASS else ''}{'IntentGnd_' if INTENTGND else ''}"
        f"{'WidgetList_' if WIDGETLIST else ''}scale{SCALE}_{len(samples)//1000}k"
        f"{'_debug' if DEBUG else ''}.json"
    )
    print(f"Save to {file_name}")

    save_run_info(
        dataset=DATASET_NAME,
        output_path=file_name,
        samples=samples,
        counters=task_counters,
        config={
            'debug': DEBUG,
            'scale': SCALE,
            'prob_box': PROB_BOX,
            'dataset_parts': DATASET_PARTS,
        },
        images={
            'processed': processed_img_cnt,
            'valid': valid_img_cnt,
        },
        elements={
            'iterated': iterated_elem_cnt,
            'invalid': sum(node_invalid_types.values()),
            'invalid_breakdown': dict(
                sorted(node_invalid_types.items(), key=lambda kv: -kv[1])
            ),
            'by_type': dict(elem_type_cnt.most_common()),
        },
        tasks_enabled={
            TASK_DISPLAY_NAMES[key]: enabled
            for key, enabled in {
                'text_loc_cnt': TEXTLOC,
                'ocr_cnt': OCR,
                'elemclass_cnt': ELEMCLASS,
                'intentgnd_cnt': INTENTGND,
                'widgetlist_cnt': WIDGETLIST,
            }.items()
        },
        extra_sections={
            'per_part': dict(per_part_stats),
            'top_packages': dict(package_cnt.most_common(20)),
            'errors_summary': summarize_errors(errors),
            'errors_total': len(errors),
        },
    )

    # Full dataset.
    with open(file_name, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to {file_name}")


if __name__ == '__main__':
    make_mobileviews_data()