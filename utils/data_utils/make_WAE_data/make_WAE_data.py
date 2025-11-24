"""
WAE dataset preprocessing utilities.

This module processes WAE dataset screenshots and view hierarchies to generate
training samples for multiple tasks:
1) Text localization: Given text content, predict its location (box or center point).
2) OCR: Extract text from UI elements using OCR.
3) Icon Grounding: Given an icon description, predict its location.
4) Icon Referencing: Given a location, predict the icon description.
5) Intent grounding: Given an intent description, predict the target element location.
6) Widget list: Generate a structured list of all UI elements on the screen.

The code filters invalid UI elements, validates bounding boxes, handles overlapping
elements, and generates normalized coordinates for training.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Optional

import cv2
import magic
import numpy as np
import pytesseract
from PIL import Image
from rapidfuzz import fuzz
from tqdm import tqdm

# Ensure sibling imports resolve at runtime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils.task_prompt_lib import *  # noqa: F403,F401
from utils.data_utils.misc import (  # noqa: F401
    classify_node,
    detect_invalid_lang,
    find_all_elem_texts_boxes,
    is_box_overlapping_np,
    is_pure_color,
    is_valid_string,
)

# Configuration constants
DATASET_NAME: str = 'WAE'
DEBUG: bool = False

# Task flags - enable/disable specific task generation
TEXTLOC: bool = True
OCR: bool = True
ICONGND: bool = True
ICONREF: bool = True
ELEMCLASS: bool = True
INTENTGND: bool = True
WIDGETLIST: bool = True

# Generation parameters
PROB_BOX: float = 0.0
SCALE: int = 100
SKIP_CHECKING: bool = False
CHUNK_SIZE: int = 25 if not DEBUG else 1

# Paths
WAE_DIR: str = f"/mnt/vdb1/hongxin_li/{DATASET_NAME}/tmp"
ROOT_DIR: str = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"
SAVE_IMAGES_TO: str = os.path.join(ROOT_DIR, f"{DATASET_NAME}{'_debug' if DEBUG else ''}")
SAVE_ROOT: str = os.path.join(ROOT_DIR, f"{DATASET_NAME}_processed")
CHUNK_DIR: str = os.path.join(SAVE_ROOT, 'chunks')


def _normalize_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Tuple[List[int], str]:
    """Normalize bounding box coordinates to SCALE range."""
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
    """Normalize center point coordinates to SCALE range."""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    normalized_center = [
        min(max(0, round(center_x / width * SCALE)), SCALE - 1),
        min(max(0, round(center_y / height * SCALE)), SCALE - 1),
    ]
    center_str = f"({normalized_center[0]},{normalized_center[1]})"
    return normalized_center, center_str


def _validate_node_bounds(
    node: Dict,
    width: int,
    height: int,
    all_valid_nodes: List[Dict],
    sample_identifier: str,
    invalid_elem_tracker: Dict[str, Set]
) -> bool:
    """Validate a node's bounding box and check for duplicates/invalid properties."""
    if SKIP_CHECKING:
        return True

    x1, y1, x2, y2 = node["box"]

    # Check for duplicate boxes
    for existing_node in all_valid_nodes:
        x3, y3, x4, y4 = existing_node["box"]
        if x1 == x3 and y1 == y3 and x2 == x4 and y2 == y4:
            invalid_elem_tracker[DUPLICATE_ELEMEMNT].add(sample_identifier)
            return False

    # Check for invalid coordinates
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
        invalid_elem_tracker[INVALID_ELEM_BOX].add(sample_identifier)
        return False

    node_w, node_h = x2 - x1, y2 - y1

    # Check for oversize elements
    if node_w * node_h / (height * width) >= 0.65:
        invalid_elem_tracker[OVERSIZED_ELEMENT].add(sample_identifier)
        return False

    # Check for extreme aspect ratio
    if node_w / node_h < 0.05 or node_w / node_h > 20:
        invalid_elem_tracker[EXTREME_ASPECT_RATIO].add(sample_identifier)
        return False

    # Check for too small elements
    if node_h / height <= 0.005 or node_w / width <= 0.005:
        invalid_elem_tracker[TOO_SMALL_ELEMENT].add(sample_identifier)
        return False

    # Remove meaningless nodes
    if node['tag'] in ['LinearLayout', 'FrameLayout', 'GridView', 'View', 'ViewGroup', 'TextView'] and \
            len(node.get('text', '')) == 0 and len(node.get('content-desc', '')) == 0:
        invalid_elem_tracker[INVALID_ELEM_CONTENT].add(sample_identifier)
        return False

    return True


def _get_node_description(node: Dict) -> Optional[str]:
    """Extract the best available description for a UI node."""
    elem_text = node.get('text', None)
    node_desc = None

    if elem_text:
        node_desc = elem_text

    content_desc = node.get('content-desc', None)
    if content_desc is not None:
        content_desc = content_desc.strip()
        if content_desc:
            # If content-desc is more detailed, use it combined with node_text as the target description
            # to avoid node_text not providing enough information for localization.
            # Example: In some cases, the text on the seekbar points is "4.0", "5.0",
            # while content-desc is "media volume" "call volume", which is more detailed.
            if elem_text is not None and len(elem_text) and len(content_desc) > len(elem_text):
                node_desc = f"{content_desc}, {elem_text}"
            else:
                node_desc = content_desc

    if node_desc is None and node.get('resource_id', None) is not None:
        raw_node_text = node['resource_id'].split('/')[-1].strip()
        if raw_node_text:
            node_desc = raw_node_text

    return node_desc


def _process_node_for_textloc(
    elem_text: str,
    all_node_texts: List[str],
    box_str: str,
    center_str: str,
    unnormalized_box: List[int],
    short_img_path: str,
    package: str,
    width: int,
    height: int,
    with_box: bool,
    samples: List[Dict],
    counter: int
) -> int:
    """Generate text localization sample."""
    # To avoid ambiguity, a text localization task is created if the elem text is unique.
    if TEXTLOC and all_node_texts.count(elem_text) <= 1:
        task_id = f'autogui_{DATASET_NAME}_textloc_{counter}'
        sample = make_textloc_sample(
            task_id,
            text=elem_text,
            loc=box_str if with_box else center_str,
            output_tag=''
        )
        sample.update({
            'task_attr': elem_text,
            'unnormalized_box': unnormalized_box,
            'image': short_img_path,
            'wxh': f"{width}x{height}",
            'package': package
        })
        samples.append(sample)
        return counter + 1
    return counter


def _process_node_for_ocr(
    elem_text: str,
    box_str: str,
    center_str: str,
    unnormalized_box: List[int],
    short_img_path: str,
    package: str,
    width: int,
    height: int,
    with_box: bool,
    samples: List[Dict],
    counter: int
) -> int:
    """Generate OCR sample."""
    if OCR:
        task_id = f'autogui_{DATASET_NAME}_ocr_{counter}'
        loc = box_str if with_box else center_str
        sample = make_ocr_sample(task_id, text=elem_text, loc=loc, with_box=with_box)
        sample.update({
            'task_attr': loc,
            'unnormalized_box': unnormalized_box,
            'image': short_img_path,
            'wxh': f"{width}x{height}",
            'package': package
        })
        samples.append(sample)
        return counter + 1
    return counter


def _process_node_for_icon_tasks(
    node_content_desc: str,
    all_node_content_descs: List[str],
    box_str: str,
    center_str: str,
    unnormalized_box: List[int],
    short_img_path: str,
    package: str,
    width: int,
    height: int,
    samples: List[Dict],
    icongnd_cnt: int,
    iconref_cnt: int
) -> Tuple[int, int]:
    """Generate Icon Grounding and Icon Referencing samples."""
    # Grounding tasks require unique targets, so check if the current element's
    # node_content_desc appears repeatedly. If so, skip it.
    if ICONGND and all_node_content_descs.count(node_content_desc) <= 1:
        task_id = f'autogui_{DATASET_NAME}_icongnd_{icongnd_cnt}'
        with_box = random.random() < PROB_BOX
        sample = make_icongnd_sample(
            task_id,
            icon_desc=node_content_desc,
            loc=box_str if with_box else center_str,
            output_tag=''
        )
        sample.update({
            'task_attr': node_content_desc,
            'unnormalized_box': unnormalized_box,
            'image': short_img_path,
            'wxh': f"{width}x{height}",
            'package': package
        })
        samples.append(sample)
        icongnd_cnt += 1

        if ICONREF:
            task_id = f'autogui_{DATASET_NAME}_iconref_{iconref_cnt}'
            with_box = random.random() < PROB_BOX
            loc = box_str if with_box else center_str
            sample = make_iconref_sample(
                task_id,
                icon_desc=node_content_desc,
                loc=loc,
                with_box=with_box
            )
            sample.update({
                'task_attr': loc,
                'unnormalized_box': unnormalized_box,
                'image': short_img_path,
                'wxh': f"{width}x{height}",
                'package': package
            })
            samples.append(sample)
            iconref_cnt += 1

    return icongnd_cnt, iconref_cnt


def _process_node_for_intentgnd(
    node: Dict,
    node_desc: str,
    elem_text: str,
    all_node_descs: List[str],
    all_node_texts: List[str],
    box_str: str,
    center_str: str,
    normalized_center: List[int],
    unnormalized_box: List[int],
    short_img_path: str,
    package: str,
    width: int,
    height: int,
    samples: List[Dict],
    counter: int
) -> int:
    """Generate Intent Grounding sample."""
    if INTENTGND and node_desc and all_node_descs.count(node_desc) <= 1 and 0 < len(node_desc) <= 200:
        # Although the node might provide an HTML tag, we use the Android tag type here
        tag = node['tag']
        with_box = random.random() < PROB_BOX
        task_id = f'autogui_{DATASET_NAME}_intentgnd_{counter}'

        if elem_text not in [None, 'None', 'none'] and all_node_texts.count(elem_text) <= 1:
            intent = gen_naive_action_gnd_anno(
                node_desc.strip(' ,.'), tag, normalized_center, scale=SCALE
            )
            sample = make_intentgnd_sample(
                task_id,
                intent=intent,
                loc=box_str if with_box else center_str,
                output_tag=''
            )
            sample.update({
                'task_attr': intent,
                'unnormalized_box': unnormalized_box,
                'image': short_img_path,
                'wxh': f"{width}x{height}",
                'package': package
            })
            samples.append(sample)
            return counter + 1
    return counter


def _save_chunk_data(
    chunk_idx: int,
    chunk_samples: List[Dict],
    invalid_elem_tracker: Dict[str, Set],
    counts: Dict[str, int],
    chunk_traj_names: List[str],
    completely_proc_traj_names: List[str],
    ckpt_file: str
) -> None:
    """Save the current chunk of processed data."""
    chunk_file = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx}_s{SCALE}{'_debug' if DEBUG else ''}.json")
    
    print(f"Saving chunk {chunk_idx} with {len(chunk_samples)} samples to {chunk_file}")
    
    with open(chunk_file, 'w') as f:
        json.dump(chunk_samples, f)

    # Save invalid element records for this chunk
    with open(os.path.join(CHUNK_DIR, f'chunk_{chunk_idx}_invalid_elems.json'), 'w') as f:
        json.dump({k: list(v) for k, v in invalid_elem_tracker.items()}, f, indent=2)

    # Update checkpoint
    completely_proc_traj_names.extend(chunk_traj_names)
    with open(ckpt_file, 'w') as f:
        json.dump({'completely_proc_traj_names': completely_proc_traj_names}, f)

    # Generate and save report
    num_invalid_elem = sum(len(v) for v in invalid_elem_tracker.values())
    processed_img_cnt = counts['processed_img_cnt']
    valid_img_cnt = counts['valid_img_cnt']
    iterated_elem_cnt = counts['iterated_elem_cnt']

    valid_ratio = valid_img_cnt / processed_img_cnt if processed_img_cnt > 0 else 0.0
    invalid_ratio = num_invalid_elem / iterated_elem_cnt if iterated_elem_cnt > 0 else 0.0

    report = (
        f"Valid image ratio: {valid_img_cnt} / {processed_img_cnt} = {valid_ratio:.2f}\n"
        f"Invalid elem ratio: {num_invalid_elem} / {iterated_elem_cnt} = {invalid_ratio:.2f}\n"
        f"text_loc: {counts['text_loc']} | ocr: {counts['ocr']} | "
        f"icongnd: {counts['icongnd']} | iconref: {counts['iconref']} | "
        f"elemclass: {counts['elemclass']} | intentgnd: {counts['intentgnd']} | "
        f"widgetlist: {counts['widgetlist']}"
    )
    print(report)

    # Save info file
    info_data = {
        'num_samples': len(chunk_samples),
        'valid_img_cnt': valid_img_cnt,
        'processed_img_cnt': processed_img_cnt,
        'num_invalid_elements': num_invalid_elem,
        'iterated_elem_cnt': iterated_elem_cnt,
        'text_loc_cnt': counts['text_loc'],
        'ocr_cnt': counts['ocr'],
        'icongnd_cnt': counts['icongnd'],
        'iconref_cnt': counts['iconref'],
        'elemclass_cnt': counts['elemclass'],
        'intentgnd_cnt': counts['intentgnd'],
        'widgetlist_cnt': counts['widgetlist'],
        'report': report,
        'invalid_elem_types': {k: len(v) for k, v in invalid_elem_tracker.items()}
    }
    with open(chunk_file.replace('.json', '_info.json'), "w") as f:
        json.dump(info_data, f, indent=2)


def make_wae_data() -> None:
    """Main function to generate WAE training data."""
    # Setup directories
    os.makedirs(SAVE_IMAGES_TO, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)

    # Load checkpoint
    ckpt_file = os.path.join(SAVE_ROOT, f"ckpt_s{SCALE}{'_debug' if DEBUG else ''}.json")
    if os.path.exists(ckpt_file):
        with open(ckpt_file, 'r') as f:
            ckpt = json.load(f)
            completely_proc_traj_names = ckpt['completely_proc_traj_names']
    else:
        completely_proc_traj_names = []

    # Get trajectories
    all_traj_names = sorted(os.listdir(WAE_DIR))
    if DEBUG:
        all_traj_names = all_traj_names[:50]

    # Initialize counters and trackers
    counts = defaultdict(int)
    last_sample_cnt = 0
    chunk_samples = []
    chunk_traj_names = []
    unique_elems = defaultdict(list)

    # Initialize invalid element tracker
    invalid_elem_meta = {
        TOO_SMALL_ELEMENT: set(), INVALID_ELEM_BOX: set(), INVALID_ELEM_CONTENT: set(),
        BLANK_ELEM: set(), EMPTY_ELEM_TEXT: set(), OVERLY_LENGTHY_ELEM_TEXT: set(),
        DUPLICATE_ELEMEMNT: set(), OVERSIZED_ELEMENT: set(), EXTREME_ASPECT_RATIO: set(),
        GHOST_ELEMENT: set()
    }
    
    # Load previous invalid element records if available
    # Note: This logic is slightly simplified from original to assume start fresh or from last chunk
    invalid_elem_tracker = deepcopy(invalid_elem_meta)

    # Processing Loop
    for traj_idx, traj_name in tqdm(enumerate(all_traj_names), total=len(all_traj_names), desc="Processing trajectories"):
        if traj_name in completely_proc_traj_names:
            continue
        
        chunk_traj_names.append(traj_name)
        screenshot_files = glob.glob(os.path.join(WAE_DIR, traj_name, '*', 'ui', '*.png'))

        if DEBUG:
            screenshot_files = screenshot_files[:70]

        for screenshot_file in screenshot_files:
            screenshot_file = os.path.join(WAE_DIR, screenshot_file)
            if not os.path.exists(screenshot_file):
                continue

            vh_file = screenshot_file.replace(".png", ".xml")
            if not os.path.exists(vh_file):
                continue

            counts['processed_img_cnt'] += 1

            # Parse XML
            with open(vh_file, 'r') as f:
                xml_content = f.read()
                dom_tree = ET.fromstring(xml_content)
            
            nodes = find_all_elem_texts_boxes(dom_tree)

            # Detect illegal language types (except English and Chinese)
            contain_invalid_characters = False
            for node_idx, node in enumerate(nodes):
                text = node['text']
                node['idx'] = node_idx
                if text is None: continue
                if not is_valid_string(text):
                    contain_invalid_characters = True
                    break
            
            num_nodes = len([x for x in nodes if x['is_leaf']])
            if contain_invalid_characters:
                counts['iterated_elem_cnt'] += num_nodes
                continue

            # Get image dimensions
            # Since the maximum UI resolution in this dataset is 1280, no resizing is performed.
            hw_info = re.search(r'(\d+) x (\d+)', magic.from_file(screenshot_file))
            if hw_info is None:
                counts['iterated_elem_cnt'] += num_nodes
                continue
            W, H = list(map(int, hw_info.groups()))
            
            img = None
            broken_img = False
            all_valid_nodes = []

            # First pass validation
            for node in nodes:
                if not node['is_leaf'] or node.get("box", None) is None:
                    continue
                
                counts['iterated_elem_cnt'] += 1
                
                # Construct unique identifier for invalid checking
                sample_identifier = f"{screenshot_file}|{node['idx']}|{node['box']}"
                
                # Check if already marked invalid in previous runs
                is_invalid = False
                for v in invalid_elem_tracker.values():
                    if sample_identifier in v:
                        is_invalid = True
                        break
                if is_invalid:
                    continue

                node["box"] = list(node["box"]) # Ensure list
                if node["box"] not in unique_elems[screenshot_file]:
                    unique_elems[screenshot_file].append(node["box"])

                if not _validate_node_bounds(node, W, H, all_valid_nodes, sample_identifier, invalid_elem_tracker):
                    continue

                # Detect successfully displayed nodes (pure color check)
                if img is None:
                    img = cv2.imread(screenshot_file)
                if img is None:
                    broken_img = True
                    break
                
                x1, y1, x2, y2 = node["box"]
                if is_pure_color(img, [x1, y1, x2, y2]):
                    invalid_elem_tracker[BLANK_ELEM].add(sample_identifier)
                    continue

                all_valid_nodes.append(node)

            if broken_img:
                continue

            # Second pass: Overlap checking with OCR
            # Compare OCR results with element text attributes to remove invalid elements
            # where the box covers other elements.
            if len(all_valid_nodes) > 1:
                all_valid_nodes_after_checking_overlap = []
                for node_idx, node in enumerate(all_valid_nodes):
                    x1, y1, x2, y2 = node["box"]
                    sample_identifier = f"{screenshot_file}|{node['idx']}|{node['box']}"
                    
                    # Re-check invalidity (defensive)
                    is_invalid = False
                    for v in invalid_elem_tracker.values():
                        if sample_identifier in v:
                            is_invalid = True
                            break
                    if is_invalid:
                        continue

                    if not SKIP_CHECKING:
                        other_boxes = [x['box'] for cur_idx, x in enumerate(all_valid_nodes) if cur_idx != node_idx]
                        is_overlap = is_box_overlapping_np(target_box=[x1, y1, x2, y2], other_boxes=other_boxes, threshold=0.01)

                        if is_overlap:
                            elem_type = classify_node(node)
                            if elem_type in ['Icon', 'Text']:
                                elem_text = node.get('text', None)
                                if elem_text is not None:
                                    elem_text = elem_text.strip()
                                    lower_node_text = elem_text.lower()
                                    elem_text_is_description = any(k in lower_node_text for k in ['icon', 'button', 'back'])
                                    
                                    if not elem_text_is_description:
                                        roi = img[y1:y2, x1:x2]
                                        ocr_result = pytesseract.image_to_string(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)).strip()
                                        similarity_ratio = fuzz.ratio(ocr_result, elem_text)
                                        node['ocr'] = [ocr_result, similarity_ratio]
                                        
                                        if similarity_ratio < 22:
                                            invalid_elem_tracker[GHOST_ELEMENT].add(sample_identifier)
                                            continue
                    
                    all_valid_nodes_after_checking_overlap.append(node)
                all_valid_nodes = all_valid_nodes_after_checking_overlap

            if not all_valid_nodes:
                continue

            # Prepare data for task generation
            all_node_texts = [node['text'] for node in all_valid_nodes if 'text' in node]
            
            # Extract package name
            # example: .../com.twansoftware.pdfconverterpro_6000006-output/stoat_fsm_output/ui/S_1.png
            traj_name_part = screenshot_file.split('/')[-4]
            package = traj_name_part.split('_')[0]
            
            short_img_path = f'{DATASET_NAME}/' + screenshot_file.split(f'{DATASET_NAME}/tmp/')[1]
            
            # Process nodes for individual tasks
            used_boxes = []
            
            for node in all_valid_nodes:
                sample_identifier = f"{screenshot_file}|{node['idx']}|{node['box']}"
                
                # Final invalid check
                is_invalid = False
                for v in invalid_elem_tracker.values():
                    if sample_identifier in v:
                        is_invalid = True
                        break
                if is_invalid:
                    continue

                # Skip overlapping boxes
                if not SKIP_CHECKING and node["box"] in used_boxes:
                    invalid_elem_tracker[DUPLICATE_ELEMEMNT].add(sample_identifier) # Note: original script counted integer here, but set expected
                    continue
                else:
                    used_boxes.append(node["box"])

                # Prepare node attributes
                elem_text = node['text']
                node['node_desc'] = _get_node_description(node)
                
                x1, y1, x2, y2 = node["box"]
                box, box_str = _normalize_box(x1, y1, x2, y2, W, H)
                normalized_center, center_str = _normalize_center(x1, y1, x2, y2, W, H)
                unnormalized_box = [x1, y1, x2, y2]
                box_w, box_h = x2 - x1, y2 - y1
                center = [(x1 + x2) // 2, (y1 + y2) // 2]

                # Text Localization & OCR
                if elem_text is not None and 0 < len(elem_text) <= 200:
                    elem_text = elem_text.strip()
                    with_box = random.random() < PROB_BOX
                    
                    # If the element text does not populate the whole box, use the box as reference
                    if img is None: img = cv2.imread(screenshot_file)
                    center_roi = [
                        max(0, center[0] - box_w // 10),
                        max(0, center[1] - box_h // 10),
                        center[0] + box_w // 10,
                        center[1] + box_h // 10
                    ]
                    if is_pure_color(img, center_roi):
                        with_box = True

                    counts['text_loc'] = _process_node_for_textloc(
                        elem_text, all_node_texts, box_str, center_str, unnormalized_box,
                        short_img_path, package, W, H, with_box, chunk_samples, counts['text_loc']
                    )

                    counts['ocr'] = _process_node_for_ocr(
                        elem_text, box_str, center_str, unnormalized_box,
                        short_img_path, package, W, H, with_box, chunk_samples, counts['ocr']
                    )

                # Icon Grounding & Reference
                node_content_desc = node.get('content-desc', None)
                all_node_content_descs = [n.get('content-desc') for n in all_valid_nodes if n.get('content-desc')]
                
                if (elem_text is None or len(elem_text) == 0) and node_content_desc and 0 < len(node_content_desc) <= 200:
                    counts['icongnd'], counts['iconref'] = _process_node_for_icon_tasks(
                        node_content_desc, all_node_content_descs, box_str, center_str,
                        unnormalized_box, short_img_path, package, W, H, chunk_samples,
                        counts['icongnd'], counts['iconref']
                    )

                # Intent Grounding
                node_desc = node.get('node_desc', None)
                all_node_descs = [n.get('node_desc') for n in all_valid_nodes if n.get('node_desc')]
                
                counts['intentgnd'] = _process_node_for_intentgnd(
                    node, node_desc, elem_text, all_node_descs, all_node_texts,
                    box_str, center_str, normalized_center, unnormalized_box,
                    short_img_path, package, W, H, chunk_samples, counts['intentgnd']
                )

            # Widget List
            if WIDGETLIST and len(all_valid_nodes) >= 2:
                task_id = f'autogui_{DATASET_NAME}_widgetlist_{counts["widgetlist"]}'
                node_texts_boxes = []
                for node in all_valid_nodes:
                    desc = node['node_desc'] if node['node_desc'] else ''
                    u_box = node['box']
                    n_box, _ = _normalize_box(u_box[0], u_box[1], u_box[2], u_box[3], W, H)
                    
                    clean_desc = ' '.join(x for x in desc.strip(' ,.').split('\n') if x.strip())
                    node_texts_boxes.append((node['tag'], clean_desc, u_box, n_box))
                
                # Sort by position: top-to-bottom, left-to-right
                node_texts_boxes.sort(key=lambda x: (x[2][1] + x[2][3], x[2][0] + x[2][2]))
                
                elem_list_str = '\n'.join(
                    f"{i} {nodeclass} '{nodetext}' ({nb[0]},{nb[1]},{nb[2]},{nb[3]})"
                    for i, (nodeclass, nodetext, _, nb) in enumerate(node_texts_boxes)
                )
                
                sample = make_widgetlist_sample(task_id, elem_list=elem_list_str)
                sample.update({
                    'task_attr': None,
                    'image': short_img_path,
                    'wxh': f"{W}x{H}",
                    'package': package,
                    'unnormalized_box': [x[2] for x in node_texts_boxes]
                })
                chunk_samples.append(sample)
                counts['widgetlist'] += 1

            # Update image validity stats
            if DEBUG:
                for i in range(last_sample_cnt, len(chunk_samples)):
                    chunk_samples[i]['original_img_file'] = screenshot_file

            if len(chunk_samples) > last_sample_cnt:
                counts['valid_img_cnt'] += 1
            last_sample_cnt = len(chunk_samples)

        # Chunk Saving Logic
        if traj_idx > 0 and (traj_idx % CHUNK_SIZE == 0 or traj_idx == len(all_traj_names) - 1):
            chunk_idx = traj_idx // CHUNK_SIZE
            _save_chunk_data(
                chunk_idx, chunk_samples, invalid_elem_tracker, counts,
                chunk_traj_names, completely_proc_traj_names, ckpt_file
            )
            
            # Reset for next chunk
            # Note: The original script reset specific counters here but accumulated total invalid element counts
            # via file loading/saving. Here we keep the counters running in `counts` but `chunk_samples` clears.
            # The original script's behavior of resetting *all* counters (processed_img_cnt etc) every chunk 
            # suggests the report is per-chunk.
            
            counts['processed_img_cnt'] = 0
            counts['valid_img_cnt'] = 0
            counts['iterated_elem_cnt'] = 0
            # Reset task counters as well to match original logic?
            # Original script resets them: text_loc_cnt = ... = 0
            # But it passes them into task generation. This implies task IDs might overlap between chunks 
            # if we reset to 0? 
            # Looking at original: `task_id = f'autogui_{DATASET_NAME}_textloc_{text_loc_cnt}'`
            # If we reset `text_loc_cnt` to 0, we get duplicate IDs across chunks.
            # However, the original script *does* reset them (line 492). 
            # This implies task IDs are unique only within a chunk or the user doesn't care about global uniqueness.
            # I will follow the original logic and reset them.
            for key in ['text_loc', 'ocr', 'icongnd', 'iconref', 'elemclass', 'intentgnd', 'widgetlist']:
                counts[key] = 0
            
            last_sample_cnt = 0
            chunk_samples.clear()
            chunk_traj_names.clear()

            # Handle invalid elem tracker persistence for next chunk
            # The original script loads the *next* chunk's invalid elem file if it exists, else deepcopies meta.
            # This implies a pre-computation or multi-pass approach where we might resume.
            next_invalid_file = os.path.join(CHUNK_DIR, f'chunk_{chunk_idx+1}_invalid_elems.json')
            if os.path.exists(next_invalid_file):
                with open(next_invalid_file, 'r') as f:
                    loaded = json.load(f)
                invalid_elem_tracker = {k: set(v) for k, v in loaded.items()}
            else:
                invalid_elem_tracker = deepcopy(invalid_elem_meta)

    # Final Aggregation
    _aggregate_and_save_final_results(unique_elems)


def _aggregate_and_save_final_results(unique_elems: Dict[str, List]) -> None:
    """Aggregate all chunk results and save the final dataset."""
    all_samples = []
    all_chunks_num_invalid_elem = defaultdict(int)
    
    # Aggregated counters
    agg_counts = defaultdict(int)
    
    chunk_info_files = glob.glob(os.path.join(CHUNK_DIR, f'*_s{SCALE}{"_debug" if DEBUG else ""}_info.json'))
    
    for chunk_info_file in chunk_info_files:
        chunk_file = chunk_info_file.replace("_info", "")
        with open(chunk_file, 'r') as f:
            samples = json.load(f)
            all_samples.extend(samples)

        with open(chunk_info_file, 'r') as f:
            info = json.load(f)
            agg_counts['processed_img_cnt'] += info['processed_img_cnt']
            agg_counts['valid_img_cnt'] += info['valid_img_cnt']
            agg_counts['iterated_elem_cnt'] += info['iterated_elem_cnt']
            agg_counts['text_loc'] += info['text_loc_cnt']
            agg_counts['ocr'] += info['ocr_cnt']
            agg_counts['icongnd'] += info['icongnd_cnt']
            agg_counts['iconref'] += info['iconref_cnt']
            agg_counts['elemclass'] += info['elemclass_cnt']
            agg_counts['intentgnd'] += info['intentgnd_cnt']
            agg_counts['widgetlist'] += info['widgetlist_cnt']
            
            for k, v in info['invalid_elem_types'].items():
                all_chunks_num_invalid_elem[k] += v

    all_invalid_elem = sum(all_chunks_num_invalid_elem.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k, v in unique_elems.items() if len(v)])

    report = {
        'num_samples': len(all_samples),
        '#valid_unique_images': num_valid_imgs,
        '#all_unique_images': len(unique_elems),
        'Valid image ratio': f"{agg_counts['valid_img_cnt']} / {agg_counts['processed_img_cnt']} = "
                             f"{agg_counts['valid_img_cnt']/agg_counts['processed_img_cnt']:.2f}",
        'processed_img_cnt': agg_counts['processed_img_cnt'],
        '#num_unique_elems': num_unique_elems,
        'iterated_elem_cnt': agg_counts['iterated_elem_cnt'],
        'num_invalid_elements': all_invalid_elem,
        'Invalid elem ratio': f"{all_invalid_elem} / {agg_counts['iterated_elem_cnt']} = "
                              f"{all_invalid_elem/agg_counts['iterated_elem_cnt']:.2f}",
        'text_loc_cnt': agg_counts['text_loc'],
        'ocr_cnt': agg_counts['ocr'],
        'icongnd_cnt': agg_counts['icongnd'],
        'iconref_cnt': agg_counts['iconref'],
        'elemclass_cnt': agg_counts['elemclass'],
        'intentgnd_cnt': agg_counts['intentgnd'],
        'widgetlist_cnt': agg_counts['widgetlist'],
        'invalid_elem_types': dict(all_chunks_num_invalid_elem)
    }

    print(json.dumps(report, indent=2))

    os.makedirs(SAVE_ROOT, exist_ok=True)
    file_name = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_scale{SCALE}_{len(all_samples)// 1000}k{'_debug' if DEBUG else ''}.json")
    print(f"Saving {len(all_samples)} samples to {file_name}")

    with open(file_name.replace('.json', '_info.json'), "w") as f:
        json.dump(report, f, indent=2)

    with open(file_name.replace(".json", "_sample.json"), 'w') as f:
        json.dump(random.sample(all_samples, min(len(all_samples), 128)), f, indent=2)
    
    with open(file_name, 'w') as f:
        json.dump(all_samples, f)


if __name__ == '__main__':
    make_wae_data()
