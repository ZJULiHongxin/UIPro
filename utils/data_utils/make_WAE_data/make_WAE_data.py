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
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Set, Tuple, Optional

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
from utils.data_utils.run_info import (
    TASK_DISPLAY_NAMES,
    print_error_summary,
    renumber_sample_ids,
    save_run_info,
    summarize_errors,
)


def _wae_canonical_counters(counts: Dict[str, int]) -> Dict[str, int]:
    """Translate WAE's short counter keys (e.g. 'text_loc', 'icongnd') to the
    canonical ``<task>_cnt`` form used by ``run_info.save_run_info``."""
    return {
        'text_loc_cnt': counts.get('text_loc', 0),
        'ocr_cnt': counts.get('ocr', 0),
        'icongnd_cnt': counts.get('icongnd', 0),
        'iconref_cnt': counts.get('iconref', 0),
        'elemclass_cnt': counts.get('elemclass', 0),
        'intentgnd_cnt': counts.get('intentgnd', 0),
        'widgetlist_cnt': counts.get('widgetlist', 0),
    }

# Configuration constants
DATASET_NAME: str = 'WAE'
DEBUG: bool = False

# Task flags - enable/disable specific task generation
TEXTLOC: bool = True
OCR: bool = True
ICONGND: bool = True
ICONREF: bool = True
ELEMCLASS: bool = False
INTENTGND: bool = True
WIDGETLIST: bool = True

# Generation parameters
PROB_BOX: float = 0.3
SCALE: int = 1000
SKIP_CHECKING: bool = False
CHUNK_SIZE: int = 25 if not DEBUG else 1

# --- Parallelism ---
# Cap derived from the same lesson as GUIEnv: the overlap-check path runs
# ``pytesseract.image_to_string`` per ambiguous element, which shells out
# to the tesseract binary. Past ~32 concurrent workers the subprocess /
# tempfile pressure becomes the bottleneck and throughput collapses. Set
# NUM_WORKERS=1 to force the sequential code path.
NUM_WORKERS: int = min(32, max(1, (os.cpu_count() or 1) - 1))

# Populated by ``_init_worker`` in every Pool child. Holds the flat
# frozenset of all known-invalid sample identifiers so per-element skip
# checks stay O(1) instead of scanning every registry bucket per call.
_WAE_INVALID_IDS_CACHE: frozenset = frozenset()


def _init_wae_worker(invalid_ids: frozenset) -> None:
    """Pool initialiser: install the cumulative skip-list in every worker.

    A single frozenset broadcast at pool creation is far cheaper than
    re-pickling the (potentially huge) set as part of every task arg.
    """
    global _WAE_INVALID_IDS_CACHE
    _WAE_INVALID_IDS_CACHE = invalid_ids


# Bucket names this worker may write to. Mirrors the meta init in
# ``make_wae_data`` so a worker's local tracker has the right shape.
_WAE_INVALID_BUCKETS: Tuple[str, ...] = (
    TOO_SMALL_ELEMENT,
    INVALID_ELEM_BOX,
    INVALID_ELEM_CONTENT,
    BLANK_ELEM,
    EMPTY_ELEM_TEXT,
    OVERLY_LENGTHY_ELEM_TEXT,
    DUPLICATE_ELEMEMNT,
    OVERSIZED_ELEMENT,
    EXTREME_ASPECT_RATIO,
    GHOST_ELEMENT,
)

# Paths
WAE_DIR: str = f"/volume/pt-coder/users/gji/projects/UIPro/data/{DATASET_NAME}"
ROOT_DIR: str = "/volume/pt-coder/users/gji/projects/UIPro/data"
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
            output_tag=WITHBOX_TAG if with_box else WITHPOINT_TAG
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
        sample = make_ocr_sample(task_id, text=elem_text, loc=loc, output_tag=WITHBOX_TAG if with_box else WITHPOINT_TAG)
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
            output_tag=WITHPOINT_TAG
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
                output_tag=WITHBOX_TAG if with_box else WITHPOINT_TAG 
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
                output_tag=WITHBOX_TAG if with_box else WITHPOINT_TAG
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


def _wae_process_trajectory(traj_name: str) -> Dict[str, Any]:
    """Worker: process every screenshot in one WAE trajectory.

    The per-screenshot validation/sample-generation logic is preserved
    verbatim from the original sequential loop — including the quirky
    pytesseract overlap check, the BLANK_ELEM / GHOST_ELEMENT
    distinctions, and the per-traj ``img`` lazy-load — so the output is
    bit-for-bit equivalent to the sequential run except for two changes:

      1. Sample task IDs use worker-local counters (collisions across
         trajectories are renumbered globally by ``renumber_sample_ids``
         after aggregation).
      2. The cumulative skip-list lookup is O(1) against the worker-global
         frozenset ``_WAE_INVALID_IDS_CACHE``, not the original O(N) scan
         across every registry bucket.

    Returned dict keys (all picklable):
      - ``traj_name``: echo of the input for ordered checkpoint updates.
      - ``samples``: list of generated sample dicts.
      - ``counts``: per-trajectory counter increments (``processed_img_cnt``,
        ``valid_img_cnt``, ``iterated_elem_cnt``, plus one per task type).
      - ``invalid_updates``: ``{bucket: [identifier, ...]}`` to union into
        the cumulative invalid-elem record.
      - ``unique_elem_entries``: ``{screenshot_path: [unnorm_box, ...]}``
        for cross-trajectory dedup in the parent.
      - ``errors``: ``[[traj_name, traceback_str], ...]`` if anything
        threw — caught locally so a poisonous trajectory never hangs the
        pool.
    """
    samples: List[Dict] = []
    counts: Dict[str, int] = defaultdict(int)
    invalid_updates: Dict[str, Set[str]] = {b: set() for b in _WAE_INVALID_BUCKETS}
    unique_elem_entries: Dict[str, List[List[int]]] = defaultdict(list)
    errors: List[List[str]] = []
    last_sample_cnt = 0

    try:
        screenshot_files = glob.glob(
            os.path.join(WAE_DIR, traj_name, '*', 'ui', '*.png')
        )
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
                if text is None:
                    continue
                if not is_valid_string(text):
                    contain_invalid_characters = True
                    break

            num_nodes = len([x for x in nodes if x['is_leaf']])
            if contain_invalid_characters:
                counts['iterated_elem_cnt'] += num_nodes
                continue

            # Image dimensions via libmagic (cheap header read, no decode).
            hw_info = re.search(r'(\d+) x (\d+)', magic.from_file(screenshot_file))
            if hw_info is None:
                counts['iterated_elem_cnt'] += num_nodes
                continue
            W, H = list(map(int, hw_info.groups()))

            img = None
            broken_img = False
            all_valid_nodes: List[Dict] = []

            # First pass validation
            for node in nodes:
                if not node['is_leaf'] or node.get("box", None) is None:
                    continue

                counts['iterated_elem_cnt'] += 1
                sample_identifier = f"{screenshot_file}|{node['idx']}|{node['box']}"

                # O(1) cumulative skip-list check (was an O(N) scan
                # across every invalid-elem bucket in the sequential code).
                if sample_identifier in _WAE_INVALID_IDS_CACHE:
                    continue

                node["box"] = list(node["box"])
                if node["box"] not in unique_elem_entries[screenshot_file]:
                    unique_elem_entries[screenshot_file].append(node["box"])

                if not _validate_node_bounds(
                    node, W, H, all_valid_nodes, sample_identifier,
                    invalid_updates,
                ):
                    continue

                # Detect successfully displayed nodes (pure colour check).
                if img is None:
                    img = cv2.imread(screenshot_file)
                if img is None:
                    broken_img = True
                    break

                x1, y1, x2, y2 = node["box"]
                if is_pure_color(img, [x1, y1, x2, y2]):
                    invalid_updates[BLANK_ELEM].add(sample_identifier)
                    continue

                all_valid_nodes.append(node)

            if broken_img:
                continue

            # Second pass: overlap check using OCR. This is the heavy
            # pytesseract call — same workload that capped GUIEnv at 32
            # workers. We honour the original behaviour exactly.
            if len(all_valid_nodes) > 1:
                all_valid_nodes_after_checking_overlap = []
                for node_idx, node in enumerate(all_valid_nodes):
                    x1, y1, x2, y2 = node["box"]
                    sample_identifier = f"{screenshot_file}|{node['idx']}|{node['box']}"

                    # Defensive re-check (preserve original behaviour even
                    # though new local additions are append-only).
                    if sample_identifier in _WAE_INVALID_IDS_CACHE:
                        continue
                    is_invalid = any(
                        sample_identifier in v for v in invalid_updates.values()
                    )
                    if is_invalid:
                        continue

                    if not SKIP_CHECKING:
                        other_boxes = [
                            x['box'] for cur_idx, x in enumerate(all_valid_nodes)
                            if cur_idx != node_idx
                        ]
                        is_overlap = is_box_overlapping_np(
                            target_box=[x1, y1, x2, y2],
                            other_boxes=other_boxes, threshold=0.01,
                        )

                        if is_overlap:
                            elem_type = classify_node(node)
                            if elem_type in ['Icon', 'Text']:
                                elem_text = node.get('text', None)
                                if elem_text is not None:
                                    elem_text = elem_text.strip()
                                    lower_node_text = elem_text.lower()
                                    elem_text_is_description = any(
                                        k in lower_node_text
                                        for k in ['icon', 'button', 'back']
                                    )
                                    if not elem_text_is_description:
                                        roi = img[y1:y2, x1:x2]
                                        ocr_result = pytesseract.image_to_string(
                                            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                        ).strip()
                                        similarity_ratio = fuzz.ratio(
                                            ocr_result, elem_text,
                                        )
                                        node['ocr'] = [ocr_result, similarity_ratio]
                                        if similarity_ratio < 22:
                                            invalid_updates[GHOST_ELEMENT].add(
                                                sample_identifier,
                                            )
                                            continue

                    all_valid_nodes_after_checking_overlap.append(node)
                all_valid_nodes = all_valid_nodes_after_checking_overlap

            if not all_valid_nodes:
                continue

            all_node_texts = [
                node['text'] for node in all_valid_nodes if 'text' in node
            ]
            traj_name_part = screenshot_file.split('/')[-4]
            package = traj_name_part.split('_')[0]
            short_img_path = (
                f'{DATASET_NAME}/' + screenshot_file.split(f'{DATASET_NAME}/')[1]
            )

            used_boxes: List[List[int]] = []
            for node in all_valid_nodes:
                sample_identifier = f"{screenshot_file}|{node['idx']}|{node['box']}"

                # Final invalid-check (preserves the original defensive triple-check).
                if sample_identifier in _WAE_INVALID_IDS_CACHE:
                    continue
                if any(sample_identifier in v for v in invalid_updates.values()):
                    continue

                if not SKIP_CHECKING and node["box"] in used_boxes:
                    invalid_updates[DUPLICATE_ELEMEMNT].add(sample_identifier)
                    continue
                else:
                    used_boxes.append(node["box"])

                elem_text = node['text']
                node['node_desc'] = _get_node_description(node)

                x1, y1, x2, y2 = node["box"]
                box, box_str = _normalize_box(x1, y1, x2, y2, W, H)
                normalized_center, center_str = _normalize_center(
                    x1, y1, x2, y2, W, H,
                )
                unnormalized_box = [x1, y1, x2, y2]
                box_w, box_h = x2 - x1, y2 - y1
                center = [(x1 + x2) // 2, (y1 + y2) // 2]

                # Text Localization & OCR
                if elem_text is not None and 0 < len(elem_text) <= 200:
                    elem_text = elem_text.strip()
                    with_box = random.random() < PROB_BOX

                    if img is None:
                        img = cv2.imread(screenshot_file)
                    center_roi = [
                        max(0, center[0] - box_w // 10),
                        max(0, center[1] - box_h // 10),
                        center[0] + box_w // 10,
                        center[1] + box_h // 10,
                    ]
                    if is_pure_color(img, center_roi):
                        with_box = True

                    counts['text_loc'] = _process_node_for_textloc(
                        elem_text, all_node_texts, box_str, center_str,
                        unnormalized_box, short_img_path, package, W, H,
                        with_box, samples, counts['text_loc'],
                    )
                    counts['ocr'] = _process_node_for_ocr(
                        elem_text, box_str, center_str, unnormalized_box,
                        short_img_path, package, W, H, with_box,
                        samples, counts['ocr'],
                    )

                # Icon Grounding & Reference
                node_content_desc = node.get('content-desc', None)
                all_node_content_descs = [
                    n.get('content-desc') for n in all_valid_nodes
                    if n.get('content-desc')
                ]

                if (
                    (elem_text is None or len(elem_text) == 0)
                    and node_content_desc
                    and 0 < len(node_content_desc) <= 200
                ):
                    counts['icongnd'], counts['iconref'] = _process_node_for_icon_tasks(
                        node_content_desc, all_node_content_descs,
                        box_str, center_str, unnormalized_box,
                        short_img_path, package, W, H,
                        samples, counts['icongnd'], counts['iconref'],
                    )

                # Intent Grounding
                node_desc = node.get('node_desc', None)
                all_node_descs = [
                    n.get('node_desc') for n in all_valid_nodes if n.get('node_desc')
                ]
                counts['intentgnd'] = _process_node_for_intentgnd(
                    node, node_desc, elem_text, all_node_descs, all_node_texts,
                    box_str, center_str, normalized_center, unnormalized_box,
                    short_img_path, package, W, H, samples, counts['intentgnd'],
                )

            # Widget list (per-screen, one sample if ≥2 valid nodes).
            if WIDGETLIST and len(all_valid_nodes) >= 2:
                task_id = f'autogui_{DATASET_NAME}_widgetlist_{counts["widgetlist"]}'
                node_texts_boxes = []
                for node in all_valid_nodes:
                    desc = node['node_desc'] if node['node_desc'] else ''
                    u_box = node['box']
                    n_box, _ = _normalize_box(u_box[0], u_box[1], u_box[2], u_box[3], W, H)
                    clean_desc = ' '.join(
                        x for x in desc.strip(' ,.').split('\n') if x.strip()
                    )
                    node_texts_boxes.append((node['tag'], clean_desc, u_box, n_box))

                node_texts_boxes.sort(
                    key=lambda x: (x[2][1] + x[2][3], x[2][0] + x[2][2]),
                )
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
                    'unnormalized_box': [x[2] for x in node_texts_boxes],
                })
                samples.append(sample)
                counts['widgetlist'] += 1

            if DEBUG:
                for i in range(last_sample_cnt, len(samples)):
                    samples[i]['original_img_file'] = screenshot_file

            if len(samples) > last_sample_cnt:
                counts['valid_img_cnt'] += 1
            last_sample_cnt = len(samples)

    except Exception:
        # Catch *everything* — the pool must never see an exception from a
        # worker, or ``imap_unordered`` can stall waiting for tasks that
        # never produce a result.
        errors.append([traj_name, traceback.format_exc()])

    return {
        'traj_name': traj_name,
        'samples': samples,
        'counts': dict(counts),
        'invalid_updates': {k: list(v) for k, v in invalid_updates.items() if v},
        'unique_elem_entries': {k: list(v) for k, v in unique_elem_entries.items()},
        'errors': errors,
    }


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
    
    # Route through tqdm.write so the message never corrupts an active
    # progress bar (a raw print() to stdout shatters the bar's redraw).
    tqdm.write(f"Saving chunk {chunk_idx} with {len(chunk_samples)} samples to {chunk_file}")
    
    with open(chunk_file, 'w') as f:
        json.dump(chunk_samples, f)

    # Save invalid element records for this chunk
    with open(os.path.join(CHUNK_DIR, f'chunk_{chunk_idx}_invalid_elems.json'), 'w') as f:
        json.dump({k: list(v) for k, v in invalid_elem_tracker.items()}, f, indent=2)

    # Update checkpoint
    completely_proc_traj_names.extend(chunk_traj_names)
    with open(ckpt_file, 'w') as f:
        json.dump({'completely_proc_traj_names': completely_proc_traj_names}, f)

    # Unified per-chunk reporting via run_info.save_run_info.
    num_invalid_elem = sum(len(v) for v in invalid_elem_tracker.values())
    save_run_info(
        dataset=f"{DATASET_NAME} (chunk {chunk_idx})",
        output_path=chunk_file,
        samples=chunk_samples,
        counters=_wae_canonical_counters(counts),
        config={'debug': DEBUG, 'scale': SCALE, 'chunk_idx': chunk_idx},
        images={
            'processed': counts['processed_img_cnt'],
            'valid': counts['valid_img_cnt'],
        },
        elements={
            'iterated': counts['iterated_elem_cnt'],
            'invalid': num_invalid_elem,
            'invalid_breakdown': {
                k: len(v) for k, v in invalid_elem_tracker.items()
            },
        },
        write_preview=False,
        # Per-chunk: persist the chunk's info.json for resumability/aggregation
        # but don't print a report block — only the final aggregated run prints
        # (mirrors make_webui_data.py, which calls save_run_info once at the end).
        print_report=False,
    )


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
        all_traj_names = all_traj_names[:100]

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

    # ------------------------------------------------------------------
    # Parallel per-trajectory dispatch — the per-screenshot logic the
    # original loop ran sequentially is now in ``_wae_process_trajectory``.
    # Behavioural notes:
    #   - Trajectories already listed in ``completely_proc_traj_names``
    #     are skipped (original RESUME semantics).
    #   - Chunks are flushed every ``CHUNK_SIZE`` *completed* trajectories,
    #     not every ``CHUNK_SIZE`` trajectories on the input list — with
    #     parallel processing the completion order is non-deterministic,
    #     and what matters for the saved chunk files is the unit of work,
    #     not the source ordering.
    #   - Task IDs are worker-local (collisions across trajectories within
    #     one chunk are possible). The final aggregator
    #     :func:`_aggregate_and_save_final_results` calls
    #     ``renumber_sample_ids`` on the full output to give every sample
    #     a globally unique id.
    #   - The cumulative invalid-elem skip-list is broadcast to every
    #     worker as a frozenset via ``_init_wae_worker`` — O(1) per-elem
    #     lookups instead of the original O(N) scan over registry buckets.
    # ------------------------------------------------------------------
    work_items = [t for t in all_traj_names if t not in completely_proc_traj_names]
    errors: List[List[str]] = []

    # Build the read-only skip-list snapshot before pool creation. New
    # invalid IDs found by workers are merged into ``invalid_elem_tracker``
    # in the parent — they don't propagate back to other workers in the
    # same chunk, but the chunk's saved record captures them all.
    def _refresh_invalid_ids_frozen() -> frozenset:
        return frozenset(
            sid for bucket in invalid_elem_tracker.values() for sid in bucket
        )

    chunk_idx_counter = 0

    def _flush_chunk_if_due(force: bool = False) -> None:
        """Flush the current chunk to disk if we've hit ``CHUNK_SIZE`` or
        the caller marks the run as done. Preserves the original chunk
        cadence and the "reset everything per chunk" semantics (the chunk
        info JSON is per-chunk, not cumulative)."""
        nonlocal chunk_idx_counter, invalid_elem_tracker, last_sample_cnt
        if not chunk_traj_names:
            return
        if not force and len(chunk_traj_names) < CHUNK_SIZE:
            return

        chunk_idx_counter += 1
        _save_chunk_data(
            chunk_idx_counter, chunk_samples, invalid_elem_tracker, counts,
            chunk_traj_names, completely_proc_traj_names, ckpt_file,
        )

        # Reset per-chunk counters (matches the original sequential
        # behaviour at line ~1137 — task counters reset so each chunk's
        # info.json reports its own slice, not the cumulative roll-up).
        counts['processed_img_cnt'] = 0
        counts['valid_img_cnt'] = 0
        counts['iterated_elem_cnt'] = 0
        for key in ['text_loc', 'ocr', 'icongnd', 'iconref',
                    'elemclass', 'intentgnd', 'widgetlist']:
            counts[key] = 0
        last_sample_cnt = 0
        chunk_samples.clear()
        chunk_traj_names.clear()

        # Resumability: same hook as the original — if a previous run
        # already wrote ``chunk_{N+1}_invalid_elems.json`` we adopt it.
        next_invalid_file = os.path.join(
            CHUNK_DIR, f'chunk_{chunk_idx_counter + 1}_invalid_elems.json',
        )
        if os.path.exists(next_invalid_file):
            with open(next_invalid_file, 'r') as f:
                loaded = json.load(f)
            invalid_elem_tracker = {k: set(v) for k, v in loaded.items()}
        else:
            invalid_elem_tracker = deepcopy(invalid_elem_meta)

    def _aggregate_result(result: Dict[str, Any]) -> None:
        """Merge one worker result into the in-flight chunk."""
        nonlocal last_sample_cnt
        chunk_samples.extend(result['samples'])
        for k, v in result['counts'].items():
            counts[k] += v
        for bucket, ids in result['invalid_updates'].items():
            invalid_elem_tracker.setdefault(bucket, set()).update(ids)
        for img, boxes in result['unique_elem_entries'].items():
            existing = unique_elems[img]
            for b in boxes:
                if b not in existing:
                    existing.append(b)
        chunk_traj_names.append(result['traj_name'])
        last_sample_cnt = len(chunk_samples)
        errors.extend(result['errors'])

    use_parallel = NUM_WORKERS > 1 and len(work_items) > 1
    print(
        f"Processing {len(work_items)} trajectories "
        f"{'in parallel across ' + str(NUM_WORKERS) + ' workers' if use_parallel else 'sequentially'} ..."
    )

    if not work_items:
        print("Nothing to process (all trajectories already complete in checkpoint).")
    elif use_parallel:
        # chunksize=1 here: each task is one whole trajectory (already a
        # batch of many screenshots), so per-task IPC overhead is small
        # relative to processing time. Keeping chunksize=1 also bounds
        # the work lost if a worker dies mid-chunk.
        with mp.Pool(
            processes=NUM_WORKERS,
            initializer=_init_wae_worker,
            initargs=(_refresh_invalid_ids_frozen(),),
            # Recycle workers periodically — pytesseract leaves /tmp
            # artifacts behind, and long-lived workers grow their heap
            # holding image buffers. 100 trajectories per worker is a
            # reasonable balance.
            maxtasksperchild=100,
        ) as pool:
            pbar = tqdm(
                pool.imap_unordered(_wae_process_trajectory, work_items, chunksize=1),
                total=len(work_items),
                desc=f'WAE ×{NUM_WORKERS}',
                unit='traj',
                smoothing=0.05,
                dynamic_ncols=True,
                mininterval=0.3,
            )
            for result in pbar:
                _aggregate_result(result)
                _flush_chunk_if_due()
    else:
        # Sequential fallback also needs the worker global installed so
        # the in-worker O(1) skip check works in-process.
        _init_wae_worker(_refresh_invalid_ids_frozen())
        for traj_name in tqdm(work_items, total=len(work_items), desc='WAE'):
            _aggregate_result(_wae_process_trajectory(traj_name))
            _flush_chunk_if_due()

    # Force-flush whatever is left in the in-flight chunk.
    _flush_chunk_if_due(force=True)

    # Loud end-of-run summary — surfaces mass failures (poison trajectory,
    # missing tesseract, bad XML) before the aggregator buries them.
    print_error_summary(errors, total_tasks=len(work_items))

    # Final aggregation (reads all chunk info.json files, writes the
    # combined samples + info.json).
    _aggregate_and_save_final_results(unique_elems)


def _aggregate_and_save_final_results(unique_elems: Dict[str, List]) -> None:
    """Aggregate all chunk results and save the final dataset.

    Reads each chunk's unified info.json (written via run_info.save_run_info)
    and rebuilds canonical counters / element stats for a final unified
    info.json over the whole dataset.
    """
    all_samples: List[Dict] = []
    all_chunks_num_invalid_elem: Dict[str, int] = defaultdict(int)
    agg_counters: Dict[str, int] = defaultdict(int)
    agg_processed_img = 0
    agg_valid_img = 0
    agg_iterated_elem = 0

    # CamelCase task name -> canonical counter key for write-back.
    display_to_key = {v: k for k, v in TASK_DISPLAY_NAMES.items()}

    chunk_info_files = glob.glob(
        os.path.join(CHUNK_DIR, f'*_s{SCALE}{"_debug" if DEBUG else ""}_info.json')
    )

    for chunk_info_file in chunk_info_files:
        chunk_file = chunk_info_file.replace("_info", "")
        with open(chunk_file, 'r') as f:
            all_samples.extend(json.load(f))

        with open(chunk_info_file, 'r') as f:
            info = json.load(f)

        images = info.get('images', {})
        elements = info.get('elements', {})
        agg_processed_img += images.get('processed', 0)
        agg_valid_img += images.get('valid', 0)
        agg_iterated_elem += elements.get('iterated', 0)

        for display_name, count in info.get('samples', {}).get('by_task', {}).items():
            key = display_to_key.get(display_name)
            if key is not None:
                agg_counters[key] += count

        for k, v in elements.get('invalid_breakdown', {}).items():
            all_chunks_num_invalid_elem[k] += v

    all_invalid_elem = sum(all_chunks_num_invalid_elem.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k, v in unique_elems.items() if len(v)])

    os.makedirs(SAVE_ROOT, exist_ok=True)
    file_name = os.path.join(
        SAVE_ROOT,
        f"{DATASET_NAME}_scale{SCALE}_{len(all_samples)// 1000}k"
        f"{'_debug' if DEBUG else ''}.json"
    )
    print(f"Saving {len(all_samples)} samples to {file_name}")

    # Workers used per-trajectory local counters for task IDs, so the
    # aggregated ``all_samples`` may have duplicates like
    # ``autogui_WAE_textloc_0`` from multiple workers. Renumber globally
    # per-prefix so the final dataset has unique, contiguous IDs.
    renumber_sample_ids(all_samples)

    save_run_info(
        dataset=DATASET_NAME,
        output_path=file_name,
        samples=all_samples,
        counters=dict(agg_counters),
        config={
            'debug': DEBUG, 'scale': SCALE,
            'num_chunks': len(chunk_info_files),
            'num_workers': NUM_WORKERS,
        },
        images={
            'processed': agg_processed_img,
            'valid': agg_valid_img,
            'unique_seen': len(unique_elems),
            'unique_valid': num_valid_imgs,
        },
        elements={
            'iterated': agg_iterated_elem,
            'invalid': all_invalid_elem,
            'unique': num_unique_elems,
            'invalid_breakdown': dict(all_chunks_num_invalid_elem),
        },
    )

    with open(file_name, 'w') as f:
        json.dump(all_samples, f)


if __name__ == '__main__':
    make_wae_data()
