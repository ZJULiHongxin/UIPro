"""SeeClick-Web dataset preprocessing utilities.

This module loads the SeeClick-Web annotations, filters invalid UI elements,
and generates element grounding samples that can be consumed by downstream
training pipelines. The code mirrors the structure of
`extract_and_generate_mobilebiews_data.py` to keep preprocessing logic
consistent across datasets.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional, Set, Tuple

import cv2
import magic
import numpy as np
from tqdm import tqdm

from utils.data_utils.task_prompt_lib import *  # noqa: F403,F401
from utils.data_utils.misc import is_pure_color

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALE: int = 1000
IMG_DIR: str = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/images/seeclick_web_imgs/"
ANNO_FILE: str = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/raw/seeclick_web.json"

DATASET_NAME: str = "SeeClick-Web"
SAVE_ROOT: str = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"
os.makedirs(SAVE_ROOT, exist_ok=True)

USE_ACTION_PROMPT: bool = False
DEBUG: bool = False
SKIP_CHECKING: bool = True

INVALID_RECORD_NAME: str = "invalid_elem_record.json"

InvalidRegistry = Dict[str, Set[str]]
ImgType = Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _read_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def _load_invalid_registry(path: str) -> InvalidRegistry:
    if not os.path.exists(path):
        return {
            TOO_SMALL_ELEMENT: set(),
            INVALID_ELEM_BOX: set(),
            INVALID_ELEM_CONTENT: set(),
            BLANK_ELEM: set(),
            EMPTY_ELEM_TEXT: set(),
            OVERLY_LENGTHY_ELEM_TEXT: set(),
            DUPLICATE_ELEMEMNT: set(),
        }

    raw_registry = _read_json(path)
    return {k: set(v) for k, v in raw_registry.items()}


def _save_invalid_registry(path: str, registry: InvalidRegistry) -> None:
    with open(path, "w") as f:
        json.dump({k: sorted(v) for k, v in registry.items()}, f, indent=2)


def _extract_image_size(img_path: str) -> Tuple[int, int]:
    width, height = map(
        int,
        re.search(r"(\d+)\s*x\s*(\d+)", magic.from_file(img_path)).groups(),
    )
    return width, height


def _denormalize_bbox(bbox: List[float], width: int, height: int) -> List[int]:
    return list(
        map(
            round,
            [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height],
        )
    )


def _normalize_center(bbox: List[float], width: int, height: int) -> Tuple[List[int], str]:
    norm_center = [
        max(0, min(SCALE - 1, round((bbox[0] + bbox[2]) / 2 * SCALE))),
        max(0, min(SCALE - 1, round((bbox[1] + bbox[3]) / 2 * SCALE))),
    ]
    center_str = f"({norm_center[0]},{norm_center[1]})"
    return norm_center, center_str


def _validate_element(
    bbox: List[float],
    instruc: str,
    sample_identifier: str,
    width: int,
    height: int,
    img_path: str,
    img_cache: ImgType,
    unnorm_boxes: List[int],
    invalid_registry: InvalidRegistry,
) -> Tuple[ImgType, bool]:
    """Run sanity checks on an element. Returns (img_cache, is_valid)."""
    if abs(bbox[0] - bbox[2]) <= 0.005 or abs(bbox[1] - bbox[3]) <= 0.005:
        invalid_registry[TOO_SMALL_ELEMENT].add(sample_identifier)
        return img_cache, False

    if not (
        0 <= bbox[0] <= 1
        and 0 <= bbox[1] <= 1
        and 0 <= bbox[2] <= 1
        and 0 <= bbox[3] <= 1
        and bbox[0] < bbox[2]
        and bbox[1] < bbox[3]
    ):
        invalid_registry[INVALID_ELEM_BOX].add(sample_identifier)
        return img_cache, False

    if ("{" in instruc) or ("}" in instruc):
        invalid_registry[INVALID_ELEM_CONTENT].add(sample_identifier)
        return img_cache, False

    if len(instruc) == 0:
        invalid_registry[EMPTY_ELEM_TEXT].add(sample_identifier)
        return img_cache, False

    if len(instruc) > 60:
        invalid_registry[OVERLY_LENGTHY_ELEM_TEXT].add(sample_identifier)
        return img_cache, False

    if img_cache is None:
        img_cache = cv2.imread(img_path)

    if img_cache is None or is_pure_color(img_cache, unnorm_boxes):
        invalid_registry[BLANK_ELEM].add(sample_identifier)
        return img_cache, False

    return img_cache, True


def _create_sample(
    instruc: str,
    norm_center_str: str,
    norm_center: List[int],
    unnorm_boxes: List[int],
    elem_role: str,
    img_filename: str,
    url: str,
    sample_idx: int,
) -> Dict:
    """Create either action planning or element grounding sample."""
    if USE_ACTION_PROMPT:
        action = CLICK_TEMPLATE.format(target_x=norm_center[0], target_y=norm_center[1])
        sample = make_actionplanning_sample(  # noqa: F405
            task_id=f"autogui_{DATASET_NAME}_textloc_{sample_idx}",
            global_task=instruc,
            gt_action=action,
            history="None",
            prompt_format_type="aguvis",
        )
    else:
        sample = make_elemgnd_sample(  # noqa: F405
            task_id=f"autogui_{DATASET_NAME}_elemgnd_{sample_idx}",
            text=instruc,
            loc=norm_center_str,
            output_tag="",
        )

    sample["image"] = f"{DATASET_NAME}/{img_filename}"
    sample["unnormalized_box"] = unnorm_boxes
    sample["task_attr"] = instruc
    sample["elem_role"] = elem_role
    sample["url"] = url
    return sample


def _should_skip_sample(sample_identifier: str, invalid_registry: InvalidRegistry) -> bool:
    return any(sample_identifier in records for records in invalid_registry.values())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def make_seeclickweb_data() -> None:
    annotations = _read_json(ANNO_FILE)
    if DEBUG:
        annotations = random.sample(annotations, min(len(annotations), 200))

    invalid_record_file = os.path.join(SAVE_ROOT, INVALID_RECORD_NAME)
    invalid_registry = _load_invalid_registry(invalid_record_file)

    samples: List[Dict] = []
    unique_elems: Dict[str, List[List[int]]] = {}

    iterated_elem_cnt = 0

    for group_idx, screen in tqdm(
        enumerate(annotations),
        total=len(annotations),
        desc="SeeClick-Web",
    ):
        img_filename = screen["img_filename"]
        img_path = os.path.join(IMG_DIR, img_filename)
        width, height = _extract_image_size(img_path)
        iterated_elem_cnt += len(screen["elements"])
        unique_elems.setdefault(img_filename, [])

        used_instructions: Set[str] = set()
        img_cache = None

        for elem_info in screen["elements"]:
            bbox: List[float] = elem_info["bbox"]
            instruc: str = elem_info["instruction"].strip()
            sample_identifier = f"{img_filename}|{elem_info['instruction']}"

            if _should_skip_sample(sample_identifier, invalid_registry):
                continue

            if not SKIP_CHECKING:
                if instruc in used_instructions:
                    invalid_registry[DUPLICATE_ELEMEMNT].add(sample_identifier)
                    continue
            used_instructions.add(instruc)

            unnorm_boxes = _denormalize_bbox(bbox, width, height)
            if unnorm_boxes not in unique_elems[img_filename]:
                unique_elems[img_filename].append(unnorm_boxes)

            if not SKIP_CHECKING:
                img_cache, is_valid = _validate_element(
                    bbox=bbox,
                    instruc=instruc,
                    sample_identifier=sample_identifier,
                    width=width,
                    height=height,
                    img_path=img_path,
                    img_cache=img_cache,
                    unnorm_boxes=unnorm_boxes,
                    invalid_registry=invalid_registry,
                )
                if not is_valid:
                    continue

            norm_center, center_str = _normalize_center(bbox, width, height)
            sample = _create_sample(
                instruc=instruc,
                norm_center_str=center_str,
                norm_center=norm_center,
                unnorm_boxes=unnorm_boxes,
                elem_role=elem_info.get("data_type", ""),
                img_filename=img_filename,
                url=screen.get("url", ""),
                sample_idx=len(samples),
            )
            samples.append(sample)

        if (group_idx > 0 and group_idx % 10000 == 0) or group_idx == len(annotations) - 1:
            _save_invalid_registry(invalid_record_file, invalid_registry)

    num_invalid_elem = sum(len(v) for v in invalid_registry.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = sum(1 for v in unique_elems.values() if v)

    report = (
        f"#samples: {len(samples)}\n"
        f"#Unique elements: {num_unique_elems}\n"
        f"#Valid unique images: {num_valid_imgs}\n"
        f"#All unique images: {len(unique_elems)}\n"
        f"Invalid elem ratio: {num_invalid_elem} / {iterated_elem_cnt} = "
        f"{num_invalid_elem / iterated_elem_cnt:.2f}\n"
        f"text_loc_cnt: {len(samples)} | ocr_cnt: 0 | widgetlist_cnt: 0"
    )
    print(report)

    file_name = os.path.join(
        SAVE_ROOT,
        f"{DATASET_NAME}_{len(samples)//1000}k"
        f"{'_debug' if DEBUG else ''}"
        f"{'_actformat' if USE_ACTION_PROMPT else ''}.json",
    )
    print(f"save to {file_name}")

    info_payload = {
        "num_samples": len(samples),
        "#num_unique_elems": num_unique_elems,
        "#all_elems": iterated_elem_cnt,
        "#valid_unique_images": num_valid_imgs,
        "#all_unique_images": len(unique_elems),
        "text_loc_cnt": len(samples),
        "ocr_cnt": 0,
        "elemclass_cnt": 0,
        "intentgnd_cnt": 0,
        "widgetlist_cnt": 0,
        "num_invalid_elements": num_invalid_elem,
        "invalid_elem_types": {k: len(v) for k, v in invalid_registry.items()},
    }

    with open(file_name.replace(".json", "_info.json"), "w") as f:
        json.dump(info_payload, f, indent=2)

    with open(file_name.replace(".json", "_sample.json"), "w") as f:
        json.dump(random.sample(samples, min(len(samples), 128)), f, indent=2)

    with open(file_name, "w") as f:
        json.dump(samples, f, indent=2)


if __name__ == "__main__":
    make_seeclickweb_data()