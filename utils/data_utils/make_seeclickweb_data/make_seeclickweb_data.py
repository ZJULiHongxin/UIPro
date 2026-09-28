"""SeeClick-Web dataset preprocessing utilities.

This module loads the SeeClick-Web annotations, filters invalid UI elements,
and generates element grounding samples that can be consumed by downstream
training pipelines. The code mirrors the structure of
`extract_and_generate_mobilebiews_data.py` to keep preprocessing logic
consistent across datasets.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import re
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import magic
import numpy as np
from tqdm import tqdm

from utils.data_utils.task_prompt_lib import *  # noqa: F403,F401
from utils.data_utils.misc import is_pure_color
from utils.data_utils.run_info import (
    print_error_summary,
    renumber_sample_ids,
    save_run_info,
    summarize_errors,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALE: int = 1000
IMG_DIR: str = "/volume/pt-coder/users/gji/projects/UIPro/data/SeeClick-Web"
ANNO_FILE: str = "/volume/pt-coder/users/gji/projects/UIPro/data/seeclick_web.json"

DATASET_NAME: str = "SeeClick-Web"
SAVE_ROOT: str = f"/volume/pt-coder/users/gji/projects/UIPro/data/{DATASET_NAME}_processed"
os.makedirs(SAVE_ROOT, exist_ok=True)

USE_ACTION_PROMPT: bool = False
DEBUG: bool = True
SKIP_CHECKING: bool = False

# --- Task Switches ---
# Toggle which task variants get generated per element. Mirrors the
# per-task flag pattern in make_webui_data.py so the generated reports use
# the canonical CamelCase task labels (ElemGnd / ElemRef). When
# USE_ACTION_PROMPT is True, these are ignored and the action-planning
# shortcut is emitted instead.
ELEMGND: bool = True    # Element Grounding: text -> element location.
ELEMREF: bool = True    # Element Referring: location -> element text.

INVALID_RECORD_NAME: str = "invalid_elem_record.json"

# --- Parallelism ---
# Number of worker processes. Set to 1 (or any value <= 1) to force the
# sequential code path — useful when debugging.
NUM_WORKERS: int = 1 # max(1, (os.cpu_count() or 1) - 1)
# How often the cumulative `invalid_elem_record.json` is flushed to disk
# (counted in completed screens).
CHECKPOINT_EVERY: int = 10000

# Populated by ``_init_worker`` in every Pool child. Holds the flat
# frozenset of all known-invalid sample identifiers so per-element skip
# checks stay O(1) instead of scanning all registry buckets per call.
_INVALID_IDS_CACHE: frozenset = frozenset()


def _init_worker(invalid_ids: frozenset) -> None:
    """Pool initializer: install the cumulative skip-list in every worker.

    The skip-list is read-only inside workers, so a single broadcast at pool
    creation avoids re-pickling the (potentially huge) set for every task.
    Works under both ``fork`` (Linux default) and ``spawn`` start methods.
    """
    global _INVALID_IDS_CACHE
    _INVALID_IDS_CACHE = invalid_ids

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


def _stamp_sample(
    sample: Dict,
    *,
    task_attr: object,
    unnorm_boxes: List[int],
    elem_role: str,
    img_filename: str,
    url: str,
) -> Dict:
    """Attach the dataset-specific extra fields shared by every sample shape."""
    sample["image"] = f"{DATASET_NAME}/{img_filename}"
    sample["unnormalized_box"] = unnorm_boxes
    sample["task_attr"] = task_attr
    sample["elem_role"] = elem_role
    sample["url"] = url
    return sample


def _create_samples(
    instruc: str,
    norm_center_str: str,
    norm_center: List[int],
    unnorm_boxes: List[int],
    elem_role: str,
    img_filename: str,
    url: str,
    counters: Dict[str, int],
) -> List[Dict]:
    """Generate every enabled sample variant for a single element.

    Returns one or more samples depending on the task flags. Mirrors the
    per-task generation pattern in ``make_webui_data._create_samples_from_block``
    so the canonical CamelCase task labels and ``<task>_cnt`` counter keys
    line up across datasets.

    Counter keys updated in ``counters`` (passed by reference):
      - ``text_loc_cnt`` (action-planning shortcut when ``USE_ACTION_PROMPT``)
      - ``elemgnd_cnt``  (when ``ELEMGND``)
      - ``elemref_cnt``  (when ``ELEMREF``)
    """
    out: List[Dict] = []

    # Action-planning shortcut keeps the original behaviour: one sample per
    # element, location encoded into a ``click(x,y)`` action.
    if USE_ACTION_PROMPT:
        action = CLICK_TEMPLATE.format(  # noqa: F405
            target_x=norm_center[0], target_y=norm_center[1]
        )
        sample = make_actionplanning_sample(  # noqa: F405
            task_id=f"autogui_{DATASET_NAME}_textloc_{counters['text_loc_cnt']}",
            global_task=instruc,
            gt_action=action,
            history="None",
            prompt_format_type="aguvis",
        )
        out.append(_stamp_sample(
            sample, task_attr=instruc, unnorm_boxes=unnorm_boxes,
            elem_role=elem_role, img_filename=img_filename, url=url,
        ))
        counters['text_loc_cnt'] += 1
        return out

    # Element Grounding: "Where is <instruc>?" -> location.
    if ELEMGND:
        sample = make_elemgnd_sample(  # noqa: F405
            task_id=f"autogui_{DATASET_NAME}_elemgnd_{counters['elemgnd_cnt']}",
            text=instruc,
            loc=norm_center_str,
            output_tag=WITHPOINT_TAG
        )
        out.append(_stamp_sample(
            sample, task_attr=instruc, unnorm_boxes=unnorm_boxes,
            elem_role=elem_role, img_filename=img_filename, url=url,
        ))
        counters['elemgnd_cnt'] += 1

    # Element Referring: "What is at <location>?" -> instruc.
    # task_attr is the query (the location string) — opposite of Gnd, to
    # match the convention from WebUI.
    if ELEMREF:
        sample = make_elemref_sample(  # noqa: F405
            task_id=f"autogui_{DATASET_NAME}_elemref_{counters['elemref_cnt']}",
            text=instruc,
            loc=norm_center_str,
            output_tag=WITHPOINT_TAG
        )
        out.append(_stamp_sample(
            sample, task_attr=norm_center_str, unnorm_boxes=unnorm_boxes,
            elem_role=elem_role, img_filename=img_filename, url=url,
        ))
        counters['elemref_cnt'] += 1

    return out


# ---------------------------------------------------------------------------
# Parallel-processing worker
# ---------------------------------------------------------------------------
# All buckets the worker may write to. Workers seed a fresh defaultdict-like
# structure so missing buckets don't crash; the main process unions them in.
_INVALID_BUCKETS: Tuple[str, ...] = (
    TOO_SMALL_ELEMENT,
    INVALID_ELEM_BOX,
    INVALID_ELEM_CONTENT,
    BLANK_ELEM,
    EMPTY_ELEM_TEXT,
    OVERLY_LENGTHY_ELEM_TEXT,
    DUPLICATE_ELEMEMNT,
)


def _process_one_screen(screen: Dict) -> Dict[str, Any]:
    """Worker: process one screen and return deltas.

    The function is side-effect free with respect to the parent process —
    all mutation happens against worker-local containers and only the
    aggregated deltas are returned. Any exception raised mid-processing is
    caught locally and emitted in ``errors`` so a poisonous screen never
    crashes the pool (which would otherwise hang ``imap_unordered``).

    Returned dict keys:
      - ``samples``: list of generated samples (worker-local task IDs).
      - ``counters``: per-task local counts keyed by canonical ``<task>_cnt``.
      - ``invalid_updates``: ``{bucket: [identifier, ...]}`` to union into
        the cumulative invalid-elem record.
      - ``unique_elem_entries``: ``{img_filename: [unnormalized_box, ...]}``
        (image keys are unique across screens, so the parent merges with
        a plain ``dict.update``).
      - ``iterated_elem_cnt``: number of raw elements seen in this screen.
      - ``errors``: ``[[img_filename, traceback_str], ...]`` (empty on success).
    """
    samples: List[Dict] = []
    counters: Dict[str, int] = defaultdict(int)
    new_invalid: Dict[str, Set[str]] = {b: set() for b in _INVALID_BUCKETS}
    unique_elems_local: Dict[str, List[List[int]]] = {}
    iterated_elem_cnt = 0
    errors: List[List[str]] = []

    img_filename = screen.get("img_filename", "<unknown>")
    try:
        img_path = os.path.join(IMG_DIR, img_filename)
        width, height = _extract_image_size(img_path)
        iterated_elem_cnt = len(screen["elements"])
        unique_elems_local[img_filename] = []

        used_instructions: Set[str] = set()
        img_cache: ImgType = None

        for elem_info in screen["elements"]:
            bbox: List[float] = elem_info["bbox"]
            instruc: str = elem_info["instruction"].strip()
            sample_identifier = f"{img_filename}|{elem_info['instruction']}"

            # O(1) cumulative skip-list check (was O(N) over registry buckets).
            if sample_identifier in _INVALID_IDS_CACHE:
                continue

            if not SKIP_CHECKING:
                if instruc in used_instructions:
                    new_invalid[DUPLICATE_ELEMEMNT].add(sample_identifier)
                    continue
            used_instructions.add(instruc)

            unnorm_boxes = _denormalize_bbox(bbox, width, height)
            if unnorm_boxes not in unique_elems_local[img_filename]:
                unique_elems_local[img_filename].append(unnorm_boxes)

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
                    invalid_registry=new_invalid,
                )
                if not is_valid:
                    continue

            norm_center, center_str = _normalize_center(bbox, width, height)
            # One element can yield multiple samples (ElemGnd + ElemRef) so we
            # extend the list rather than append a single dict. The local
            # ``counters`` drive worker-local task IDs; the main process
            # renumbers them globally per-prefix after aggregation.
            samples.extend(_create_samples(
                instruc=instruc,
                norm_center_str=center_str,
                norm_center=norm_center,
                unnorm_boxes=unnorm_boxes,
                elem_role=elem_info.get("data_type", ""),
                img_filename=img_filename,
                url=screen.get("url", ""),
                counters=counters,
            ))

    except Exception:
        errors.append([img_filename, traceback.format_exc()])

    return {
        'samples': samples,
        'counters': dict(counters),
        'invalid_updates': {k: list(v) for k, v in new_invalid.items() if v},
        'unique_elem_entries': unique_elems_local,
        'iterated_elem_cnt': iterated_elem_cnt,
        'errors': errors,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def make_seeclickweb_data() -> None:
    annotations = _read_json(ANNO_FILE)
    if DEBUG:
        annotations = random.sample(annotations, min(len(annotations), 200))

    invalid_record_file = os.path.join(SAVE_ROOT, INVALID_RECORD_NAME)
    invalid_registry = _load_invalid_registry(invalid_record_file)

    # Flatten the cumulative skip-list into a frozenset once for O(1) lookups
    # in workers — far cheaper than the original per-element scan across the
    # registry buckets.
    invalid_ids_frozen = frozenset(
        sid for bucket in invalid_registry.values() for sid in bucket
    )

    samples: List[Dict] = []
    counters: Dict[str, int] = defaultdict(int)
    unique_elems: Dict[str, List[List[int]]] = {}
    iterated_elem_cnt = 0
    errors: List[List[str]] = []

    # Streaming aggregator: merge a worker result into the main accumulators.
    # Kept as a closure so the parallel and sequential branches share one
    # implementation and the bookkeeping logic doesn't drift.
    def _aggregate(result: Dict[str, Any]) -> None:
        nonlocal iterated_elem_cnt
        samples.extend(result['samples'])
        for key, value in result['counters'].items():
            counters[key] += value
        for bucket, ids in result['invalid_updates'].items():
            invalid_registry.setdefault(bucket, set()).update(ids)
        # Each screen owns a unique ``img_filename`` so a plain update is safe
        # (no cross-worker key collisions).
        unique_elems.update(result['unique_elem_entries'])
        iterated_elem_cnt += result['iterated_elem_cnt']
        errors.extend(result['errors'])

    use_parallel = NUM_WORKERS > 1
    print(
        f"Processing {len(annotations)} screens "
        f"{'in parallel across ' + str(NUM_WORKERS) + ' workers' if use_parallel else 'sequentially'}..."
    )

    if use_parallel:
        # chunksize ~ N / (workers * 4): high enough to amortize IPC, low
        # enough that imap_unordered streams progress per item. Exceptions
        # are caught inside the worker and emitted as ``errors`` entries, so
        # a poisonous screen cannot hang the pool.
        chunksize = max(1, min(64, len(annotations) // max(1, NUM_WORKERS * 4)))
        completed = 0
        with mp.Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(invalid_ids_frozen,),
        ) as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    _process_one_screen, annotations, chunksize=chunksize
                ),
                total=len(annotations),
                desc=f"SeeClick-Web ×{NUM_WORKERS}",
                smoothing=0.05,
            )
            for result in pbar:
                _aggregate(result)
                completed += 1
                if completed % CHECKPOINT_EVERY == 0:
                    _save_invalid_registry(invalid_record_file, invalid_registry)
    else:
        # Sequential path also needs the worker global initialised so the
        # in-worker skip check works in-process.
        _init_worker(invalid_ids_frozen)
        for screen in tqdm(annotations, total=len(annotations), desc="SeeClick-Web"):
            _aggregate(_process_one_screen(screen))

    # Final flush of the cumulative invalid-element record.
    _save_invalid_registry(invalid_record_file, invalid_registry)

    # Surface mass-failures loudly (e.g. a missing IMG_DIR or a typo that
    # crashes every worker call) — same pattern as webui / mobileviews.
    print_error_summary(errors, total_tasks=len(annotations))

    # Workers used local 0-based ``sample_idx`` counters, so IDs collide
    # globally after aggregation. Renumber per-prefix.
    renumber_sample_ids(samples)

    num_invalid_elem = sum(len(v) for v in invalid_registry.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = sum(1 for v in unique_elems.values() if v)

    # Encode the enabled tasks into the filename so each variant lands in its
    # own file (mirrors webui's `*_actformat` / mobileviews's `*_TextLoc_OCR_`
    # naming so downstream tooling can tell flag combinations apart).
    task_suffix = ''
    if USE_ACTION_PROMPT:
        task_suffix = '_actformat'
    else:
        if ELEMGND:
            task_suffix += '_ElemGnd'
        if ELEMREF:
            task_suffix += '_ElemRef'

    file_name = os.path.join(
        SAVE_ROOT,
        f"{DATASET_NAME}{task_suffix}_{len(samples)//1000}k"
        f"{'_debug' if DEBUG else ''}.json",
    )
    print(f"save to {file_name}")

    save_run_info(
        dataset=DATASET_NAME,
        output_path=file_name,
        samples=samples,
        counters=counters,
        config={
            'debug': DEBUG,
            'scale': SCALE,
            'use_action_prompt': USE_ACTION_PROMPT,
            'skip_checking': SKIP_CHECKING,
            'num_workers': NUM_WORKERS,
        },
        images={
            'processed': len(unique_elems),
            'valid': num_valid_imgs,
        },
        elements={
            'iterated': iterated_elem_cnt,
            'invalid': num_invalid_elem,
            'unique': num_unique_elems,
            'invalid_breakdown': {
                k: len(v) for k, v in invalid_registry.items()
            },
        },
        tasks_enabled=(
            # Action-planning shortcut emits a single TextLoc-style sample;
            # otherwise the enabled ElemGnd / ElemRef flags drive what the
            # report surfaces.
            {'TextLoc': True}
            if USE_ACTION_PROMPT
            else {'ElemGnd': ELEMGND, 'ElemRef': ELEMREF}
        ),
        extra_sections={
            'errors_summary': summarize_errors(errors),
            'errors_total': len(errors),
        },
    )

    with open(file_name, "w") as f:
        json.dump(samples, f, indent=2)


if __name__ == "__main__":
    make_seeclickweb_data()