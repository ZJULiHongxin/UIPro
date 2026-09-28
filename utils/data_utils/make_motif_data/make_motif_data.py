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
import multiprocessing as mp
import random
import sys
import traceback
import numpy as np
from typing import Any, Dict, List, Set, Tuple
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
from run_info import (
    print_error_summary,
    renumber_sample_ids,
    save_run_info,
    summarize_errors,
)

# ==============================================================================
# Configuration & Constants
# ==============================================================================

DATASET_NAME = 'MOTIF'
SCALE = 1000

# Paths
# Note: Adjust paths as necessary for your environment
IMG_DIR = "/volume/pt-coder/users/gji/projects/UIPro/data/motif/"
SAVE_ROOT_DIR = f"/volume/pt-coder/users/gji/projects/UIPro/data/{DATASET_NAME}_processed"

# Processing Flags
DEBUG = False
SKIP_CHECKING = False
USE_ACTION_PROMPT = False # If True, format as action planning; else element grounding
OUTPUT_TAG = WITHPOINT_TAG # Tag for element grounding output, e.g., 'Please output its center coordinates.'

# --- Parallelism ---
# Number of worker processes. Set to 1 to force the sequential code path.
# Each worker shells out a single `magic.from_file` call per item (or falls
# back to cv2.imread for dimension extraction), so file I/O dominates and
# multiprocessing gives a large speed-up on networked storage.
NUM_WORKERS: int = 1# max(1, (os.cpu_count() or 1) - 1)

# Populated by ``_init_worker`` in every Pool child. Holds the flat frozenset
# of all known-invalid sample identifiers so per-element skip checks stay
# O(1) instead of scanning every registry bucket per call.
_INVALID_IDS_CACHE: frozenset = frozenset()


def _init_worker(invalid_ids: frozenset) -> None:
    """Pool initializer: install the cumulative skip-list in every worker."""
    global _INVALID_IDS_CACHE
    _INVALID_IDS_CACHE = invalid_ids


# Every bucket the worker may add to. Workers seed all of them so writes
# don't need defensive ``setdefault`` calls.
_INVALID_BUCKETS: Tuple[str, ...] = (
    TOO_SMALL_ELEMENT,
    INVALID_ELEM_BOX,
    INVALID_ELEM_CONTENT,
    BLANK_ELEM,
    EMPTY_ELEM_TEXT,
    OVERLY_LENGTHY_ELEM_TEXT,
    DUPLICATE_ELEMEMNT,
)


def _process_one_motif_item(item: Dict) -> Dict[str, Any]:
    """Worker: process a single MOTIF row and return deltas.

    Pure function with respect to the parent process — all mutation happens
    against worker-local containers. Any exception is caught locally and
    emitted in ``errors`` so a poisonous row never hangs ``imap_unordered``.

    Returned dict keys:
      - ``sample``: the generated sample (or None if filtered out).
      - ``image_id``: image identifier used to dedupe ``unique_elems`` boxes.
      - ``unnorm_box``: the bbox tracked for the image's unique-elem list.
      - ``invalid_updates``: ``{bucket: [identifier, ...]}`` to union into
        the cumulative invalid-elem record.
      - ``errors``: ``[[image_id, traceback_str], ...]`` (empty on success).
    """
    sample: Dict | None = None
    image_id = item.get('image_id', '<unknown>')
    unnorm_box: List[int] | None = None
    new_invalid: Dict[str, Set[str]] = {b: set() for b in _INVALID_BUCKETS}
    errors: List[List[str]] = []

    try:
        img_path = os.path.join(IMG_DIR, image_id + '.jpg')
        if not os.path.exists(img_path):
            return _bundle_motif_result(
                sample, image_id, unnorm_box, new_invalid, errors,
            )

        # Dimensions: try magic first (cheap header read), fall back to cv2.
        W: int
        H: int
        try:
            file_info = magic.from_file(img_path)
            res = re.search(r'precision 8, (\d+)x(\d+)', file_info)
            if res:
                W, H = list(map(int, res.groups()))
            else:
                img = cv2.imread(img_path)
                if img is None:
                    return _bundle_motif_result(
                        sample, image_id, unnorm_box, new_invalid, errors,
                    )
                H, W = img.shape[:2]
        except Exception:
            return _bundle_motif_result(
                sample, image_id, unnorm_box, new_invalid, errors,
            )

        img = None  # lazy-load only when the pure-color check needs it.

        if len(item['bbox_mc']) == 0:
            return _bundle_motif_result(
                sample, image_id, unnorm_box, new_invalid, errors,
            )

        bbox = item['bbox_mc'][0]
        instruction = item['instr'].strip()
        sample_identifier = f"{image_id}|{instruction}"

        # O(1) cumulative skip-list lookup.
        if sample_identifier in _INVALID_IDS_CACHE:
            return _bundle_motif_result(
                sample, image_id, unnorm_box, new_invalid, errors,
            )

        unnorm_box = list(map(round, [bbox[0]*W, bbox[1]*H, bbox[2]*W, bbox[3]*H]))

        if not SKIP_CHECKING:
            if any(k in instruction for k in ['flip', 'swipe']):
                new_invalid[INVALID_ELEM_CONTENT].add(sample_identifier)
                return _bundle_motif_result(
                    sample, image_id, unnorm_box, new_invalid, errors,
                )

            if abs(bbox[0] - bbox[2]) <= 0.005 or abs(bbox[1] - bbox[3]) <= 0.005:
                new_invalid[TOO_SMALL_ELEMENT].add(sample_identifier)
                return _bundle_motif_result(
                    sample, image_id, unnorm_box, new_invalid, errors,
                )

            if not (
                0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1
                and 0 <= bbox[2] <= 1 and 0 <= bbox[3] <= 1
                and bbox[0] < bbox[2] and bbox[1] < bbox[3]
            ):
                new_invalid[INVALID_ELEM_BOX].add(sample_identifier)
                return _bundle_motif_result(
                    sample, image_id, unnorm_box, new_invalid, errors,
                )

            if len(instruction) == 0:
                new_invalid[EMPTY_ELEM_TEXT].add(sample_identifier)
                return _bundle_motif_result(
                    sample, image_id, unnorm_box, new_invalid, errors,
                )

            # Pure-color check — load the image only if we made it this far.
            if img is None:
                img = cv2.imread(img_path)
            if img is not None and is_pure_color(img, unnorm_box):
                new_invalid[BLANK_ELEM].add(sample_identifier)
                return _bundle_motif_result(
                    sample, image_id, unnorm_box, new_invalid, errors,
                )

        norm_center = [
            max(0, min(SCALE - 1, round((bbox[0]+bbox[2])/2*SCALE))),
            max(0, min(SCALE - 1, round((bbox[1]+bbox[3])/2*SCALE))),
        ]
        center_str = f'({norm_center[0]},{norm_center[1]})'

        if USE_ACTION_PROMPT:
            action_str = CLICK_TEMPLATE.format(target_x=norm_center[0], target_y=norm_center[1])
            sample = make_actionplanning_sample(
                task_id=f'autogui_{DATASET_NAME}_intentgnd_0',  # local; main renumbers.
                global_task=instruction,
                gt_action=action_str,
                history='None',
                prompt_format_type='aguvis',
            )
        else:
            sample = make_intentgnd_sample(
                task_id=f'autogui_{DATASET_NAME}_intentgnd_0',  # local; main renumbers.
                intent=instruction,
                loc=center_str,
                output_tag=OUTPUT_TAG,
            )

        sample['image'] = f'motif/{image_id}.jpg'
        sample['unnormalized_box'] = unnorm_box
        sample['task_attr'] = instruction

    except Exception:
        errors.append([image_id, traceback.format_exc()])

    return _bundle_motif_result(
        sample, image_id, unnorm_box, new_invalid, errors,
    )


def _bundle_motif_result(
    sample: Dict | None,
    image_id: str,
    unnorm_box: List[int] | None,
    new_invalid: Dict[str, Set[str]],
    errors: List[List[str]],
) -> Dict[str, Any]:
    """Pack a worker's locals into a small pickleable result dict."""
    return {
        'sample': sample,
        'image_id': image_id,
        'unnorm_box': unnorm_box,
        'invalid_updates': {k: list(v) for k, v in new_invalid.items() if v},
        'errors': errors,
    }


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
    unique_elems: Dict[str, List[List[int]]] = defaultdict(list)
    samples: List[Dict] = []
    errors: List[List[str]] = []

    # Load or initialize invalid element records
    invalid_elem_record_file = os.path.join(SAVE_ROOT_DIR, 'invalid_elem_record.json')
    if os.path.exists(invalid_elem_record_file):
        invalid_elem = json.load(open(invalid_elem_record_file))
        for k, v in invalid_elem.items():
            invalid_elem[k] = set(v)
    else:
        invalid_elem = {b: set() for b in _INVALID_BUCKETS}

    # Flatten the cumulative skip-list once for O(1) worker lookups.
    invalid_ids_frozen = frozenset(
        sid for bucket in invalid_elem.values() for sid in bucket
    )

    # Streaming aggregator: merge a worker result into the main accumulators.
    # Shared by parallel + sequential branches.
    def _aggregate(result: Dict[str, Any]) -> None:
        if result['sample'] is not None and result['unnorm_box'] is not None:
            img_id = result['image_id']
            boxes = unique_elems[img_id]
            if result['unnorm_box'] not in boxes:
                boxes.append(result['unnorm_box'])
            samples.append(result['sample'])
        for bucket, ids in result['invalid_updates'].items():
            invalid_elem.setdefault(bucket, set()).update(ids)
        errors.extend(result['errors'])

    use_parallel = NUM_WORKERS > 1
    total = len(data) if hasattr(data, '__len__') else None
    print(
        f"Processing {total or '?'} items "
        f"{'in parallel across ' + str(NUM_WORKERS) + ' workers' if use_parallel else 'sequentially'}..."
    )

    if use_parallel:
        # chunksize ~ N / (workers * 4): high enough to amortize IPC, low
        # enough that imap_unordered streams progress per item. Exceptions
        # are caught inside the worker and emitted via ``errors``, so a
        # poisonous row cannot hang the pool.
        chunksize = max(1, min(64, (total or NUM_WORKERS) // max(1, NUM_WORKERS * 4)))
        with mp.Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(invalid_ids_frozen,),
        ) as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    _process_one_motif_item, data, chunksize=chunksize
                ),
                total=total,
                desc=f"{DATASET_NAME} ×{NUM_WORKERS}",
                smoothing=0.05,
            )
            for result in pbar:
                _aggregate(result)
    else:
        # Sequential fallback also needs the worker global initialised.
        _init_worker(invalid_ids_frozen)
        for item in tqdm(data, total=total, desc=f"Processing {DATASET_NAME}"):
            _aggregate(_process_one_motif_item(item))

    # Final flush of the cumulative invalid-element record.
    with open(invalid_elem_record_file, 'w') as f:
        json.dump({k: list(v) for k, v in invalid_elem.items()}, f, indent=2)

    # Surface mass-failures loudly before the run-summary block.
    print_error_summary(errors, total_tasks=total or 0)

    # Workers used a fixed local task_id placeholder; renumber globally.
    renumber_sample_ids(samples)

    # --------------------------------------------------------------------------
    # Statistics and Saving
    # --------------------------------------------------------------------------
    
    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k, v in unique_elems.items() if len(v)])

    # Construct filenames
    suffix_debug = '_debug' if DEBUG else ''
    suffix_format = '_actformat' if USE_ACTION_PROMPT else ''
    # Naming convention: {DATASET}_{COUNT}k{FLAGS}.json
    base_filename = f"{DATASET_NAME}_{len(samples)//1000}k{suffix_debug}{suffix_format}.json"
    save_path = os.path.join(SAVE_ROOT_DIR, base_filename)
    print(f"Saving to {save_path}")

    save_run_info(
        dataset=DATASET_NAME,
        output_path=save_path,
        samples=samples,
        counters={'intentgnd_cnt': len(samples)},
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
            'iterated': len(data),
            'invalid': num_invalid_elem,
            'unique': num_unique_elems,
            'invalid_breakdown': {k: len(v) for k, v in invalid_elem.items()},
        },
        tasks_enabled={'IntentGnd': True},
        extra_sections={
            'errors_summary': summarize_errors(errors),
            'errors_total': len(errors),
        },
    )

    # Save Full Dataset
    with open(save_path, 'w') as f:
        json.dump(samples, f, indent=2)

if __name__ == "__main__":
    process_motif_dataset()
