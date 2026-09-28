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
import multiprocessing as mp
import os
import random
import re
import sys
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import magic
from datasets import load_dataset
from tqdm import tqdm

from utils.data_utils.misc import VerbExtactor, is_pure_color
from utils.data_utils.task_prompt_lib import *  # noqa: F403,F401
from utils.data_utils.run_info import (
    print_error_summary,
    renumber_sample_ids,
    save_run_info,
    summarize_errors,
)

# ============================================================================
# Configuration Constants
# ============================================================================

# Dataset configuration
DATASET_NAME: str = 'RefExp'
SCALE: int = 1000  # Coordinate normalization scale (output coordinates in range [0, SCALE-1])

# Directory paths
IMG_DIR: str = "/volume/pt-coder/users/gji/projects/UIPro/data/rico/"
SAVE_ROOT: str = f"/volume/pt-coder/users/gji/projects/UIPro/data/{DATASET_NAME}_processed"

# Task generation flags
USE_ACTION_PROMPT: bool = False  # If True, use action planning format instead of intent grounding
DEBUG: bool = False  # If True, only process first 200 samples for testing
OUTPUT_TAG: str = ''  # Optional output format tag (e.g., 'Please output its center coordinates.')

# Validation flags
SKIP_CHECKING: bool = False  # If True, skip all validation checks (not recommended)

# --- Parallelism ---
# Number of worker processes. Set to 1 to force the sequential code path.
# Each worker initialises its own ``VerbExtactor`` (spaCy-based) at startup —
# that's a ~1-2 s cost per worker, paid once.
NUM_WORKERS: int = max(1, (os.cpu_count() or 1) - 1)

# Per-worker globals populated by ``_init_worker``.
_INVALID_IDS_CACHE: frozenset = frozenset()
_INSTRUC_TYPE_CACHE: Dict[str, Optional[str]] = {}
_VERB_EXTRACTOR: Optional[VerbExtactor] = None


def _ensure_spacy_model_available() -> None:
    """Verify the spaCy verb-extraction model loads before any pool work.

    Without this pre-flight check every Pool worker independently tries to
    construct ``VerbExtactor()`` in its initializer; with the default
    ``cpu_count - 1`` workers that yields ~100 identical ``OSError``
    tracebacks before the run dies. Failing fast in the parent with a
    single, actionable install command is much friendlier.
    """
    try:
        VerbExtactor()
    except OSError as exc:
        sys.exit(
            "\nERROR: required spaCy model 'en_core_web_md' is not installed.\n"
            "RefExp uses it to classify instructions as 'element' (no verb)\n"
            "vs. 'intent' (action verb) grounding tasks.\n\n"
            "Install with:\n"
            "    python -m spacy download en_core_web_md\n\n"
            f"Underlying error: {exc}\n"
        )


def _init_worker(
    invalid_ids: frozenset,
    instruc_cache_seed: Dict[str, Optional[str]],
) -> None:
    """Pool initializer: install the skip-list + verb-cache seed.

    The verb extractor is created here rather than passed in because spaCy
    models do not pickle cleanly. Each worker pays the model-load cost once
    at startup; subsequent calls reuse the live extractor.
    """
    global _INVALID_IDS_CACHE, _INSTRUC_TYPE_CACHE, _VERB_EXTRACTOR
    _INVALID_IDS_CACHE = invalid_ids
    # Seed each worker's cache with the persisted mapping so re-runs benefit
    # from prior verb-extraction work. New entries the worker computes stay
    # local and are returned in the per-task result for merging in the parent.
    _INSTRUC_TYPE_CACHE = dict(instruc_cache_seed)
    _VERB_EXTRACTOR = VerbExtactor()

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


# ============================================================================
# Parallel-processing worker
# ============================================================================
# All invalid buckets the worker may write to. Pre-seeding the dict here means
# the per-element check paths don't need defensive ``setdefault`` calls.
_INVALID_BUCKETS: Tuple[str, ...] = (
    TOO_SMALL_ELEMENT,
    INVALID_ELEM_BOX,
    INVALID_ELEM_CONTENT,
    BLANK_ELEM,
    EMPTY_ELEM_TEXT,
    OVERLY_LENGTHY_ELEM_TEXT,
    DUPLICATE_ELEMEMNT,
)


def _process_one_refexp_sample(raw_sample: Dict) -> Dict[str, Any]:
    """Worker: process a single RefExp row and return deltas.

    Pure function with respect to the parent process — all mutation happens
    against worker-local containers. The verb extractor and cumulative
    skip-list live in module-level globals (set by ``_init_worker``).

    Returned dict keys:
      - ``sample``: the generated sample (or None if filtered out).
      - ``task_type``: ``'elemgnd'`` | ``'intentgnd'`` | None
      - ``image_id`` / ``unnorm_box``: tracked for the per-image unique-elem list.
      - ``invalid_updates``: ``{bucket: [identifier, ...]}`` to union into
        the cumulative invalid-elem record.
      - ``new_instruc_types``: ``{instruction: verb_or_None}`` learned this call.
      - ``errors``: ``[[identifier, traceback_str], ...]`` (empty on success).
    """
    sample: Optional[Dict] = None
    task_type: Optional[str] = None
    image_id = raw_sample.get('image_id', '<unknown>')
    unnorm_boxes: Optional[List[int]] = None
    new_invalid: Dict[str, Set[str]] = {b: set() for b in _INVALID_BUCKETS}
    new_instruc_types: Dict[str, Optional[str]] = {}
    errors: List[List[str]] = []

    try:
        instruc = raw_sample['prompt'].strip()
        bbox_raw = eval(raw_sample['target_bounding_box'])
        bbox = (bbox_raw['xmin'], bbox_raw['ymin'], bbox_raw['xmax'], bbox_raw['ymax'])
        sample_identifier = f"{image_id}|{raw_sample['prompt']}"

        # O(1) cumulative skip-list lookup.
        if sample_identifier in _INVALID_IDS_CACHE:
            return _bundle_refexp_result(
                sample, task_type, image_id, unnorm_boxes,
                new_invalid, new_instruc_types, errors,
            )

        img_path = os.path.join(IMG_DIR, f'{image_id}.jpg')
        try:
            W, H = _get_image_dimensions(img_path)
        except (ValueError, AttributeError):
            return _bundle_refexp_result(
                sample, task_type, image_id, unnorm_boxes,
                new_invalid, new_instruc_types, errors,
            )

        unnorm_boxes = [
            round(bbox[0] * W), round(bbox[1] * H),
            round(bbox[2] * W), round(bbox[3] * H),
        ]

        if not SKIP_CHECKING:
            # Inline the validation checks so the worker can tag rejections
            # into the local ``new_invalid`` dict directly.
            if any(keyword in instruc for keyword in ['flip', 'swipe']):
                new_invalid[INVALID_ELEM_CONTENT].add(sample_identifier)
                return _bundle_refexp_result(
                    sample, task_type, image_id, unnorm_boxes,
                    new_invalid, new_instruc_types, errors,
                )
            if abs(bbox[0] - bbox[2]) <= 0.005 or abs(bbox[1] - bbox[3]) <= 0.005:
                new_invalid[TOO_SMALL_ELEMENT].add(sample_identifier)
                return _bundle_refexp_result(
                    sample, task_type, image_id, unnorm_boxes,
                    new_invalid, new_instruc_types, errors,
                )
            if not (
                0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1
                and 0 <= bbox[2] <= 1 and 0 <= bbox[3] <= 1
                and bbox[0] < bbox[2] and bbox[1] < bbox[3]
            ):
                new_invalid[INVALID_ELEM_BOX].add(sample_identifier)
                return _bundle_refexp_result(
                    sample, task_type, image_id, unnorm_boxes,
                    new_invalid, new_instruc_types, errors,
                )
            if len(instruc) == 0:
                new_invalid[EMPTY_ELEM_TEXT].add(sample_identifier)
                return _bundle_refexp_result(
                    sample, task_type, image_id, unnorm_boxes,
                    new_invalid, new_instruc_types, errors,
                )
            img = cv2.imread(img_path)
            if img is not None and is_pure_color(img, unnorm_boxes):
                new_invalid[BLANK_ELEM].add(sample_identifier)
                return _bundle_refexp_result(
                    sample, task_type, image_id, unnorm_boxes,
                    new_invalid, new_instruc_types, errors,
                )

        # Classify instruction (cache hit avoids the spaCy parse).
        if instruc in _INSTRUC_TYPE_CACHE:
            verb = _INSTRUC_TYPE_CACHE[instruc]
        else:
            assert _VERB_EXTRACTOR is not None, "Pool was not initialised — call _init_worker first."
            verb, _ = _VERB_EXTRACTOR.find_first_verb(instruc)
            _INSTRUC_TYPE_CACHE[instruc] = verb
            new_instruc_types[instruc] = verb

        # Normalise centre.
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        norm_center = [
            max(0, min(SCALE - 1, round(center_x * SCALE))),
            max(0, min(SCALE - 1, round(center_y * SCALE))),
        ]
        center_str = f'({norm_center[0]},{norm_center[1]})'

        if verb is None:
            # No verb -> Element Grounding.
            sample = make_elemgnd_sample(
                task_id=f'autogui_{DATASET_NAME}_elemgnd_0',  # local; main renumbers.
                text=instruc,
                loc=center_str,
                output_tag=OUTPUT_TAG,
            )
            task_type = 'elemgnd'
        else:
            # Verb -> Intent Grounding.
            if USE_ACTION_PROMPT:
                action = CLICK_TEMPLATE.format(target_x=norm_center[0], target_y=norm_center[1])
                query = (
                    TURN_GND_INTO_PLANNING_PROMPT.format(instruc=instruc)
                    if len(instruc.strip().split()) == 1 else instruc
                )
                sample = make_actionplanning_sample(
                    task_id=f'autogui_{DATASET_NAME}_intentgnd_0',
                    global_task=query,
                    gt_action=action,
                    history='None',
                    prompt_format_type='aguvis',
                )
            else:
                sample = make_intentgnd_sample(
                    task_id=f'autogui_{DATASET_NAME}_intentgnd_0',
                    intent=instruc,
                    loc=center_str,
                    output_tag=OUTPUT_TAG,
                )
            task_type = 'intentgnd'

        sample['image'] = f'rico/{image_id}.jpg'
        sample['unnormalized_box'] = unnorm_boxes
        sample['task_attr'] = instruc

    except Exception:
        errors.append([f"{image_id}", traceback.format_exc()])

    return _bundle_refexp_result(
        sample, task_type, image_id, unnorm_boxes,
        new_invalid, new_instruc_types, errors,
    )


def _bundle_refexp_result(
    sample: Optional[Dict],
    task_type: Optional[str],
    image_id: str,
    unnorm_box: Optional[List[int]],
    new_invalid: Dict[str, Set[str]],
    new_instruc_types: Dict[str, Optional[str]],
    errors: List[List[str]],
) -> Dict[str, Any]:
    """Pack a worker's locals into a small pickleable result dict."""
    return {
        'sample': sample,
        'task_type': task_type,
        'image_id': image_id,
        'unnorm_box': unnorm_box,
        'invalid_updates': {k: list(v) for k, v in new_invalid.items() if v},
        'new_instruc_types': new_instruc_types,
        'errors': errors,
    }


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
    # Step 0: Pre-flight checks — fail fast with one actionable error rather
    # than ~100 worker tracebacks for the same underlying problem.
    # ========================================================================
    _ensure_spacy_model_available()

    # ========================================================================
    # Step 1: Initialize data structures and load cached records
    # ========================================================================
    unique_elems: Dict[str, List[List[int]]] = defaultdict(list)
    samples: List[Dict] = []
    elemgnd_cnt: int = 0
    intentgnd_cnt: int = 0
    errors: List[List[str]] = []

    invalid_elem_record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    invalid_elem = _load_invalid_records(invalid_elem_record_file)
    instruc_type_file = os.path.join(SAVE_ROOT, 'instruc_type_record.json')
    instruc_type_dict = _load_instruction_types(instruc_type_file)

    # ========================================================================
    # Step 2: Shuffle + materialise the work list
    # ========================================================================
    # ``ds.shuffle()`` returns a shuffled view in newer ``datasets`` versions
    # but mutates in place in older ones; this works either way.
    shuffled = ds.shuffle() or ds
    if DEBUG:
        # Materialise only the slice we'll actually process so the parallel
        # iteration has an accurate length for tqdm and chunksize sizing.
        work_items = [shuffled[i] for i in range(min(200, len(shuffled)))]
    else:
        work_items = shuffled

    total = len(work_items) if hasattr(work_items, '__len__') else None

    # Flatten the cumulative skip-list once for O(1) worker lookups.
    invalid_ids_frozen = frozenset(
        sid for bucket in invalid_elem.values() for sid in bucket
    )

    # ========================================================================
    # Step 3: Streaming aggregator + parallel dispatch
    # ========================================================================
    def _aggregate(result: Dict[str, Any]) -> None:
        nonlocal elemgnd_cnt, intentgnd_cnt
        if result['sample'] is not None and result['unnorm_box'] is not None:
            img_id = result['image_id']
            boxes = unique_elems[img_id]
            if result['unnorm_box'] not in boxes:
                boxes.append(result['unnorm_box'])
            samples.append(result['sample'])
            if result['task_type'] == 'elemgnd':
                elemgnd_cnt += 1
            elif result['task_type'] == 'intentgnd':
                intentgnd_cnt += 1
        for bucket, ids in result['invalid_updates'].items():
            invalid_elem.setdefault(bucket, set()).update(ids)
        # Merge newly-learned instruction → verb mappings into the persistent
        # cache. Same instruction text from different workers maps to the
        # same verb (verb extraction is deterministic), so collisions are
        # fine.
        for instruc, verb in result['new_instruc_types'].items():
            instruc_type_dict[instruc] = verb
        errors.extend(result['errors'])

    use_parallel = NUM_WORKERS > 1
    print(
        f"\nProcessing {total or '?'} samples "
        f"{'in parallel across ' + str(NUM_WORKERS) + ' workers' if use_parallel else 'sequentially'}..."
    )

    if use_parallel:
        chunksize = max(1, min(64, (total or NUM_WORKERS) // max(1, NUM_WORKERS * 4)))
        with mp.Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(invalid_ids_frozen, instruc_type_dict),
        ) as pool:
            pbar = tqdm(
                pool.imap_unordered(
                    _process_one_refexp_sample, work_items, chunksize=chunksize
                ),
                total=total,
                desc=f"{DATASET_NAME} ×{NUM_WORKERS}",
                smoothing=0.05,
            )
            for result in pbar:
                _aggregate(result)
    else:
        _init_worker(invalid_ids_frozen, instruc_type_dict)
        for raw_sample in tqdm(work_items, total=total, desc=f"Processing {DATASET_NAME}"):
            _aggregate(_process_one_refexp_sample(raw_sample))

    # Final flush of the cumulative records.
    _save_invalid_records(invalid_elem_record_file, invalid_elem)
    _save_instruction_types(instruc_type_file, instruc_type_dict)

    # Surface mass-failures loudly before the run-summary block.
    print_error_summary(errors, total_tasks=total or 0)

    # Workers used a fixed local placeholder for the task-id integer suffix;
    # renumber globally per-prefix so each task has unique, contiguous IDs.
    renumber_sample_ids(samples)

    # ========================================================================
    # Step 4: Generate statistics and summary report
    # ========================================================================
    
    # Calculate statistics
    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k, v in unique_elems.items() if len(v) > 0])
    num_total_imgs = len(unique_elems)
    
    # ========================================================================
    # Step 5: Save results to disk
    # ========================================================================
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

    save_run_info(
        dataset=DATASET_NAME,
        output_path=file_name,
        samples=samples,
        counters={
            'elemgnd_cnt': elemgnd_cnt,
            'intentgnd_cnt': intentgnd_cnt,
        },
        config={
            'scale': SCALE,
            'use_action_prompt': USE_ACTION_PROMPT,
            'debug': DEBUG,
            'skip_checking': SKIP_CHECKING,
            'num_workers': NUM_WORKERS,
        },
        images={
            'processed': num_total_imgs,
            'valid': num_valid_imgs,
        },
        elements={
            'iterated': total or len(ds),
            'invalid': num_invalid_elem,
            'unique': num_unique_elems,
            'invalid_breakdown': {k: len(v) for k, v in invalid_elem.items()},
        },
        tasks_enabled={'ElemGnd': True, 'IntentGnd': True},
        extra_sections={
            'errors_summary': summarize_errors(errors),
            'errors_total': len(errors),
        },
    )

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