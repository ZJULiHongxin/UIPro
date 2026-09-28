"""
AndroidControl dataset conversion and sample generation utilities.

This module reads AndroidControl TFRecord episodes, decodes screenshots and
accessibility trees, filters/validates UI elements, and generates model-ready
samples for multiple tasks, including:

- Action planning (high-level and high+low-level variants)
- Intent grounding (optional)

Performance / caching
---------------------
The TFRecord stream is GZIP-compressed and single-threaded to read; a single
pass over ~15k episodes is the dominant cost. To avoid paying that cost on
every run, the first run extracts each episode's raw bytes into a per-episode
pickle file at ``{ANDROIDCONTROL_ROOT}/_episode_cache/{ep_id}.pkl`` and writes
a ``_complete.marker`` once the full pass has been done. Subsequent runs skip
TensorFlow entirely and dispatch episodes to worker processes that load their
pickle and process it locally.

Multiprocessing notes
---------------------
The per-episode body is a pure-Python worker (``_process_episode``) — no
``tf.*`` calls. The TF import is lazy (inside ``_build_episode_cache``) so
spawned workers that re-import this module never load TF. The Pool is
created with ``mp.get_context('spawn')`` to keep workers independent of the
parent's loaded modules.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import json, os, random, cv2, re, pickle, time
import multiprocessing as mp
import traceback
from functools import partial
from tqdm import tqdm
from android_env.proto.a11y import android_accessibility_forest_pb2
import numpy as np
import ast
import xml.etree.ElementTree as ET

from collections import Counter, defaultdict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *
from misc import (
    parse_axtrees_proto,
    resize_image,
    find_smallest_box_containing_point,
    is_pure_color,
    box2center,
    random_substring,
    revise_swipe_action,
    keep_unique_actions,
    lower_first_letter,
)
from run_info import (
    print_error_summary,
    renumber_sample_ids,
    save_run_info,
    summarize_errors,
)

# Treat very short windows as status bars; skip strict validation for them
MAX_STATUSBAR_HEIGHT = 130

# Per-episode pickle cache, sitting beside the raw TFRecords.
EPISODE_CACHE_DIRNAME = '_episode_cache'
CACHE_COMPLETE_MARKER = '_complete.marker'

# Number of worker processes for the per-episode loop. Workers are TF-free so
# memory cost is modest (mostly task_prompt_lib + cv2 + numpy). Tune to taste.
NUM_WORKERS: int = 32

# Bucket names for per-element rejection tracking. Constants are imported via
# ``from task_prompt_lib import *``; ``UI_NO_MAIN_WINDOW`` is an
# AndroidControl-specific bucket (no nearby equivalent in WAE/webui).
UI_NO_MAIN_WINDOW = 'UIs without main app windows'
_INVALID_BUCKETS: Tuple[str, ...] = (
    DUPLICATE_ELEMEMNT,
    INVALID_ELEM_BOX,
    TOO_SMALL_ELEMENT,
    OVERSIZED_ELEMENT,
    INVALID_ELEM_CONTENT,
    BLANK_ELEM,
    UI_NO_MAIN_WINDOW,
)

# Filename of the cumulative invalid-element record (mirrors webui/WAE).
INVALID_ELEM_RECORD_FILENAME = 'invalid_elem_record.json'


def generate_negative_action_plans(
    gt_act_type: str,
    W: int,
    H: int,
    scale: int,
    gt_center: Optional[Sequence[float]] = None,
    neg_boxes: Optional[Sequence[Sequence[int]]] = None,
    direction: str = '',
    text: str = '',
    goal_status: str = ''
) -> List[List[List[str]]]:
    """Generate negative action plans for contrastive/evaluation purposes."""

    neg_action_plans: List[List[List[str]]] = []

    if neg_boxes is None:
        neg_boxes = []

    neg_actions_other_types: List[List[str]] = []

    if len(neg_boxes) >= 1:
        for act_type, template in zip(['click', 'long_press'], [CLICK_TEMPLATE, LONG_PRESS_TEMPLATE]):
            if gt_act_type == act_type:
                neg_cands: List[List[str]] = []
                selected_idxs = random.sample(list(range(len(neg_boxes))), min(len(neg_boxes), 9))
                for idx in selected_idxs:
                    box = neg_boxes[idx]
                    normalized_center = box2center(box, W, H, scale)
                    neg_click = template.format(target_x=normalized_center[0], target_y=normalized_center[1])
                    neg_cands.append([INCORRECT_CLICK_TARGET.format(action=act_type), neg_click])

                random.shuffle(neg_cands)
                neg_action_plans.append(neg_cands)
            else:
                if gt_center is not None:
                    normalized_center = [
                        max(0, min(scale, round(gt_center[0] / W * scale))),
                        max(0, min(scale, round(gt_center[1] / H * scale)))
                    ]
                else:
                    normalized_center = box2center(random.choice(neg_boxes), W, H, scale)

                neg_act = template.format(target_x=normalized_center[0], target_y=normalized_center[1])
                if gt_act_type in ['click', 'long_press']:
                    incorr_reason = INCORRECT_TOUCH_MODE + f". Should {gt_act_type} instead of {act_type}"
                else:
                    incorr_reason = INCORRECT_ACTION

                neg_actions_other_types.append([incorr_reason, neg_act])

    if gt_act_type == 'scroll':
        neg_cands = []
        for cand_dir in ['up', 'down', 'left', 'right']:
            if direction == cand_dir:
                continue
            swipe_dir, start, end = format_swiping_dual_points(cand_dir, scale=scale, scroll2swipe=False)
            neg_swipe = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=swipe_dir, distance="medium")
            neg_cands.append([INCORRECT_SWIPE_DIRECTION + f'. Should swipe {swipe_dir} instead of {cand_dir}', neg_swipe])

        random.shuffle(neg_cands)
        neg_action_plans.append(neg_cands)
    else:
        rand_dir = random.choice(['up', 'down', 'left', 'right'])
        swipe_dir, start, end = format_swiping_dual_points(rand_dir, scale=scale, scroll2swipe=False)
        neg_swipe = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=swipe_dir, distance="medium")
        neg_actions_other_types.append([INCORRECT_ACTION, neg_swipe])

    if gt_act_type == 'input_text':
        neg_cands = []
        used = [text]
        for _ in range(9):
            rand_text = random_substring(text)
            if rand_text in used:
                continue
            used.append(rand_text)
            neg_type = INPUT_TEMPLATE.format(text=rand_text)
            neg_cands.append([INCORRECT_INPUT_TEXT + f'. Should be "{text}" instead of "{rand_text}"', neg_type])

    app = random.choice([
        "Facebook", "Instagram", "WhatsApp", "TikTok", "Snapchat",
        "YouTube", "Twitter", "Spotify", "Netflix", "Zoom",
        "Google Maps", "Amazon", "Gmail", "Pinterest", "Reddit",
        "LinkedIn", "Telegram", "Discord", "Uber", "Airbnb"
    ])
    neg_open_app = OPEN_APP_TEMPLATE.format(app_name=app)
    incorr_reason = INCORRECT_OPEN_APP + f'. Should be {text} instead of {app}.' if gt_act_type == 'open_app' else INCORRECT_ACTION
    neg_actions_other_types.append([incorr_reason, neg_open_app])

    if gt_act_type != 'navigate_back':
        neg_back = NAVIGATE_BACK_TEMPLATE
        incorr_reason = INCORRECT_NAVIGATION_ACTION + '. Should navigate home instead of navigate back.' if 'home' in gt_act_type else INCORRECT_ACTION
        neg_actions_other_types.append([incorr_reason, neg_back])

    if gt_act_type != 'navigate_home':
        neg_home = NAVIGATE_HOME_TEMPLATE
        incorr_reason = INCORRECT_NAVIGATION_ACTION + '. Should navigate back instead of navigate home.' if 'back' in gt_act_type else INCORRECT_ACTION
        neg_actions_other_types.append([incorr_reason, neg_home])

    if gt_act_type == 'status':
        if goal_status == 'successful':
            neg_status = 'infeasible'
        elif goal_status == 'infeasible':
            neg_status = 'successful'
        else:
            neg_status = random.choice(['successful', 'infeasible'])
        incorr_reason = INCORRECT_STATUS
    else:
        neg_status = random.choice(['successful', 'infeasible'])
        incorr_reason = INCORRECT_ACTION

    neg_status = STATUS_TEMPLATE.format(goal_status=neg_status, answer='')
    neg_actions_other_types.append([incorr_reason, neg_status])
    random.shuffle(neg_actions_other_types)
    neg_action_plans.append(neg_actions_other_types)

    return neg_action_plans


# =============================================================================
# Episode cache: extract raw bytes from TFRecords into per-episode pickles
# =============================================================================
def _build_episode_cache(androidcontrol_root: str, cache_dir: str) -> int:
    """Walk the TFRecord stream once and write one pickle per episode_id.

    Idempotent and resumable: pre-existing ``{ep_id}.pkl`` files are skipped.
    Returns the number of episodes seen.
    """
    import tensorflow as tf  # lazy: keeps workers TF-free

    os.makedirs(cache_dir, exist_ok=True)

    filenames = tf.io.gfile.glob(os.path.join(androidcontrol_root, 'android_control*'))
    if not filenames:
        raise FileNotFoundError(f"no android_control* shards under {androidcontrol_root}")

    # Parallel reads + prefetch — GZIP decompression is the bottleneck and
    # AUTOTUNE will spin up multiple decompressors per file.
    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP',
        num_parallel_reads=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    cnt = 0
    for d in tqdm(dataset, total=15283, desc='building episode cache'):
        ex = tf.train.Example.FromString(d)
        ep_id = int(ex.features.feature['episode_id'].int64_list.value[0])
        cnt += 1

        cache_path = os.path.join(cache_dir, f'{ep_id}.pkl')
        if os.path.exists(cache_path):
            continue

        ep_data = {
            'episode_id': ep_id,
            'goal': bytes(ex.features.feature['goal'].bytes_list.value[0]),
            'accessibility_trees': [bytes(b) for b in ex.features.feature['accessibility_trees'].bytes_list.value],
            'screenshots': [bytes(b) for b in ex.features.feature['screenshots'].bytes_list.value],
            'actions': [bytes(b) for b in ex.features.feature['actions'].bytes_list.value],
            'step_instructions': [bytes(b) for b in ex.features.feature['step_instructions'].bytes_list.value],
        }

        tmp = cache_path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(ep_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(tmp, cache_path)

    return cnt


# =============================================================================
# Per-episode worker (pure Python — no TF)
# =============================================================================
def _process_episode(ep_data: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Process one episode and return aggregation deltas for the main process.

    The body mirrors the original sequential implementation; only the data
    source (``ep_data`` dict instead of ``tf.train.Example``) and the
    global-state accumulators (returned in the result dict) differ.
    """
    ep_id = ep_data['episode_id']

    SPLIT = cfg['SPLIT']
    SCALE = cfg['SCALE']
    LONGEST = cfg['LONGEST']
    PLANNING = cfg['PLANNING']
    ADD_DUMMY_COMPLETE = cfg['ADD_DUMMY_COMPLETE']
    INTENTGND = cfg['INTENTGND']
    POINT_FORMAT = cfg['POINT_FORMAT']
    USE_ACTION_REFEXP = cfg['USE_ACTION_REFEXP']
    DISCARD_OPENAPP = cfg['DISCARD_OPENAPP']
    DEVICE_TAG = cfg['DEVICE_TAG']
    DATASET_NAME = cfg['DATASET_NAME']
    SAVE_DIR = cfg['SAVE_DIR']
    RETAIN_STATUSBAR_IN_XML = cfg['RETAIN_STATUSBAR_IN_XML']
    RESUME = cfg['RESUME']
    DRAW = cfg['DRAW']
    TEXTLOC = cfg['TEXTLOC']
    OCR = cfg['OCR']
    WIDGETLIST = cfg['WIDGETLIST']
    PROB_BOX = cfg['PROB_BOX']

    samples: List[Dict[str, Any]] = []
    open_app_indices: set = set()
    invalid_UIs: List[List[Any]] = []
    # WAE/webui-style invalid-sample recording. Per-element rejections land in
    # ``invalid_elem_tracker[bucket]`` keyed by a stable ``sample_identifier``;
    # step-level rejections (e.g. step discarded for too many invalid elements,
    # or a UI with no main-app window) are counted in ``extra_invalid_counts``
    # since they don't correspond to a single node.
    invalid_elem_tracker: Dict[str, Set[str]] = {b: set() for b in _INVALID_BUCKETS}
    extra_invalid_counts: Dict[str, int] = defaultdict(int)
    intentgnd_cnt = 0
    planning_cnt = 0
    textloc_cnt = 0
    ocr_cnt = 0
    widgetlist_cnt = 0
    valid_step_cnt = 0
    total_step_cnt = 0
    valid_elem_cnt = 0

    # Timing breakdown (seconds, summed across steps of this episode).
    t_decode = 0.0
    t_parse = 0.0
    t_filter = 0.0
    t_write_img = 0.0
    t_write_xml = 0.0
    t_other = 0.0
    n_steps = 0

    task = ep_data['goal'].decode().replace(" .", ".").replace(" ,", ",")
    raw_AXTrees = ep_data['accessibility_trees']
    screenshots = ep_data['screenshots']
    actions = [x.decode() for x in ep_data['actions']]
    step_instructions = [x.decode() for x in ep_data['step_instructions']]
    total_step_cnt += len(step_instructions)

    new_actions, new_step_instructions, remained_indices = [], [], []
    is_openapp = False
    for step_idx, (act, inst) in enumerate(zip(actions, step_instructions)):
        if 'open_app' not in act or not is_openapp:
            new_actions.append(act)
            new_step_instructions.append(inst)
            remained_indices.append(step_idx)
            is_openapp = 'open_app' in act

    save_img_to = None

    for step_idx, act in enumerate(actions):
        if 'open_app' in act:
            open_app_indices.add(step_idx)

    action_meta: List[Dict[str, Any]] = []
    valid_sc_list: List[str] = []

    WAITING_STEP_INSTRUC = random.choice(WAIT_INSTRUC)

    # Track per-step locals that the trailing `else` of the outer loop reads.
    retained_history: List[str] = []
    short_nextimg_path = ''
    short_img_path = ''
    new_img_path = ''
    action_refexp = ''
    W = H = 0
    bboxes: List[List[int]] = []

    for step_idx, (raw_AXTree, screenshot) in enumerate(zip(raw_AXTrees, screenshots)):
        n_steps += 1
        t0 = time.perf_counter()
        forest = android_accessibility_forest_pb2.AndroidAccessibilityForest().FromString(raw_AXTree)
        t_parse += time.perf_counter() - t0
        total_step_cnt += 1
        action_meta.append({"action_type": "meaningless action"})

        if step_idx == 0:
            app_name = list(forest.windows[0].tree.nodes)[0].package_name
            save_img_to = os.path.join(SAVE_DIR, 'images', app_name, str(ep_id))
            os.makedirs(save_img_to, exist_ok=True)

        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(screenshot, dtype=np.uint8), cv2.IMREAD_COLOR)
        ORIG_H, ORIG_W = img.shape[:2]
        img, ratio = resize_image(img, LONGEST)
        t_decode += time.perf_counter() - t0

        new_img_path = os.path.join(save_img_to, f'{step_idx}.png')
        short_img_path = new_img_path[new_img_path.find(DATASET_NAME):]
        valid_sc_list.append(short_img_path)
        if not (RESUME and os.path.exists(new_img_path)):
            t0 = time.perf_counter()
            cv2.imwrite(new_img_path, img)
            t_write_img += time.perf_counter() - t0

        H, W = img.shape[:2]

        try:
            t0 = time.perf_counter()
            tree_str_lst, node_key_attrs_lst = parse_axtrees_proto(forest.windows, ratio)
            t_parse += time.perf_counter() - t0
        except Exception as e:
            print(e)
            invalid_UIs.append([ep_id, step_idx, new_img_path, str(e)])
            continue

        t_filter_start = time.perf_counter()
        new_node_key_attrs_lst = []
        checked_cnt = 0
        invalid_elem_cnt = 0
        for win_idx, node_key_attrs in enumerate(node_key_attrs_lst):
            window_H = forest.windows[win_idx].bounds_in_screen.bottom - forest.windows[win_idx].bounds_in_screen.top
            window_W = forest.windows[win_idx].bounds_in_screen.right - forest.windows[win_idx].bounds_in_screen.left
            window_H = round(window_H * ratio)
            window_W = round(window_W * ratio)
            # Tag each element with its source-window role. Element-level
            # grounding tasks (TextLoc, OCR, IntentGnd) skip status-bar /
            # system-overlay elements downstream — they're rarely useful
            # grounding targets and create cross-screen noise (the same
            # battery / wifi / clock icons repeat across every episode).
            # Heuristic mirrors the existing filter: a window short enough
            # to be a status bar gets the tag.
            is_statusbar_window = window_H <= MAX_STATUSBAR_HEIGHT
            if is_statusbar_window:
                for nka in node_key_attrs:
                    nka['is_statusbar'] = True
                new_node_key_attrs_lst.append(node_key_attrs)
                continue

            new_node_key_attrs, used_boxes = [], []
            checked_cnt += len(node_key_attrs)
            for node_key_attr in node_key_attrs:
                # Stable per-node identifier (mirrors WAE):
                #   {short_img_path}|{win_idx}|{box}
                sample_identifier = f"{short_img_path}|{win_idx}|{node_key_attr['box']}"
                is_invalid = False

                if node_key_attr['box'] in used_boxes:
                    invalid_elem_tracker[DUPLICATE_ELEMEMNT].add(sample_identifier)
                    is_invalid = True
                    continue

                x1, y1, x2, y2 = node_key_attr['box']
                if x1 < 0 or x2 > W or y1 < 0 or y2 > H or x1 >= x2 or y1 >= y2:
                    invalid_elem_tracker[INVALID_ELEM_BOX].add(sample_identifier)
                    is_invalid = True
                    continue
                if (x2 - x1) / W <= 0.005 or (y2 - y1) / H <= 0.005:
                    invalid_elem_tracker[TOO_SMALL_ELEMENT].add(sample_identifier)
                    is_invalid = True
                    continue
                if (x2 - x1) * (y2 - y1) / (window_H * window_W) >= 0.65:
                    invalid_elem_tracker[OVERSIZED_ELEMENT].add(sample_identifier)
                    is_invalid = True
                    continue
                if len(node_key_attr['node_desc']) == 0 and len(node_key_attr['resource_id']) == 0:
                    invalid_elem_tracker[INVALID_ELEM_CONTENT].add(sample_identifier)
                    is_invalid = True
                    continue
                if is_pure_color(img, [x1, y1, x2, y2]):
                    invalid_elem_tracker[BLANK_ELEM].add(sample_identifier)
                    is_invalid = True
                    invalid_elem_cnt += 1
                    continue

                used_boxes.append(node_key_attr['box'])
                if is_invalid:
                    node_key_attr['invalid'] = True
                new_node_key_attrs.append(node_key_attr)
            new_node_key_attrs_lst.append(new_node_key_attrs)

        t_filter += time.perf_counter() - t_filter_start
        valid_elem_cnt_this_step = sum(len(v) for v in new_node_key_attrs_lst)

        if invalid_elem_cnt >= 10:
            extra_invalid_counts['others'] += checked_cnt - invalid_elem_cnt
            invalid_UIs.append([ep_id, step_idx, new_img_path, 'Discard the traj due to too many invalid elements', f'invalid_elem_cnt:{invalid_elem_cnt}'])

            for node_info in new_node_key_attrs_lst[-1]:
                inter_color = (0, random.randint(150, 250), random.randint(150, 250))
                noninter_color = (random.randint(150, 250), random.randint(150, 250), 0)
                elem_box = node_info['box']
                is_interactable = node_info['is_interactable']
                if is_interactable:
                    cv2.rectangle(img, (round(elem_box[0]), round(elem_box[1])), (round(elem_box[2]), round(elem_box[3])), inter_color, 2)
                else:
                    cv2.rectangle(img, (elem_box[0], elem_box[1]), (elem_box[2], elem_box[3]), noninter_color, 2)
            continue

        save_xml_to = os.path.join(save_img_to, f'{step_idx}.xml')
        if not (RESUME and os.path.exists(save_xml_to)):
            t_xml_start = time.perf_counter()
            tree_str_list_wo_statusbar = []
            for win_idx, tree_str in enumerate(tree_str_lst):
                if (not RETAIN_STATUSBAR_IN_XML and forest.windows[win_idx].window_type != 1) or len(tree_str.strip()) == 0:
                    continue
                tree_str_list_wo_statusbar.append(tree_str)

            if len(tree_str_list_wo_statusbar) == 0:
                extra_invalid_counts[UI_NO_MAIN_WINDOW] += valid_elem_cnt_this_step
                continue

            windows_tree_str = []
            for win_idx, tree in enumerate(tree_str_list_wo_statusbar, start=1):
                window_tag = f'window_{win_idx}' + ('_statusbar' if forest.windows[win_idx-1].window_type != 1 else '_main-app')
                windows_tree_str.append(f'<{window_tag}>\n{tree}\n</{window_tag}>')

            tree_str = f"<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>\n" + '<hierarchy>\n' + '\n'.join(windows_tree_str) + '</hierarchy>'

            with open(save_xml_to, 'w') as f:
                f.write(tree_str)
            t_write_xml += time.perf_counter() - t_xml_start

        valid_elem_cnt += valid_elem_cnt_this_step

        # ---- Per-step element-level samples (textloc / ocr / widgetlist) ----
        # Mirrors the WAE pattern: flatten windows into one list, then generate
        # samples per element. Boxes are normalized to ``SCALE``; ``with_box``
        # is randomly toggled per element using ``PROB_BOX``. Textloc requires
        # unique text within the screen; widgetlist requires ≥2 valid nodes.
        if (TEXTLOC or OCR or WIDGETLIST) and valid_elem_cnt_this_step > 0:
            all_valid_nodes: List[Dict[str, Any]] = []
            for window_elems in new_node_key_attrs_lst:
                all_valid_nodes.extend(window_elems)

            # Element-level grounding/OCR tasks target a specific element and
            # only leaves are valid targets — a non-leaf container often
            # duplicates the text of its leaf child (e.g. a View wrapping a
            # TextView with the same content-desc), so iterating containers
            # would (a) generate noisy duplicate samples and (b) make the
            # ambiguity check falsely reject genuinely-unique leaf targets.
            # The candidate group for uniqueness is therefore leaves only.
            # We also exclude elements from the top status bar (battery /
            # wifi / clock / notification icons): those repeat across every
            # episode, dominate the sample distribution, and aren't useful
            # grounding targets for app-level tasks. ``is_statusbar`` is
            # tagged upstream in the per-window filter.
            leaf_nodes: List[Dict[str, Any]] = [
                n for n in all_valid_nodes
                if n.get('is_leaf') and not n.get('is_statusbar')
            ]
            all_leaf_descs = [(n.get('node_desc') or '').strip() for n in leaf_nodes]

            for elem in leaf_nodes:
                x1, y1, x2, y2 = elem['box']
                # Normalized box + center.
                nb = [
                    min(max(0, round(x1 / W * SCALE)), SCALE - 1),
                    min(max(0, round(y1 / H * SCALE)), SCALE - 1),
                    min(max(0, round(x2 / W * SCALE)), SCALE - 1),
                    min(max(0, round(y2 / H * SCALE)), SCALE - 1),
                ]
                cx = min(max(0, round(((x1 + x2) / 2) / W * SCALE)), SCALE - 1)
                cy = min(max(0, round(((y1 + y2) / 2) / H * SCALE)), SCALE - 1)
                box_str = f'({nb[0]},{nb[1]},{nb[2]},{nb[3]})'
                center_str = f'({cx},{cy})'

                elem_text = (elem.get('node_desc') or '').strip()
                if not elem_text or len(elem_text) > 200:
                    continue
                with_box = random.random() < PROB_BOX
                loc_str = box_str if with_box else center_str

                # Text localization: only when the text is unique among leaf
                # candidates on this screen so the model has an unambiguous
                # target to locate.
                if TEXTLOC and all_leaf_descs.count(elem_text) <= 1:
                    task_id = f'autogui_{DATASET_NAME}_textloc_{ep_id}-{step_idx}-{textloc_cnt}'
                    sample = make_textloc_sample(
                        task_id, text=elem_text,
                        loc=box_str if with_box else center_str,
                        output_tag=WITHBOX_TAG if with_box else WITHPOINT_TAG,
                    )
                    sample.update({
                        'task_attr': elem_text,
                        'unnormalized_box': [x1, y1, x2, y2],
                        'image': short_img_path,
                        'wxh': f"{W}x{H}",
                        'ep_id': ep_id,
                        'step_id': step_idx,
                    })
                    samples.append(sample)
                    textloc_cnt += 1

                if OCR:
                    task_id = f'autogui_{DATASET_NAME}_ocr_{ep_id}-{step_idx}-{ocr_cnt}'
                    sample = make_ocr_sample(
                        task_id, text=elem_text, loc=loc_str,
                        output_tag=WITHBOX_TAG if with_box else WITHPOINT_TAG,
                    )
                    sample.update({
                        'task_attr': loc_str,
                        'unnormalized_box': [x1, y1, x2, y2],
                        'image': short_img_path,
                        'wxh': f"{W}x{H}",
                        'ep_id': ep_id,
                        'step_id': step_idx,
                    })
                    samples.append(sample)
                    ocr_cnt += 1

            # Widget list: one sample per step when there are ≥2 valid nodes.
            # 80 nodes is the cap from webui to keep prompts in context window.
            if WIDGETLIST and 2 <= len(all_valid_nodes) <= 80:
                items = []
                for elem in all_valid_nodes:
                    ux1, uy1, ux2, uy2 = elem['box']
                    nb = [
                        min(max(0, round(ux1 / W * SCALE)), SCALE - 1),
                        min(max(0, round(uy1 / H * SCALE)), SCALE - 1),
                        min(max(0, round(ux2 / W * SCALE)), SCALE - 1),
                        min(max(0, round(uy2 / H * SCALE)), SCALE - 1),
                    ]
                    desc = (elem.get('node_desc') or '').strip(' ,.')
                    clean_desc = ' '.join(s for s in desc.split('\n') if s.strip())
                    tag = elem.get('class') or 'View'
                    items.append((tag, clean_desc, [ux1, uy1, ux2, uy2], nb))

                # Reading order: top-to-bottom then left-to-right.
                items.sort(key=lambda x: (x[2][1] + x[2][3], x[2][0] + x[2][2]))
                elem_list_str = '\n'.join(
                    f"{i} {tag} '{desc}' ({nb[0]},{nb[1]},{nb[2]},{nb[3]})"
                    for i, (tag, desc, _, nb) in enumerate(items)
                )

                task_id = f'autogui_{DATASET_NAME}_widgetlist_{ep_id}-{step_idx}-{widgetlist_cnt}'
                sample = make_widgetlist_sample(task_id, elem_list=elem_list_str)
                sample.update({
                    'task_attr': None,
                    'unnormalized_box': [it[2] for it in items],
                    'image': short_img_path,
                    'wxh': f"{W}x{H}",
                    'ep_id': ep_id,
                    'step_id': step_idx,
                })
                samples.append(sample)
                widgetlist_cnt += 1

        click_action_added = False
        if (PLANNING or INTENTGND) and step_idx in remained_indices and valid_elem_cnt_this_step:
            cur_action = ast.literal_eval(actions[step_idx])

            if SPLIT == 'train' and DISCARD_OPENAPP and 'open_app' in actions[step_idx]:
                action_meta[-1] = cur_action
            else:
                action_type = cur_action['action_type']

                if INTENTGND and action_type not in ['click', 'long_press']:
                    continue

                main_app_elems = []
                for window_elems_lst in new_node_key_attrs_lst:
                    main_app_elems.extend(window_elems_lst)

                bboxes = [x['box'] for x in main_app_elems]

                if action_type in ['click', 'long_press']:
                    click_x, click_y = int(cur_action["x"]) * ratio, int(cur_action["y"]) * ratio
                    if SCALE == -1:
                        normalized_center = [round(click_x), round(click_y)]
                    else:
                        normalized_center = [max(0, min(SCALE-1, round(click_x / W * SCALE))), max(0, min(SCALE-1, round(click_y / H * SCALE)))]

                    target = f'({normalized_center[0]},{normalized_center[1]})'

                    if action_type == 'click':
                        action_str = CLICK_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1])
                    else:
                        action_str = LONG_PRESS_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1])

                    interacted_box, index = find_smallest_box_containing_point(np.array([click_x, click_y]), boxes=np.array(bboxes))

                    if index is None:
                        #print(f"Bad bbox: cannot locate the bbox")
                        continue
                    elif not (interacted_box[0] <= click_x <= interacted_box[2] and interacted_box[1] <= click_y <= interacted_box[3]):
                        print(f"Bad bbox: [{click_x}, {click_y}] not in bbox {interacted_box}")
                        continue
                    else:
                        interacted_node = main_app_elems[index]

                        if DRAW:
                            print(step_instructions[step_idx], cur_action)
                            cv2.circle(img, (round(click_x), round(click_y)), 5, (0, 255, 0), 2)
                            cv2.circle(img, (round(click_x), round(click_y)), 3, (0, 255, 0), -3)
                            cv2.rectangle(img, (interacted_box[0], interacted_box[1]), (interacted_box[2], interacted_box[3]), (0, 0, 255), 3)
                            cv2.imwrite("test.png", img)

                        if not ('Open' in step_instructions[step_idx] and 'app' in step_instructions[step_idx]) and not INTENTGND:
                            step_instructions[step_idx] = f'Click on "{interacted_node["node_desc"]}" {interacted_node["class"]}.'

                        action_refexp = random.choice(ACTION_PREFIXES[action_type]['specific']) + f' {interacted_node["class"]} "{interacted_node["node_desc"].strip()}"'

                        assert 0 <= interacted_box[0] <= W + 10 and 0 <= interacted_box[2] <= W + 10
                        action_meta[-1] = {"action_type": "DualPoint", "touch_point": [click_x, click_y], "lift_point": [click_x, click_y], "typed_text": f"TAP:{interacted_node['class']} Box:[{interacted_box[0]},{interacted_box[1]},{interacted_box[2]},{interacted_box[3]}]", "action_refexp": action_refexp}
                        click_action_added = True

                    if INTENTGND and SPLIT != 'test':
                        # Skip when the target's description is ambiguous on
                        # this screen. Only leaves are valid grounding
                        # targets; non-leaf containers often duplicate their
                        # leaf child's text, which would falsely flag unique
                        # leaves as ambiguous. So both the target check and
                        # the candidate group are restricted to leaves.
                        # Status-bar elements (battery / wifi / notification
                        # icons) are also excluded — both as targets and as
                        # candidates — since those repeat across episodes
                        # and aren't useful grounding signal for app-level
                        # intents.
                        target_desc = (interacted_node.get('node_desc') or '').strip()
                        target_is_leaf = bool(interacted_node.get('is_leaf'))
                        target_is_statusbar = bool(interacted_node.get('is_statusbar'))
                        leaf_descs_step = [
                            (n.get('node_desc') or '').strip()
                            for n in main_app_elems
                            if n.get('is_leaf') and not n.get('is_statusbar')
                        ]
                        if (target_is_leaf and not target_is_statusbar
                                and target_desc
                                and leaf_descs_step.count(target_desc) <= 1):
                            intent = lower_first_letter(action_refexp if USE_ACTION_REFEXP else step_instructions[step_idx])
                            # Bbox format switching: with prob PROB_BOX, emit
                            # the full normalized bbox of the interacted
                            # element; otherwise emit its center point.
                            with_box = random.random() < PROB_BOX
                            if with_box:
                                normalized_box = [
                                    min(max(0, round(interacted_box[0] / W * SCALE)), SCALE - 1),
                                    min(max(0, round(interacted_box[1] / H * SCALE)), SCALE - 1),
                                    min(max(0, round(interacted_box[2] / W * SCALE)), SCALE - 1),
                                    min(max(0, round(interacted_box[3] / H * SCALE)), SCALE - 1),
                                ]
                                ig_loc = normalized_box
                                ig_tag = WITHBOX_TAG
                            else:
                                ig_loc = normalized_center
                                ig_tag = WITHPOINT_TAG
                            sample = make_intentgnd_sample(task_id=f"autogui_{DATASET_NAME}_intentgnd_{ep_id}-{step_idx}", intent=intent, loc=ig_loc, output_tag=ig_tag, point_format=POINT_FORMAT)
                            sample['task_attr'], sample['image'] = intent, short_img_path
                            samples.append(sample)
                            intentgnd_cnt += 1
                elif action_type == 'scroll':
                    direction = cur_action['direction']
                    distance = random.choice(["medium", "short", "long"])
                    direction, start, end = format_swiping_dual_points(direction, scale=SCALE, scroll2swipe=True, distance=distance)
                    action_str = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=direction, distance=distance)
                    action_meta[-1] = ast.literal_eval(action_str)
                    step_instructions[step_idx] = revise_swipe_action(step_instructions[step_idx], f"Swipe {direction}")
                    action_refexp = random.choice(SWIPE_PHRASES).format(direction=direction)
                elif action_type == 'navigate_back':
                    action_str = NAVIGATE_BACK_TEMPLATE
                    action_refexp = random.choice(NAVIGATE_BACK_PREFIXES)
                elif action_type == 'navigate_home':
                    action_str = NAVIGATE_HOME_TEMPLATE
                    action_refexp = random.choice(NAVIGATE_HOME_PREFIXES)
                elif action_type == 'wait':
                    step_instructions[step_idx] = WAITING_STEP_INSTRUC
                    action_str = WAIT_TEMPLATE
                    action_refexp = random.choice(WAIT_INSTRUC)
                elif action_type == 'input_text':
                    text = cur_action['text'].strip(' \\').replace("\n", "\\n").replace('"', '\\"')
                    action_str = INPUT_TEMPLATE.format(text=text)
                    action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target="the text box")
                elif action_type == 'status':
                    action_str = STATUS_TEMPLATE.format(goal_status=cur_action['goal_status'], answer='')
                    action_refexp = random.choice(TASK_STATUS_SENTENCES[cur_action['goal_status']])
                elif action_type == 'open_app':
                    action_str = OPEN_APP_TEMPLATE.format(app_name=cur_action['app_name'])
                    action_refexp = random.choice(OPEN_APP_PREFIXES).format(app_name=cur_action['app_name'])
                else:
                    action_str = actions[step_idx]
                    action_meta[-1] = cur_action

                retained_idxs, retained_history = keep_unique_actions(step_instructions[:step_idx])
                history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(retained_history, start=max(1, len(retained_history) - MAX_PREV_ACT + 1))) if len(retained_history) > 0 else 'None'

                action_type = ast.literal_eval(action_str)['action_type']

                task_attr = ast.literal_eval(actions[step_idx])
                if click_action_added:
                    task_attr['x'] *= ratio
                    task_attr['y'] *= ratio
                    task_attr['bbox'] = interacted_box

                if USE_ACTION_REFEXP:
                    action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"

                if PLANNING:
                    sample = make_actionplanning_sample(
                        task_id=f"autogui_androidcontrol_planning_{ep_id}-{step_idx}-H",
                        global_task=task,
                        history=history_str,
                        gt_action='Action: ' + action_str,
                        device_tag=DEVICE_TAG,
                        use_action_refexp=USE_ACTION_REFEXP,
                    )
                    sample['ep_id'], sample['step_id'], sample['task'], sample['step_instruction'], sample['action_refexp'], sample["action_type"], sample["history"], sample["image"], sample["task_attr"], sample["wxh"], sample['orig_wxh'], sample['bboxes'], sample['#total_steps'] = ep_id, step_idx, task, step_instructions[step_idx], action_refexp, action_type, retained_history, short_img_path, task_attr, f"{W}x{H}", f"{ORIG_W}x{ORIG_H}", [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bboxes], len(raw_AXTrees) + ADD_DUMMY_COMPLETE

                    samples.append(sample)
                    planning_cnt += 1

                    assert os.path.exists(os.path.join(os.path.dirname(SAVE_DIR), sample["image"]))

                    if step_idx < min(len(raw_AXTrees), len(screenshots)) - 1:
                        next_img_path = os.path.join(save_img_to, f'{step_idx+1}.png')
                    else:
                        next_img_path = new_img_path
                    short_nextimg_path = next_img_path[next_img_path.find(DATASET_NAME):]

                    if not (step_idx > 1 and step_instructions[step_idx] == step_instructions[step_idx-1]):
                        sample = make_actionplanning_sample(task_id=f"autogui_androidcontrol_planning_{ep_id}-{step_idx}-HL", global_task=task, history=history_str, gt_action='Action: ' + action_str, step_instruction=f"The next step instruction: {step_instructions[step_idx]}\n", device_tag=DEVICE_TAG, use_action_refexp=USE_ACTION_REFEXP)

                        sample['ep_id'], sample['step_id'], sample['task'], sample['step_instruction'], sample['action_refexp'], sample["action_type"], sample["history"], sample["image"], sample['next_image'], sample["task_attr"], sample["wxh"], sample['bboxes'], sample['#total_steps'] = ep_id, step_idx, task, step_instructions[step_idx], action_refexp, action_type, retained_history, short_img_path, short_nextimg_path, task_attr, f"{W}x{H}", [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bboxes], len(raw_AXTrees) + ADD_DUMMY_COMPLETE
                        samples.append(sample)
                        planning_cnt += 1

                    valid_step_cnt += 1

    else:
        if PLANNING and step_idx == len(raw_AXTrees) - 1:
            step_instruc_complete = "Task completed"
            last_step_history = retained_history + ([step_instructions[-1]] if len(step_instructions) else [])
            history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(last_step_history, start=max(1, len(last_step_history) - MAX_PREV_ACT + 1))) if len(last_step_history) > 0 else 'None'
            action_str = STATUS_TEMPLATE.format(goal_status='successful', answer='')
            task_attr = eval(action_str)

            sample = make_actionplanning_sample(task_id=f"autogui_androidcontrol_planning_{ep_id}-{step_idx+1}-HL", global_task=task, history=history_str, gt_action=f'Action: {action_str}', step_instruction=f"The next step instruction: {step_instruc_complete}\n", device_tag=DEVICE_TAG, use_action_refexp=USE_ACTION_REFEXP)
            sample['ep_id'], sample['step_id'], sample['task'], sample['step_instruction'], sample['action_refexp'], sample["action_type"], sample["history"], sample["image"], sample['next_image'], sample["task_attr"], sample["wxh"], sample['bboxes'], sample['#total_steps'] = ep_id, step_idx+1, task, step_instruc_complete, action_refexp, 'status', last_step_history, short_img_path, short_nextimg_path, task_attr, f"{W}x{H}", [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bboxes], len(raw_AXTrees) + ADD_DUMMY_COMPLETE
            samples.append(sample)
            planning_cnt += 1

            sample = make_actionplanning_sample(task_id=f"autogui_androidcontrol_planning_{ep_id}-{step_idx+1}-H", global_task=task, history=history_str, gt_action=f'Action: {action_str}', device_tag=DEVICE_TAG, use_action_refexp=USE_ACTION_REFEXP)
            sample['ep_id'], sample['step_id'], sample['task'], sample['step_instruction'], sample['action_refexp'], sample["action_type"], sample["history"], sample["image"], sample['next_image'], sample["task_attr"], sample["wxh"], sample['bboxes'], sample['#total_steps'] = ep_id, step_idx+1, task, step_instruc_complete, action_refexp, 'status', last_step_history, short_img_path, short_nextimg_path, task_attr, f"{W}x{H}", [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bboxes], len(raw_AXTrees) + ADD_DUMMY_COMPLETE
            samples.append(sample)
            planning_cnt += 1

    assert len(action_meta) == len(screenshots)

    meta_file = os.path.join(save_img_to, 'meta.json')
    if not (os.path.exists(meta_file) and RESUME):
        with open(meta_file, 'w') as f:
            json.dump(action_meta, f)

    return {
        'samples': samples,
        'open_app_indices': open_app_indices,
        'invalid_UIs': invalid_UIs,
        # Sets are sent as lists to keep the pickle small and the IPC simple;
        # the parent unions them back into a global ``Dict[str, Set[str]]``.
        'invalid_elem_tracker': {k: list(v) for k, v in invalid_elem_tracker.items() if v},
        'extra_invalid_counts': dict(extra_invalid_counts),
        'intentgnd_cnt': intentgnd_cnt,
        'planning_cnt': planning_cnt,
        'textloc_cnt': textloc_cnt,
        'ocr_cnt': ocr_cnt,
        'widgetlist_cnt': widgetlist_cnt,
        'valid_step_cnt': valid_step_cnt,
        'total_step_cnt': total_step_cnt,
        'valid_elem_cnt': valid_elem_cnt,
        '_timing': {
            'decode_s': t_decode,
            'parse_s': t_parse,
            'filter_s': t_filter,
            'write_img_s': t_write_img,
            'write_xml_s': t_write_xml,
            'n_steps': n_steps,
        },
    }


def _process_episode_id(ep_id: int, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Worker entry: load the cached episode pickle and process it.

    Returns the delta dict, or an error stub on exception so the main process
    can report and continue rather than killing the whole pool.
    """
    cache_path = os.path.join(cfg['cache_dir'], f'{ep_id}.pkl')
    try:
        t0 = time.perf_counter()
        with open(cache_path, 'rb') as f:
            ep_data = pickle.load(f)
        load_s = time.perf_counter() - t0
        cache_bytes = os.path.getsize(cache_path)

        t1 = time.perf_counter()
        result = _process_episode(ep_data, cfg)
        total_s = time.perf_counter() - t1

        # Augment the timing block with load/total/size.
        timing = result.get('_timing', {})
        timing['load_s'] = load_s
        timing['total_s'] = total_s
        timing['cache_bytes'] = cache_bytes
        timing['ep_id'] = ep_id
        result['_timing'] = timing
        return result
    except Exception as e:
        tb = traceback.format_exc()
        # Echo to stderr so the failure is visible live; the full traceback
        # is also returned so the main process can group/summarize them via
        # ``summarize_errors`` in the final info.json.
        sys.stderr.write(tb)
        return {'error': str(e), 'traceback': tb, 'ep_id': ep_id}


# =============================================================================
# Multiprocessing-safe entry point
# =============================================================================
def main() -> None:
    SPLIT = ['train', 'test'][0]

    RESUME = True

    DEBUG = False

    PLANNING = False; ADD_DUMMY_COMPLETE = True
    INTENTGND = True; POINT_FORMAT = ['plain', 'qwen'][0]
    # Element-level tasks (mirroring make_webui_data / make_WAE_data).
    TEXTLOC = True
    OCR = True
    WIDGETLIST = True
    PROB_BOX = 0.3  # Probability of using a bounding box instead of a point as loc.

    SCALE = 1000
    LONGEST = -1

    RETAIN_STATUSBAR_IN_XML = True

    DISCARD_OPENAPP = False

    USE_ACTION_REFEXP = False

    DEVICE_TAG = "Android"

    DRAW = True
    
    ANDROIDCONTROL_ROOT = "/volume/pt-coder/users/gji/projects/UIPro/data/AndroidControl_raw"

    DATASET_NAME = "AndroidControl" + ("_debug" if DEBUG else "") + ("_test" if SPLIT == 'test' else "")

    SAVE_DIR = f"/volume/pt-coder/users/gji/projects/UIPro/data/{DATASET_NAME}"
    print(f"save imgs to {SAVE_DIR}")

    with open(os.path.join(ANDROIDCONTROL_ROOT, "splits.json"), "r") as f:
        splits = json.load(f)




    save_invalidcache_to = os.path.join(SAVE_DIR, "invalid_cache")
    os.makedirs(save_invalidcache_to, exist_ok=True)

    # ---- Episode cache: build on first run, reuse afterwards -------------
    cache_dir = os.path.join(ANDROIDCONTROL_ROOT, EPISODE_CACHE_DIRNAME)
    cache_marker = os.path.join(cache_dir, CACHE_COMPLETE_MARKER)

    if not os.path.exists(cache_marker):
        print(f"building episode cache at {cache_dir} (first run, one-time)")
        n = _build_episode_cache(ANDROIDCONTROL_ROOT, cache_dir)
        with open(cache_marker, 'w') as f:
            f.write(f"episodes={n}\n")
        print(f"cache built: {n} episodes at {cache_dir}")
    else:
        print(f"using cached episodes at {cache_dir} (delete {CACHE_COMPLETE_MARKER} to rebuild)")

    # ---- Resolve the set of ep_ids to process ---------------------------
    split_ep_ids = set(int(x) for x in splits[SPLIT])
    cached_ids: set = set()
    for fname in os.listdir(cache_dir):
        if fname.endswith('.pkl'):
            try:
                cached_ids.add(int(fname[:-4]))
            except ValueError:
                pass
    ep_ids_to_process = sorted(cached_ids & split_ep_ids)

    if DEBUG:
        ep_ids_to_process = ep_ids_to_process[:40]

    missing = split_ep_ids - cached_ids
    if missing:
        print(f"warning: {len(missing)} ep_ids in split={SPLIT} are not in the cache "
              f"(first {min(5, len(missing))}: {sorted(missing)[:5]})")

    print(f"processing {len(ep_ids_to_process)} episodes (split={SPLIT}, workers={NUM_WORKERS})")

    # DRAW writes a shared "test.png" — racy with multiple workers. Disable.
    cfg = {
        'SPLIT': SPLIT,
        'SCALE': SCALE,
        'LONGEST': LONGEST,
        'PLANNING': PLANNING,
        'ADD_DUMMY_COMPLETE': ADD_DUMMY_COMPLETE,
        'INTENTGND': INTENTGND,
        'POINT_FORMAT': POINT_FORMAT,
        'USE_ACTION_REFEXP': USE_ACTION_REFEXP,
        'DISCARD_OPENAPP': DISCARD_OPENAPP,
        'DEVICE_TAG': DEVICE_TAG,
        'DATASET_NAME': DATASET_NAME,
        'SAVE_DIR': SAVE_DIR,
        'RETAIN_STATUSBAR_IN_XML': RETAIN_STATUSBAR_IN_XML,
        'RESUME': RESUME,
        'DRAW': DRAW if NUM_WORKERS <= 1 else False,
        'cache_dir': cache_dir,
        'TEXTLOC': TEXTLOC,
        'OCR': OCR,
        'WIDGETLIST': WIDGETLIST,
        'PROB_BOX': PROB_BOX,
    }

    # ---- Aggregators ----------------------------------------------------
    samples: List[Dict[str, Any]] = []
    open_app_indices: set = set()
    invalid_UIs: List[List[Any]] = []
    # WAE/webui-style cumulative invalid-elem record: bucket → set of IDs.
    invalid_elem_tracker: Dict[str, Set[str]] = {b: set() for b in _INVALID_BUCKETS}
    extra_invalid_counts: Dict[str, int] = defaultdict(int)
    # Worker exceptions in the [identifier, traceback] format expected by
    # ``summarize_errors``. We never re-raise so a bad episode never kills
    # the pool — the failure is captured here and surfaced in info.json.
    errors: List[List[str]] = []
    intentgnd_cnt = 0
    planning_cnt = 0
    textloc_cnt = 0
    ocr_cnt = 0
    widgetlist_cnt = 0
    valid_step_cnt = 0
    total_step_cnt = 0
    valid_elem_cnt = 0
    total_ep_cnt = 0
    valid_ep_cnt = 0

    # Rolling timing aggregates for logging.
    LOG_EVERY = 200
    timing_totals = {
        'load_s': 0.0, 'decode_s': 0.0, 'parse_s': 0.0, 'filter_s': 0.0,
        'write_img_s': 0.0, 'write_xml_s': 0.0, 'total_s': 0.0,
        'cache_bytes': 0, 'n_steps': 0, 'n_eps': 0,
    }
    wall_start = time.perf_counter()
    last_log_at = wall_start

    def _absorb(result: Optional[Dict[str, Any]]) -> None:
        nonlocal intentgnd_cnt, planning_cnt, textloc_cnt, ocr_cnt, widgetlist_cnt
        nonlocal valid_step_cnt, total_step_cnt, valid_elem_cnt, valid_ep_cnt, last_log_at
        if result is None:
            return
        if 'error' in result:
            ep_id = result.get('ep_id')
            print(f"[worker error ep_id={ep_id}] {result['error']}")
            errors.append([f'ep_id={ep_id}', result.get('traceback', result['error'])])
            return
        samples.extend(result['samples'])
        open_app_indices.update(result['open_app_indices'])
        invalid_UIs.extend(result['invalid_UIs'])
        # Union the set-based invalid-elem record.
        for bucket, ids in result.get('invalid_elem_tracker', {}).items():
            invalid_elem_tracker.setdefault(bucket, set()).update(ids)
        # Sum the step-level extra counts.
        for k, v in result.get('extra_invalid_counts', {}).items():
            extra_invalid_counts[k] += v
        intentgnd_cnt += result['intentgnd_cnt']
        planning_cnt += result['planning_cnt']
        textloc_cnt += result.get('textloc_cnt', 0)
        ocr_cnt += result.get('ocr_cnt', 0)
        widgetlist_cnt += result.get('widgetlist_cnt', 0)
        valid_step_cnt += result['valid_step_cnt']
        total_step_cnt += result['total_step_cnt']
        valid_elem_cnt += result['valid_elem_cnt']
        valid_ep_cnt += 1

        # Accumulate timing.
        timing = result.get('_timing') or {}
        for k in ('load_s', 'decode_s', 'parse_s', 'filter_s', 'write_img_s', 'write_xml_s', 'total_s'):
            timing_totals[k] += timing.get(k, 0.0) or 0.0
        timing_totals['cache_bytes'] += timing.get('cache_bytes', 0)
        timing_totals['n_steps'] += timing.get('n_steps', 0)
        timing_totals['n_eps'] += 1

        # Rolling per-block log.
        if timing_totals['n_eps'] % LOG_EVERY == 0:
            n = timing_totals['n_eps']
            wall = time.perf_counter() - wall_start
            block_wall = time.perf_counter() - last_log_at
            last_log_at = time.perf_counter()
            avg = {k: timing_totals[k] / max(n, 1) for k in ('load_s', 'decode_s', 'parse_s', 'filter_s', 'write_img_s', 'write_xml_s', 'total_s')}
            avg_steps = timing_totals['n_steps'] / max(n, 1)
            avg_mb = (timing_totals['cache_bytes'] / max(n, 1)) / (1024 * 1024)
            eps_per_sec = LOG_EVERY / block_wall if block_wall > 0 else 0.0
            print(
                f"\n[timing @ {n} eps, wall={wall:.0f}s, block={block_wall:.1f}s, {eps_per_sec:.2f} eps/s]\n"
                f"  per-episode (avg): load={avg['load_s']*1000:.0f}ms ({avg_mb:.1f}MB) "
                f"decode={avg['decode_s']*1000:.0f}ms parse={avg['parse_s']*1000:.0f}ms "
                f"filter={avg['filter_s']*1000:.0f}ms write_img={avg['write_img_s']*1000:.0f}ms "
                f"write_xml={avg['write_xml_s']*1000:.0f}ms total={avg['total_s']*1000:.0f}ms (steps={avg_steps:.1f})"
            )

    # ---- Dispatch -------------------------------------------------------
    print(f"[dispatch] {len(ep_ids_to_process)} episodes via "
          f"{'sequential' if NUM_WORKERS <= 1 else f'{NUM_WORKERS} spawn workers'}")
    if NUM_WORKERS <= 1:
        for ep_id in tqdm(ep_ids_to_process, desc='processing'):
            total_ep_cnt += 1
            _absorb(_process_episode_id(ep_id, cfg))
    else:
        ctx = mp.get_context('spawn')
        worker = partial(_process_episode_id, cfg=cfg)
        # chunksize=4: workers grab tasks in small batches; reduces IPC
        # overhead vs chunksize=1 without blowing load balancing.
        with ctx.Pool(NUM_WORKERS) as pool:
            for result in tqdm(
                pool.imap_unordered(worker, ep_ids_to_process, chunksize=4),
                total=len(ep_ids_to_process),
                desc='processing',
            ):
                total_ep_cnt += 1
                _absorb(result)

    # Final timing summary.
    n = timing_totals['n_eps']
    if n > 0:
        wall = time.perf_counter() - wall_start
        avg = {k: timing_totals[k] / n for k in ('load_s', 'decode_s', 'parse_s', 'filter_s', 'write_img_s', 'write_xml_s', 'total_s')}
        print(
            f"\n[timing FINAL: {n} eps in {wall:.0f}s = {n/wall:.2f} eps/s]\n"
            f"  per-episode (avg): load={avg['load_s']*1000:.0f}ms "
            f"decode={avg['decode_s']*1000:.0f}ms parse={avg['parse_s']*1000:.0f}ms "
            f"filter={avg['filter_s']*1000:.0f}ms write_img={avg['write_img_s']*1000:.0f}ms "
            f"write_xml={avg['write_xml_s']*1000:.0f}ms total={avg['total_s']*1000:.0f}ms"
        )

    # ---- Reporting ------------------------------------------------------
    # Combine set-based per-element rejections with step-level extra counts
    # for the breakdown.
    invalid_breakdown = {k: len(v) for k, v in invalid_elem_tracker.items() if v}
    for k, v in extra_invalid_counts.items():
        invalid_breakdown[k] = invalid_breakdown.get(k, 0) + v
    num_invalid_elem = sum(invalid_breakdown.values())
    iterated_elem_cnt = valid_elem_cnt + num_invalid_elem

    # Persist the cumulative invalid-element record (set of identifiers per
    # bucket) so downstream tooling can match webui/WAE conventions.
    invalid_record_path = os.path.join(SAVE_DIR, INVALID_ELEM_RECORD_FILENAME)
    with open(invalid_record_path, 'w') as f:
        json.dump({k: sorted(v) for k, v in invalid_elem_tracker.items() if v}, f, indent=2)
    print(f"wrote invalid-element record to {invalid_record_path}")

    unique_img_cnt = len(set(x['image'] for x in samples))

    report = (
        f"#Unique images: {unique_img_cnt} | "
        f"Valid eps / Total eps: {valid_ep_cnt} / {total_ep_cnt} = "
        f"{valid_ep_cnt/max(total_ep_cnt,1):.2f} | "
        f"Valid Steps / Total steps: {valid_step_cnt} / {total_step_cnt} = "
        f"{valid_step_cnt / max(total_step_cnt,1):.2f} | "
        f"Valid elems / Total iterated elems: {valid_elem_cnt} / {iterated_elem_cnt} = "
        f"{valid_elem_cnt / max(iterated_elem_cnt,1):.2f}"
    )

    # ---- Per-task analytics (matches the MultiUI / MobileViews pattern) -----
    # Action-type distribution for planning samples (extracted from the gpt
    # value of each ``*_planning_*`` sample).
    act_stats: Dict[str, int] = defaultdict(int)
    # Per-task box-vs-point split, derived from each sample's question. This
    # is the post-hoc audit of ``PROB_BOX`` — the textloc/ocr/intentgnd
    # output should be ~PROB_BOX with bbox and ~(1-PROB_BOX) with point.
    task_loc_format: Dict[str, Counter] = defaultdict(Counter)
    # Top apps by sample count (image path is ``…/AndroidControl/images/
    # {package}/{ep_id}/{step}.png``; the 3rd component is the app name).
    package_cnt: Counter = Counter()
    # Samples per episode — used for a tiny stats line in extra_sections.
    samples_per_episode: Counter = Counter()

    for x in samples:
        sid = x['id']
        if 'planning' in sid:
            m = re.search(r'"action_type":\s*"([^"]+)"', x['conversations'][1]['value'])
            if m:
                act_stats[m.group(1)] += 1

        for task_tag in ('textloc', 'ocr', 'intentgnd'):
            if f'_{task_tag}_' in sid:
                q = x['conversations'][0]['value']
                if '(with bbox)' in q:
                    task_loc_format[task_tag]['box'] += 1
                elif '(with point)' in q:
                    task_loc_format[task_tag]['point'] += 1
                else:
                    task_loc_format[task_tag]['none'] += 1
                break

        # Extract the app package from the image path.
        img_path = x.get('image')
        if isinstance(img_path, str):
            parts = img_path.split('/')
            # Expected layout: ``AndroidControl[…]/images/<package>/<ep>/<step>.png``
            if len(parts) >= 3 and parts[-4] == 'images':
                package_cnt[parts[-3]] += 1

        ep_id_x = x.get('ep_id')
        if ep_id_x is not None:
            samples_per_episode[ep_id_x] += 1

    if samples_per_episode:
        per_ep_vals = list(samples_per_episode.values())
        per_ep_vals.sort()
        n_eps = len(per_ep_vals)
        samples_per_episode_stats = {
            'episodes_with_samples': n_eps,
            'min': per_ep_vals[0],
            'p50': per_ep_vals[n_eps // 2],
            'p90': per_ep_vals[int(n_eps * 0.9)],
            'max': per_ep_vals[-1],
            'mean': round(sum(per_ep_vals) / n_eps, 2),
        }
    else:
        samples_per_episode_stats = {}

    # Format the per-task box ratio for human-readable audit.
    task_loc_format_report: Dict[str, Dict[str, Any]] = {}
    for task, counts in task_loc_format.items():
        total = sum(counts.values()) or 1
        task_loc_format_report[task] = {
            'box': counts['box'],
            'point': counts['point'],
            'box_ratio': round(counts['box'] / total, 4),
        }

    scale_str = f"_s{SCALE}" if SCALE != -1 else "_unnorm"
    save_file = os.path.join(SAVE_DIR, f"{DATASET_NAME}{'_wActRef' if USE_ACTION_REFEXP else ''}{scale_str}_{len(samples)}.json")

    # NB: the legacy ``_stats.json`` writer was retired here — every counter
    # it carried now lives in the unified ``_info.json`` produced by
    # ``save_run_info``, matching the convention used by MultiUI / GUIEnv /
    # MobileViews. See ``run_info.py``.

    # Cap the invalid-id sample at 200 to keep info.json human-readable, but
    # surface the total so the reader knows how much is hidden.
    invalid_elem_total = sum(len(v) for v in invalid_elem_tracker.values())
    invalid_elem_sample: List[str] = []
    for bucket, ids in invalid_elem_tracker.items():
        for sid in list(ids)[:max(0, 200 - len(invalid_elem_sample))]:
            invalid_elem_sample.append(f"{bucket}: {sid}")
        if len(invalid_elem_sample) >= 200:
            break

    save_run_info(
        dataset=DATASET_NAME,
        output_path=save_file,
        samples=samples,
        counters={
            'actionplanning_cnt': planning_cnt,
            'intentgnd_cnt': intentgnd_cnt,
            # Canonical key is ``text_loc_cnt`` per run_info.TASK_DISPLAY_NAMES.
            'text_loc_cnt': textloc_cnt,
            'ocr_cnt': ocr_cnt,
            'widgetlist_cnt': widgetlist_cnt,
        },
        config={
            'debug': DEBUG,
            'split': SPLIT,
            'scale': SCALE,
            'longest': LONGEST,
            'planning': PLANNING,
            'intentgnd': INTENTGND,
            'textloc': TEXTLOC,
            'ocr': OCR,
            'widgetlist': WIDGETLIST,
            'prob_box': PROB_BOX,
            'use_action_refexp': USE_ACTION_REFEXP,
            'discard_openapp': DISCARD_OPENAPP,
            'add_dummy_complete': ADD_DUMMY_COMPLETE,
            'device_tag': DEVICE_TAG,
            'num_workers': NUM_WORKERS,
        },
        images={
            'processed': total_ep_cnt,
            'valid': valid_ep_cnt,
            'unique_image_files': unique_img_cnt,
        },
        elements={
            'iterated': iterated_elem_cnt,
            'invalid': num_invalid_elem,
            'valid': valid_elem_cnt,
            'invalid_breakdown': invalid_breakdown,
            'steps_total': total_step_cnt,
            'steps_valid': valid_step_cnt,
        },
        tasks_enabled={
            'ActionPlanning': PLANNING,
            'IntentGnd': INTENTGND,
            'TextLoc': TEXTLOC,
            'OCR': OCR,
            'WidgetList': WIDGETLIST,
        },
        extra_sections={
            # App-level distribution (top 20 packages by sample count).
            'top_packages': dict(package_cnt.most_common(20)),
            'unique_packages': len(package_cnt),
            # Action-type breakdown of planning samples (empty unless
            # ``PLANNING`` is enabled).
            'action_type_distribution': dict(act_stats),
            # Box-vs-point audit of the ``PROB_BOX`` toggle, per task. Useful
            # for spotting silent regressions in the loc-format switching.
            'loc_format_by_task': task_loc_format_report,
            # Per-episode sample density (min / p50 / p90 / max / mean).
            'samples_per_episode_stats': samples_per_episode_stats,
            # WAE-style invalid-element record stats. The full record (all
            # IDs, grouped by bucket) lives in ``invalid_elem_record.json``;
            # info.json holds a 200-id sample so a quick eyeball check is
            # possible without loading the larger file.
            'invalid_elem_total': invalid_elem_total,
            'invalid_elem_sample': invalid_elem_sample,
            # Step-level discards (UIs where the strict element filter
            # threw out too many candidates).
            'invalid_ui_count': len(invalid_UIs),
            'invalid_ui_examples': invalid_UIs[:50],
            # Worker exceptions, grouped + counted via the shared helper.
            'errors_summary': summarize_errors(errors),
            'errors_total': len(errors),
        },
        write_preview=False,
    )

    print(f'save {len(samples)} samples to {save_file.replace(".json", "_sample.json")}')
    with open(save_file.replace(".json", "_sample.json"), "w") as f:
        json.dump(random.sample(samples, min(160, len(samples))), f, indent=2)

    print(f"save {len(samples)} samples to {save_file}")
    with open(save_file, "w") as f:
        json.dump(samples, f, indent=2)


if __name__ == "__main__":
    main()
