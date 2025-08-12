"""
Build AITW dataset variants.

Generates two task types:
1) Given image, global task and history, predict next action plan.
2) Given current click intent, predict ground-truth coordinates.

This module emphasizes clarity, type safety, and robust I/O.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import cv2  # type: ignore
from datasets import Dataset  # type: ignore
from tqdm import tqdm  # type: ignore

# Make local libs importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from misc import keep_unique_actions, lower_first_letter  # noqa: E402
from task_prompt_lib import *  # noqa: E402, F403


# ------------------------------
# Configuration
# ------------------------------

DATASET_NAME: str = "AITW"

ROOT: str = "/mnt/vdb1/hongxin_li//AITW/"
SAVE_DIR: str = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"

SCALE: int = 1000
SPLIT: str = ["train", "val", "trainval", "test"][2]

APPS: str = ["all", "general", "install", "googleapps", "single", "webshopping"][0]

ONLY_ACTION: str = "click"  # Example: 'click' or '' to allow all
PUSH2HUB: bool = False
PLANNING: bool = False
INTENTGND: bool = True

POINT_FORMAT: str = ["plain", "qwen2", "florence"][0]

USE_ACTION_REFEXP: bool = True # Set as True when generating Qwen2-format training samples

DEVICE_TAG: str = "Android"

AITW_IMAGES_DIR: str = os.path.join(ROOT, "aitw_images")
SPLIT_NAME: str = f"aitw_data_{SPLIT}"

logger = logging.getLogger(__name__)


# ------------------------------
# Utilities
# ------------------------------

def _load_split(root: str, split: str) -> List[List[Dict[str, Any]]]:
    """Load AITW split and return a list of episodes."""
    if split == "trainval":
        episodes: List[List[Dict[str, Any]]] = []
        for value in json.load(open(os.path.join(root, "aitw_data_train.json"), "r")).values():
            episodes.extend(value)  # type: ignore[arg-type]
        for value in json.load(open(os.path.join(root, "aitw_data_val.json"), "r")).values():
            episodes.extend(value)  # type: ignore[arg-type]
        return episodes

    split_payload: Dict[str, List[List[Dict[str, Any]]]] = json.load(
        open(os.path.join(root, f"{SPLIT_NAME}.json"), "r")
    )
    # Merge all app categories in a fixed order for determinism
    merged: List[List[Dict[str, Any]]] = (
        split_payload.get("general", [])
        + split_payload.get("single", [])
        + split_payload.get("webshopping", [])
        + split_payload.get("install", [])
        + split_payload.get("googleapps", [])
    )
    return merged


def _episode_has_excessive_repeats(episode: Sequence[Dict[str, Any]], threshold: int = 2) -> bool:
    """Detect trajectories with a repeated identical action >= threshold times in a row."""
    consecutive_repeats = 0
    last_action = ""
    for step in episode:
        current_action = step["action_addition"]
        if current_action == last_action:
            consecutive_repeats += 1
        else:
            consecutive_repeats = 0
        if consecutive_repeats >= threshold:
            return True
        last_action = current_action
    return False


def _compute_image_size(img_path: str) -> Tuple[int, int]:
    """Return (width, height) for the given image path.
    
    Falls back to magic library if cv2 fails, maintaining original behavior.
    """
    # Try cv2 first (more reliable)
    try:
        image = cv2.imread(img_path)
        if image is not None:
            height, width = image.shape[:2]
            return width, height
    except Exception:
        pass
    
    # Fallback to magic library (original method)
    try:
        import magic
        size_match = re.search(r'(\d+) x (\d+)', magic.from_file(img_path))
        if size_match:
            return int(size_match.group(1)), int(size_match.group(2))
    except Exception:
        pass
    
    raise FileNotFoundError(f"Failed to read image dimensions: {img_path}")


def _scale_point_to_grid(point01: Tuple[float, float], scale: int) -> Tuple[int, int]:
    sx = max(0, min(scale - 1, round(point01[0] * scale)))
    sy = max(0, min(scale - 1, round(point01[1] * scale)))
    return sx, sy


def _build_history(normalized_step_instructions: Sequence[str], until_index: int) -> str:
    # Deduplicate while preserving the last MAX_PREV_ACT
    _, deduped = keep_unique_actions(normalized_step_instructions[:until_index])
    retained = deduped[-MAX_PREV_ACT:]
    if not retained:
        return "None"
    start_idx = max(1, len(deduped) - MAX_PREV_ACT + 1)
    return " ".join(
        f"Step {i}. {instr.strip(' .')}." for i, instr in enumerate(retained, start=start_idx)
    )


def _action_from_step(
    step_info: Dict[str, Any],
    *,
    scale: int,
) -> Tuple[str, str, str]:
    """Return (action_str, action_refexp, resolved_action_type_text).

    resolved_action_type_text may be adjusted (e.g., scroll -> swipe direction)
    """
    action_type_text: str = step_info["action_type_text"]
    from_point: Tuple[float, float] = step_info["touch"]
    to_point: Tuple[float, float] = step_info["lift"]

    if action_type_text == "click":
        sx, sy = _scale_point_to_grid(from_point, scale)
        action_str = CLICK_TEMPLATE.format(target_x=sx, target_y=sy)  # type: ignore[name-defined]
        action_intent = step_info["action_addition"].strip(" .")
        click_target = action_intent[action_intent.find(" ") + 1 :]
        action_refexp = random.choice(ACTION_PREFIXES[action_type_text]["specific"]) + f' the element "{click_target}"'  # type: ignore[name-defined]
        return action_str, action_refexp, action_type_text

    if "scroll" in action_type_text:
        sx, sy = _scale_point_to_grid(from_point, scale)
        scroll_direction = action_type_text.split()[-1]
        if scroll_direction in ["down", "up"]:
            shift = to_point[1] - from_point[1]
            shift_abs = abs(shift)
            swipe_direction = "down" if shift > 0 else "up"
        else:
            shift = to_point[0] - from_point[0]
            shift_abs = abs(shift)
            swipe_direction = "right" if shift > 0 else "left"
        step_info["action_addition"] = f"swipe {swipe_direction}"
        distance = discretize_dist(shift_abs)  # type: ignore[name-defined]
        action_str = SWIPE_TEMPLATE.format(  # type: ignore[name-defined]
            start_x=sx, start_y=sy, direction=swipe_direction, distance=distance
        )
        action_refexp = random.choice(SWIPE_PHRASES).format(direction=swipe_direction)  # type: ignore[name-defined]
        return action_str, action_refexp, action_type_text

    if action_type_text == "type":
        text_raw = step_info["type_text"]
        # Normalize quotes and escapes
        if text_raw.count('"') % 2 != 0:
            text_raw = text_raw.strip('"')
        if text_raw.count("'") % 2 != 0:
            text_raw = text_raw.strip("'")
        text = text_raw.strip(" \\").replace("\n", "\\n").replace('"', '\\"')
        if not text:
            return "", "", action_type_text
        action_str = INPUT_TEMPLATE.format(text=text)  # type: ignore[name-defined]
        action_refexp = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT["specific"]).format(  # type: ignore[name-defined]
            text=text, target="the text box"
        )
        return action_str, action_refexp, action_type_text

    if action_type_text == "status task complete":
        action_str = STATUS_TEMPLATE.format(goal_status="successful", answer="")  # type: ignore[name-defined]
        action_refexp = random.choice(TASK_STATUS_SENTENCES["successful"])  # type: ignore[name-defined]
        return action_str, action_refexp, action_type_text

    if action_type_text == "status task impossible":
        action_str = STATUS_TEMPLATE.format(goal_status="infeasible", answer="")  # type: ignore[name-defined]
        action_refexp = random.choice(TASK_STATUS_SENTENCES["infeasible"])  # type: ignore[name-defined]
        return action_str, action_refexp, action_type_text

    if action_type_text == "press back":
        action_str = NAVIGATE_BACK_TEMPLATE  # type: ignore[name-defined]
        action_refexp = random.choice(NAVIGATE_BACK_PREFIXES)  # type: ignore[name-defined]
        return action_str, action_refexp, action_type_text

    if action_type_text == "press home":
        action_str = NAVIGATE_HOME_TEMPLATE  # type: ignore[name-defined]
        action_refexp = random.choice(NAVIGATE_HOME_PREFIXES)  # type: ignore[name-defined]
        return action_str, action_refexp, action_type_text

    if action_type_text == "press enter":
        action_str = PRESSKEY_TEMPLATE.format(key="Enter")  # type: ignore[name-defined]
        action_refexp = random.choice(PRESSKEY_PREFIXES["Enter"])  # type: ignore[name-defined]
        return action_str, action_refexp, action_type_text

    raise ValueError(f"Unknown action type: {action_type_text}")


def make_aitw_data() -> None:
    episodes = _load_split(ROOT, SPLIT)

    planning_count = 0
    intentgnd_count = 0
    samples: List[Dict[str, Any]] = []
    invalid_samples: List[str] = []

    for ep_index, episode in tqdm(
        enumerate(episodes), total=len(episodes), desc=f"{ROOT}/{SPLIT_NAME}.json"
    ):
        # Normalize scroll wording for history and stats to keep behavior parity
        step_instructions = [
            x["action_addition"].replace("scroll down", "swipe up")
            .replace("scroll up", "swipe down")
            .replace("scroll left", "swipe right")
            .replace("scroll right", "swipe left")
            for x in episode
        ]

        # Drop episodes with too many consecutive identical actions (non-test only)
        if SPLIT != "test" and _episode_has_excessive_repeats(episode):
            invalid_samples.extend([f"{x['ep_id']}-{x['step']}" for x in episode])
            continue

        for step_index, step_info in enumerate(episode):
            img_filename = f"{step_info['img_filename']}.png"
            img_path = os.path.join(AITW_IMAGES_DIR, img_filename)

            if not os.path.exists(img_path):
                logger.warning("Image not found: %s", img_path)
                continue

            try:
                width, height = _compute_image_size(img_path)
            except FileNotFoundError:
                logger.warning("Failed to read image: %s", img_path)
                continue

            short_img_path = f"{DATASET_NAME}/" + img_path.split("aitw_images/")[1]

            if step_index < len(episode) - 1:
                next_img_path = os.path.join(
                    AITW_IMAGES_DIR, f"{episode[step_index + 1]['img_filename']}.png"
                )
                short_next_img_path = f"{DATASET_NAME}/" + next_img_path.split("aitw_images/")[1]
            else:
                short_next_img_path = short_img_path

            try:
                action_str, action_refexp, action_type_text = _action_from_step(
                    step_info, scale=SCALE
                )
            except ValueError as err:
                logger.error("%s", err)
                continue

            # Skip empty actions (e.g., empty type text)
            if not action_str:
                continue

            if len(ONLY_ACTION) > 0 and ONLY_ACTION not in action_str:
                continue

            history_str = _build_history(step_instructions, step_index)

            # Derive structured action type from the action_str JSON
            action_type = ast.literal_eval(action_str)["action_type"] if action_str else ""

            if USE_ACTION_REFEXP and action_str:
                action_str = f"{QWEN_OBJ_REF_TAG_START}{action_refexp}{QWEN_OBJ_REF_TAG_END}\n{action_str}"  # type: ignore[name-defined]

            if APPS != "all" and img_filename.split("/")[0] != APPS:
                continue

            if PUSH2HUB:
                samples.append(
                    {
                        "image": img_filename,
                        "step": step_info,
                        "history": step_instructions[:step_index],
                        "step_instruction": step_info["action_addition"],
                        "action_type": action_type,
                        "action_refexp": action_refexp,
                    }
                )
            else:
                gt_action = f"Action: {action_str}"

                if PLANNING:
                    # H
                    sample = make_actionplanning_sample(  # type: ignore[name-defined]
                        task_id=f"autogui_{DATASET_NAME}_planning_{step_info['ep_id']}-{step_info['step']}-H",
                        global_task=step_info["goal"],
                        history=history_str,
                        gt_action=gt_action,
                        with_cot=False,
                        device_tag=DEVICE_TAG,
                        use_action_refexp=USE_ACTION_REFEXP,
                    )
                    sample.update(
                        {
                            "action_type": action_type,
                            "step_info": step_info,
                            "image": short_img_path,
                            "next_image": short_next_img_path,
                            "history": step_instructions[:step_index],
                            "step_instruction": step_info["action_addition"],
                            "action_refexp": action_refexp,
                        }
                    )
                    samples.append(sample)
                    planning_count += 1

                    # HL
                    sample = make_actionplanning_sample(  # type: ignore[name-defined]
                        task_id=f"autogui_{DATASET_NAME}_planning_{step_info['ep_id']}-{step_info['step']}-HL",
                        global_task=step_info["goal"],
                        history=history_str,
                        gt_action=gt_action,
                        step_instruction=f"The next step instruction: {step_info['action_addition']}\n",
                        with_cot=False,
                        device_tag=DEVICE_TAG,
                        use_action_refexp=USE_ACTION_REFEXP,
                    )
                    sample.update(
                        {
                            "action_type": action_type,
                            "step_info": step_info,
                            "image": short_img_path,
                            "next_image": short_next_img_path,
                            "history": step_instructions[:step_index],
                            "step_instruction": step_info["action_addition"],
                            "action_refexp": action_refexp,
                        }
                    )
                    samples.append(sample)
                    planning_count += 1

                # Intent grounding samples (non-test only, and only for click)
                if INTENTGND and SPLIT != "test" and action_type_text == "click":
                    sx, sy = _scale_point_to_grid(step_info["touch"], SCALE)
                    intent_text = lower_first_letter(
                        action_refexp if USE_ACTION_REFEXP else step_info["action_addition"]
                    )
                    sample = make_intentgnd_sample(  # type: ignore[name-defined]
                        task_id=f"autogui_{DATASET_NAME}_intentgnd_{step_info['ep_id']}-{step_info['step']}",
                        intent=intent_text,
                        loc=(sx, sy),
                        output_tag=WITHPOINT_TAG_LONG,  # type: ignore[name-defined]
                        point_format=POINT_FORMAT,
                    )
                    sample["step_info"], sample["task_attr"], sample["image"] = (
                        step_info,
                        intent_text,
                        short_img_path,
                    )
                    samples.append(sample)
                    intentgnd_count += 1

    app_suffix = "" if APPS == "all" else f"-{APPS}"
    act_limit_suffix = "" if len(ONLY_ACTION) == 0 else f"-{ONLY_ACTION}"

    if PUSH2HUB:
        dataset = Dataset.from_list(samples)
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN environment variable not set; cannot push to hub.")
            return
        dataset.push_to_hub(
            f"HongxinLi/AITW_test{app_suffix}{act_limit_suffix}_v2",
            private=False,
            token=hf_token,
            split=SPLIT,
        )
        return

    action_stats: Dict[str, int] = defaultdict(int)
    for item in samples:
        if "planning" not in item.get("id", ""):
            continue
        match = re.search(r'"action_type":\s*"([^"]+)"', item["conversations"][1]["value"])  # type: ignore[index]
        if match:
            action_stats[match.group(1)] += 1

    report = (
        f"Total samples: {len(samples)+len(invalid_samples)} Valid samples: {len(samples)} | "
        f"#Unique imgs: {len(set(x['image'] for x in samples if 'image' in x))} | "
        f"planning: {planning_count} | intentgnd: {intentgnd_count}"
    )
    logger.info(report)

    save_dir = f"{SAVE_DIR}/{DATASET_NAME}_processed"
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(
        save_dir,
        f"{DATASET_NAME}{app_suffix}{act_limit_suffix}_{SPLIT}"
        f"{'_wActRef' if PLANNING and USE_ACTION_REFEXP else ''}"
        f"{'_IntengGnd' if INTENTGND else ''}_s{SCALE}_{len(samples)}.json",
    )

    stats_path = save_file.replace(".json", "_stats.json")
    sample_path = save_file.replace(".json", "_sample.json")

    with open(stats_path, "w") as f:
        json.dump(
            {
                "total_sample_cnt": len(samples) + len(invalid_samples),
                "valid_sample_cnt": len(samples),
                "planning": planning_count,
                "action_stats": action_stats,
                "intentgnd": intentgnd_count,
                "invalid_samples": invalid_samples,
            },
            f,
            indent=2,
        )
    logger.info("Saved stats to %s", stats_path)

    with open(sample_path, "w") as f:
        json.dump(random.sample(samples, min(len(samples), 160)), f, indent=2)
    logger.info("Saved sample to %s", sample_path)

    with open(save_file, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info("Saved %d samples to %s", len(samples), save_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    make_aitw_data()