
"""
AndroidControl evaluation script.
Refactored for readability and structured evaluation flow.
"""

import os
import time
import random
import torch
import json
import logging
import traceback
import ast
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image
from pprint import pprint
from colorama import Fore, Style
import transformers.data.metrics.squad_metrics as squad_metrics

from action_matching import *
from utils.data_utils.task_prompt_lib import *
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.osatlas import OSATLAS
from utils.openai_utils.showui import SHOWUI, to_showui_action, showui_to_original_action
from utils.openai_utils.osatlas4b import OSATLAS4B
from utils.data_utils.misc import keep_unique_actions, scroll2swipe, get_swipe_direction
from utils.openai_utils.misc import extract_protocol_components
from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models on AndroidControl.")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=[
            "HongxinLi/UIPro-7B_Stage2_MobileB",
            "Qwen/Qwen2-VL-2B-Instruct",
        ][-1],
    )
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--wo_openapp", type=bool, default=False)
    parser.add_argument("--action_refexp", type=bool, default=True)
    parser.add_argument("--max_prev_acts", type=int, default=6)
    parser.add_argument("--device_tag", type=str, default="Android")
    parser.add_argument(
        "--preset_id_file",
        type=str,
        default=["utils/eval_utils/androidcontrol_test/selected_andcon_idx.json", ""][-1],
    )
    parser.add_argument(
        "--prompt_format_type",
        type=str,
        default="simple"
    )
    parser.add_argument("--use_qwen_actspace", type=bool, default=False)
    parser.add_argument(
        "--testset_path",
        type=str,
        default="utils/eval_utils/AndroidControl-test_12685.json",
        help="Path to AndroidControl test JSON.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/vdb1/hongxin_li",
        help="Root directory used to resolve image paths.",
    )
    args, _ = parser.parse_known_args()
    return args


def get_model_postfix(model_path: str) -> str:
    model_path = model_path.replace("merged/", "").replace("lora/", "")
    if "snapshots" in model_path:
        return model_path[model_path.find("models--") + 8 : model_path.find("snapshots") - 1]
    return "/".join(model_path.split("/")[-2:])


def init_model(args: argparse.Namespace) -> Tuple[Any, Dict[str, Any], str, int, int]:
    model_path = args.pretrained.rstrip("/ ")
    logger.info(f"Loading model from {model_path}")

    extra: Dict[str, Any] = {}
    scale = 1000
    max_prev_act = args.max_prev_acts

    if "slime" in model_path.lower():
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, use_flash_attn=True
        )
        model.generation_config.eos_token_id = 107  # <end_of_turn>
        extra["tokenizer"] = tokenizer
        extra["image_processor"] = image_processor
        extra["gen_kwargs"] = {"temperature": 0, "top_p": None, "num_beams": 1}
        scale = 100
    elif "uipro" in model_path.lower():
        model = QWen2VL(device="cuda", model_name=model_path)
        max_prev_act = 6
        scale = 1000
    elif "atlas" in model_path.lower():
        if "7b" in model_path.lower():
            model = OSATLAS(device="cuda", model_name=model_path)
        else:
            model = OSATLAS4B(device="cuda", model_name=model_path)
        max_prev_act = 999
        scale = 1000
        args.prompt_format_type = "atlas"
    elif any(k in model_path.lower() for k in ["qwen2.5", "qwen2p5"]):
        model = QWen2VL(device="cuda", model_name=model_path)
        max_prev_act = 999
        scale = -1
    elif "qwen2" in model_path.lower():
        model = QWen2VL(device="cuda", model_name=model_path)
        max_prev_act = 999
        scale = 1000
    elif "show" in model_path.lower():
        model = SHOWUI(device="cuda", model_name=model_path)
        max_prev_act = 999
        scale = 1
    else:
        model = QWen2VL(device="cuda", model_name=model_path)

    if any(k in args.pretrained.lower() for k in ["qwen2.5", "qwen2p5"]):
        scale = -1

    return model, extra, model_path, max_prev_act, scale


def load_androidcontrol(testset_path: str):
    return json.load(open(testset_path, "r"))


def group_by_trajectory(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Tuple[str, Dict[str, Any]]]]]:
    trajectory_groups: Dict[str, Dict[str, List[Tuple[str, Dict[str, Any]]]]] = {}
    for step in samples:
        step_traj_id, step_idx, instr_type = step["id"].split("-")
        if step_traj_id not in trajectory_groups:
            trajectory_groups[step_traj_id] = {"HL": [], "H": []}
        trajectory_groups[step_traj_id][instr_type].append((step_idx, step))
    for step_traj_id in trajectory_groups:
        for instr_type in trajectory_groups[step_traj_id]:
            trajectory_groups[step_traj_id][instr_type].sort(key=lambda x: x[0])
    return trajectory_groups


def build_history_str(
    step: Dict[str, Any],
    model_path: str,
    max_prev_act: int,
    scale: int,
    trajectory_groups: Dict[str, Dict[str, List[Tuple[str, Dict[str, Any]]]]],
    img_size: Tuple[int, int],
    args: argparse.Namespace,
) -> Any:
    step_id = step["id"]
    cur_step_traj_id = step_id.split("-")[0]
    cur_step_traj_idx, cur_step_idx, cur_instr_type = step_id.split("_")[-1].split("-")

    if "uipro" in model_path.lower():
        _, clean_step_instructions = keep_unique_actions(step["history"])
        history = clean_step_instructions[max(0, len(clean_step_instructions) - max_prev_act) :]
        return (
            " ".join(
                f"Step {i}. {instruc.strip(' .')}."
                for i, instruc in enumerate(
                    history, start=max(1, len(clean_step_instructions) - max_prev_act + 1)
                )
            )
            if len(history) > 0
            else "None"
        )

    if "protocol" in model_path.lower() and args.use_qwen_actspace:
        W, H = img_size
        return make_AndroidWorld_official_history_str(
            [
                to_qwen_action(x["conversations"][-1]["value"], None, W, H)
                for x in trajectory_groups[cur_step_traj_id][cur_instr_type]
            ],
            [x["outcome"] for x in trajectory_groups[cur_step_traj_id][cur_instr_type]],
        )

    if "showui" in model_path.lower():
        W, H = img_size
        return [
            to_showui_action(
                x[1]["conversations"][-1]["value"].replace("Action:", ""),
                W,
                H,
                "phone",
                scale=scale,
            )
            for x in trajectory_groups[cur_step_traj_id][cur_instr_type]
        ]

    return (
        " ".join(
            f"Step {i}. {action.strip(' .')}."
            for i, action in enumerate(step["history"][-max_prev_act:], start=1)
        )
        if step["step_id"] > 0
        else "None"
    )


def build_prompt(
    step: Dict[str, Any],
    goal: str,
    history_str: Any,
    args: argparse.Namespace,
    model_path: str,
    mode: str,
) -> str:
    if model_path in ["Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-2B-Instruct"]:
        return ANDROIDCONTROL_PLANNING_PROMPT_COT.format(
            global_task=goal,
            history=history_str,
            step_instruction=f"The next step instruction: {step['step_instruction']}\n"
            if mode == "HL"
            else "",
        )
    if "minicpm" in model_path.lower():
        return GUICOURSE_PROMPT.format(goal=goal, history=history_str)
    if args.prompt_format_type == "reflec":
        return make_planning_reflec_protocol(
            "AndroidControl",
            goal,
            history_str,
            device_type="smartphone",
            use_unnorm_xy="Qwen2.5" in args.pretrained,
            use_qwen_actspace=True,
            use_guidelines=False,
        )
    if args.prompt_format_type == "protocol":
        return make_planning_protocol(
            "AndroidWorld",
            goal,
            history_str,
            device_type="smartphone",
            use_unnorm_xy="Qwen2.5" in args.pretrained,
        )
    if "showui" in model_path.lower():
        return goal + (f" The next step instruction: {step['step_instruction']}" if mode == "HL" else "")
    return make_actionplanning_prompt(
        goal,
        history_str,
        step_instruction=step["step_instruction"] if mode == "HL" else "",
        device_tag=args.device_tag,
        prompt_format_type=args.prompt_format_type,
        with_cot=args.cot,
        without_action_space=True,
        use_action_refexp=args.action_refexp,
    )


def get_response(
    model: Any,
    model_path: str,
    prompt: str,
    img_path: str,
    history_str: Any,
    extra: Dict[str, Any],
) -> str:
    if "slime" in model_path.lower():
        tokenizer = extra["tokenizer"]
        image_processor = extra["image_processor"]
        gen_kwargs = extra["gen_kwargs"]
        conv = conv_templates["gemma"].copy()
        conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()
        img = [Image.open(img_path).convert("RGB")]
        img_tensor = process_images(img, image_processor, model.config).to(
            dtype=model.dtype, device=model.device
        )
        gen_kwargs["image_sizes"] = [img[0].size]
        input_ids = tokenizer_image_token(
            prompt_formatted, tokenizer, constants.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device=model.device)
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=img_tensor,
                image_sizes=gen_kwargs["image_sizes"],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=2048,
                use_cache=True,
            )
        return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

    if "showui" in model_path.lower() and "qwen" not in model_path.lower():
        return model.get_model_response(
            prompt,
            f"file://{img_path}",
            max_new_tokens=256,
            history=history_str,
            scenario="phone",
            task_type="nav2",
        )

    if "atlas" in model_path.lower():
        return model.get_model_response(
            prompt,
            f"file://{img_path}",
            max_new_tokens=8192,
            sys_prompt=OSATLAS_ANDROIDCONTROL_SYS_PROMPT
            if "atlas" in model_path.lower()
            else OSATLAS_SYS_PROMPT,
        )

    return model.get_model_response(prompt, f"file://{img_path}", max_new_tokens=4096)


def parse_action(
    response: str,
    model_path: str,
    prompt_format_type: str,
    action_type_ref: str,
    step_result: Dict[str, Any],
) -> Dict[str, Any]:
    if "atlas" in model_path.lower():
        return parse_atlas_action(response)
    if prompt_format_type == "reflec":
        return ast.literal_eval(response[response.rfind('{"action') : response.rfind("}") + 1])
    if prompt_format_type == "protocol":
        resp_parts = extract_protocol_components(response)
        action_raw = resp_parts.pop("action")
        step_result["thought_parts"] = resp_parts
        return ast.literal_eval(action_raw)
    if "showui" in model_path.lower() and "qwen" not in model_path.lower():
        action_pred_raw = ast.literal_eval(response)
        action_pred = showui_to_original_action(action_pred_raw)
        if "<&>" in action_pred:
            click_action_pred, input_action_pred = action_pred.split("<&>")
            action_pred = click_action_pred if action_type_ref == "click" else input_action_pred
        return ast.literal_eval(action_pred)
    return ast.literal_eval(response[response.rfind('{"action') : response.rfind("}") + 1])


def normalize_action_type(
    action_pred: Dict[str, Any],
    action_type_ref: str,
) -> Tuple[str, bool]:
    action_type_pred = action_pred.get("action_type", action_pred.get("action"))
    special_match = False

    if action_type_ref == "enter" and action_pred.get("action_type") == "press_key":
        if action_pred.get("key", "").lower() == "enter":
            special_match = True

    if action_type_pred == "terminate" and action_type_ref == "status":
        special_match = True

    if action_type_pred == "scroll" and action_type_ref == "swipe":
        special_match = True

    if action_type_pred in ["open"]:
        action_type_pred = action_pred["action"] = "open_app"

    if action_type_pred in ["type", "input_text"]:
        action_type_pred = action_pred["action_type"] = "input_text"

    if action_type_pred in ["answer"]:
        action_type_pred = action_pred["action_type"] = "status"
        action_pred["goal_status"] = (
            "successful" if "complete" in action_pred.get("text", "") else "infeasible"
        )

    if action_type_pred == "system_button":
        button = action_pred.get("button", "").lower()
        if button == "home":
            action_type_pred = "navigate_home"
        elif button == "back":
            action_type_pred = "navigate_back"

    return action_type_pred, special_match


def match_action(
    step: Dict[str, Any],
    action_pred: Dict[str, Any],
    action_type_ref: str,
    action_type_pred: str,
    special_match: bool,
    scale: int,
    img_size: Tuple[int, int],
    step_result: Dict[str, Any],
) -> None:
    if action_type_ref != action_type_pred and not special_match:
        return

    step_result["metrics"]["type_match"] = True
    W, H = img_size

    if action_type_ref in ["click", "long_press"]:
        step_result["metrics"]["need_gnd"] = True
        target = action_pred.get("target", action_pred.get("coordinate"))
        if isinstance(target, str):
            target = eval(target)
        if scale == -1:
            target_pred = [target[0] / W, target[1] / H]
        else:
            target_pred = list(map(lambda p: p / scale, target))

        gt_box = step["task_attr"]["bbox"]
        gt_box_normalized = list(
            map(lambda p: round(p, 3), [gt_box[0] / W, gt_box[1] / H, gt_box[2] / W, gt_box[3] / H])
        )
        assert all(
            0 <= p <= 1.0 for p in gt_box_normalized + target_pred
        ), f"Invalid box or target: {gt_box_normalized} {target_pred}"
        if (
            gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2]
            and gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]
        ):
            step_result["metrics"]["action_match"] = True
            step_result["metrics"]["elem_acc"] = True
            step_result["metrics"][f"{action_type_ref}_match"] = True
        return

    if action_type_ref == "input_text":
        text_ref = step["task_attr"]["text"].lower().strip()
        text_pred = action_pred["text"].lower().strip()
        step_result["metrics"]["action_match"] = (
            squad_metrics.compute_f1(text_pred, text_ref) > 0.5
        )
        step_result["metrics"]["input_text_match"] = step_result["metrics"]["action_match"]
        return

    if action_type_ref == "swipe":
        direction_ref = scroll2swipe(step["task_attr"]["direction"])
        if "direction" in action_pred:
            direction_pred = action_pred["direction"]
        else:
            direction_pred, _ = get_swipe_direction(
                action_pred["coordinate"], action_pred["coordinate2"], is_swipe=True
            )
        if direction_ref == direction_pred:
            step_result["metrics"]["action_match"] = True
            step_result["metrics"]["swipe_match"] = True
        return

    if action_type_ref == "status":
        status_ref = step["task_attr"]["goal_status"]
        status_pred = action_pred["goal_status"]
        if status_ref == status_pred:
            step_result["metrics"]["action_match"] = True
            step_result["metrics"]["status_match"] = True
        return

    if action_type_ref == "open_app":
        app_name_ref = step["task_attr"]["app_name"]
        app_name_pred = action_pred.get("app_name", action_pred.get("text", None))
        if app_name_ref == app_name_pred:
            step_result["metrics"]["action_match"] = True
            step_result["metrics"]["open_app_match"] = True
        return

    step_result["metrics"]["action_match"] = True
    step_result["metrics"][f"{action_type_ref}_match"] = True


def compute_metrics(results: List[Dict[str, Any]], counts: Dict[str, int]) -> Dict[str, Any]:
    num_sample = counts["total"]
    num_need_gnd = sum(x["metrics"]["need_gnd"] for x in results)
    num_action_match = sum(x["metrics"]["action_match"] for x in results)
    num_type_match = sum(x["metrics"]["type_match"] for x in results)
    num_elem_match = sum(x["metrics"]["elem_acc"] for x in results)

    final_metrics = {
        "step_acc": [num_action_match / num_sample if num_sample > 0 else 0, num_action_match, num_sample],
        "action_type_acc": [
            num_type_match / num_sample if num_sample > 0 else 0,
            num_type_match,
            num_sample,
        ],
        "elem_acc": [num_elem_match / num_need_gnd if num_need_gnd > 0 else 0, num_elem_match, num_need_gnd],
    }

    for k in counts.keys():
        if k == "total":
            continue
        cnt = counts[k]
        acc_cnt = sum(x["metrics"][f"{k}_match"] for x in results)
        final_metrics[f"{k}_acc"] = [round(acc_cnt / cnt, 4) if cnt > 0 else 0, acc_cnt, cnt]

    final_metrics["num_wrong_format"] = sum(1 for x in results if "wrong_format" in x)
    return final_metrics


def aggregate_metrics(metrics_all_repeats: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    aggr_metrics: Dict[str, Dict[str, List[float]]] = {"HL": {}, "H": {}}
    for mode in aggr_metrics.keys():
        for repeat_result in metrics_all_repeats:
            if not repeat_result[mode]:
                continue
            for metric_name, info in repeat_result[mode].items():
                if metric_name == "num_wrong_format":
                    continue
                if metric_name not in aggr_metrics[mode]:
                    aggr_metrics[mode][metric_name] = [0, 0, 0]
                aggr_metrics[mode][metric_name][1] += info[1]
                aggr_metrics[mode][metric_name][2] += info[2]
        for metric_name in aggr_metrics[mode].keys():
            if metric_name == "num_wrong_format":
                continue
            acc_cnt = aggr_metrics[mode][metric_name][1]
            cnt = aggr_metrics[mode][metric_name][2]
            aggr_metrics[mode][metric_name][0] = acc_cnt / cnt if cnt > 0 else 0
    return aggr_metrics


def evaluate_mode(
    samples: List[Dict[str, Any]],
    mode: str,
    args: argparse.Namespace,
    model: Any,
    model_path: str,
    max_prev_act: int,
    scale: int,
    extra: Dict[str, Any],
    trajectory_groups: Dict[str, Dict[str, List[Tuple[str, Dict[str, Any]]]]],
    root: str,
    postfix: str,
    repeat_idx: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    counts = {
        "total": 0,
        "click": 0,
        "input_text": 0,
        "swipe": 0,
        "long_press": 0,
        "enter": 0,
        "navigate_home": 0,
        "navigate_back": 0,
        "status": 0,
        "open_app": 0,
        "wait": 0,
    }

    results = []
    for step_idx, step in tqdm(
        enumerate(samples), total=len(samples), desc=f"{postfix} | Repeat {repeat_idx + 1} {mode}"
    ):
        step_id = step["id"]
        goal = step["task"].strip(" .") + "."
        counts["total"] += 1
        action_type_ref = step["action_type"]
        counts[action_type_ref] += 1

        img_path = os.path.join(root, step["image"])
        image = Image.open(img_path)
        W, H = image.size

        history_str = build_history_str(
            step, model_path, max_prev_act, scale, trajectory_groups, (W, H), args
        )
        prompt = build_prompt(step, goal, history_str, args, model_path, mode)

        step_result = {
            "img_path": img_path,
            "task": goal,
            "prompt": prompt,
            "response": None,
            "GT_action": step["task_attr"],
            "action_pred": None,
            "metrics": {
                k: False
                for k in [
                    "action_match",
                    "type_match",
                    "elem_acc",
                    "click_match",
                    "input_text_match",
                    "swipe_match",
                    "enter_match",
                    "status_match",
                    "navigate_home_match",
                    "navigate_back_match",
                    "open_app_match",
                    "wait_match",
                    "long_press_match",
                    "need_gnd",
                ]
            },
        }

        try:
            response = get_response(model, model_path, prompt, img_path, history_str, extra)
            step_result["response"] = response
            action_pred = parse_action(
                response, model_path, args.prompt_format_type, action_type_ref, step_result
            )
            step_result["action_pred"] = action_pred

            action_type_pred, special_match = normalize_action_type(action_pred, action_type_ref)
            match_action(
                step,
                action_pred,
                action_type_ref,
                action_type_pred,
                special_match,
                scale,
                (W, H),
                step_result,
            )

            is_match = step_result["metrics"]["action_match"]
            print(
                (Fore.GREEN if is_match else Fore.RED)
                + f"{is_match}"
                + Style.RESET_ALL
                + f": GT: {step['task_attr']} <=> Pred: {action_pred}"
            )

            if step_idx % 2 == 0:
                print(
                    Fore.YELLOW
                    + f"\nUser: <img>{img_path}</img> {prompt}\n"
                    + Fore.CYAN
                    + f"GPT: {response}"
                    + Style.RESET_ALL
                )
        except Exception:
            traceback.print_exc()
            print("format wrong")
            step_result["wrong_format"] = True

        results.append(step_result)

    final_metrics = compute_metrics(results, counts)
    pprint(final_metrics)
    return results, final_metrics


def main() -> None:
    args = parse_args()
    set_seeds(0)

    model, extra, model_path, max_prev_act, scale = init_model(args)
    postfix = get_model_postfix(model_path)

    root, androidcontrol_test_raw = args.data_root, load_androidcontrol(args.testset_path)
    trajectory_groups = group_by_trajectory(androidcontrol_test_raw)

    preset_ids = json.load(open(args.preset_id_file)) if args.preset_id_file else {}
    hl_ids = set(
        x["id"].split("-H")[0]
        for x in androidcontrol_test_raw
        if "-HL" in x["id"]
        and not (args.wo_openapp and x["action_type"] == "open_app")
        and not (
            len(preset_ids) > 0 and f"{x['task']}-{x['step_id']}" not in preset_ids["HL"]
        )
    )
    h_ids = set(
        x["id"].split("-H")[0]
        for x in androidcontrol_test_raw
        if "-HL" not in x["id"]
        and not (args.wo_openapp and x["action_type"] == "open_app")
        and not (
            len(preset_ids) > 0 and f"{x['task']}-{x['step_id']}" not in preset_ids["H"]
        )
    )
    hl_h_ids = hl_ids.intersection(h_ids)

    eval_result_dir = os.path.join(os.path.dirname(__file__), "eval_results/androidcontrol")
    os.makedirs(eval_result_dir, exist_ok=True)
    save_to = os.path.join(eval_result_dir, postfix)
    os.makedirs(save_to, exist_ok=True)
    save_file = os.path.join(save_to, datetime.now().strftime("%m-%d-%H-%M-%S")) + ".json"

    meta = vars(args)
    meta["max_prev_actions"] = max_prev_act

    results_all_repeats: List[Dict[str, Any]] = []
    metrics_all_repeats: List[Dict[str, Any]] = []

    repeat = 3
    for rep in range(repeat):
        selected_ids = random.sample(list(hl_h_ids), 500 if not args.debug else 7)
        hl_samples = [
            x
            for x in androidcontrol_test_raw
            if "-HL" in x["id"] and x["id"].split("-H")[0] in selected_ids
        ]
        h_samples = [
            x
            for x in androidcontrol_test_raw
            if "-HL" not in x["id"] and x["id"].split("-H")[0] in selected_ids
        ]

        metrics_this_repeat = {"HL": {}, "H": {}}
        results = {"HL": [], "H": []}

        for mode, samples in zip(["HL", "H"][1:], [hl_samples, h_samples][1:]):
            mode_results, mode_metrics = evaluate_mode(
                samples,
                mode,
                args,
                model,
                model_path,
                max_prev_act,
                scale,
                extra,
                trajectory_groups,
                root,
                postfix,
                rep,
            )
            results[mode] = mode_results
            metrics_this_repeat[mode] = mode_metrics

        results_all_repeats.append(results)
        metrics_all_repeats.append(metrics_this_repeat)

        with open(save_file, "w") as f:
            json.dump(
                {
                    "meta": meta,
                    "metrics_each_repeat": metrics_all_repeats,
                    "logs": results_all_repeats,
                },
                f,
                indent=2,
            )

    aggr_metrics = aggregate_metrics(metrics_all_repeats)
    print("\nFinal:")
    pprint(aggr_metrics)

    with open(save_file, "w") as f:
        json.dump(
            {
                "meta": meta,
                "overall_results": aggr_metrics,
                "metrics_each_repeat": metrics_all_repeats,
                "logs": results_all_repeats,
            },
            f,
            indent=2,
        )

    print(f"Finised evaluation {args.pretrained} on AndroidControl. Save to {save_file}")


if __name__ == "__main__":
    main()