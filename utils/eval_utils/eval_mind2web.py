"""
Evaluation script for Mind2Web dataset.
This script evaluates models on the Mind2Web web automation dataset.
Reference: https://arxiv.org/abs/2306.06070
"""

import os
import time
import random
import torch
import json
import ast
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image

# Local imports
from eval_utils import mind2web_action2step
from utils.data_utils.misc import keep_unique_actions, remove_redundant_spaces
from utils.data_utils.task_prompt_lib import (
    ATLAS_PROMPT, make_actionplanning_prompt, OSATLAS_MIND2WEB_PROMPT
)
from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.misc import extract_thought_components
import transformers.data.metrics.squad_metrics as squad_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_f1(pred: str, label: str) -> float:
    """Calculate F1 score following Mind2Web evaluation protocol."""
    pred_tokens = set(pred.strip().split())
    label_tokens = set(label.strip().split())

    if len(pred_tokens) == 0 and len(label_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(label_tokens) == 0:
        return 0.0

    tp = len(pred_tokens & label_tokens)
    fp = len(pred_tokens - label_tokens)
    fn = len(label_tokens - pred_tokens)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0 or recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def action2step(step_data: Dict[str, Any]) -> str:
    """Convert Mind2Web action data to prediction format."""
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':
            # For click action, calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
            click_point = [f"{item:.2f}" for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "{{\"action_type\": {}, \"click_point\": {}}}".format(action_type_new, click_point)
        else:
            # For scroll action, assign an action_type_id for each scroll direction
            if step_data["action_type_text"] == 'scroll down':
                action_type_new = 0
            elif step_data["action_type_text"] == 'scroll up':
                action_type_new = 1
            elif step_data["action_type_text"] == 'scroll left':
                action_type_new = 8
            elif step_data["action_type_text"] == 'scroll right':
                action_type_new = 9
            action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 3:
        typed_text = step_data["type_text"]
        action_type_new = action_type
        action = "{{\"action_type\": {}, \"typed_text\": \"{}\"}}".format(action_type_new, typed_text)
    else:
        action_type_new = action_type
        action = "{{\"action_type\": {}}}".format(action_type_new)

    return action


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for Mind2Web evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model on Mind2Web dataset.")
    parser.add_argument('--pretrained', type=str,
                        default=['/mnt/vdb1/hongxin_li/uipro_ckpt/0123_UIPro_Qwen2-VL-7B_gnd2planning4336k+Mind2Web-GUIAct_wActRef_s1000_108k/lora/checkpoint-3368',
                                 'HongxinLi/UIPro-7B_Stage2_Web'][-1],
                        help="Path or name of the pretrained model.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode (limited episodes).")
    parser.add_argument('--cot', action='store_true', help="Whether to use Chain-of-Thought prompting.")
    parser.add_argument('--scale', type=int, default=1000, help="Coordinate scale.")
    parser.add_argument('--action_refexp', action='store_true', default=True,
                        help="Whether to use action referring expression.")
    parser.add_argument('--task', type=str, default='website',
                        choices=['website', 'task', 'domain'],
                        help="Evaluation task type.")
    parser.add_argument('--device_tag', type=str, default='Web', help="Device tag.")
    parser.add_argument('--max_prev_acts', type=int, default=66,
                        help="Maximum number of previous actions in history.")

    args = parser.parse_args()
    return args


def initialize_model(model_path: str) -> Tuple[Any, Any, str, Dict[str, Any]]:
    """
    Initialize the model based on the model path.
    Returns: (model, tokenizer/processor, postfix, gen_kwargs)
    """
    model_path = model_path.rstrip('/ ')

    # Generate a postfix for identification from the model path
    model_path_clean = model_path.replace("lora/", "").replace("merged/", "")
    if "snapshots" in model_path_clean:
        postfix = model_path_clean[model_path_clean.find("models--") + 8: model_path_clean.find("snapshots") - 1]
    elif len(model_path_clean.split('/')) == 2:
        postfix = model_path_clean.replace('/', '--')
    elif 'checkpoint-' in model_path_clean:
        postfix = '/'.join(model_path_clean.replace("lora/merged/", "").split('/')[-2:])
    else:
        postfix = model_path_clean.replace('/', '-')

    logger.info(f"Initializing model: {model_path}")

    # Default generation kwargs
    gen_kwargs = {
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
    }

    if 'slime' in model_path.lower():
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name, use_flash_attn=True
        )
        model.generation_config.eos_token_id = 107  # '<end_of_turn>'
        return model, (tokenizer, image_processor), postfix, gen_kwargs

    elif 'qwen2' in model_path.lower():
        model = QWen2VL(device='cuda', model_name=model_path)
        return model, None, postfix, gen_kwargs

    elif 'atlas' in model_path.lower():
        model = QWen2VL(device='cuda', model_name=model_path)
        gen_kwargs["temperature"] = 0
        return model, None, postfix, gen_kwargs

    else:
        # Default fallback
        model = QWen2VL(device='cuda', model_name=model_path)
        return model, None, postfix, gen_kwargs


def format_previous_actions(action_reprs: List[str]) -> List[str]:
    """Convert Mind2Web action representations to readable action strings."""
    prev_actions = []

    for action_repr in action_reprs:
        elem, act = action_repr.split('->')
        act = act.strip()
        elem = elem.replace('  ', ' ').strip()

        if 'TYPE:' in act:
            split_id = act.find(':')
            act_name, text = act[:split_id], act[split_id + 1:]
            text = text.strip(' \n\\').replace('"', '\\"').replace('\n', '\\n')
            prev_act_str = f"type \"{text}\" into the {elem}"
        elif act == 'ENTER':
            prev_act_str = f"press enter on {elem}"
        elif act == 'CLICK':
            prev_act_str = f"click on {elem}"
        elif act == 'HOVER':
            prev_act_str = f"hover over {elem}"
        elif 'SELECT:' in act:
            split_id = act.find(':')
            act_name, value = act[:split_id], act[split_id + 1:]
            value = value.strip()
            prev_act_str = f"select {value} in the {elem}"
        else:
            raise ValueError(f"Unknown action type: {act}")

        prev_actions.append(prev_act_str)

    return prev_actions


def format_history(prev_actions: List[str], step_idx: int, max_prev_acts: int,
                   postfix: str) -> str:
    """Format the interaction history into a string based on model type."""
    if step_idx == 0:
        return 'None'

    if 'qwen2' in postfix.lower():
        clean_prev_step_instructions = keep_unique_actions(prev_actions[:step_idx])
        retained_idxs, retained_history = clean_prev_step_instructions[-max_prev_acts:]
        history_str = ' '.join(
            f"Step {i}. {remove_redundant_spaces(instruc.replace('  ', ' ').replace('[', ' ', 1).replace(']', ' ', 1).strip(' .'))}."
            for i, instruc in enumerate(retained_history, start=max(1, len(clean_prev_step_instructions) - max_prev_acts + 1))
        ) if len(retained_history) > 0 else 'None'
    elif 'atlas' in postfix.lower():
        history_str = '\n'.join(
            f'Step {i}: {step.strip(" .")}.'
            for i, step in enumerate(prev_actions[max(0, step_idx - max_prev_acts):step_idx], start=1)
        )
    else:
        history_str = ' '.join(
            f'Step {i}. {step.strip(" .")}.'
            for i, step in enumerate(prev_actions[max(0, step_idx - max_prev_acts):step_idx], start=1)
        )

    return history_str


def get_model_response(model: Any, postfix: str, prompt: str, img_path: str,
                       gen_kwargs: Dict[str, Any], tokenizer_processor: Optional[Tuple] = None) -> str:
    """Get response from the model with handling for different model types."""

    if 'slime' in postfix.lower():
        tokenizer, image_processor = tokenizer_processor
        conv = conv_templates['gemma'].copy()
        conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        img = [Image.open(img_path).convert('RGB')]
        img_tensor = process_images(img, image_processor, model.config).to(dtype=model.dtype, device=model.device)

        gen_kwargs["image_sizes"] = [img[0].size]

        input_ids = tokenizer_image_token(
            prompt_formatted, tokenizer, constants.IMAGE_TOKEN_INDEX, return_tensors='pt'
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
                max_new_tokens=256,
                use_cache=True,
            )
            return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

    else:
        # QWen2VL or Atlas models
        return model.get_model_response(
            prompt, f"file://{img_path}", max_new_tokens=4096,
            sys_prompt=OSATLAS_MIND2WEB_PROMPT if 'atlas' in postfix.lower() else ''
        )


def parse_predicted_action(response: str, postfix: str) -> Dict[str, Any]:
    """Parse the action from the model's response string."""
    if 'atlas' in postfix.lower():
        from utils.data_utils.task_prompt_lib import parse_atlas_action
        return parse_atlas_action(response)
    else:
        # Standard format: "Action: { ... }"
        if 'Action:' in response:
            action_str = response.split('Action:')[1].strip().split('\n')[-1]
        else:
            action_str = response.strip().split('\n')[-1]

        try:
            return ast.literal_eval(action_str)
        except (ValueError, SyntaxError):
            raise ValueError(f"Failed to parse action string: {action_str}")


def evaluate_step(step_data: Dict[str, Any], action_step_ref: Dict[str, Any],
                  bbox_ref: List[float], response: str, postfix: str,
                  scale: int) -> Tuple[Dict[str, Any], bool]:
    """Evaluate a single step and return results."""

    step_result = {
        "img_path": None,  # Will be set by caller
        "task": None,      # Will be set by caller
        "prompt": None,    # Will be set by caller
        "response": response,
        "GT_action": None, # Will be set by caller
        "GT_box": bbox_ref,
        "Op_match": False,
        "Ele_match": False,
        "Op_F1": [0, action_step_ref["action_type"]],
        "status": ""
    }

    try:
        action_pred = parse_predicted_action(response, postfix)

        # Normalize action types
        if action_pred["action_type"] in ['click', 'hover']:
            action_pred["action_type"] = 'click'

        # Operation matching
        if (action_pred["action_type"] == action_step_ref["action_type"] or
            action_pred["action_type"] == action_step_ref.get("ori_act", "").lower()):
            step_result["Op_match"] = True

        # Element matching for click actions
        click_point = action_pred.get("target", (-1.0, -1.0))

        if action_pred["action_type"] == 'enter':
            step_result["Ele_match"] = step_result["Op_match"]
            step_result["Op_F1"][0] = 1.0
        else:
            # Check if click point is within bounding box
            if (bbox_ref[0] <= click_point[0] / scale <= bbox_ref[2] and
                bbox_ref[1] <= click_point[1] / scale <= bbox_ref[3]):
                step_result["Ele_match"] = True

            # Convert actions to strings for F1 calculation (following Mind2Web protocol)
            pred_str = str(action_pred["action_type"])
            if action_pred["action_type"] in [3, "input_text"] or action_pred["action_type"] in [2, "select"]:
                pred_str += ' '
                pred_str += action_pred.get("text", action_pred.get("value", "")).lower()

            ref_str = str(action_step_ref["action_type"])
            if action_step_ref["action_type"] in [3, "input_text"] or action_step_ref["action_type"] in [2, "select"]:
                ref_str += ' '
                ref_str += action_step_ref["value"].lower()

            op_f1 = squad_metrics.compute_f1(pred_str, ref_str)
            step_result["Op_F1"][0] = op_f1

        return step_result, True

    except Exception as e:
        step_result["status"] = f"wrong format: {str(e)}"
        logger.error(f"Error parsing action: {e}")
        return step_result, False


def calculate_metrics(results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Calculate final metrics from evaluation results."""

    num_step = 0
    num_episode = 0
    num_op = 0
    num_ele = 0
    op_f1 = {"click": [], "select": [], "input_text": []}
    macro_ele_acc = {}
    macro_step_acc = {}
    macro_action_f1 = {}
    num_step_success = 0
    num_episode_success = 0

    for i, item in enumerate(results):
        macro_ele_acc[i] = []
        macro_step_acc[i] = []
        macro_action_f1[i] = []
        num_episode += 1
        episode_success = True

        for step_result in item:
            num_step += 1

            if step_result["Op_match"]:
                num_op += 1

            if step_result["Ele_match"]:
                num_ele += 1
                macro_ele_acc[i].append(1)
            else:
                macro_ele_acc[i].append(0)

            # Collect F1 scores by action type
            if step_result["Op_F1"][1] in op_f1:
                op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
            macro_action_f1[i].append(step_result["Op_F1"][0])

            # Step success: perfect operation F1 and element match
            if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
                num_step_success += 1
                macro_step_acc[i].append(1)
            else:
                macro_step_acc[i].append(0)
                episode_success = False

        if episode_success:
            num_episode_success += 1

    # Calculate macro averages
    marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values() if len(x) > 0])
    macro_ele_acc_avg = np.mean([np.mean(x) for x in macro_ele_acc.values()])
    macro_action_f1_avg = np.mean([np.mean(x) for x in macro_action_f1.values()])
    macro_step_acc_avg = np.mean([np.mean(x) for x in macro_step_acc.values()])

    results = {
        "Operation F1": marco_op_f1,
        "Element Acc": num_ele / num_step if num_step > 0 else 0,
        "Step Success": num_step_success / num_step if num_step > 0 else 0,
        "Episode Success": num_episode_success / num_episode if num_episode > 0 else 0,
        "Operation F1 cate": [np.mean(x).item() if len(x) > 0 else 0 for x in op_f1.values()],
        "Macro Ele Acc": macro_ele_acc_avg,
        "Macro Op F1": macro_action_f1_avg,
        "Macro Step SR": macro_step_acc_avg
    }

    return results

def evaluate_episode(episode: Dict[str, Any], model: Any, args: argparse.Namespace,
                     postfix: str, gen_kwargs: Dict[str, Any], tokenizer_processor: Optional[Tuple],
                     mind2web_imgs_dir: str) -> Tuple[List[Dict[str, Any]], float]:
    """Evaluate all steps in a single episode."""

    goal = episode["confirmed_task"]
    annot_id = episode["annotation_id"]

    # Format previous actions
    prev_actions = format_previous_actions(episode['action_reprs'])

    results_actions = []
    total_time = 0

    for step_i, step in enumerate(episode["actions"]):
        if "bbox" not in step:
            logger.warning("Action not found in step data")
            continue

        filename = annot_id + '-' + step["action_uid"] + '.jpg'
        img_path = os.path.join(mind2web_imgs_dir, filename)

        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue

        try:
            image = Image.open(img_path)
        except Exception as e:
            logger.error(f"Failed to open image {img_path}: {e}")
            continue

        # Convert action to reference format
        action_step, bbox_ref = mind2web_action2step(step, image.size, scale=args.scale, return_bbox=True)

        try:
            action_step_ref = ast.literal_eval(action_step)
        except Exception as e:
            logger.error(f"Failed to parse action step: {e}")
            continue

        # Format history
        history_str = format_history(prev_actions, step_i, args.max_prev_acts, postfix)

        # Build prompt
        if 'atlas' in postfix.lower():
            prompt = ATLAS_PROMPT.format(global_task=goal, history='None')
        else:
            prompt = make_actionplanning_prompt(
                goal, history_str, device_tag=args.device_tag,
                prompt_format_type='simple', with_cot=args.cot,
                without_action_space=True, use_action_refexp=args.action_refexp
            )

        # Get model response
        start_time = time.time()
        try:
            response = get_model_response(model, postfix, prompt, img_path, gen_kwargs, tokenizer_processor)
            inference_time = time.time() - start_time
            total_time += inference_time
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            continue

        # Evaluate step
        step_result, success = evaluate_step(step, action_step_ref, bbox_ref, response, postfix, args.scale)

        if success:
            # Update step result with additional info
            step_result.update({
                "img_path": os.path.basename(img_path),
                "task": goal,
                "prompt": prompt,
                "GT_action": action_step
            })

            action_step_ref['box'] = bbox_ref

            print(f"Op: {step_result['Op_match']} | Elem: {step_result['Ele_match']} | "
                  f"Op_F1: {step_result['Op_F1']} | GT:{action_step_ref} <=> {parse_predicted_action(response, postfix) if success else 'N/A'}")

        results_actions.append(step_result)

    return results_actions, total_time


def main():
    """Main execution entry point."""
    args = parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)

    # Load model and components
    model, tokenizer_processor, postfix, gen_kwargs = initialize_model(args.pretrained)

    # Set up directories and paths
    index = 0  # Could be made configurable
    ROOT = [
        "/mnt/nvme0n1p1/hongxin_li/UI_training_data/Mind2Web",
        "/data2/hongxin_li/UI_training_data/Mind2Web",
        "/data0/jingran/workspace/UI_training_data/Mind2Web",
        "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/Mind2Web"
    ][index]

    mind2web_imgs_dir = os.path.join(ROOT, "mind2web_images")
    eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results/mind2web', postfix)
    os.makedirs(eval_result_dir, exist_ok=True)

    # Load test data
    test_file = f'{ROOT}/mind2web_data_test_{args.task}.json'
    logger.info(f"Loading Mind2Web test dataset from: {test_file}")
    mind2web_test = json.load(open(test_file, 'r'))

    if args.debug:
        mind2web_test = mind2web_test[:5]  # Limit to first 5 episodes for debugging
        logger.info(f"Debug mode: using {len(mind2web_test)} episodes")

    # Main evaluation loop
    results = []
    total_inference_time = 0
    total_steps = 0

    pbar = tqdm(enumerate(mind2web_test), total=len(mind2web_test),
                desc=f'Evaluating {postfix} on Mind2Web {args.task} (Max prev acts: {args.max_prev_acts})')

    for ep_idx, episode in pbar:
        episode_results, episode_time = evaluate_episode(
            episode, model, args, postfix, gen_kwargs, tokenizer_processor, mind2web_imgs_dir
        )

        results.append(episode_results)
        total_inference_time += episode_time
        total_steps += len(episode_results)

    # Calculate final metrics
    final_metrics = calculate_metrics(results)
    avg_inference_time = total_inference_time / total_steps if total_steps > 0 else 0
    final_metrics['time_per_step'] = avg_inference_time

    # Print results
    print(f"\nResults for {args.task}:")
    print(f"Operation F1: {final_metrics['Operation F1']:.4f}")
    print(f"Element Acc: {final_metrics['Element Acc']:.4f}")
    print(f"Step Success: {final_metrics['Step Success']:.4f}")
    print(f"Episode Success: {final_metrics['Episode Success']:.4f}")
    print(f"Operation F1 by category: {final_metrics['Operation F1 cate']}")
    print(f"Macro Element Acc: {final_metrics['Macro Ele Acc']:.4f}")
    print(f"Macro Operation F1: {final_metrics['Macro Op F1']:.4f}")
    print(f"Macro Step Success Rate: {final_metrics['Macro Step SR']:.4f}")
    print(f"Average Inference Time per Step: {avg_inference_time:.4f}s")

    # Save results
    time_str = datetime.now().strftime("%m-%d-%H-%M-%S")
    save_file = os.path.join(eval_result_dir, f"{args.task}-{time_str}{'_debug' if args.debug else ''}.json")

    output_data = {
        "meta": vars(args),
        "overall_results": final_metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log": results
    }

    with open(save_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Evaluation finished for {args.pretrained} on Mind2Web {args.task}. "
                f"Results saved to: {save_file}")


if __name__ == '__main__':
    main()