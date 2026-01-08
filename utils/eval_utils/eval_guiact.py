"""
GUIAct Dataset Evaluation Script

Evaluates multimodal models on GUIAct dataset by comparing predicted actions
with ground truth actions across different action types (click, scroll, input, etc.).
Supports Web and Mobile device types with various model architectures.
"""

import os
import time
import random
import torch
import json
import re
import ast
import argparse
import logging
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image
from pprint import pprint
from colorama import Fore, Style

# Local imports
from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates
from utils.data_utils.misc import average_iou, keep_unique_actions, restore_unified_actions
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.osatlas import OSATLAS
from utils.openai_utils.showui import SHOWUI, showui_to_original_action
from utils.openai_utils.misc import extract_thought_components
from utils.data_utils.task_prompt_lib import (
    GUIACTWEB_PLANNING_PROMPT_COT, GUIACTMOBILE_PLANNING_PROMPT_COT,
    make_actionplanning_prompt, parse_atlas_action
)
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.data_utils.misc import keep_unique_actions, restore_unified_actions, qwen2vl_to_nornal_action
from utils.openai_utils.misc import extract_thought_components
from colorama import Fore, Style

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_answer(text: str) -> str:
    """Clean answer text by removing punctuation and converting to lowercase."""
    text = text.lower().strip(' .?!').replace('"', '').replace("'", '').replace(',', '').replace(";", '')
    return text


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for GUIAct evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model on GUIAct dataset.")
    parser.add_argument('--pretrained', type=str,
                        default=['showlab/ShowUI-2B', 'Qwen/Qwen2-VL-7B-Instruct', 'HongxinLi/UIPro-7B_Stage2_Web'][-1],
                        help="Path or name of the pretrained model.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode (limited steps).")
    parser.add_argument('--cot', action='store_true', help="Whether to use Chain-of-Thought prompting.")
    parser.add_argument('--scale', type=int, default=1000, help="Coordinate scale (e.g., 1 for ShowUI, 1000 for others).")
    parser.add_argument('--action_refexp', action='store_true', default=True,
                        help="Whether to use action referring expression.")
    parser.add_argument('--device_type', type=str, default='Web',
                        choices=['Web', 'Mobile'], help="Device type for evaluation.")
    parser.add_argument('--max_prev_acts', type=int, default=6,
                        help="Maximum number of previous actions in history.")
    parser.add_argument('--original_actspace', action='store_true',
                        help="Whether to use original GUIAct action space.")

    args, _ = parser.parse_known_args()
    return args


def initialize_model(model_path: str) -> Tuple[Any, Any, str, Dict[str, Any]]:
    """
    Initialize the model based on the model path.

    Returns:
        Tuple of (model, tokenizer/processor, postfix, gen_kwargs)
    """
    model_path = model_path.rstrip('/ ')

    # Generate postfix for result file identification
    if "snapshots" in model_path:
        postfix = model_path[model_path.find("models--") + 8: model_path.find("snapshots") - 1]
    elif len(model_path.split('/')) == 2:
        postfix = model_path.replace('/', '--')
    elif 'checkpoint-' in model_path:
        postfix = '/'.join(model_path.replace("lora/", "").replace("merged/", "").split('/')[-2:])
    else:
        postfix = os.path.basename(model_path)

    logger.info(f"Initializing model: {model_path}")

    # Default generation parameters
    gen_kwargs = {
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
    }

    # Initialize different model types
    if 'slime' in model_path.lower():
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, use_flash_attn=True
        )
        model.generation_config.eos_token_id = 107  # '<end_of_turn>'
        return model, (tokenizer, image_processor), postfix, gen_kwargs

    elif 'qwen2' in model_path.lower():
        model = QWen2VL(device='cuda', model_name=model_path)
        return model, None, postfix, gen_kwargs

    elif 'atlas' in model_path.lower():
        model = OSATLAS(device='cuda', model_name=model_path)
        return model, None, postfix, gen_kwargs

    elif 'show' in model_path.lower():
        model = SHOWUI(device='cuda', model_name=model_path)
        return model, None, postfix, gen_kwargs

    else:
        # Default fallback to QWen2VL
        model = QWen2VL(device='cuda', model_name=model_path)
        return model, None, postfix, gen_kwargs


def initialize_model(model_path: str) -> Tuple[Any, Any, str, Dict[str, Any]]:
    """
    Initialize the model based on the model path.
    Returns: (model, tokenizer/processor, postfix, gen_kwargs)
    """
    model_path = model_path.rstrip('/ ')

    # Generate a postfix for identification from the model path
    if "snapshots" in model_path:
        postfix = model_path[model_path.find("models--") + 8: model_path.find("snapshots") - 1]
    elif len(model_path.split('/')) == 2:
        postfix = model_path.replace('/', '--')
    elif 'checkpoint-' in model_path:
        postfix = '/'.join(model_path.replace("lora/", "").replace("merged/","").split('/')[-2:])
    else:
        postfix = os.path.basename(model_path)

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
        gen_kwargs["temperature"] = 0
        return model, None, postfix, gen_kwargs

    elif 'atlas' in model_path.lower():
        model = OSATLAS(device='cuda', model_name=model_path)
        gen_kwargs["temperature"] = 0
        return model, None, postfix, gen_kwargs

    elif 'show' in model_path.lower():
        model = SHOWUI(device='cuda', model_name=model_path)
        gen_kwargs["temperature"] = 0
        return model, None, postfix, gen_kwargs

    else:
        # Default fallback
        model = QWen2VL(device='cuda', model_name=model_path)
        return model, None, postfix, gen_kwargs


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters using regex pattern."""
    pattern = re.compile('[\u4e00-\u9fff]')
    return bool(pattern.search(text))


def format_history(step_data: Dict[str, Any], max_prev_acts: int) -> str:
    """
    Format the interaction history into a readable string.

    Args:
        step_data: Step data containing history
        max_prev_acts: Maximum number of previous actions to include

    Returns:
        Formatted history string
    """
    raw_history = step_data['history']

    if isinstance(raw_history, list):
        # Keep only unique actions and limit history length
        _, clean_step_instructions = keep_unique_actions(raw_history)
        history = clean_step_instructions[max(0, len(clean_step_instructions) - max_prev_acts):]

        # Format as numbered steps
        history_str = ' '.join(
            f"Step {i}. {instruc.strip(' .')}."
            for i, instruc in enumerate(history, start=max(1, len(clean_step_instructions) - max_prev_acts + 1))
        ) if len(history) > 0 else 'None'
        return history_str
    else:
        return str(raw_history)


def get_model_response(model: Any, model_type: str, prompt: str, img_path: str,
                       gen_kwargs: Dict[str, Any], tokenizer_processor: Optional[Tuple] = None) -> str:
    """
    Get response from the model with handling for different model architectures.

    Args:
        model: Initialized model instance
        model_type: String identifying model type (slime, showui, atlas, etc.)
        prompt: Formatted prompt text
        img_path: Path to input image
        gen_kwargs: Generation parameters
        tokenizer_processor: Tuple of (tokenizer, image_processor) for certain models

    Returns:
        Model's text response
    """

    if 'slime' in model_type:
        # Handle SLIME models with custom conversation template
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

    elif 'showui' in model_type and 'qwen' not in model_type:
        # Handle ShowUI models
        return model.get_model_response(
            prompt,
            f"file://{img_path}",
            max_new_tokens=256,
            history="",  # GUIAct doesn't use history in the same way
            scenario='web',
            task_type='nav2'
        )

    elif 'atlas' in model_type:
        # Handle OS-Atlas models
        return model.get_model_response(
            prompt,
            f"file://{img_path}",
            max_new_tokens=4096,
            sys_prompt=OSATLAS_ANDROIDCONTROL_SYS_PROMPT if 'atlas' in model_type.lower() else OSATLAS_SYS_PROMPT,
        )

    else:
        # Handle QWen2VL and other models
        return model.get_model_response(
            prompt, f"file://{img_path}", max_new_tokens=4096,
            sys_prompt=OSATLAS_MIND2WEB_PROMPT if 'atlas' in model_type.lower() else ''
        )


def parse_predicted_action(response: str, postfix: str, scale: int) -> Dict[str, Any]:
    """
    Parse the action prediction from model's response string.

    Args:
        response: Raw model response text
        postfix: Model identifier for parsing logic
        scale: Coordinate scaling factor (unused in some parsers)

    Returns:
        Parsed action dictionary
    """
    if 'atlas' in postfix.lower():
        return parse_atlas_action(response)
    elif 'Qwen--' in postfix:  # Official Qwen models
        obs, thought, _, action_pred, _ = extract_thought_components(response.replace('}}', '}').replace('"target": [', '"target": ('))
        action_pred = ast.literal_eval(action_pred)
        # Convert target dict to list format if needed
        if isinstance(action_pred.get('target', None), dict) and 'x' in action_pred['target']:
            action_pred['target'] = [action_pred['target']['x'], action_pred['target']['y']]
        return action_pred
    elif 'showui' in postfix.lower() and 'qwen' not in postfix.lower():
        action_pred_raw = ast.literal_eval(response)
        action_pred = showui_to_original_action(action_pred_raw)
        return ast.literal_eval(action_pred)
    else:
        # Standard JSON-like format parsing with fallback
        try:
            return ast.literal_eval(response[response.rfind('{"action'):response.rfind('}')+1])
        except:
            response += '"}'  # Add missing closing brace
            return ast.literal_eval(response[response.find('{"act'):])


def evaluate_step(step_data: Dict[str, Any], model: Any, args: argparse.Namespace,
                  postfix: str, gen_kwargs: Dict[str, Any], tokenizer_processor: Optional[Tuple],
                  guiact_imgs_dir: str) -> Tuple[Dict[str, Any], float]:
    """
    Evaluate a single GUI action step and return results with inference time.

    Args:
        step_data: Single step data from test set
        model: Initialized model instance
        args: Parsed command line arguments
        postfix: Model identifier string
        gen_kwargs: Model generation parameters
        tokenizer_processor: Tokenizer and processor for certain models
        guiact_imgs_dir: Directory containing GUIAct images

    Returns:
        Tuple of (step_result_dict, inference_time_seconds)
    """

    goal = step_data["task"]
    action_type_ref = step_data['action_type']
    img_path = os.path.join(guiact_imgs_dir, step_data["image"])

    try:
        image = Image.open(img_path)
    except Exception as e:
        logger.warning(f"Failed to open image {img_path}: {e}")
        return None, 0

    # Format history
    history_str = format_history(step_data, args.max_prev_acts)

    # Build prompt
    if 'Qwen--' in postfix:
        prompt = {'Web': GUIACTWEB_PLANNING_PROMPT_COT, 'Mobile': GUIACTMOBILE_PLANNING_PROMPT_COT}[args.device_type].format(
            global_task=goal,
            history=history_str,
            step_instruction=''
        )
    else:
        prompt = make_actionplanning_prompt(
            goal, history_str, device_tag=args.device_type,
            prompt_format_type='simple', with_cot=args.cot,
            without_action_space=True, use_action_refexp=args.action_refexp
        )

    if args.original_actspace:
        prompt = ('[GUIAct-Web] ' if args.device_type == 'Web' else '[GUIAct] ') + prompt

    # Initialize metrics tracking for all action types
    actions = ['click', 'scroll', 'hover', 'drag', 'press_key', 'hotkey', 'status', 'swipe', 'tap', 'input_text', 'enter']
    metrics = {f'{act}_match': False for act in actions}
    metrics.update({'action_match': False, 'type_match': False, 'elem_acc': False, 'need_gnd': False})

    step_result = {
        "img_path": os.path.basename(img_path),
        "task": goal,
        "gt_action": step_data['step_info'],
        "prompt": prompt,
        "response": None,
        "original_action": step_data['task_attr'],
        "action_pred": None,
        "metrics": deepcopy(metrics),
        'wrong_format': False
    }

    start_time = time.time()
    try:
        response = get_model_response(model, postfix.lower(), prompt, img_path, gen_kwargs, tokenizer_processor)
        inference_time = time.time() - start_time

        step_result["response"] = response

        # Parse action prediction
        action_pred = parse_predicted_action(response, postfix, args.scale)
        step_result["action_pred"] = action_pred

        # Restore unified action definitions if needed
        if args.original_actspace:
            action_pred = restore_unified_actions(action_pred)

        action_type_pred = action_pred['action_type']
        task_attr = step_data['task_attr']

        # Type matching
        if action_type_ref == action_type_pred:
            step_result['metrics']['type_match'] = True

            # Element accuracy for click and hover
            if action_type_ref in ['click', 'hover']:
                step_result['metrics']['need_gnd'] = True
                target = eval(action_pred['target']) if isinstance(action_pred['target'], str) else action_pred['target']
                target_pred = list(map(lambda p: p / args.scale, target))

                if 'bbox' in task_attr:
                    gt_box_normalized = task_attr['bbox']
                    if (gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2] and
                        gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]):
                        step_result['metrics']['action_match'] = step_result['metrics']['elem_acc'] = step_result['metrics'][f'{action_type_ref}_match'] = True
                elif 'center' in task_attr:
                    center_normalized = task_attr['center']
                    if np.linalg.norm(np.array(center_normalized) - np.array(target_pred)) < 0.14:
                        step_result['metrics']['action_match'] = step_result['metrics']['elem_acc'] = step_result['metrics'][f'{action_type_ref}_match'] = True

            # Text input matching
            elif action_type_ref == 'input_text':
                if squad_metrics.compute_f1(task_attr['text'], action_pred['text']) > 0.5:
                    step_result['metrics']['action_match'] = step_result['metrics']['input_text_match'] = True

            # Scroll and swipe matching
            elif action_type_ref in ['scroll', 'swipe']:
                step_result['metrics']['action_match'] = step_result['metrics'][f'{action_type_ref}_match'] = task_attr['direction'] == action_pred['direction']

            # Status matching
            elif action_type_ref == 'status':
                status_ref, status_pred = task_attr['goal_status'], action_pred['goal_status']
                if status_ref == status_pred:
                    answer_ref, answer_pred = clean_answer(task_attr['answer']), clean_answer(action_pred['answer'])
                    if answer_ref in ['', 'task complete', 'task impossible']:
                        answer_f1 = 1.0
                    else:
                        answer_f1 = squad_metrics.compute_f1(answer_ref, answer_pred)
                    step_result['metrics']['action_match'] = step_result['metrics']['status_match'] = answer_f1 > 0.5

            # Drag matching
            elif action_type_ref == 'drag':
                drag_start, drag_end = list(map(lambda p: p/args.scale, action_pred['start'])), list(map(lambda p: p/args.scale, action_pred['end']))

                # Post-process drag points
                if drag_start[0] == drag_end[0]:
                    drag_start[0] -= 0.01
                    drag_end[0] += 0.01
                elif drag_start[1] == drag_end[1]:
                    drag_start[1] -= 0.01
                    drag_end[1] += 0.01

                gt_box = (min(task_attr['from'][0], task_attr['to'][0]), min(task_attr['from'][1], task_attr['to'][1]),
                         max(task_attr['from'][0], task_attr['to'][0]), max(task_attr['from'][1], task_attr['to'][1]))

                iou = average_iou(np.array([gt_box, [drag_start[0], drag_start[1], drag_end[0], drag_end[1]]])).item()
                step_result['metrics']['action_match'] = step_result['metrics']['drag_match'] = iou > 0.5

            # Hotkey matching
            elif action_type_ref == 'hotkey':
                keycomb_ref = task_attr['key_comb'].replace("_","").replace("-","").replace("+","")
                keycomb_pred = action_pred['key_comb'].replace("_","").replace("-","").replace("+","")
                step_result['metrics']['action_match'] = step_result['metrics']['hotkey_match'] = keycomb_ref == keycomb_pred

            # Other actions (press_key, enter, etc.)
            else:
                step_result['metrics']['action_match'] = step_result['metrics'][f'{action_type_ref}_match'] = True

    except Exception as e:
        inference_time = time.time() - start_time
        step_result['wrong_format'] = True
        logger.error(f"Error evaluating step: {e}")

    return step_result, inference_time


def calculate_metrics(results: Dict[str, List[Dict]], actions: List[str]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Calculate final metrics for each language and overall."""

    metrics_each_lang = {}

    for lang in results.keys():
        num_sample = len([x for x in results[lang] if x is not None])
        num_need_gnd = sum(x['metrics']['need_gnd'] for x in results[lang] if x)
        num_action_match = sum(x['metrics']['action_match'] for x in results[lang] if x)
        num_type_match = sum(x['metrics']['type_match'] for x in results[lang] if x)
        num_elem_match = sum(x['metrics']['elem_acc'] for x in results[lang] if x)

        final_metrics = {
            'step_acc': [(num_action_match / num_sample) if num_sample > 0 else 0., num_action_match, num_sample],
            'action_type_acc': [(num_type_match / num_sample) if num_sample > 0 else 0., num_type_match, num_sample],
            'elem_acc': [(num_elem_match / num_need_gnd) if num_need_gnd > 0 else 0., num_elem_match, num_need_gnd]
        }

        for act in actions:
            if act == 'total':
                continue
            cnt = sum(1 for x in results[lang] if x and x['gt_action'].get('action_type') == act)
            acc_cnt = sum(x['metrics'][f'{act}_match'] for x in results[lang] if x)
            final_metrics[f'{act}_acc'] = [round(acc_cnt / cnt, 4) if cnt > 0 else 0, acc_cnt, cnt]

        final_metrics['num_wrong_format'] = sum(1 for x in results[lang] if x and 'wrong_format' in x)
        metrics_each_lang[lang] = final_metrics

    # Aggregate metrics
    aggr_metrics = {}
    for lang, metrics_subset in metrics_each_lang.items():
        for metric_name, info in metrics_subset.items():
            if metric_name == 'num_wrong_format':
                aggr_metrics['num_wrong_format'] = aggr_metrics.get('num_wrong_format', 0) + metrics_subset['num_wrong_format']
                continue

            if metric_name not in aggr_metrics:
                aggr_metrics[metric_name] = [0, 0, 0]

            aggr_metrics[metric_name][1] += metrics_subset[metric_name][1]
            aggr_metrics[metric_name][2] += metrics_subset[metric_name][2]

    for metric_name in aggr_metrics.keys():
        if metric_name == 'num_wrong_format':
            continue
        acc_cnt, cnt = aggr_metrics[metric_name][1], aggr_metrics[metric_name][2]
        aggr_metrics[metric_name][0] = acc_cnt / cnt if cnt > 0 else 0

    return aggr_metrics, metrics_each_lang

def main():
    """Main execution entry point."""
    args = parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)

    # Load model and components
    model, tokenizer_processor, postfix, gen_kwargs = initialize_model(args.pretrained)

    # Set up directories and paths
    index = 0  # Could be made configurable
    guiact_imgs_dir = ROOT = ["/mnt/vdb1/hongxin_li", ""][index]
    #guiact_imgs_dir = os.path.join(ROOT, "GUICourse/GUIAct")

    eval_result_dir = os.path.join(os.path.dirname(__file__), f'eval_results/GUIAct-{args.device_type}')
    os.makedirs(eval_result_dir, exist_ok=True)
    save_to = os.path.join(eval_result_dir, postfix)

    # Load test data
    test_file = {
        'Web': ['/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse_processed/',
                '/data/hongxin_li/scaling_exp/GUICourse_processed/'][index] + 'guiact-web-test_wActRef_s1000_2346.json',
        'Mobile': ['/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse_processed/',
                   '/data/hongxin_li/scaling_exp/GUICourse_processed/'][index] + 'guiact-smartphone-test_wActRef_s1000_2070.json'
    }[args.device_type]

    logger.info(f"Loading GUIAct test dataset from: {test_file}")
    guiact_test = json.load(open(test_file, 'r'))

    if args.debug:
        guiact_test = random.sample(guiact_test, 30)
        logger.info(f"Debug mode: using {len(guiact_test)} samples")

    # Initialize results structure
    results = {'en': [], 'zh': []}
    actions = ['total', 'click', 'scroll', 'hover', 'drag', 'press_key', 'hotkey', 'status', 'swipe', 'tap', 'input_text', 'enter']
    counts = {k: {act: 0 for act in actions} for k in ['en', 'zh']}

    total_inference_time = 0
    total_steps = 0

    # Main evaluation loop
    pbar = tqdm(enumerate(guiact_test), total=len(guiact_test),
                desc=f"{postfix} on {args.device_type}")
    for step_idx, step in pbar:
        goal = step["task"]
        lang = 'zh' if contains_chinese(goal) else 'en'

        counts[lang]['total'] += 1
        counts[lang][step['action_type']] += 1

        # Evaluate step
        step_result, inference_time = evaluate_step(
            step, model, args, postfix, gen_kwargs, tokenizer_processor, guiact_imgs_dir
        )

        if step_result is None:
            continue

        total_inference_time += inference_time
        total_steps += 1

        results[lang].append(step_result)

        # Logging
        if step_idx % 3 == 0:
            print(f"{Fore.CYAN}{step_result['prompt']}{Style.RESET_ALL} => "
                  f"{Fore.GREEN}{step_result['response']}{Style.RESET_ALL}")

        is_match = step_result['metrics']['action_match']
        print(f"{step_idx}: " + (Fore.GREEN if is_match else Fore.RED) + f"{is_match}" +
              Style.RESET_ALL + f": {step_result['gt_action']} <=> {step_result['action_pred']}")

    # Calculate final metrics
    aggr_metrics, metrics_each_lang = calculate_metrics(results, actions)

    # Add timing information
    avg_inference_time = total_inference_time / total_steps if total_steps > 0 else 0
    aggr_metrics['time_per_step'] = avg_inference_time

    # Print results
    for lang, metrics in metrics_each_lang.items():
        print(f"\n{lang.upper()} Results:")
        pprint(metrics)

    print("\nFinal Overall Results:")
    pprint(aggr_metrics)
    print(f"Average inference time per step: {avg_inference_time:.4f}s")

    # Save results
    os.makedirs(save_to, exist_ok=True)
    time_str = datetime.now().strftime("%m-%d-%H-%M-%S")
    save_file = os.path.join(save_to, f"{args.device_type}-{time_str}{'_debug' if args.debug else ''}.json")

    output_data = {
        "meta": vars(args),
        "overall_results": aggr_metrics,
        "metrics_each_lang": metrics_each_lang,
        "logs": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(save_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Evaluation finished for {args.pretrained} on GUIAct-{args.device_type}. "
                f"Results saved to: {save_file}")


if __name__ == '__main__':
    main()
    

