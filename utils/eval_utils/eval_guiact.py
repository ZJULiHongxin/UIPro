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
from utils.data_utils.misc import (
    average_iou, keep_unique_actions, restore_unified_actions, 
    qwen2vl_to_nornal_action, contains_chinese
)
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.osatlas import OSATLAS
from utils.openai_utils.showui import SHOWUI, showui_to_original_action
from utils.openai_utils.misc import extract_thought_components
from utils.data_utils.task_prompt_lib import (
    GUIACTWEB_PLANNING_PROMPT_COT, GUIACTMOBILE_PLANNING_PROMPT_COT,
    make_actionplanning_prompt, parse_atlas_action
)
import transformers.data.metrics.squad_metrics as squad_metrics

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
                        default=['showlab/ShowUI-2B', 'Qwen/Qwen2-VL-7B-Instruct', 'HongxinLi/UIPro-7B_Stage2_Web'][1],
                        help="Path or name of the pretrained model.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode (limited steps).")
    parser.add_argument('--cot', action='store_true', help="Whether to use Chain-of-Thought prompting.")
    parser.add_argument('--scale', type=int, default=1000, help="Coordinate scale (e.g., 1 for ShowUI, 1000 for others).")
    parser.add_argument('--action_refexp', action='store_true', default=True,
                        help="Whether to use action referring expression.")
    parser.add_argument('--device_type', type=str, default='Web',
                        choices=['Web', 'Mobile'], help="Device type for evaluation.")
    parser.add_argument(
        '--root',
        type=str,
        default='/mnt/vdb1/hongxin_li/GUICourse',
        help="Root directory of processed GUIAct data (overrides auto-detect).",
    )
    parser.add_argument(
        '--imgs_dir',
        type=str,
        default=None,
        help="Optional images base directory (overrides auto-detect).",
    )
    parser.add_argument('--max_prev_acts', type=int, default=6,
                        help="Maximum number of previous actions in history.")
    parser.add_argument('--original_actspace', action='store_true',
                        help="Whether to use original GUIAct action space.")

    args, _ = parser.parse_known_args()
    return args


def initialize_model(model_path: str) -> Tuple[Any, Any, str, Dict[str, Any]]:
    """
    Initialize the model based on the model path.
    
    Args:
        model_path: Path or name of the pretrained model
        
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
        model_type: String identifying model type (slime, showui, atlas, qwen, etc.)
        prompt: Formatted prompt text
        img_path: Path to input image
        gen_kwargs: Generation parameters
        tokenizer_processor: Tuple of (tokenizer, image_processor) for certain models

    Returns:
        Model's text response
        
    Raises:
        ValueError: If model type is unsupported or required parameters are missing
    """
    if 'slime' in model_type:
        # Handle SLIME models with custom conversation template
        if tokenizer_processor is None:
            raise ValueError("tokenizer_processor is required for SLIME models")
            
        tokenizer, image_processor = tokenizer_processor
        conv = conv_templates['gemma'].copy()
        conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        img = [Image.open(img_path).convert('RGB')]
        img_tensor = process_images(
            img, image_processor, model.config
        ).to(dtype=model.dtype, device=model.device)

        gen_kwargs_copy = gen_kwargs.copy()
        gen_kwargs_copy["image_sizes"] = [img[0].size]
        
        input_ids = tokenizer_image_token(
            prompt_formatted, tokenizer, constants.IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device=model.device)

        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=img_tensor,
                image_sizes=gen_kwargs_copy["image_sizes"],
                do_sample=True if gen_kwargs_copy["temperature"] > 0 else False,
                temperature=gen_kwargs_copy["temperature"],
                top_p=gen_kwargs_copy["top_p"],
                num_beams=gen_kwargs_copy["num_beams"],
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
        # Handle OS-Atlas models (not typically used for GUIAct but included for completeness)
        # Note: OSATLAS_ANDROIDCONTROL_SYS_PROMPT and similar constants should be imported if needed
        return model.get_model_response(
            prompt,
            f"file://{img_path}",
            max_new_tokens=4096,
            sys_prompt='',  # Can be customized based on model requirements
        )

    else:
        # Handle QWen2VL and other models
        return model.get_model_response(
            prompt, 
            f"file://{img_path}", 
            max_new_tokens=4096,
            sys_prompt=''
        )


def parse_predicted_action(response: str, postfix: str, scale: int) -> Dict[str, Any]:
    """
    Parse the action prediction from model's response string.

    Args:
        response: Raw model response text
        postfix: Model identifier for parsing logic
        scale: Coordinate scaling factor

    Returns:
        Parsed action dictionary
        
    Raises:
        ValueError: If the response cannot be parsed into a valid action
    """
    try:
        if 'atlas' in postfix.lower():
            # Handle OS-Atlas specific format
            return parse_atlas_action(response)
            
        elif 'Qwen--' in postfix:  # Official Qwen models
            # Extract structured components from response
            obs, thought, _, action_pred_str, _ = extract_thought_components(
                response.replace('}}', '}').replace('"target": [', '"target": (')
            )
            action_pred = ast.literal_eval(action_pred_str)
            
            # Convert target dict to list format if needed
            if isinstance(action_pred.get('target'), dict) and 'x' in action_pred['target']:
                action_pred['target'] = [
                    action_pred['target']['x'], 
                    action_pred['target']['y']
                ]
            return action_pred
            
        elif 'showui' in postfix.lower() and 'qwen' not in postfix.lower():
            # Handle ShowUI specific format
            action_pred_raw = ast.literal_eval(response)
            action_pred = showui_to_original_action(action_pred_raw)
            return ast.literal_eval(action_pred)
            
        else:
            # Standard JSON-like format parsing
            # Try to find the action dictionary in the response
            start_idx = response.rfind('{"action')
            if start_idx == -1:
                start_idx = response.find('{"act')
            
            if start_idx == -1:
                raise ValueError(f"No action dictionary found in response: {response[:100]}")
            
            end_idx = response.rfind('}', start_idx) + 1
            
            if end_idx <= start_idx:
                # No closing brace found, try to recover
                action_str = response[start_idx:] + '"}'
            else:
                action_str = response[start_idx:end_idx]
            
            return ast.literal_eval(action_str)
            
    except (ValueError, SyntaxError, KeyError, IndexError) as e:
        logger.error(f"Failed to parse action from response: {response[:200]}")
        raise ValueError(f"Action parsing failed: {str(e)}")


def initialize_step_metrics() -> Dict[str, bool]:
    """Initialize metrics dictionary for tracking action matching."""
    actions = ['click', 'scroll', 'hover', 'drag', 'press_key', 'hotkey', 
               'status', 'swipe', 'tap', 'input_text', 'enter']
    metrics = {f'{act}_match': False for act in actions}
    metrics.update({
        'action_match': False, 
        'type_match': False, 
        'elem_acc': False, 
        'need_gnd': False
    })
    return metrics


def build_prompt(goal: str, history_str: str, args: argparse.Namespace, postfix: str) -> str:
    """
    Build the evaluation prompt based on model type and arguments.
    
    Args:
        goal: The task goal/instruction
        history_str: Formatted history string
        args: Command line arguments
        postfix: Model identifier
        
    Returns:
        Formatted prompt string
    """
    if 'Qwen--' in postfix:
        # Use official Qwen prompt templates
        prompt_template = {
            'Web': GUIACTWEB_PLANNING_PROMPT_COT, 
            'Mobile': GUIACTMOBILE_PLANNING_PROMPT_COT
        }[args.device_type]
        prompt = prompt_template.format(
            global_task=goal,
            history=history_str,
            step_instruction=''
        )
    else:
        # Use general action planning prompt
        prompt = make_actionplanning_prompt(
            goal, history_str, 
            device_tag=args.device_type,
            prompt_format_type='simple', 
            with_cot=args.cot,
            without_action_space=True, 
            use_action_refexp=args.action_refexp
        )

    # Add dataset prefix if using original action space
    if args.original_actspace:
        prefix = '[GUIAct-Web] ' if args.device_type == 'Web' else '[GUIAct] '
        prompt = prefix + prompt
        
    return prompt


def match_click_hover_action(action_pred: Dict, task_attr: Dict, 
                             action_type_ref: str, scale: int) -> Tuple[bool, bool]:
    """
    Match click or hover actions based on target coordinates.
    
    Args:
        action_pred: Predicted action dictionary
        task_attr: Ground truth task attributes
        action_type_ref: Reference action type
        scale: Coordinate scale factor
        
    Returns:
        Tuple of (element_accuracy, action_match)
    """
    target = (eval(action_pred['target']) if isinstance(action_pred['target'], str) 
             else action_pred['target'])
    target_pred = [p / scale for p in target]

    if 'bbox' in task_attr:
        # Check if point is inside bounding box
        gt_box = task_attr['bbox']
        is_match = (gt_box[0] <= target_pred[0] <= gt_box[2] and
                   gt_box[1] <= target_pred[1] <= gt_box[3])
    elif 'center' in task_attr:
        # Check if point is close to center (within threshold)
        center = task_attr['center']
        distance = np.linalg.norm(np.array(center) - np.array(target_pred))
        is_match = distance < 0.14
    else:
        is_match = False
        
    return is_match, is_match


def match_drag_action(action_pred: Dict, task_attr: Dict, scale: int) -> bool:
    """
    Match drag action based on IoU between predicted and ground truth drag regions.
    
    Args:
        action_pred: Predicted action dictionary
        task_attr: Ground truth task attributes
        scale: Coordinate scale factor
        
    Returns:
        True if IoU > 0.5, False otherwise
    """
    drag_start = [p / scale for p in action_pred['start']]
    drag_end = [p / scale for p in action_pred['end']]

    # Post-process drag points to ensure they form a valid rectangle
    if drag_start[0] == drag_end[0]:
        drag_start[0] -= 0.01
        drag_end[0] += 0.01
    elif drag_start[1] == drag_end[1]:
        drag_start[1] -= 0.01
        drag_end[1] += 0.01

    # Create bounding box from ground truth (order not fixed)
    gt_box = (
        min(task_attr['from'][0], task_attr['to'][0]),
        min(task_attr['from'][1], task_attr['to'][1]),
        max(task_attr['from'][0], task_attr['to'][0]),
        max(task_attr['from'][1], task_attr['to'][1])
    )
    
    pred_box = [drag_start[0], drag_start[1], drag_end[0], drag_end[1]]
    iou = average_iou(np.array([gt_box, pred_box])).item()
    
    return iou > 0.5


def match_status_action(action_pred: Dict, task_attr: Dict) -> bool:
    """
    Match status action based on goal status and answer text.
    
    Args:
        action_pred: Predicted action dictionary
        task_attr: Ground truth task attributes
        
    Returns:
        True if status and answer match, False otherwise
    """
    status_ref = task_attr['goal_status']
    status_pred = action_pred['goal_status']
    
    if status_ref != status_pred:
        return False
        
    answer_ref = clean_answer(task_attr['answer'])
    answer_pred = clean_answer(action_pred.get('answer', ''))
    
    # Special cases where exact text match is not required
    if answer_ref in ['', 'task complete', 'task impossible']:
        return True
        
    # Use F1 score for text matching
    answer_f1 = squad_metrics.compute_f1(answer_ref, answer_pred)
    return answer_f1 > 0.5


def evaluate_action_match(action_type_ref: str, action_type_pred: str,
                          action_pred: Dict, task_attr: Dict, 
                          scale: int, metrics: Dict) -> None:
    """
    Evaluate if predicted action matches ground truth and update metrics.
    
    Args:
        action_type_ref: Reference action type
        action_type_pred: Predicted action type
        action_pred: Predicted action dictionary
        task_attr: Ground truth task attributes
        scale: Coordinate scale factor
        metrics: Metrics dictionary to update (modified in place)
    """
    # Type matching
    if action_type_ref != action_type_pred:
        return
        
    metrics['type_match'] = True

    # Element accuracy for click and hover
    if action_type_ref in ['click', 'hover']:
        metrics['need_gnd'] = True
        elem_acc, action_match = match_click_hover_action(
            action_pred, task_attr, action_type_ref, scale
        )
        if action_match:
            metrics['action_match'] = True
            metrics['elem_acc'] = True
            metrics[f'{action_type_ref}_match'] = True

    # Text input matching
    elif action_type_ref == 'input_text':
        text_f1 = squad_metrics.compute_f1(task_attr['text'], action_pred['text'])
        if text_f1 > 0.5:
            metrics['action_match'] = True
            metrics['input_text_match'] = True

    # Scroll and swipe matching
    elif action_type_ref in ['scroll', 'swipe']:
        is_match = task_attr['direction'] == action_pred['direction']
        metrics['action_match'] = is_match
        metrics[f'{action_type_ref}_match'] = is_match

    # Status matching
    elif action_type_ref == 'status':
        is_match = match_status_action(action_pred, task_attr)
        metrics['action_match'] = is_match
        metrics['status_match'] = is_match

    # Drag matching
    elif action_type_ref == 'drag':
        is_match = match_drag_action(action_pred, task_attr, scale)
        metrics['action_match'] = is_match
        metrics['drag_match'] = is_match

    # Hotkey matching
    elif action_type_ref == 'hotkey':
        keycomb_ref = task_attr['key_comb'].replace("_", "").replace("-", "").replace("+", "")
        keycomb_pred = action_pred['key_comb'].replace("_", "").replace("-", "").replace("+", "")
        is_match = keycomb_ref == keycomb_pred
        metrics['action_match'] = is_match
        metrics['hotkey_match'] = is_match

    # Other actions (press_key, enter, tap, etc.)
    else:
        metrics['action_match'] = True
        metrics[f'{action_type_ref}_match'] = True


def evaluate_step(step_data: Dict[str, Any], model: Any, args: argparse.Namespace,
                  postfix: str, gen_kwargs: Dict[str, Any], tokenizer_processor: Optional[Tuple],
                  guiact_imgs_dir: str) -> Tuple[Optional[Dict[str, Any]], float]:
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
        Returns (None, 0) if image cannot be loaded
    """
    goal = step_data["task"]
    action_type_ref = step_data['action_type']
    img_path = os.path.join(guiact_imgs_dir, step_data["image"])

    # Validate image exists
    try:
        Image.open(img_path).close()
    except Exception as e:
        logger.warning(f"Failed to open image {img_path}: {e}")
        return None, 0

    # Format history and build prompt
    history_str = format_history(step_data, args.max_prev_acts)
    prompt = build_prompt(goal, history_str, args, postfix)

    # Initialize result structure
    step_result = {
        "img_path": os.path.basename(img_path),
        "task": goal,
        "gt_action": step_data['step_info'],
        "prompt": prompt,
        "response": None,
        "original_action": step_data['task_attr'],
        "action_pred": None,
        "metrics": initialize_step_metrics(),
        'wrong_format': False
    }

    start_time = time.time()
    try:
        # Get model response
        response = get_model_response(
            model, postfix.lower(), prompt, img_path, gen_kwargs, tokenizer_processor
        )
        step_result["response"] = response

        # Parse action prediction
        action_pred = parse_predicted_action(response, postfix, args.scale)
        
        # Restore unified action definitions if needed
        if args.original_actspace:
            action_pred = restore_unified_actions(action_pred)
            
        step_result["action_pred"] = action_pred

        # Evaluate action matching
        evaluate_action_match(
            action_type_ref, action_pred['action_type'],
            action_pred, step_data['task_attr'],
            args.scale, step_result['metrics']
        )

    except Exception as e:
        step_result['wrong_format'] = True
        logger.error(f"Error evaluating step: {e}")

    inference_time = time.time() - start_time
    return step_result, inference_time


def calculate_metrics(results: Dict[str, List[Dict]], actions: List[str]) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
    """
    Calculate final metrics for each language and overall.
    
    Args:
        results: Dictionary mapping language codes to lists of step results
        actions: List of action types to evaluate
        
    Returns:
        Tuple of (aggregated_metrics, metrics_per_language)
    """
    metrics_each_lang = {}

    for lang in results.keys():
        # Filter out None results
        valid_results = [x for x in results[lang] if x is not None]
        
        num_sample = len(valid_results)
        num_need_gnd = sum(x['metrics']['need_gnd'] for x in valid_results)
        num_action_match = sum(x['metrics']['action_match'] for x in valid_results)
        num_type_match = sum(x['metrics']['type_match'] for x in valid_results)
        num_elem_match = sum(x['metrics']['elem_acc'] for x in valid_results)

        # Calculate overall metrics
        final_metrics = {
            'step_acc': [
                (num_action_match / num_sample) if num_sample > 0 else 0., 
                num_action_match, num_sample
            ],
            'action_type_acc': [
                (num_type_match / num_sample) if num_sample > 0 else 0., 
                num_type_match, num_sample
            ],
            'elem_acc': [
                (num_elem_match / num_need_gnd) if num_need_gnd > 0 else 0., 
                num_elem_match, num_need_gnd
            ]
        }

        # Calculate per-action metrics
        for act in actions:
            if act == 'total':
                continue
            cnt = sum(1 for x in valid_results if x['gt_action'].get('action_type') == act)
            acc_cnt = sum(x['metrics'][f'{act}_match'] for x in valid_results)
            final_metrics[f'{act}_acc'] = [
                round(acc_cnt / cnt, 4) if cnt > 0 else 0, 
                acc_cnt, cnt
            ]

        final_metrics['num_wrong_format'] = sum(
            1 for x in valid_results if x.get('wrong_format', False)
        )
        metrics_each_lang[lang] = final_metrics

    # Aggregate metrics across languages
    aggr_metrics = {}
    for lang, metrics_subset in metrics_each_lang.items():
        for metric_name, info in metrics_subset.items():
            if metric_name == 'num_wrong_format':
                aggr_metrics['num_wrong_format'] = (
                    aggr_metrics.get('num_wrong_format', 0) + 
                    metrics_subset['num_wrong_format']
                )
                continue

            if metric_name not in aggr_metrics:
                aggr_metrics[metric_name] = [0, 0, 0]

            aggr_metrics[metric_name][1] += metrics_subset[metric_name][1]
            aggr_metrics[metric_name][2] += metrics_subset[metric_name][2]

    # Calculate final percentages
    for metric_name in aggr_metrics.keys():
        if metric_name == 'num_wrong_format':
            continue
        acc_cnt, cnt = aggr_metrics[metric_name][1], aggr_metrics[metric_name][2]
        aggr_metrics[metric_name][0] = acc_cnt / cnt if cnt > 0 else 0

    return aggr_metrics, metrics_each_lang

def get_dataset_paths(device_type: str, base_path: Optional[str] = None, imgs_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Get the paths for test data and images directory based on environment.
    
    Args:
        device_type: 'Web' or 'Mobile'
        
    Returns:
        Tuple of (test_file_path, images_dir)
    """

    test_files = {
        'Web': 'guiact-web-test_wActRef_s1000_2346.json',
        'Mobile': 'guiact-smartphone-test_wActRef_s1000_2070.json'
    }
    
    test_file = os.path.join(base_path, test_files[device_type])
    
    # Images directory (can be empty string if images are in the same base path)
    if imgs_dir is None:
        imgs_dir = '/mnt/vdb1/hongxin_li' if os.path.exists('/mnt/vdb1/hongxin_li') else ''

    return test_file, imgs_dir


def evaluate_dataset(guiact_test: List[Dict], model: Any, args: argparse.Namespace,
                     postfix: str, gen_kwargs: Dict[str, Any], 
                     tokenizer_processor: Optional[Tuple], 
                     guiact_imgs_dir: str) -> Tuple[Dict[str, List], float, int]:
    """
    Evaluate the entire dataset.
    
    Args:
        guiact_test: List of test samples
        model: Initialized model instance
        args: Parsed command line arguments
        postfix: Model identifier string
        gen_kwargs: Model generation parameters
        tokenizer_processor: Tokenizer and processor for certain models
        guiact_imgs_dir: Directory containing GUIAct images
        
    Returns:
        Tuple of (results_dict, total_inference_time, total_steps)
    """
    results = {'en': [], 'zh': []}
    total_inference_time = 0
    total_steps = 0

    pbar = tqdm(enumerate(guiact_test), total=len(guiact_test),
                desc=f"{postfix} on {args.device_type}")
    
    for step_idx, step in pbar:
        goal = step["task"]
        lang = 'zh' if contains_chinese(goal) else 'en'

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
        if args.debug or step_idx % 3 == 0:
            print(f"{Fore.CYAN}{step_result['prompt']}{Style.RESET_ALL} => "
                  f"{Fore.GREEN}{step_result['response']}{Style.RESET_ALL}")

        is_match = step_result['metrics']['action_match']
        status_color = Fore.GREEN if is_match else Fore.RED
        print(f"{step_idx}: {status_color}{is_match}{Style.RESET_ALL}: "
              f"{step_result['gt_action']} <=> {step_result['action_pred']}")

    return results, total_inference_time, total_steps


def save_results(save_dir: str, device_type: str, args: argparse.Namespace,
                 aggr_metrics: Dict, metrics_each_lang: Dict, 
                 results: Dict, is_debug: bool) -> str:
    """
    Save evaluation results to JSON file.
    
    Args:
        save_dir: Directory to save results
        device_type: 'Web' or 'Mobile'
        args: Parsed command line arguments
        aggr_metrics: Aggregated metrics across languages
        metrics_each_lang: Metrics for each language
        results: Detailed results for all steps
        is_debug: Whether in debug mode
        
    Returns:
        Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    time_str = datetime.now().strftime("%m-%d-%H-%M-%S")
    save_file = os.path.join(save_dir, f"{device_type}-{time_str}{'_debug' if is_debug else ''}.json")

    output_data = {
        "meta": vars(args),
        "overall_results": aggr_metrics,
        "metrics_each_lang": metrics_each_lang,
        "logs": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(save_file, "w") as f:
        json.dump(output_data, f, indent=2)
        
    return save_file


def print_results(metrics_each_lang: Dict, aggr_metrics: Dict, avg_inference_time: float) -> None:
    """Print evaluation results to console."""
    for lang, metrics in metrics_each_lang.items():
        print(f"\n{lang.upper()} Results:")
        pprint(metrics)

    print("\nFinal Overall Results:")
    pprint(aggr_metrics)
    print(f"Average inference time per step: {avg_inference_time:.4f}s")


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
    test_file, guiact_imgs_dir = get_dataset_paths(
        args.device_type, root=args.root, imgs_dir=args.imgs_dir
    )
    
    eval_result_dir = os.path.join(os.path.dirname(__file__), f'eval_results/GUIAct-{args.device_type}')
    os.makedirs(eval_result_dir, exist_ok=True)
    save_dir = os.path.join(eval_result_dir, postfix)

    # Load test data
    logger.info(f"Loading GUIAct test dataset from: {test_file}")
    with open(test_file, 'r') as f:
        guiact_test = json.load(f)

    if args.debug:
        guiact_test = random.sample(guiact_test, min(30, len(guiact_test)))
        logger.info(f"Debug mode: using {len(guiact_test)} samples")

    # Main evaluation
    results, total_inference_time, total_steps = evaluate_dataset(
        guiact_test, model, args, postfix, gen_kwargs, 
        tokenizer_processor, guiact_imgs_dir
    )

    # Calculate final metrics
    actions = ['total', 'click', 'scroll', 'hover', 'drag', 'press_key', 
               'hotkey', 'status', 'swipe', 'tap', 'input_text', 'enter']
    aggr_metrics, metrics_each_lang = calculate_metrics(results, actions)

    # Add timing information
    avg_inference_time = total_inference_time / total_steps if total_steps > 0 else 0
    aggr_metrics['time_per_step'] = avg_inference_time

    # Print and save results
    print_results(metrics_each_lang, aggr_metrics, avg_inference_time)
    
    save_file = save_results(save_dir, args.device_type, args, aggr_metrics, 
                            metrics_each_lang, results, args.debug)

    logger.info(f"Evaluation finished for {args.pretrained} on GUIAct-{args.device_type}. "
                f"Results saved to: {save_file}")


if __name__ == '__main__':
    main()
    

