"""
Evaluation script for Android In The Wild (AITW) dataset.
This script calculates action matching scores by comparing model predictions with ground truth actions.
Reference: https://github.com/google-research/google-research/tree/master/android_in_the_wild
"""

import os
import time
import cv2
import random
import torch
import json
import logging
import ast
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import datasets
from tqdm import tqdm
from PIL import Image
from colorama import Style, Fore

# Local imports
from action_matching import *
from utils.data_utils.task_prompt_lib import *
from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.showui import SHOWUI, to_showui_action, showui_to_original_action
from utils.data_utils.misc import keep_unique_actions, qwen2vl_to_nornal_action

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model on AITW dataset.")
    parser.add_argument('--pretrained', type=str, 
                        default=['HongxinLi/UIPro_2stage_Mobile','Qwen/Qwen2-VL-7B-Instruct'][0],
                        help="Path or name of the pretrained model.")
    parser.add_argument('--imgs_dir', type=str, 
                        default='/mnt/vdb1/hongxin_li/AITW/aitw_images/',
                        help="Directory containing AITW images.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode (limited steps).")
    parser.add_argument('--cot', action='store_true', help="Whether to use Chain-of-Thought prompting.")
    parser.add_argument('--action_refexp', action='store_true', default=True, help="Whether to use action referring expression.")
    parser.add_argument('--scale', type=int, default=1000, help="Coordinate scale (e.g., 1000).")
    parser.add_argument('--device_tag', type=str, default='Android', help="Device tag used in prompts.")
    parser.add_argument('--max_prev_acts', type=int, default=6, help="Maximum number of previous actions in history.")
    parser.add_argument('--original_actspace', action='store_true', help="Whether to use original AITW action space.")
    
    args, _ = parser.parse_known_args()
    return args


def initialize_model(model_path: str) -> Tuple[Any, Any, str, int]:
    """
    Initialize the model based on the model path.
    Returns: (model, tokenizer/processor, postfix, max_prev_act)
    """
    model_path = model_path.rstrip('/ ')
    
    # Generate a postfix for identification from the model path
    if "snapshots" in model_path:
        postfix = model_path[model_path.find("models--") + 8: model_path.find("snapshots") - 1]
    elif len(model_path.split('/')) == 2:
        postfix = model_path.replace('/', '--')
    elif 'checkpoint-' in model_path:
        postfix = '/'.join(model_path.replace("lora/", "").replace("merged/", "").split('/')[-2:])
    else:
        postfix = os.path.basename(model_path)

    logger.info(f"Initializing model: {model_path}")
    
    if 'slime' in model_path.lower():
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, use_flash_attn=True)
        model.generation_config.eos_token_id = 107  # '<end_of_turn>'
        return model, (tokenizer, image_processor), postfix, 4
    
    elif any(k in model_path.lower() for k in ['qwen2', 'atlas', 'uipro']):
        model = QWen2VL(device='cuda', model_name=model_path)
        return model, None, postfix, 6
        
    elif 'showui' in model_path.lower():
        model = SHOWUI(device='cuda', model_name=model_path)
        return model, None, postfix, 6
        
    else:
        # Default to QWen2VL for other models
        model = QWen2VL(device='cuda', model_name=model_path)
        return model, None, postfix, 6


def format_history(step_data: Dict, max_prev_acts: int, postfix: str, step_idx: int) -> str:
    """Format the interaction history into a string."""
    raw_history = step_data['history']
    
    if isinstance(raw_history, list):
        # Filter and clean history steps
        _, clean_step_instructions = keep_unique_actions(raw_history)
        history = clean_step_instructions[max(0, len(clean_step_instructions) - max_prev_acts):]

        # Use different separators based on model type
        is_atlas = 'atlas' in postfix.lower()
        end_punc = ':' if is_atlas else '.'
        line_split = '\n' if is_atlas else ' '
        
        if not history:
            return 'None'
            
        history_str = line_split.join(
            f"Step {i}{end_punc} {instruc.strip(' .')}." 
            for i, instruc in enumerate(history, start=max(1, step_idx - max_prev_acts + 1))
        )
        return history_str
        
    elif isinstance(raw_history, str) and f'Step {max_prev_acts+1}.' in raw_history:
        # Parse history from string format
        steps = []
        curr_pos = 0
        while True:
            next_step_pos = raw_history.find('Step ', curr_pos + 1)
            is_last = next_step_pos == -1
            
            if is_last:
                next_step_pos = raw_history.find('\n', curr_pos + 1)
                if next_step_pos == -1:
                    next_step_pos = len(raw_history)
            
            dot_pos = raw_history.find('. ', curr_pos)
            if dot_pos != -1:
                this_step = raw_history[dot_pos + 2:next_step_pos].strip(' .')
                steps.append(this_step)
            
            if is_last or next_step_pos >= len(raw_history):
                break
            curr_pos = next_step_pos
            
        return ' '.join(
            f'Step {i}. {s}.' 
            for i, s in enumerate(steps[-max_prev_acts:], start=max(1, len(steps) - max_prev_acts + 1))
        )
    else:
        return str(raw_history)


def get_model_response(model: Any, model_type: str, prompt: str, img_path: str, 
                       history_str: str, postfix: str, tokenizer_processor: Optional[Tuple] = None) -> str:
    """Get response from the model with handling for different model types."""
    
    if 'slime' in model_type:
        tokenizer, image_processor = tokenizer_processor
        conv = conv_templates['gemma'].copy()
        conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        img = [Image.open(img_path).convert('RGB')]
        img_tensor = process_images(img, image_processor, model.config).to(dtype=model.dtype, device=model.device)
        
        input_ids = tokenizer_image_token(
            prompt_formatted, tokenizer, constants.IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device=model.device)
        
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=img_tensor,
                image_sizes=[img[0].size],
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                use_cache=True,
            )
            return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            
    elif 'showui' in model_type:
        return model.get_model_response(
            prompt,
            f"file://{img_path}",
            max_new_tokens=256,
            history=history_str,
            scenario='phone',
            task_type='nav2'
        )
    else:
        # QWen2VL or Atlas
        temp, retry = 0.0, 0
        curr_prompt = prompt
        sys_prompt = OSATLAS_AITW_SYS_PROMPT if 'atlas' in postfix.lower() else ''
        
        while retry <= 5:
            response = model.get_model_response(
                curr_prompt, f"file://{img_path}", temperature=temp, sys_prompt=sys_prompt
            )
            # Special handling for Atlas to avoid 'open_app' loop
            if 'open_app' in response:
                avoid_msg = random.choice([
                    'You SHOULD not', 'You MUST not', 'You are not allowed to', 'Do NOT', 'You can not'
                ])
                curr_prompt = prompt + f" ({avoid_msg} use the open_app action)"
                temp = 1.0
                retry += 1
            else:
                break
        return response


def parse_predicted_action(response: str, postfix: str, scale: int) -> Dict:
    """Parse the action from the model's response string."""
    if 'atlas' in postfix.lower():
        from utils.openai_utils.misc import parse_atlas_action
        return pred_2_format(parse_atlas_action(response, device='mobile'), scale=scale)
    elif 'showui' in postfix.lower() and 'qwen' not in postfix.lower():
        action_pred_raw = ast.literal_eval(response)
        return pred_2_format(showui_to_original_action(action_pred_raw, scale=scale), scale=scale)
    else:
        # Standard format: "Action: { ... }"
        if 'Action:' in response:
            action_str = response.split('Action:')[1].strip().split('\n')[-1]
        elif '{"action' in response:
            action_str = response[response.rfind('{"action'): response.rfind('}')+1]
            if 'Qwen2-VL' in postfix:
                action_str = qwen2vl_to_nornal_action(action_str.replace('}}','}'))
        else:
            action_str = response.strip().split('\n')[-1]
            
        # Clean up string if it contains markdown code blocks
        if '```' in action_str:
            action_str = action_str.replace('```json', '').replace('```', '').strip()
            
        try:
            return pred_2_format(ast.literal_eval(action_str), scale=scale)
        except (ValueError, SyntaxError):
            # Fallback if evaluation fails
            raise ValueError(f"Failed to parse action string: {action_str}")


def evaluate_task(task_name: str, steps: List[Dict], model: Any, args: argparse.Namespace, 
                  postfix: str, max_prev_act: int, tokenizer_processor: Optional[Tuple], 
                  aitw_imgs_dir: str) -> Tuple[Dict, List[Dict], float]:
    """Evaluate all steps in a single task."""
    logger.info(f"Starting evaluation for task: {task_name}")

    task_logs = []
    stats = {
        'corr_action': 0, 'corr_type': 0, 'num_text': 0, 'corr_text': 0,
        'num_scroll': 0, 'corr_scroll': 0, 'num_click': 0, 'corr_click': 0,
        'num_both_click': 0, 'corr_both_click': 0, 'num_wrong_format': 0, 'num': 0
    }
    
    # AITW tests these action types: {'press_key', 'click', 'navigate_back', 'input_text', 'status', 'swipe', 'navigate_home'}

    total_time = 0
    
    pbar = tqdm(enumerate(steps), total=len(steps), 
                desc=f"{task_name} | CoT: {args.cot} | ActRef: {args.action_refexp}")
                
    for step_i, step_data in pbar:
        if args.debug and step_i >= 6:
            break
        # Action debug
        # if step_data['action_type'] not in ['swipe']:continue

        img_filename = step_data["image"]
        step_idx = int(img_filename.split('_')[-1][:-4])
        img_path = os.path.join(aitw_imgs_dir, img_filename)
        
        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue

        goal = step_data["step"]["goal"]
        action_ref = action_2_format(step_data["step"])
        
        # Prepare history string
        history_str = format_history(step_data, max_prev_act, postfix, step_idx)
        
        # Build prompt
        if 'atlas' in postfix.lower():
            prompt = ATLAS_PROMPT.format(global_task=goal, history=history_str)
        elif 'showui' in postfix.lower() and 'qwen' not in postfix.lower():
            prompt = goal
        elif 'Qwen2-VL' in postfix:
            prompt = make_planning_protocol(
                bmk_name='AITW',
                task=goal,
                history=history_str,
                protocol_type='no',
                use_guidelines=False,
                use_qwen_actspace=True,
                use_obs=False, use_progress=False, use_index=False, use_outcome=False
            )
        else:
            prompt = make_actionplanning_prompt(
                goal, history_str, device_tag=args.device_tag, 
                prompt_format_type='simple', with_cot=args.cot, 
                without_action_space=True, use_action_refexp=args.action_refexp
            )

        if args.original_actspace:
            prompt = '[AITW] ' + prompt

        start_time = time.time()
        response = ""
        try:
            response = get_model_response(
                model, postfix.lower(), prompt, img_path, 
                history_str, postfix, tokenizer_processor
            )
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            
        inference_time = time.time() - start_time
        total_time += inference_time
        stats['num'] += 1
        
        if args.debug or step_i % 5 == 0:
            print(Fore.CYAN + f"Prompt: {prompt}\n" + Fore.YELLOW + f"Response: {response}\n" + Style.RESET_ALL)
            
        # Logging entry
        log_entry = {
            "step_id": f"{step_data['step']['ep_id']}-{step_data['step']['step']}",
            "img_path": img_path,
            "goal": goal,
            "prompt": prompt,
            "response": response,
            "action_ref": action_ref,
            "status": ""
        }

        try:
            action_pred = parse_predicted_action(response, postfix, args.scale)
            log_entry["action_pred"] = action_pred
            
            # Ground truth annotations for matching
            annot_pos = np.array([
                step_data["step"]["annot_position"][i:i + 4] 
                for i in range(0, len(step_data["step"]["annot_position"]), 4)
            ])
            
            # Action matching logic
            is_match = check_actions_match(
                action_pred["touch_point"], action_pred["lift_point"], action_pred["action_type"],
                action_ref["touch_point"], action_ref["lift_point"], action_ref["action_type"],
                annot_pos
            )
            
            # Update statistics
            if is_match:
                stats['corr_action'] += 1
                log_entry["status"] = "correct"
            else:
                log_entry["status"] = "wrong"
                
            # Type accuracy
            if action_pred["action_type"] == action_ref["action_type"]:
                stats['corr_type'] += 1
                log_entry["status"] += ",type_correct"
            else:
                log_entry["status"] += ",type_wrong"
                
            # Text accuracy (Type 3 is TYPE)
            if action_ref["action_type"] == 3:
                stats['num_text'] += 1
                pred_text = action_pred.get("typed_text", "").lower()
                ref_text = action_ref.get("typed_text", "").lower()
                if pred_text == ref_text or pred_text in ref_text or ref_text in pred_text:
                    stats['corr_text'] += 1
                    log_entry["status"] += ",text_correct"
                else:
                    log_entry["status"] += ",text_wrong"
                    
            # Click vs Scroll accuracy (Type 4 is DUAL_POINT)
            if action_ref["action_type"] == 4:
                if is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                    stats['num_click'] += 1
                    if is_match: stats['corr_click'] += 1
                else:
                    stats['num_scroll'] += 1
                    if is_match: stats['corr_scroll'] += 1
                    
                # Both identified as click
                if (action_pred["action_type"] == 4 and 
                    is_tap_action(action_ref["touch_point"], action_ref["lift_point"]) and 
                    is_tap_action(action_pred["touch_point"], action_pred["lift_point"])):
                    stats['num_both_click'] += 1
                    if is_match: stats['corr_both_click'] += 1
            
            status_color = Fore.GREEN if is_match else Fore.RED
            print(f"{status_color}{log_entry['status']}{Style.RESET_ALL} | "
                  f"{Fore.CYAN}GT: {action_ref['action_type']} {action_ref.get('typed_text', '')}{Style.RESET_ALL} | "
                  f"{Fore.YELLOW}Pred: {action_pred['action_type']} {action_pred.get('typed_text', '')}{Style.RESET_ALL}")
            
        except Exception as e:
            stats['num_wrong_format'] += 1
            log_entry["status"] = f"wrong format: {str(e)}"
            logger.error(f"Error parsing action: {e}")

        task_logs.append(log_entry)
        
    # Calculate final task metrics
    num = stats['num'] if stats['num'] > 0 else 1
    results = {
        "action_acc": f"{100 * stats['corr_action'] / num:.2f}% ({stats['corr_action']}/{num})",
        "type_acc": f"{100 * stats['corr_type'] / num:.2f}% ({stats['corr_type']}/{num})",
        "text_acc": f"{100 * stats['corr_text'] / (stats['num_text'] or 1):.2f}%",
        "click_acc": f"{100 * stats['corr_click'] / (stats['num_click'] or 1):.2f}%",
        "scroll_acc": f"{100 * stats['corr_scroll'] / (stats['num_scroll'] or 1):.2f}%",
        "dual_click_acc": f"{100 * stats['corr_both_click'] / (stats['num_both_click'] or 1):.2f}%",
        "num_wrong_format": stats['num_wrong_format']
    }
    
    return results, task_logs, total_time


def main():
    """Main execution entry point."""
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    
    # Load model and components
    model, components, postfix, max_prev_act = initialize_model(args.pretrained)
    
    # Load dataset
    logger.info("Loading AITW test dataset...")
    aitw_test = datasets.load_dataset("HongxinLi/AITW_test", split='test')
    
    # Group steps by application/task
    aitw_test_by_app = {}
    for entry in aitw_test:
        app = entry['image'].split('/')[0]
        aitw_test_by_app.setdefault(app, []).append(entry)
        
    all_results = {}
    all_logs = {}
    total_corr_action = 0
    total_steps = 0
    total_inference_time = 0
    
    # Iterate through each app/task
    for task_name, steps in aitw_test_by_app.items():
        task_results, task_logs, task_time = evaluate_task(
            task_name, steps, model, args, postfix, 
            max_prev_act, components, args.imgs_dir
        )
        
        all_results[task_name] = task_results
        all_logs[task_name] = task_logs
        total_inference_time += task_time
        
        # Aggregate for average score
        # Extract numerators and denominators for overall accuracy
        try:
            corr_part = task_results['action_acc'].split('(')[1].split('/')[0]
            num_part = task_results['action_acc'].split('/')[1].split(')')[0]
            total_corr_action += int(corr_part)
            total_steps += int(num_part)
        except (IndexError, ValueError):
            logger.error(f"Failed to parse results for task {task_name}")
            
        logger.info(f"Task {task_name} Results: {task_results}")

    # Final summary
    avg_score = total_corr_action / total_steps if total_steps > 0 else 0
    avg_time = total_inference_time / total_steps if total_steps > 0 else 0
    
    all_results['avg_score'] = f"{100 * avg_score:.2f}%"
    all_results['time_per_step'] = f"{avg_time:.4f}s"
    
    logger.info(f"Evaluation finished. Average Score: {all_results['avg_score']}")
    
    # Save results
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_result_dir = os.path.join(
        os.path.dirname(__file__), 'eval_results/AITW', 
        f"{postfix}{'_CoT' if args.cot else ''}{'_wActRef' if args.action_refexp else ''}"
    )
    os.makedirs(eval_result_dir, exist_ok=True)
    save_path = os.path.join(eval_result_dir, f"{time_str}.json")
    
    output_data = {
        "meta": vars(args),
        "eval_result": all_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "logs": all_logs
    }
    
    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to: {save_path}")


if __name__ == '__main__':
    main()
