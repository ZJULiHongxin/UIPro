import os, time, traceback, cv2
import random
import torch
import json
from copy import deepcopy
from tqdm import tqdm
import re
import logging
import ast
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
from action_matching import *
from utils.data_utils.task_prompt_lib import *
from pprint import pprint
from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from utils.data_utils.misc import average_iou
from uipro.conversation import conv_templates
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.osatlas import OSATLAS
from utils.openai_utils.showui import SHOWUI, to_showui_action, showui_to_original_action
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.data_utils.misc import keep_unique_actions, restore_unified_actions
from utils.openai_utils.misc import extract_thought_components
from colorama import Fore, Style

def clean_answer(text):
    text = text.lower().strip(' .?!').replace('"', '').replace("'", '').replace(',', '').replace(";", '')

    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default=['showlab/ShowUI-2B'][-1])
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--cot', type=bool, default=False)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--action_refexp', type=bool, default=True)
    parser.add_argument('--device_type', type=str, default='Web', choices=['Web', 'Mobile'])
    parser.add_argument('--max_prev_acts', type=int, default=6)
    parser.add_argument('--original_actspace', type=bool, default=False)

    args, _ = parser.parse_known_args()

    model_path = args.pretrained.rstrip('/ ')
    if "snapshots" in model_path:
        postfix = model_path[model_path.find("models--") + 8: model_path.find("snapshots") - 1]
    elif len(model_path.split('/')) == 2:
        postfix = model_path.replace('/', '--')
    elif 'checkpoint-' in model_path:
        postfix = '/'.join(model_path.replace("lora/", "").replace("merged/","").split('/')[-2:])

    WO_ACTSPACE, WITH_COT = True, args.cot
    if 'slime' in model_path.lower():
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, use_flash_attn=True)
        model.generation_config.eos_token_id = 107 # '<end_of_turn>'
    elif 'qwen2' in model_path.lower():
        model = QWen2VL(device='cuda', model_name=model_path)

        if 'Qwen/' in model_path:
            WO_ACTSPACE, WITH_COT = False, True
        
        args.scale = 1000
    elif 'atlas' in model_path.lower():
        model = OSATLAS(device='cuda', model_name=model_path)
        args.scale = 1000
        MAX_PREV_ACT = 6
    elif 'show' in model_path.lower():
        model = SHOWUI(device='cuda', model_name=model_path)
        args.scale = 1

    gen_kwargs = {}      

    if "temperature" not in gen_kwargs:
        gen_kwargs["temperature"] = 0
    if "top_p" not in gen_kwargs:
        gen_kwargs["top_p"] = None
    if "num_beams" not in gen_kwargs:
        gen_kwargs["num_beams"] = 1

    def contains_chinese(text):
        # Regular expression to match Chinese characters
        pattern = re.compile('[\u4e00-\u9fff]')
        return bool(pattern.search(text))

    index = 0

    ROOT = ["/mnt/vdb1/hongxin_li",
            ""][index]
    guiact_imgs_dir = os.path.join(ROOT, "GUICourse/GUIAct")

    eval_result_dir = os.path.join(os.path.dirname(__file__), f'eval_results/GUIAct-{args.device_type}')
    os.makedirs(eval_result_dir, exist_ok=True)

    save_to = os.path.join(eval_result_dir, postfix)

    test_file = {'Web': ['/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse_processed/',
                         '/data/hongxin_li/scaling_exp/GUICourse_processed/'][index] + 'guiact-web-test_wActRef_s1000_2346.json',
                 'Mobile': ['/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse_processed/',
                            '/data/hongxin_li/scaling_exp/GUICourse_processed/'][index] + 'guiact-smartphone-test_wActRef_s1000_2070.json'}[args.device_type]
    
    guiact_test = json.load(open(test_file, 'r'))

    results = {'en': [], 'zh': []}
    metrics_each_lang = {}

    # mobile
    # 'status'
    # 'press_key'
    # 'swipe'
    # 'input_text'
    # 'click'

    # Web: 1107 tasks (2346 steps) in totals
    # "click": 1461,
    # "scroll": 560,
    # "status": 197,
    # "hover": 39,
    # "drag": 194,
    # "hotkey": 14

    # press_key can be used to replace enter
    actions = ['total', 'click', 'scroll', 'hover','drag', 'press_key', 'hotkey', 'status', 'swipe', 'tap', 'input_text', 'enter']
    counts = {k:{act: 0 for act in actions} for k in ['en', 'zh']}

    if args.debug:
        guiact_test = random.sample(guiact_test, 15)

    time_record = [0, 0]
    for step_idx, step in tqdm(enumerate(guiact_test), total=len(guiact_test), desc=f"{postfix} on {args.device_type}"):
        goal = step["task"]
        
        lang = 'zh' if contains_chinese(goal) else 'en'

        counts[lang]['total'] += 1
        action_type_ref = step['action_type']
        #if action_type_ref not in ['press_key']: continue
        counts[lang][action_type_ref] += 1

        img_path = os.path.join(ROOT, step["image"])

        try:
            image = Image.open(img_path)
        except: continue
        W,H=image.size
        # Used in the prompt

        # history只有一个元素
        # prompt = SIMPLE_PROMPT.format(global_task=goal, history=step['history'][0], step_instruction='')

        raw_history = step['history']
        if isinstance(raw_history, list):
            retained_idxs, clean_step_instructions = keep_unique_actions(raw_history)
            history = clean_step_instructions[max(0,len(clean_step_instructions)-MAX_PREV_ACT):]

            history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(history, start=max(1,len(clean_step_instructions)-MAX_PREV_ACT+1))) if len(history) > 0 else 'None'
        else:
            history_str = raw_history

        t1 = time.time()

        if 'Qwen/' in model_path:
            prompt = {'Web': GUIACTWEB_PLANNING_PROMPT_COT, 'Mobile': GUIACTMOBILE_PLANNING_PROMPT_COT}[args.device_type].format(
                global_task=goal,
                history=history_str,
                step_instruction=''
            )
        else:
            prompt = make_actionplanning_prompt(goal, history_str, device_tag=args.device_type, prompt_format_type='simple', with_cot=WITH_COT, without_action_space=WO_ACTSPACE, use_action_refexp=args.action_refexp)

        if args.original_actspace:
            prompt = ('[GUIAct-Web] ' if args.device_type == 'Web' else '[GUIAct] ') + prompt

        metrics = {f'{act}_match': False for act in actions}
        metrics['action_match'] = metrics['type_match'] = metrics['elem_acc'] = metrics['need_gnd'] = False

        task_attr = step['task_attr']
        step_result = {"img_path": os.path.basename(img_path), "task": goal, "gt_action": step['step_info'], "prompt": None, "response": None, "original_action": task_attr, "action_pred": None, "metrics":  deepcopy(metrics), 'wrong_format': False}
            
        try:    # several sample's img dir lead to error, just jump it
            if 'slime' in model_path.lower():
                conv = conv_templates['gemma'].copy()
                conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
                conv.append_message(conv.roles[1], None)
                prompt_formatted = conv.get_prompt()

                step_result['prompt'] = prompt_formatted
                img = [Image.open(img_path).convert('RGB')]
                img_tensor = process_images(img, image_processor, model.config).to(dtype=model.dtype, device=model.device)
                
                gen_kwargs["image_sizes"] = [img[0].size]

                input_ids = tokenizer_image_token(prompt_formatted, tokenizer, constants.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=model.device)
                
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
                    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            elif 'showui' in model_path.lower() and 'qwen' not in model_path.lower():
                step_result['prompt'] = prompt
                response = model.get_model_response(
                    prompt,
                    f"file://{img_path}",
                    max_new_tokens=256,
                    history=history_str,
                    scenario='web',
                    task_type='nav2'
                )
            elif 'atlas' in model_path.lower():
                response = model.get_model_response(
                    prompt,
                    f"file://{img_path}",
                    max_new_tokens=4096,
                    sys_prompt=OSATLAS_ANDROIDCONTROL_SYS_PROMPT if 'atlas' in model_path.lower() else OSATLAS_SYS_PROMPT,
                    )
            else:
                step_result['prompt'] = prompt
                response = model.get_model_response(prompt, f"file://{img_path}", max_new_tokens=4096, sys_prompt=OSATLAS_MIND2WEB_PROMPT if 'atlas' in postfix.lower() else '')

            time_record[0] += time.time() - t1
            time_record[1] += 1

            step_result["response"] = response

            if 'atlas' in postfix.lower():
                action_pred = parse_atlas_action(response)
            elif 'Qwen/' in model_path: # qwen official models
                obs, thought, _, action_pred, _ = extract_thought_components(response)
                action_pred = ast.literal_eval(action_pred)
                if isinstance(action_pred.get('target', None), dict) and 'x' in action_pred['target']:
                    action_pred['target'] = [action_pred['target']['x'], action_pred['target']['y']]
                    
            elif 'showui' in model_path.lower() and 'qwen' not in model_path.lower():
                action_pred_raw = ast.literal_eval(response)
                action_pred = showui_to_original_action(action_pred_raw)
                action_pred = ast.literal_eval(action_pred)
            else:
                try:
                    action_pred = ast.literal_eval(response[response.find('{"act'):])
                except:
                    response += '"}'
                    action_pred = ast.literal_eval(response[response.find('{"act'):])
       
            step_result["action_pred"] = action_pred
            
            # restore the unified action definitions
            if args.original_actspace:
                action_pred = restore_unified_actions(action_pred)
            # matching
            action_type_pred = action_pred['action_type']

            if action_type_ref == action_type_pred:
                step_result['metrics']['type_match'] = True

                if action_type_ref in ['click', 'hover']:
                    step_result['metrics']['need_gnd'] = True
                    target_pred = list(map(lambda p: p / args.scale, action_pred['target']))

                    if 'bbox' in task_attr:
                        gt_box_normalized = task_attr['bbox']
                    
                        if gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2] and gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]:
                            step_result['metrics']['action_match'] = step_result['metrics']['elem_acc'] = step_result['metrics'][f'{action_type_ref}_match'] = True
                    elif 'center' in task_attr:
                        center_normalized = task_attr['center']

                        if np.linalg.norm(np.array(center_normalized)-np.array(target_pred)) < 0.14:
                            step_result['metrics']['action_match'] = step_result['metrics']['elem_acc'] = step_result['metrics'][f'{action_type_ref}_match'] = True

                elif action_type_ref == 'input_text':                    
                    if squad_metrics.compute_f1(task_attr['text'], action_pred['text']) > 0.5 :
                        step_result['metrics']['action_match'] = step_result['metrics']['input_text_match'] = True
                
                elif action_type_ref in ['scroll', 'swipe']:
                    step_result['metrics']['action_match'] = step_result['metrics'][f'{action_type_ref}_match'] = task_attr['direction'] == action_pred['direction']
                    
                    if  step_result['metrics'][f'{action_type_ref}_match']:
                        1+1
                
                elif action_type_ref == 'status':
                    status_ref, status_pred = task_attr['goal_status'], action_pred['goal_status']
                    
                    if status_ref == status_pred:
                        answer_ref, answer_pred = clean_answer(task_attr['answer']), clean_answer(action_pred['answer'])
                        if answer_ref in ['', 'task complete', 'task impossible']:
                            answer_f1 = 1.0
                        else:
                            answer_f1 = squad_metrics.compute_f1(answer_ref, answer_pred)
                        
                        step_result['metrics']['action_match'] = step_result['metrics']['status_match'] = answer_f1 > 0.5
                
                elif action_type_ref == 'drag':
                    drag_start, drag_end = list(map(lambda p: p/args.scale, action_pred['start'])), list(map(lambda p: p/args.scale, action_pred['end']))

                    # post-process the drage points to handle the case where the drag points do not form a rectangle
                    if drag_start[0] == drag_end[0]:
                        drag_start[0] -= 0.01
                        drag_end[0] += 0.01
                    elif drag_start[1] == drag_end[1]:
                        drag_start[1] -= 0.01
                        drag_end[1] += 0.01

                    gt_box = min(task_attr['from'][0], task_attr['to'][0]), min(task_attr['from'][1], task_attr['to'][1]), max(task_attr['from'][0], task_attr['to'][0]), max(task_attr['from'][1], task_attr['to'][1]) # the order of dragging start and end is not fixed
                    
                    iou = average_iou(np.array([gt_box, [drag_start[0], drag_start[1], drag_end[0], drag_end[1]]])).item()
                    
                    step_result['metrics']['action_match'] = step_result['metrics']['drag_match'] = iou > 0.5
                
                elif action_type_ref == 'hotkey':
                    keycomb_ref, keycomb_pred = task_attr['key_comb'].replace("_","").replace("-","").replace("+",""), action_pred['key_comb'].replace("_","").replace("-","").replace("+","")
                    
                    step_result['metrics']['action_match'] = step_result['metrics']['hotkey_match'] = keycomb_ref == keycomb_pred
                else: # For 'enter' action
                    step_result['metrics']['action_match'] = step_result['metrics'][f'{action_type_ref}_match'] = True
            
            if step_idx % 3 == 0:
                print(f"{Fore.CYAN}{step_result['prompt']}{Style.RESET_ALL} => {Fore.GREEN}{step_result['response']}{Style.RESET_ALL}")

            is_match = step_result['metrics']['action_match']
            
            print(f"{step_idx}: " + (Fore.GREEN if is_match else Fore.RED) + f"{is_match}" + Style.RESET_ALL + f": {step_result['gt_action']} <=> {action_pred}")

        except Exception as e:
            traceback.print_exc()
            logging.info("format wrong")
            step_result['wrong_format'] = True


        results[lang].append(step_result)

    # calculate metrics
    for lang in results.keys():
        num_sample = counts[lang]['total']
        num_need_gnd = sum(x['metrics']['need_gnd'] for x in results[lang])
        num_action_match = sum(x['metrics']['action_match'] for x in results[lang])
        num_type_match = sum(x['metrics']['type_match'] for x in results[lang])
        num_elem_match = sum(x['metrics']['elem_acc'] for x in results[lang])
        final_metrics = {'step_acc': [(num_action_match / num_sample) if num_sample > 0 else 0., num_action_match, num_sample], 'action_type_acc': [(num_type_match / num_sample) if num_sample > 0 else 0., num_type_match, num_sample], 'elem_acc': [(num_elem_match / num_need_gnd) if num_need_gnd > 0 else 0., num_elem_match, num_need_gnd]}

        for k in counts[lang].keys():
            if k=='total': continue
            
            cnt = counts[lang][k]
            acc_cnt = sum(x['metrics'][f'{k}_match'] for x in results[lang])
            
            final_metrics[f'{k}_acc'] = [round(acc_cnt / cnt, 4) if cnt > 0 else 0, acc_cnt, cnt]

        final_metrics['num_wrong_format'] = sum(1 for x in results[lang] if 'wrong_format' in x)

        pprint(lang + ': ' + str(final_metrics))
        
        metrics_each_lang[lang] = final_metrics

    # aggr metrics
    aggr_metrics = {}
    for lang, metrics_subset in metrics_each_lang.items():
        for metric_name, info in metrics_subset.items():
            if metric_name == 'num_wrong_format':
                aggr_metrics['num_wrong_format'] = aggr_metrics.get('num_wrong_format',0) + metrics_subset['num_wrong_format']
                continue

            if metric_name not in aggr_metrics: aggr_metrics[metric_name] = [0,0,0]
            
            aggr_metrics[metric_name][1] += metrics_subset[metric_name][1]
            aggr_metrics[metric_name][2] += metrics_subset[metric_name][2]

    for metric_name in aggr_metrics.keys():
        if metric_name == 'num_wrong_format': continue
        acc_cnt, cnt = aggr_metrics[metric_name][1], aggr_metrics[metric_name][2]
        aggr_metrics[metric_name][0] = acc_cnt / cnt if cnt > 0 else 0
    
    avg_inference_time = time_record[0] / time_record[1] if time_record[1] > 0 else 0
    aggr_metrics['time_per_step'] = avg_inference_time
        
    print("\nFinal:")
    pprint(aggr_metrics)
    print("Average inference time per step: " + str(avg_inference_time))

    os.makedirs(save_to, exist_ok=True)
    save_file = os.path.join(save_to, args.device_type + '-' + datetime.now().strftime("%m-%d-%H-%M-%S")) + ('_debug' if args.debug else '') + '.json'
    with open(save_file, "w") as f:
        json.dump(
            {
                "meta": vars(args),
                "overall_results": aggr_metrics,
                "metrics_each_lang": metrics_each_lang,
                "logs": results,
            },
            f,
            indent=2
        )

    print(f"Finised evaluation {args.pretrained} on GUIAct-{args.device_type}. Save to {save_file}")
    

