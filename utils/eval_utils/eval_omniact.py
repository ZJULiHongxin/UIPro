import os, time, magic, cv2, re
from copy import deepcopy
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
from action_matching import *
from eval_utils import mind2web_action2step
from utils.data_utils.task_prompt_lib import *
from pprint import pprint
from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates, SeparatorStyle
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.openai_utils.qwen2vl import QWen2VL
from utils.data_utils.misc import keep_unique_actions
from colorama import Fore, Style

#os.environ["CUDA_VISIBLE_DEVICES"]="7"
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='/mnt/vdb1/hongxin_li/uipro_ckpt/0126_OS-Atlas-Base-7B_FullActSpace_s1000_6478/lora/checkpoint-150')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--cot', type=bool, default=False)
parser.add_argument('--scale', type=int, default=100)
parser.add_argument('--action_refexp', type=bool, default=True)
parser.add_argument('--relaxed_gnd', type=bool, default=True)

args = parser.parse_args()

# 数据集
ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/OmniAct_processed"
ROOT2 = "/mnt/vdb1/hongxin_li"

guiact_test = json.load(open(f'{ROOT}/OmniAct_test.json', 'r'))

# Load model
model_path = args.pretrained.rstrip('/ ')
print(f"Loading model from {model_path}")

if 'slime' in model_path.lower():
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, use_flash_attn=True)

    model.generation_config.eos_token_id = 107 # '<end_of_turn>'
    gen_kwargs = {}

    if "temperature" not in gen_kwargs:
        gen_kwargs["temperature"] = 0
    if "top_p" not in gen_kwargs:
        gen_kwargs["top_p"] = None
    if "num_beams" not in gen_kwargs:
        gen_kwargs["num_beams"] = 1
    
    args.scale = 100
elif 'uipro' in model_path.lower():
    model = QWen2VL(device='cuda', model_name=model_path)
    args.scale = 1000
    MAX_PREV_ACT = 6
elif 'qwen2' in model_path.lower():
    MAX_PREV_ACT = 999
    if model_path in ['Qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2-VL-2B-Instruct']:
        model = QWen2VL(device='cuda', model_name=model_path)
    else:
        model = QWen2VL(device='cuda', model_name=model_path)

    args.scale = 1000
    MAX_PREV_ACT= 999
elif 'atlas' in model_path.lower():
    model = QWen2VL(device='cuda', model_name=model_path)
    args.scale = 1000
    MAX_PREV_ACT = 6
    ATLAS_PROMPT = "Task: {}\nHistory: \n{history}\n"


model_path = model_path.replace("merged/", "").replace("lora/","")

if "snapshots" in model_path:
    postfix = model_path[model_path.find("models--") + 8: model_path.find("snapshots") - 1]
else:
    postfix = '/'.join(model_path.split('/')[-2:])

def contains_chinese(text):
    # Regular expression to match Chinese characters
    pattern = re.compile('[\u4e00-\u9fff]')
    return bool(pattern.search(text))

results = []

# Omniact评测这些动作
# "click": 4946,
# "hover": 144,
# "right_click": 60,
# "double_click": 52,
# "hotkey": 96,
# "press_key": 12,
# "scroll": 44,
# "input_text": 4


actions = ['total', 'click', 'scroll', 'input_text', 'hotkey', 'hover','right_click', 'double_click', 'press_key']

counts = {act: 0 for act in actions}

for step_idx, step in tqdm(enumerate(guiact_test), total=len(guiact_test), desc=args.pretrained):
    goal = step["task"]
    # if step['action_type'] != 'hotkey':
    #     continue
    counts['total'] += 1
    action_type_ref = step['action_type']
    counts[action_type_ref] += 1

    img_path = os.path.join(ROOT2, step["image"])

    image = Image.open(img_path)
    W,H=image.size
    # Used in the prompt
    # history_str = '. '.join(f'Step {i}: {action}' for i, action in enumerate(step['history'][-MAX_PREV_ACT:], start=1)) if step['step_id'] > 0 else 'None'
    history_str = ''

    if 'atlas' in postfix.lower():
        prompt = ATLAS_PROMPT.format(global_task=goal, history='None')
    else:
        prompt = make_actionplanning_prompt(
                        goal,
                        'None',
                        step_instruction='',
                        device_tag='',
                        prompt_format_type='simple',
                        with_cot=args.cot,
                        without_action_space=True,
                        use_action_refexp=args.action_refexp
                    )
    # OMNIACT_PROMPT.format(global_task=goal, history=history_str, step_instruction='')
    t1 = time.time()
    
    metrics = {f'{act}_match': False for act in actions}
    metrics['action_match'] = metrics['type_match'] = metrics['elem_acc'] = False


    # 这里存每一步的图像路径、任务、prompt、模型回复、GT action with 动作参数、预测的动作、单步评测结果
    task_attr = step['task_attr']
    step_result = {"img_path": os.path.basename(img_path), "task": goal, "prompt": prompt, "response": None, "GT_action": task_attr, "action_pred": None, "metrics":  deepcopy(metrics) }

    try:    # several sample's img dir lead to error, just jump it
        if 'slime' in postfix.lower():
            conv = conv_templates['gemma'].copy()
            conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
            conv.append_message(conv.roles[1], None)
            prompt_formatted = conv.get_prompt()
            img = [Image.open(img_path).convert('RGB')]
            img_tensor = process_images(img, image_processor, model.config).to(dtype=model.dtype, device=model.device)

            step_result['prompt'] = prompt_formatted
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
        else:
            prompt = prompt.replace("Now, first describe", "You can use the action {'action_type': 'hotkey', 'key_comb': ['ctrl-shift-n' (open new tab/create a new folder), 'ctrl-a' (select all), 'winleft-a' (open new window), 'win-d' (close current window), 'ctrl-+' (zoom in), 'ctrl--' (zoom out), 'ctrl-f' (find/search), 'ctrl-w' (close tab), 'ctrl-y' (redo), 'alt-f4' (close window), 'ctrl-shift-tab' (previous tab), 'ctrl-tab' (next tab), 'ctrl-t' (new tab), 'ctrl-down' (scroll down), 'ctrl-up' (scroll up)]} to press a hotkey.\nYou can use the action {'action_type': 'press_key', 'key': ['space', 'pgdn', 'tab', 'esc', 'up', 'down', 'left', 'right', 'enter', ...]} to press a single key.\n\nNow, first describe")

            response = model.get_model_response(prompt, f"file://{img_path}", max_new_tokens=4096, sys_prompt=OSATLAS_OMNIACT_PROMPT if 'atlas' in postfix.lower() else '')


        if step_idx % 2 == 0:
            print(f"{Fore.CYAN}{prompt}{Style.RESET_ALL} -> {Fore.GREEN}{response}{Style.RESET_ALL}")

        step_result["response"] = response

        if 'atlas' in postfix.lower():
            action_pred = parse_atlas_action(response)
        else:
            action_pred = ast.literal_eval(response[response.rfind('{"action_type'):])

        step_result["action_pred"] = action_pred

        # matching
        action_type_pred = action_pred['action_type']

        if action_type_ref == action_type_pred:
            step_result['metrics']['type_match'] = True

            if action_type_ref in ['click', 'double_click', 'right_click', 'hover']:
                target_pred = list(map(lambda p: p / args.scale, action_pred['target']))

                flag = False
                if 'unnormalized_box' in step['task_attr']:
                    gt_box = step['task_attr']['unnormalized_box']
                    gt_box_normalized = list(map(lambda p:round(p, 3), [gt_box[0]/W, gt_box[1]/H, gt_box[2]/W, gt_box[3]/ H]))

                    step['task_attr']['bbox'] = gt_box_normalized

                    if args.relaxed_gnd: # 14% screen width used in OS-ATLAS
                        norm_gt_target = [gt_box_normalized[0]-0.14, gt_box_normalized[1]-0.14, gt_box_normalized[2]+0.14, gt_box_normalized[3]+0.14]
                        if norm_gt_target[0] <= target_pred[0] <= norm_gt_target[2] and norm_gt_target[1] <= target_pred[1] <= norm_gt_target[3]:
                            flag = True
                    else:
                        if gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2] and gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]:
                            flag = True
                else:
                    gt_target = step['task_attr']['target']

                    W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_path)).groups(1)))
                    # we consider the coordinates correct if they fall within a distance of 14% screen width from the ground truth.
                    norm_gt_target = [gt_target[0]/W, gt_target[1]/H]
                    if np.linalg.norm(np.array(norm_gt_target) - np.array(target_pred)) <= 0.14:
                        flag = True

                if flag:
                    step_result['metrics']['action_match'] = step_result['metrics']['elem_acc'] = step_result['metrics'][f'{action_type_ref}_match'] = True


            elif action_type_ref == 'input_text':
                text_ref, text_pred = step['task_attr']['text'].lower().strip(), action_pred['text'].lower().strip()

                if text_ref in text_pred or text_pred in text_ref:
                    step_result['metrics']['action_match'] = step_result['metrics']['input_text_match'] = True

            elif action_type_ref == 'scroll':
                direction_ref, direction_pred = step['task_attr']['direction'], action_pred['direction']
                # direction_ref = scroll2swipe(direction_ref)
                if direction_ref == direction_pred:
                    step_result['metrics']['action_match'] = step_result['metrics']['swipe_match'] = True

            elif action_type_ref == 'press_key':
                press_ref, press_pred = step['task_attr']['key'], action_pred['key']

                if press_ref == press_pred:
                    step_result['metrics']['action_match'] = step_result['metrics']['press_match'] = True

            elif action_type_ref == 'hotkey':
                hotkey_ref, hotkey_pred = step['task_attr']['key'], action_pred['key_comb']
                if hotkey_ref.replace('+', '').replace('-', '').lower() == hotkey_pred.replace('+', '').replace('-', '').replace(' ', '').lower():
                    step_result['metrics']['action_match'] = step_result['metrics']['hotkey_match'] = True
            else:
                step_result['metrics']['action_match'] = step_result['metrics'][f'{action_type_ref}_match'] = True
        else:
            print(f"{action_type_ref} != {action_type_pred}")
        print(f"{step_result['metrics']['action_match']}: {step['task_attr']} <=> {action_pred}")
        # if counts['total'] > 20:
        #     break
    except Exception as e:
        print(e, 1111111111111111111111111111111111)
        logging.info("format wrong")
        step_result['wrong_format'] = True

    results.append(step_result)

# calculate metrics
num_sample = counts['total']
num_action_match = sum(x['metrics']['action_match'] for x in results)
num_type_match = sum(x['metrics']['type_match'] for x in results)
num_elem_match = sum(x['metrics']['elem_acc'] for x in results)
final_metrics = {'step_acc': [num_action_match / num_sample, num_action_match, num_sample], 'action_type_acc': [num_type_match / num_sample, num_type_match, num_sample], 'elem_acc': [num_elem_match / num_sample, num_elem_match, num_sample]}
print('final_metrics', final_metrics)
for k in counts.keys():
    if k=='total': continue
    
    cnt = counts[k]
    acc_cnt = sum(x['metrics'][f'{k}_match'] for x in results)
    
    final_metrics[f'{k}_acc'] = [round(acc_cnt / cnt, 4) if cnt > 0 else 0, acc_cnt, cnt]

final_metrics['num_wrong_format'] = sum(1 for x in results if 'wrong_format' in x)

pprint(final_metrics)

time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results/Omniact_desktop', f"{postfix}{'_CoT' if args.cot else ''}{'_wActRef' if args.action_refexp else ''}{'_relaxedGnd' if args.relaxed_gnd else ''}")
os.makedirs(eval_result_dir, exist_ok=True)
save_to = os.path.join(eval_result_dir, datetime.now().strftime("%m-%d-%H-%M-%S") + '.json')

print(f"Finished evaluating {args.pretrained} at {time_str}. Save eval results to {save_to}")
with open(save_to, "w") as f:
    json.dump({"meta": vars(args), "eval_result": final_metrics, "time": time_str, "logs": results}, f, indent=2)