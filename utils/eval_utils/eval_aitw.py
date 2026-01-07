# evaluation on aitw
# This script refer to the official repo of AITW (https://github.com/google-research/google-research/tree/master/android_in_the_wild)
# to calculate the action matching score

import os, time, cv2
import random
import torch
import json
from tqdm import tqdm
import datasets
import logging
import ast
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
from action_matching import *
from utils.data_utils.task_prompt_lib import *
from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.showui import SHOWUI, to_showui_action, showui_to_original_action
from utils.data_utils.misc import keep_unique_actions

from colorama import Style, Fore
logging.basicConfig(level=logging.INFO)

# convert action to prediction format
def action2step(step_data):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # for click action, we calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
            click_point = [f"{item:.2f}" for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "{{\"action_type\": {}, \"click_point\": {}}}".format(action_type_new, click_point)
        else:  # for scroll action, we assign an action_type_id for each scroll
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

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default=['showlab/ShowUI-2B', '/mnt/vdb1/hongxin_li/uipro_ckpt/0122_UIPro_Qwen2-VL-7B_gnd2planning4336k+AITW-AITZ-AMEX-AndCon_wActRef_s1000_219k/lora/checkpoint-1416/','Qwen/Qwen2-VL-7B-Instruct', 'HongxinLi/UIPro_2stage_Mobile'][-1])
    parser.add_argument('--imgs_dir', type=str, default=[
        '/data0/jingran/workspace/UI_training_data/AITW/aitw_images/',
        '/mnt/vdb1/hongxin_li/AITW/aitw_images/',
        '/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/AITW/'][1])
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--cot', type=bool, default=False)
    parser.add_argument('--action_refexp', type=bool, default=True)
    parser.add_argument('--scale', type=int, default=1000)
    parser.add_argument('--device_tag', type=str, default='Android')
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

    if 'slime' in model_path.lower():
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, use_flash_attn=True)
        model.generation_config.eos_token_id = 107 # '<end_of_turn>'
        MAX_PREV_ACT = 4
    elif 'qwen2' in model_path.lower():
        model = QWen2VL(device='cuda', model_name=model_path)
        MAX_PREV_ACT = 6
    elif 'atlas' in model_path.lower():
        model = QWen2VL(device='cuda', model_name=model_path)
        args.scale = 1000
        MAX_PREV_ACT = 6
    elif 'showui' in model_path.lower():
        model = SHOWUI(device='cuda', model_name=model_path)
        args.scale = 1
        MAX_PREV_ACT = 6
    else:
        model = QWen2VL(device='cuda', model_name=model_path)
        MAX_PREV_ACT = 6

    gen_kwargs = {}      

    if "temperature" not in gen_kwargs:
        gen_kwargs["temperature"] = 0
    if "top_p" not in gen_kwargs:
        gen_kwargs["top_p"] = None
    if "num_beams" not in gen_kwargs:
        gen_kwargs["num_beams"] = 1

    if 'fullhistory' in args.pretrained.lower():
        MAX_PREV_ACT = 999

    aitw_imgs_dir = args.imgs_dir

    aitw_test = datasets.load_dataset(f"HongxinLi/AITW_test", split='test') # 578 tasks (4663 steps) in totals

    # aitw_test = json.load(open(os.path.join(os.path.dirname(aitw_imgs_dir),'aitw_data_test.json'), 'r'))
    aitw_test_each_app = {}
    for x in aitw_test:
        app = x['image'].split('/')[0]
        aitw_test_each_app.setdefault(app, []).append(x)

    score_average = 0
    time_record = [0,0]
    tasks_result = {}
    tasks_logs = {}

    for task, steps in aitw_test_each_app.items():
        tasks_logs[task] = []
        print("Task: " + task)

        corr_action = 0
        corr_type = 0
        num_text = 0
        corr_text = 0
        num_scroll = 0
        corr_scroll = 0
        num_click = 0
        corr_click = 0
        num_both_click = 0
        corr_both_click = 0
        num_wrong_format = 0
        num = 0

        for step_i, step in tqdm(enumerate(steps), total=len(steps), desc=f"{task} | DeviceTag: {args.device_tag} | CoT: {args.cot} | ActRefExp: {args.action_refexp}"):
            if args.debug and step_i >= 6: break
            img_filename = step["image"]
            step_idx = int(img_filename.split('_')[-1][:-4])
            img_path = os.path.join(aitw_imgs_dir, img_filename)
            if not os.path.exists(img_path):
                print("img not found")
                continue

            goal = step["step"]["goal"]

            action_ref = action_2_format(step["step"])

            temp = {"step_id": f"{step['step']['ep_id']}-{step['step']['step']}", "img_path": img_path, "device": task, "instruc": goal, "task": task, "action_ref": action_ref}

            prompt_lst = []
            response_lst = []
            action_pred_lst = []

            raw_history = step['history']
            if isinstance(raw_history, list):
                retained_idxs, clean_step_instructions = keep_unique_actions(raw_history)
                history = clean_step_instructions[max(0,len(clean_step_instructions)-MAX_PREV_ACT):]

                # if 'showui' in model_path.lower() and 'qwen' not in model_path.lower():
                #     history_str = [to_showui_action(x['conversations'][-1]['value'].replace('Action:', ''), W, H, 'phone') for x in steps[max(0,step_i-MAX_PREV_ACT+1):step_i]] if step_i > 0 else []
                # else:
                END_PUNC, LINE_SPLIT = ':' if 'atlas' in postfix.lower() else '.', '\n' if 'atlas' in postfix.lower() else ' '
                history_str = LINE_SPLIT.join(f"Step {i}{END_PUNC} {instruc.strip(' .')}." for i, instruc in enumerate(history, start=max(1,step_idx-MAX_PREV_ACT+1))) if len(history) > 0 else 'None'
            elif f'Step {args.max_prev_acts+1}.' in raw_history:
                steps = []      
                this_step_idx = 0
                while True:
                    next_step_idx = raw_history.find('Step ', this_step_idx+1)
                    
                    end = False
                    if next_step_idx == -1:
                        end = True
                        next_step_idx = raw_history.find('\n', this_step_idx+1)
                    
                    this_step = raw_history[raw_history.find('. ', this_step_idx)+2:next_step_idx].strip(' .')
                    steps.append(this_step)
                    
                    if end: break
                    this_step_idx = next_step_idx
                
                history_str = ' '.join(f'Step {i}. {step}.' for i, step in enumerate(steps[-args.max_prev_acts:], start=max(1,len(steps)-args.max_prev_acts+1)))
            else:
                history_str = step['history']

                    
            if 'atlas' in postfix.lower():
                prompt = ATLAS_PROMPT.format(global_task=goal, history=history_str)
            elif 'showui' in model_path.lower() and 'qwen' not in model_path.lower():
                prompt = goal
            else:    
                prompt = make_actionplanning_prompt(goal, history_str, device_tag=args.device_tag, prompt_format_type='simple', with_cot=args.cot, without_action_space=True, use_action_refexp=args.action_refexp)

            if args.original_actspace:
                prompt = '[AITW] ' + prompt

            t1 = time.time()
            try:    # several sample's img dir lead to error, just jump it
                if 'slime' in postfix.lower():
                    conv = conv_templates['gemma'].copy()
                    conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
                    conv.append_message(conv.roles[1], None)
                    prompt_formatted = conv.get_prompt()

                    img = [Image.open(img_path).convert('RGB')]
                    img_tensor = process_images(img, image_processor, model.config).to(dtype=model.dtype, device=model.device) # B x num_patches x 3 x 336 x 336
                    
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
                            max_new_tokens=512,
                            use_cache=True,
                        )
                        response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                elif 'showui' in model_path.lower() and 'qwen' not in model_path.lower():
                    image = Image.open(img_path)
                    W, H = image.size
                    response = model.get_model_response(
                        prompt,
                        f"file://{img_path}",
                        max_new_tokens=256,
                        history=history_str,
                        scenario='phone',
                        task_type='nav2'
                    )
                else:
                    tempe, retry = 0.0, 0
                    p = prompt
                    while retry <= 5:
                        response = model.get_model_response(p, f"file://{img_path}", temperature=tempe, sys_prompt=OSATLAS_AITW_SYS_PROMPT if 'atlas' in postfix.lower() else '')
                        if 'open_app' in response:
                            tempe = 1.0; p = prompt + " ({} use the open_app action)".format(random.choice(['You SHOULD not', 'You MUST not', 'You are not allowed to', 'Do NOT', 'You can not'])); retry += 1
                        else: break
                
            except Exception as e:
                print(e)
            
            if args.debug or step_i % 5 == 0:
                print(Fore.CYAN + f"User: {prompt}\n" + Fore.YELLOW + f"GPT: {response}\n" + Style.RESET_ALL)
            time_record[0] += time.time() - t1; time_record[1] += 1
            num += 1

            temp["prompt"] = prompt

            try:
                if 'atlas' in postfix.lower():
                    action_pred = pred_2_format(parse_atlas_action(response, device='mobile'), scale=args.scale)
                elif 'showui' in model_path.lower() and 'qwen' not in model_path.lower():
                    action_pred_raw = ast.literal_eval(response)
                    action_pred = pred_2_format(showui_to_original_action(action_pred_raw, scale=args.scale), scale=args.scale)
                else:
                    action_pred_str = response.split('Action:')[1].strip().split('\n')[-1]
                    action_pred = pred_2_format(ast.literal_eval(action_pred_str), scale=args.scale)
                
                annot_position = np.array(
                    [step["step"]["annot_position"][i:i + 4] for i in range(0, len(step["step"]["annot_position"]), 4)]) # [y, x, h, w, ...]
                
                if False:
                    img = cv2.imread(img_path)
                    H,W =img.shape[:2]
                    for anno in annot_position:
                        y,x,h,w = anno
                        x1,y1,x2,y2 = x*W,y*H,(x+w)*W,(y+h)*H
                        x1,y1,x2,y2 = list(map(round, [x1,y1,x2,y2]))
                        cv2.rectangle(img, (x1,y1),(x2,y2), color=(0,0,255),thickness=2)
                    cv2.imwrite("test.png", img)
                check_match = check_actions_match(action_pred["touch_point"], action_pred["lift_point"],
                                                                    action_pred["action_type"], action_ref["touch_point"],
                                                                    action_ref["lift_point"], action_ref["action_type"],
                                                                    annot_position)
                # step accuracy
                if check_match == True:
                    corr_action += 1
                    match_label = 1
                    #print("Step: " + str(j) + " right")
                    temp["status"] = "correct"
                else:
                    match_label = 0
                    #print("Step: " + str(j) + " wrong")
                    temp["status"] = "wrong"
                # type accuracy
                if action_pred["action_type"] == action_ref["action_type"]:
                    corr_type += 1
                    temp["status"] += ",action_correct"
                else: temp["status"] += ",action_wrong"
                
                # text accuracy
                if action_ref["action_type"] == 3:
                    num_text += 1
                    if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                            action_pred["typed_text"] in action_ref["typed_text"]) or (
                            action_ref["typed_text"] in action_pred["typed_text"]):
                        corr_text += 1
                        temp["status"] += ",typedText_correct"
                    else: temp["status"] += ",typedText_wrong"

                if action_ref["action_type"] == 4:
                    # click accuracy
                    if is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                        num_click += 1
                        if match_label:
                            corr_click += 1
                    # scroll accuracy
                    else:
                        num_scroll += 1
                        if match_label:
                            corr_scroll += 1
                    if (action_pred["action_type"] == 4) and is_tap_action(action_ref["touch_point"], action_ref["lift_point"]) and is_tap_action(
                            action_pred["touch_point"], action_pred["lift_point"]):
                        num_both_click += 1
                        if match_label:
                            corr_both_click += 1
                temp["response"] = response; temp["action_pred"] = action_pred
                
                print(temp["status"] + ' ' + Fore.CYAN + "GT: " + str(action_ref) + ' <=> ' + Fore.YELLOW + "Pred: " + str(action_pred) + Style.RESET_ALL)
            except:
                num_wrong_format += 1
                # print("Step: " + str(j) + " wrong format")
                temp["status"] = "wrong format"

            tasks_logs[task].append(temp)

        print("Avg Time: " + str(time_record[0] / time_record[1]))

        action_acc = f"{100 * corr_action / num if num else 0:.2f}% / {corr_action} / {num}"
        action_type_acc = f"{100 * corr_type / num if num else 0:.2f}% / {corr_type} / {num}"
        
        text_acc = f"{100 * corr_text / num_text if num_text else 0:.2f}% / {corr_text} / {num_text}"
        click_acc = f"{100 * corr_click / num_click if num_click else 0:.2f}% / {corr_click} / {num_click}"
        scroll_acc = f"{100 * corr_scroll / num_scroll if num_scroll else 0:.2f}% / {corr_scroll} / {num_scroll}"
        dual_click_acc = f"{100 * corr_both_click / num_both_click if num_both_click else 0:.2f}% / {corr_both_click} / {num_both_click}"
        
        tasks_result[task] = {"action_acc": action_acc, "action_type_acc": action_type_acc, "text_acc": text_acc, "click_acc": click_acc, "scroll_acc": scroll_acc, "dual_click_acc": dual_click_acc, "num_wrong_format": num_wrong_format}
        
        score_average += corr_action / num if num else 0

        print(f"Action Acc: {action_acc}")
        print(f"Type Acc: {action_type_acc}")
        print(f"Text Acc: {text_acc}")
        print(f"Click Acc: {click_acc}")
        print(f"Scroll Acc: {scroll_acc}")
        print(f"dual_click_acc: {dual_click_acc}")
        print(f"Num wrong format: {num_wrong_format}")

    avg_inference_time = time_record[0] / time_record[1] if time_record[1] > 0 else 0
    tasks_result['time_per_step'] = avg_inference_time
    tasks_result['avg'] = score_average / 5
    print("Average score: " + str(score_average / 5))
    print("Average inference time per step: " + str(avg_inference_time))

    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results/AITW', f"{postfix}{'_CoT' if args.cot else ''}{'_wActRef' if args.action_refexp else ''}")
    os.makedirs(eval_result_dir, exist_ok=True)
    save_to = os.path.join(eval_result_dir, datetime.now().strftime("%m-%d-%H-%M-%S") + '.json')

    print(f"Finished evaluating {args.pretrained} at {time_str}. Save eval results to {save_to}")
    with open(save_to, "w") as f:
        json.dump({"meta": vars(args), "eval_result": tasks_result, "time": time_str, "logs": tasks_logs}, f, indent=2)