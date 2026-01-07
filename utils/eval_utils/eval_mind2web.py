# evaluation on aitw
# This script refer to the official repo of AITW (https://github.com/google-research/google-research/tree/master/android_in_the_wild)
# to calculate the action matching score

# export CUDA_VISIBLE_DEVICES=$device_id
# python utils/eval_utils/eval_mind2web.py --task task --pretrained $model && python utils/eval_utils/eval_mind2web.py  --task domain --pretrained $model && python utils/eval_utils/eval_mind2web.py  --task website --pretrained $model
import os, time
import random
import torch
import json
from tqdm import tqdm
import logging
import ast
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
from action_matching import *
from eval_utils import mind2web_action2step
from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import keep_unique_actions

from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.openai_utils.qwen2vl import QWen2VL
from utils.data_utils.misc import remove_redundant_spaces

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

# calculate action f1 following mind2web
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='/mnt/vdb1/hongxin_li/uipro_ckpt/0123_UIPro_Qwen2-VL-7B_gnd2planning4336k+Mind2Web-GUIAct_wActRef_s1000_108k/lora/checkpoint-3368')
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--cot', type=bool, default=False)
    parser.add_argument('--scale', type=int, default=1000)
    parser.add_argument('--action_refexp', type=bool, default=True)
    parser.add_argument('--task', type=str, default='website', choices=['website', 'task', 'domain'])
    parser.add_argument('--device_tag', type=str, default='Web')
    parser.add_argument('--max_prev_acts', type=int, default=66)
    args = parser.parse_args()

    model_path = args.pretrained.rstrip('/ ')

    if 'slime' in model_path.lower():
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, use_flash_attn=True)
        model.generation_config.eos_token_id = 107 # '<end_of_turn>'
    elif 'qwen2' in model_path.lower():
        model = QWen2VL(device='cuda', model_name=model_path)
    elif 'atlas' in model_path.lower():
        model = QWen2VL(device='cuda', model_name=model_path)
        args.scale = 1000
        args.max_prev_acts = 6

    model_path = model_path.replace("lora/", "").replace("merged/","")
    if "snapshots" in model_path:
        postfix = model_path[model_path.find("models--") + 8: model_path.find("snapshots") - 1]
    elif len(model_path.split('/')) == 2:
        postfix = model_path.replace('/', '--')
    elif 'checkpoint-' in model_path:
        postfix = '/'.join(model_path.replace("lora/merged/","").split('/')[-2:])
    else:
        postfix = model_path.replace('/','-')

    eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results/mind2web', postfix)
    os.makedirs(eval_result_dir, exist_ok=True)

    gen_kwargs = {}      

    if "temperature" not in gen_kwargs:
        gen_kwargs["temperature"] = 0
    if "top_p" not in gen_kwargs:
        gen_kwargs["top_p"] = None
    if "num_beams" not in gen_kwargs:
        gen_kwargs["num_beams"] = 1

    index=0

    ROOT = [
        "/mnt/nvme0n1p1/hongxin_li/UI_training_data/Mind2Web",
        "/data2/hongxin_li/UI_training_data/Mind2Web",
        "/data0/jingran/workspace/UI_training_data/Mind2Web",
        "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/Mind2Web"][index]

    mind2web_imgs_dir = os.path.join(ROOT, "mind2web_images")

    # 1341 tasks (9378 steps) in total
    # website: 177 tasks (1373 steps) in total
    # domain: 912 tasks (5911 steps) in total
    # task: 252 tasks (2094 steps) in total
    mind2web_test = json.load(open(f'{ROOT}/mind2web_data_test_' + args.task + '.json', 'r'))

    time_record = [0, 0]
    results = []
    for ep_idx, episode in tqdm(enumerate(mind2web_test), total=len(mind2web_test), desc=f'Evaluating {postfix} on Mind2Web {args.task} (Max prev acts: {args.max_prev_acts})'):
        if args.debug and ep_idx > 4:
            break

        goal = episode["confirmed_task"]
        annot_id = episode["annotation_id"]
        previous_actions = []
        results_actions = []

        prev_actions = []
        for action_repr in episode['action_reprs']:
            elem, act = action_repr.split('->')
            act = act.strip()
            elem = elem.replace('  ',' ').strip()
            if 'TYPE:' in act:
                split_id = act.find(':')
                act, text = act[:split_id], act[split_id+1:]
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
                act, value = act[:split_id], act[split_id+1:]
                value = value.strip()
                prev_act_str = f"select {value} in the {elem}"
            else:
                raise Exception(f"unknown action: {act}")
            prev_actions.append(prev_act_str)
                
        for step_i, step in enumerate(episode["actions"]):
            if "bbox" not in step:
                print("action not found")
                continue

            filename = annot_id + '-' + step["action_uid"] + '.jpg'
            img_path = os.path.join(mind2web_imgs_dir, filename)
            if not os.path.exists(img_path):
                print("img not found")
                continue
            image = Image.open(img_path)

            # Used in the prompt
            action_step, bbox_ref = mind2web_action2step(step, image.size, scale=args.scale, return_bbox=True)

            try:
                action_step_ref = ast.literal_eval(action_step)
            except:
                continue
            
            if 'qwen2' in model_path.lower():
                clean_prev_step_instructions = keep_unique_actions(prev_actions[:step_i])
                retained_idxs, retained_history = clean_prev_step_instructions[-args.max_prev_acts:]
                history_str = ' '.join(f"Step {i}. {remove_redundant_spaces(instruc.replace('  ',' ').replace('[', ' ', 1).replace(']', ' ', 1).strip(' .'))}." for i, instruc in enumerate(retained_history, start=max(1,len(clean_prev_step_instructions) - MAX_PREV_ACT+1))) if len(retained_history) > 0 else 'None'
            elif 'atlas' in postfix.lower():
                history_str = 'None' if step_i == 0 else '\n'.join(f'Step {i}: {step.strip(" .")}.' for i, step in enumerate(prev_actions[max(0, step_i-args.max_prev_acts):step_i], start=1))
            else:
                history_str = 'None' if step_i == 0 else ' '.join(f'Step {i}. {step.strip(" .")}.' for i, step in enumerate(prev_actions[max(0, step_i-args.max_prev_acts):step_i], start=1))

            if 'atlas' in postfix.lower():
                prompt = ATLAS_PROMPT.format(global_task=goal, history='None')
            else:
                prompt = make_actionplanning_prompt(goal, history_str, device_tag=args.device_tag, prompt_format_type='simple', with_cot=args.cot, without_action_space=True, use_action_refexp=args.action_refexp) # SIMPLE_PROMPT.format(global_task=goal, history=history_str, step_instruction='')
            
            t1 = time.time()
            try:    # several sample's img dir lead to error, just jump it
                if 'slime' in postfix.lower():
                    conv = conv_templates['gemma'].copy()
                    conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
                    conv.append_message(conv.roles[1], None)
                    prompt_formatted = conv.get_prompt()

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
                else:
                    response = model.get_model_response(prompt, f"file://{img_path}", max_new_tokens=4096, sys_prompt=OSATLAS_MIND2WEB_PROMPT if 'atlas' in postfix.lower() else '')
                
                time_record[0] += time.time() - t1
                time_record[1] += 1
            except Exception as e:
                print(e)
                continue

            step_result = {"img_path": os.path.basename(img_path), "task": goal, "prompt": prompt, "response": response,
                        "GT_action": action_step, "GT_box": bbox_ref,
                        "Op_match": False, "Ele_match": False, "Op_F1": [0, action_step_ref["action_type"]]}
            try:
                # action_pred = ast.literal_eval(response)
                # 典型乱输出例子：# '{"action_type": "SELECT, CLICK", "target": (39,51), "value": "Things To Do"}'
                # '{"action_type": 7, "target": (80,63), "value": "Desktop Computer"}'
                # '{"action_type": CLICK, "target": (10,93}, CLICK)'
                # '{"action_type": CLICK, "target": (11,80}}'
                if 'atlas' in postfix.lower():
                    action_pred = parse_atlas_action(response)
                else:
                    action_pred_str = response.split('Action:')[1].strip().split('\n')[-1]
                    action_pred = ast.literal_eval(action_pred_str)
                
                if action_pred["action_type"] in ['click', 'hover']: action_pred["action_type"] = 'click'

                if action_pred["action_type"] == action_step_ref["action_type"] or action_pred["action_type"] == action_step_ref.get("ori_act", "").lower():
                    step_result["Op_match"] = True

                click_point = action_pred.get("target", (-1.0,-1.0))

                if action_pred["action_type"] == 'enter':
                    step_result["Ele_match"] = step_result["Op_match"]
                    step_result["Op_F1"][0] = 1.0
                else:
                    if (bbox_ref[0] <= click_point[0] / args.scale <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] / args.scale <= bbox_ref[3]):
                        step_result["Ele_match"] = True

                    # 按照mind2web的方式，把action转换成一个字符串，即如果是TYPE需要考虑字符间的F1
                    pred_str = str(action_pred["action_type"])
                    if action_pred["action_type"] in [3, "input_text"] or action_pred["action_type"] in [2, "select"]:
                        pred_str += ' '
                        pred_str += action_pred.get("text",  action_pred.get("value")).lower()
                    ref_str = str(action_step_ref["action_type"])
                    if action_step_ref["action_type"] in [3, "input_text"] or action_step_ref["action_type"] in [2, "select"]:
                        ref_str += ' '
                        ref_str += action_step_ref["value"].lower()

                    op_f1 = squad_metrics.compute_f1(pred_str, ref_str)
                    step_result["Op_F1"][0] = op_f1
                
                print(f"Op: {step_result['Op_match']} | Elem: {step_result['Ele_match']} |  Elem: {step_result['Op_F1']} | GT:{action_step_ref} <=> {action_pred}")
            except Exception as e :
                print(e)
                print("format wrong")
                step_result["status"] = "wrong format"

            action_step_ref['box'] = bbox_ref

            results_actions.append(step_result)

        results.append(results_actions)

    # calculate metrics
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

            if step_result["Op_F1"][1] in op_f1:
                op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
            macro_action_f1[i].append(step_result["Op_F1"][0])

            if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
                num_step_success += 1
                macro_step_acc[i].append(1)
            else:
                macro_step_acc[i].append(0)
                episode_success = False

        if episode_success:
            num_episode_success += 1

    marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()])




    print("Operation F1: " + str(marco_op_f1))
    print("Element Acc: " + str(num_ele / num_step))
    print("Step Success: " + str(num_step_success / num_step))
    print("Episode Success: " + str(num_episode_success / num_episode))
    print("Operation F1 cate: " + str([np.mean(x) for x in op_f1.values()]))

    macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()]).item()
    macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()]).item()
    avg_inference_time = time_record[0] / time_record[1] if time_record[1] > 0 else 0
    print("Macro Ele Acc: " + str(macro_ele_acc))
    print("Macro Op F1: " + str(macro_action_f1))
    print("Macro Step SR: " + str(macro_step_acc))
    print("Avg Inference Time: " + str(avg_inference_time))

    print(f"{macro_ele_acc*100:2f} | {macro_action_f1*100:.2f} | {macro_step_acc*100:.2f}")


    save_file = os.path.join(eval_result_dir, args.task + '-' + datetime.now().strftime("%m-%d-%H-%M-%S")) + '.json'

    with open(save_file, "w") as f:
        json.dump(
            {
                "meta": vars(args),
                "overall_results": {
                    "Operation F1": marco_op_f1,
                    "Element Acc": num_ele / num_step,
                    "Step Success": num_step_success / num_step,
                    "Episode Success": num_episode_success / num_episode,
                    "Operation F1 cate": str([np.mean(x).item() for x in op_f1.values()]),
                    "Macro Ele Acc": macro_ele_acc,
                    "Macro Op F1": macro_action_f1,
                    "Macro Step SR": macro_step_acc,
                    "time_per_step": avg_inference_time
                },
                "log": results
            },
            f,
            indent=2
        )

    print(f"Finised evaluation {args.pretrained} on AndroidControl. Save to {save_file}")