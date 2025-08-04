
import os, time, cv2
import random
import torch
import json
from tqdm import tqdm
import traceback
import ast
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
from action_matching import *
from pprint import pprint
from colorama import Fore, Style
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.data_utils.task_prompt_lib import *
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.osatlas import OSATLAS
from utils.openai_utils.showui import SHOWUI, to_showui_action, showui_to_original_action
from utils.openai_utils.osatlas4b import OSATLAS4B
from utils.data_utils.misc import keep_unique_actions, scroll2swipe, get_swipe_direction

from utils.openai_utils.misc import  extract_protocol_components
from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

# calculate action f1 following androidcontrol
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

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default=['/mnt/vdb1/hongxin_li/uipro_ckpt/0122_UIPro_Qwen2-VL-7B_gnd2planning4336k+AITW-AITZ-AMEX-AndCon_wActRef_s1000_219k/lora/checkpoint-3186/', 'showlab/ShowUI-2B', 'OS-Copilot/OS-Atlas-Pro-7B', 'OS-Copilot/OS-Atlas-Pro-4B', 'Qwen/Qwen2-VL-2B-Instruct'][1])
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--provider', type=str, default=['openai', 'step', ''][2])
parser.add_argument('--cot', type=bool, default=False)
parser.add_argument('--wo_openapp', type=bool, default=False)
parser.add_argument('--action_refexp', type=bool, default=False)
parser.add_argument('--max_prev_acts', type=int, default=6)
parser.add_argument('--device_tag', type=str, default='Android')
parser.add_argument('--preset_id_file', type=str, default=['utils/eval_utils/androidcontrol_test/selected_andcon_idx.json', ''][-1])
parser.add_argument('--prompt_format_type', type=str, default='simple', choices=['reflec', 'simple', 'protocol'])
parser.add_argument('--use_qwen_actspace', type=bool, default=False)

args, _ = parser.parse_known_args()


model_path = args.pretrained.rstrip('/ ')
print(f"Loading model from {model_path}")

if args.provider == 'step':
    model = OpenAIModel(base_url=os.environ.get('STEP_API_BASE'), api_key=os.environ.get("MODEL_PROXY_KEY", "EMPTY"), model=args.pretrained, temperature=0.0, max_tokens=2048)
else:
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
        
        SCALE = 100
    elif 'uipro' in model_path.lower():
        model = QWen2VL(device='cuda', model_name=model_path)
        SCALE = 1000
        MAX_PREV_ACT = 6
    elif 'atlas' in model_path.lower():
        if '7b' in model_path.lower():
            model = OSATLAS(device='cuda', model_name=model_path)
        else:
            model = OSATLAS4B(device='cuda', model_name=model_path)
        SCALE = 1000
        MAX_PREV_ACT = 999
        args.prompt_format_type = 'atlas'
    elif any(k in args.pretrained.lower() for k in ['qwen2.5', 'qwen2p5']):
        MAX_PREV_ACT = 999
        model = QWen2VL(device='cuda', model_name=model_path)
        SCALE = -1
    elif 'qwen2' in model_path.lower():
        MAX_PREV_ACT = 999
        model = QWen2VL(device='cuda', model_name=model_path)
        SCALE = 1000
    elif 'show' in model_path.lower():
        MAX_PREV_ACT = 999
        model = SHOWUI(device='cuda', model_name=model_path)
        SCALE = 1

# special case
if any(k in args.pretrained.lower() for k in ['qwen2.5', 'qwen2p5']):
    SCALE = -1


model_path = model_path.replace("merged/", "").replace("lora/","")
if "snapshots" in model_path:
    postfix = model_path[model_path.find("models--") + 8: model_path.find("snapshots") - 1]
else:
    postfix = '/'.join(model_path.split('/')[-2:])

index=-1

ROOT = ["/mnt/vdb1/hongxin_li", "/data0/jingran/workspace/UI_training_data/Ours-Pretrain/scaling_exp", "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data"][index]
REMOVE_RARE_ACTIONS = False

androidcontrol_test_raw = json.load(open(
    [f'{ROOT}/AndroidControl_test/AndroidControl-test_12685.json',
     f'{ROOT}/AndroidControl_test/AndroidControl-test_12685.json',
     f'{ROOT}/AndroidControl_test/AndroidControl_test_s1000_15895.json',
     ][index], 'r'))
# {'click': 7050, 'swipe': 1685, 'navigate_back': 620, 'open_app': 1190, 'input_text': 1241, 'wait': 899}

# Group all steps by the belonging trajectory
trajectory_groups = {}
for step in androidcontrol_test_raw:
    step_traj_id, step_idx, insturct_type = step['id'].split('-')
    if step_traj_id not in trajectory_groups:
        trajectory_groups[step_traj_id] = {"HL": [], "H": []}

    trajectory_groups[step_traj_id][insturct_type].append((step_idx, step))

# Sort the steps by the step_id
for step_traj_id in trajectory_groups:
    for insturct_type in trajectory_groups[step_traj_id]:
        trajectory_groups[step_traj_id][insturct_type].sort(key=lambda x: x[0])

if args.preset_id_file:
    preset_ids = json.load(open(args.preset_id_file))
else: preset_ids = {}

hl_ids = set(x['id'].split('-H')[0] for x in androidcontrol_test_raw if '-HL' in x['id'] and not (args.wo_openapp and x['action_type'] == 'open_app') and not (len(preset_ids) > 0 and f"{x['task']}-{x['step_id']}" not in preset_ids['HL']))
h_ids = set(x['id'].split('-H')[0] for x in androidcontrol_test_raw if '-HL' not in x['id'] and not (args.wo_openapp and x['action_type'] == 'open_app') and not (len(preset_ids) > 0 and f"{x['task']}-{x['step_id']}" not in preset_ids['H']))
hl_h_ids = hl_ids.intersection(h_ids)




REPEAT = 3
meta = vars(args)
meta['max_prev_actions'] = MAX_PREV_ACT

eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results/androidcontrol')
os.makedirs(eval_result_dir, exist_ok=True)


save_to = os.path.join(eval_result_dir, postfix)

os.makedirs(save_to, exist_ok=True)
save_file = os.path.join(save_to, datetime.now().strftime("%m-%d-%H-%M-%S")) + '.json'

results_all_repeats = []
metrics_all_repeats = []
for repeat in range(REPEAT):
    selected_ids = random.sample(list(hl_h_ids), 500 if not args.debug else 7)
    hl_samples = [x for x in androidcontrol_test_raw if '-HL' in x['id'] and x['id'].split('-H')[0] in selected_ids]
    h_samples = [x for x in androidcontrol_test_raw if '-HL' not in x['id'] and x['id'].split('-H')[0] in selected_ids]

    # HL
    metrics_this_repeat = {'HL': {}, 'H': {}}
    results = {'HL': [], 'H': []}
    for mode, samples in zip(['HL', 'H'][1:], [hl_samples, h_samples][1:]):
        counts = {'total': 0, 'click': 0, 'input_text': 0, 'swipe': 0, 'long_press': 0, 'enter': 0, 'navigate_home': 0, 'navigate_back': 0, 'status': 0, 'open_app': 0, 'wait': 0}

        if args.use_qwen_actspace:
            history_actions_qwen_format = []

        for step_idx, step in tqdm(enumerate(samples), total=len(samples), desc=f'{postfix} | Repeat {repeat+1} {mode}'):
            #if step['action_type'] not in ['input_text']: continue
            step_id = step['id'] # 'autogui_androidcontrol_planning_2889-0-HL'
            cur_step_traj_id = step_id.split('-')[0]
            cur_step_traj_idx, cur_step_idx, cur_insturct_type = step_id.split('_')[-1].split('-')
            goal = step["task"].strip(' .') + '.'
            counts['total'] += 1
            action_type_ref = step['action_type']

            counts[action_type_ref] += 1

            img_path = os.path.join(ROOT, step["image"])

            image = Image.open(img_path)
            W,H=image.size
            # Used in the prompt

            if 'uipro' in model_path.lower():
                retained_idxs, clean_step_instructions = keep_unique_actions(step['history'])
                history = clean_step_instructions[max(0,len(clean_step_instructions)-MAX_PREV_ACT):]
                history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(history, start=max(1,len(clean_step_instructions)-MAX_PREV_ACT+1))) if len(history) > 0 else 'None'
            elif 'protocol' in model_path.lower() and args.use_qwen_actspace:
                # convert the history to the qwen actspace format
                prev_actions = trajectory_groups[cur_step_traj_id][cur_insturct_type][:int(cur_step_idx)]
                history_str = make_AndroidWorld_official_history_str([to_qwen_action(x['conversations'][-1]['value'], None, W, H) for x in trajectory_groups[cur_step_traj_id][cur_insturct_type]], [x['outcome'] for x in trajectory_groups[cur_step_traj_id][cur_insturct_type]])
            elif 'showui' in model_path.lower():
                history_str = [to_showui_action(x[1]['conversations'][-1]['value'].replace('Action:', ''), W, H, 'phone', scale=SCALE) for x in trajectory_groups[cur_step_traj_id][cur_insturct_type]]
            else:
                history_str = ' '.join(f"Step {i}. {action.strip(' .')}." for i, action in enumerate(step['history'][-MAX_PREV_ACT:], start=1)) if step['step_id'] > 0 else 'None'
            
            if model_path in ['Qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2-VL-2B-Instruct']:
                prompt = ANDROIDCONTROL_PLANNING_PROMPT_COT.format(global_task=goal, history=history_str, step_instruction=f"The next step instruction: {step['step_instruction']}\n" if mode == 'HL' else '')
            elif 'minicpm' in model_path.lower():
                prompt = GUICOURSE_PROMPT.format(
                    goal=goal, history=history_str
                )
            elif args.prompt_format_type == 'reflec':
                prompt = make_planning_reflec_protocol('AndroidControl', goal, history_str, device_type='smartphone', use_unnorm_xy='Qwen2.5' in args.pretrained, use_qwen_actspace=True, use_guidelines=False)
            elif args.prompt_format_type == 'protocol':
                prompt = make_planning_protocol('AndroidWorld', goal, history_str, device_type='smartphone', use_unnorm_xy='Qwen2.5' in args.pretrained)
            elif 'showui' in model_path.lower():
                prompt = goal + (f" The next step instruction: {step['step_instruction']}" if mode == 'HL' else '')
            else:
                prompt = make_actionplanning_prompt(
                    goal,
                    history_str,
                    step_instruction=step['step_instruction'] if mode == 'HL' else '',
                    device_tag=args.device_tag,
                    prompt_format_type=args.prompt_format_type,
                    with_cot=args.cot,
                    without_action_space=True,
                    use_action_refexp=args.action_refexp)
            

            t1 = time.time()
                
            step_result = {"img_path": img_path, "task": goal, "prompt": prompt, "response": None, "GT_action": step['task_attr'], "action_pred": None, "metrics":  {k: False for k in ['action_match', 'type_match', 'elem_acc', 'click_match', 'input_text_match', 'swipe_match', 'enter_match', 'status_match', 'navigate_home_match', 'navigate_back_match', 'open_app_match', 'wait_match', 'long_press_match', 'need_gnd']} }

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
                            max_new_tokens=2048,
                            use_cache=True,
                        )
                        response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                elif 'showui' in model_path.lower() and 'qwen' not in model_path.lower():
                    response = model.get_model_response(
                        prompt,
                        f"file://{img_path}",
                        max_new_tokens=256,
                        history=history_str,
                        scenario='phone',
                        task_type='nav2'
                        )
                elif 'atlas' in model_path.lower():
                    response = model.get_model_response(
                        prompt,
                        f"file://{img_path}",
                        max_new_tokens=8192,
                        sys_prompt=OSATLAS_ANDROIDCONTROL_SYS_PROMPT if 'atlas' in model_path.lower() else OSATLAS_SYS_PROMPT,
                        )
                else:
                    response = model.get_model_response(
                        prompt,
                        f"file://{img_path}",
                        max_new_tokens=4096
                        )

                step_result["response"] = response

                if 'atlas' in model_path.lower():
                    action_pred = parse_atlas_action(response)
                elif args.prompt_format_type == 'reflec':
                    action_pred = ast.literal_eval(response[response.rfind('{"action'):response.rfind('}')+1]) # Use the last action
                elif args.prompt_format_type == 'protocol':
                    resp_parts = extract_protocol_components(response)
                    action_raw = resp_parts.pop('action')
                    action_pred = ast.literal_eval(action_raw)
                    step_result['thought_parts'] = resp_parts
                elif 'showui' in model_path.lower() and 'qwen' not in model_path.lower():
                    action_pred_raw = ast.literal_eval(response)
                    action_pred = showui_to_original_action(action_pred_raw)

                    # special handling for ShowUI type as this model outputs a target coordinate and a text input in one type action

                    if '<&>' in action_pred:
                        click_action_pred, input_action_pred = action_pred.split('<&>')
                        action_pred = click_action_pred if action_type_ref == 'click' else input_action_pred

                    action_pred = ast.literal_eval(action_pred)
                    
                else:
                    action_pred = ast.literal_eval(response[response.rfind('{"action'):response.rfind('}')+1])
                
                step_result["action_pred"] = action_pred
                
                # matching
                action_type_pred = action_pred.get('action_type', action_pred.get('action'))

                special_match = False
                # Special handling for enter
                if action_type_ref == 'enter' and action_pred['action_type'] == 'press_key' and action_pred['key'].lower() == 'enter':
                    special_match = True

                # Special handling for terminate
                if action_type_pred == 'terminate' and action_type_ref == 'status':
                    special_match = True
                
                # Special handling for scroll
                if action_type_pred == 'scroll' and action_type_ref == 'swipe':
                    special_match = True
                            
                if action_type_pred in ['open']:
                    action_type_pred = action_pred['action'] = 'open_app'

                if action_type_pred in ['type', 'input_text']:
                    action_type_pred = action_pred['action_type'] = 'input_text'

                if action_type_pred in ['answer']:
                    action_type_pred = action_pred['action_type'] = 'status'
                    action_pred['goal_status'] = 'successful' if 'complete' in action_pred['text'] else 'infeasible'

                if action_type_pred == 'system_button':
                    if action_pred['button'].lower() == 'home':
                        action_type_pred = 'navigate_home'
                    elif action_pred['button'].lower() == 'back':
                        action_type_pred = 'navigate_back'

                if action_type_ref == action_type_pred or special_match:
                    step_result['metrics']['type_match'] = True

                    if action_type_ref in ['click', 'long_press']:
                        step_result['metrics']['need_gnd'] = True
                        target = action_pred.get('target', action_pred.get('coordinate'))

                        if isinstance(target, str): target = eval(target)
                        if SCALE == -1:
                            target_pred = [target[0] / W, target[1] / H]
                        else:
                            target_pred = list(map(lambda p: p / SCALE, target))
                        
                        gt_box = step['task_attr']['bbox']
                        gt_box_normalized = list(map(lambda p:round(p, 3), [gt_box[0]/W, gt_box[1]/H, gt_box[2]/W, gt_box[3]/ H]))
                        
                        assert all(0<=p<=1.0 for p in gt_box_normalized + target_pred), f"Invalid box or target: {gt_box_normalized} {target_pred}"
                        
                        if gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2] and gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]:
                            step_result['metrics']['action_match'] = step_result['metrics']['elem_acc'] = step_result['metrics'][f'{action_type_ref}_match'] = True

                    elif action_type_ref == 'input_text':
                        text_ref, text_pred = step['task_attr']['text'].lower().strip(), action_pred['text'].lower().strip()
                        
                        step_result['metrics']['action_match'] = step_result['metrics']['input_text_match'] = squad_metrics.compute_f1(text_pred, text_ref) > 0.5
                    
                    elif action_type_ref == 'swipe':
                        direction_ref = step['task_attr']['direction']
                        if 'direction' in action_pred:
                            direction_pred = action_pred['direction']
                        elif 'coordinate' in action_pred: # 'action': 'swipe', 'coordinate': [345, 279], 'coordinate2': [86, 290]}
                            direction_pred, distance = get_swipe_direction(action_pred['coordinate'], action_pred['coordinate2'], is_swipe=True)
                        direction_ref = scroll2swipe(direction_ref)
                        if direction_ref == direction_pred:
                            step_result['metrics']['action_match'] = step_result['metrics']['swipe_match'] = True

                    elif action_type_ref == 'status':
                        status_ref, status_pred = step['task_attr']['goal_status'], action_pred['goal_status']

                        if status_ref == status_pred:
                            step_result['metrics']['action_match'] = step_result['metrics']['status_match'] = True
                    elif action_type_ref == 'open_app':
                        app_name_ref, app_name_pred = step['task_attr']['app_name'], action_pred.get('app_name', action_pred.get('text', None))
                        
                        if app_name_ref == app_name_pred:
                            step_result['metrics']['action_match'] = step_result['metrics']['open_app_match'] = True
                    else:
                        step_result['metrics']['action_match'] = step_result['metrics'][f'{action_type_ref}_match'] = True

                is_match = step_result['metrics']['action_match']
                print((Fore.GREEN if is_match else Fore.RED) + f"{is_match}" + Style.RESET_ALL + f": GT: {step['task_attr']} <=> Pred: {action_pred}")

                if step_idx % 2 == 0:
                    print(Fore.YELLOW + f"\nUser: <img>{img_path}</img> {prompt}\n" + Fore.CYAN + f"GPT: {response}" + Style.RESET_ALL)
            except Exception as e:
                traceback.print_exc()
                print("format wrong")
                step_result['wrong_format'] = True

            results[mode].append(step_result)

        # calculate metrics
        num_sample = counts['total']
        num_need_gnd = sum(x['metrics']['need_gnd'] for x in results[mode])
        
        num_action_match = sum(x['metrics']['action_match'] for x in results[mode])
        num_type_match = sum(x['metrics']['type_match'] for x in results[mode])
        num_elem_match = sum(x['metrics']['elem_acc'] for x in results[mode])
        final_metrics = {'step_acc': [num_action_match / num_sample if num_sample > 0 else 0, num_action_match, num_sample], 'action_type_acc': [num_type_match / num_sample if num_sample > 0 else 0, num_type_match, num_sample], 'elem_acc': [num_elem_match / num_need_gnd if num_need_gnd > 0 else 0, num_elem_match, num_need_gnd]}

        for k in counts.keys():
            if k=='total': continue
            
            cnt = counts[k]
            acc_cnt = sum(x['metrics'][f'{k}_match'] for x in results[mode])
            
            final_metrics[f'{k}_acc'] = [round(acc_cnt / cnt, 4) if cnt > 0 else 0, acc_cnt, cnt]

        final_metrics['num_wrong_format'] = sum(1 for x in results[mode] if 'wrong_format' in x)

        pprint(final_metrics)
    
        metrics_this_repeat[mode] = final_metrics
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
            indent=2
        )
# aggr
aggr_metrics = {'HL': {}, 'H': {}}

for mode in aggr_metrics.keys():
    for repeat_result in metrics_all_repeats:
        if not repeat_result[mode]: continue
        for metric_name, info in repeat_result[mode].items():
            if metric_name == 'num_wrong_format': continue
            if metric_name not in aggr_metrics[mode]: aggr_metrics[mode][metric_name] = [0,0,0]
            aggr_metrics[mode][metric_name][1] += info[1]
            aggr_metrics[mode][metric_name][2] += info[2]
    
    for metric_name in aggr_metrics[mode].keys():
        if metric_name == 'num_wrong_format': continue
        acc_cnt, cnt = aggr_metrics[mode][metric_name][1], aggr_metrics[mode][metric_name][2]
        aggr_metrics[mode][metric_name][0] = acc_cnt / cnt if cnt > 0 else 0

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
        indent=2
    )

print(f"Finised evaluation {args.pretrained} on AndroidControl. Save to {save_file}")