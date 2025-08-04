import os, json, re
from tqdm import tqdm
from utils.data_utils.task_prompt_lib import *
from collections import defaultdict

file = "/data0/jingran/workspace/UI_training_data/Ours-Pretrain/scaling_exp/AndroidControl_processed/Qwen-QwQ-32B-Preview_reasoning/AndroidControl_Qwen-QwQ-32B-Preview_reasoning_RAW.json"

data = json.load(open(file))

samples = []

FORMAT = ['Qwen2-VL', 'llava', 'Florence'][0]

MAX_PREV_ACT = 999

IMAGE_DIR = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/"

import re

# Define the regex pattern to match the category tags
pattern = r" \((?:planning|planung|planning again|reflection|causality|causal|cause|common sense|dividing-and-conquering|dividing and conquering|progress check|divide-and-conquer|execution|summarization|summary|exploration|backtracking|backtrack|reflecting|recognition|backtracking to planning|reflection and planning)\)"

keyword_pattern = r"(?:Planning|Reflection|Divide-and-Conquer|nCausality|Backtracking|Summarization): "
def remove_reasoning_phrases(text):
    text = re.sub(pattern, '', text.replace("**",""), flags=re.IGNORECASE)
    text = re.sub(keyword_pattern, '', text.replace("**",""), flags=re.IGNORECASE)
    return text

MAX_SAMPLES = 500
BALANCED = True
NUM_CLASS = 4

OBS_TYPE = ['text', 'image', 'text-image'][0]
WO_REASONING = True

action_stats = defaultdict(int)

for x in tqdm(data, total=len(data), desc=os.path.basename(file)):
    if FORMAT == 'Qwen2-VL':
        ROLE_TAG, VALUE_TAG, USER_TAG, GPT_TAG = 'role', 'content', 'user', 'assistant'
    elif FORMAT == 'llava':
        ROLE_TAG, VALUE_TAG, USER_TAG, GPT_TAG = 'from', 'value', 'human', 'gpt'

    action_type = x['task_attr']['action_type']

    if BALANCED and action_stats[action_type] >= MAX_SAMPLES / NUM_CLASS: continue
    #if action_type == 'wait' and action_stats[action_type] >= 50: continue

    action_stats[action_type] += 1
    
    history = x['history'][max(0,-MAX_PREV_ACT):]
    history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(history, start=1)) if len(history) > 0 else 'None'
    
    # Remove logical phrase category tags
    x['reasoning'] = remove_reasoning_phrases(x['reasoning'])

    answer = f"Action: {x['gt_action']}" if WO_REASONING else f"{x['reasoning']}\nAction: {x['gt_action']}"

    sample = make_actionplanning_sample(task_id=f"autogui_androidcontrol_planning_{x['ep_id']}-{x['step_id']}-H", global_task=x['task'], history=history_str, gt_action=answer, with_cot=True)

    if FORMAT == 'Qwen2-VL':
        for turn_i, turn in enumerate(sample['conversations']):
            turn.pop('from')
            turn[ROLE_TAG] = USER_TAG if turn_i % 2 == 0 else GPT_TAG
            turn[VALUE_TAG] = turn.pop('value')
            
        sample['messages'] = sample.pop('conversations')
        sample['images'] = [os.path.join(IMAGE_DIR, x['image'])]
    else:
        sample['image'] = x['image']
    
    sample['ep_id'], sample['step_id'], sample['task'], sample['step_instruction'], sample["action_type"], sample["history"], sample["task_attr"], sample["wxh"] = x['ep_id'], x['step_id'], x['task'], x['step_instruction'], x["action_type"], history, x['task_attr'], x['wxh']
    
    samples.append(sample)

print(action_stats)

if len(samples) > MAX_SAMPLES:
    samples = random.sample(samples, MAX_SAMPLES)

save_to = file.replace("_RAW.json", f"_{len(samples)}{'_woCoT' if WO_REASONING else ''}.json")
with open(save_to, 'w') as f:
    json.dump(samples, f, indent=2)