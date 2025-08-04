import json, os
from tqdm import tqdm
file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/AndroidControl_processed/AndroidControl_planning_124049.json"

data = json.load(open(file))

MAX_PREV_ACT = 3
for x in tqdm(data, total=len(data)):
    user = x['conversations'][0]['value']
    
    if 'history: Step' not in user or f'Step {MAX_PREV_ACT+1}' not in user:
        continue
    
    steps = []
    
    history_start_idx = user.find('history:')+9
    
    this_step_idx = user.find('Step ', history_start_idx)
    while True:
        next_step_idx = user.find('Step ', this_step_idx+1)
        
        end = False
        if next_step_idx == -1:
            end = True
            next_step_idx = user.find('\n', this_step_idx+1)
        
        this_step = user[user.find('. ', this_step_idx)+2:next_step_idx].strip(' .')
        steps.append(this_step)
        
        if end: break
        this_step_idx = next_step_idx
    
    new_history = ' '.join(f'Step {i}. {step}.' for i, step in enumerate(steps[-MAX_PREV_ACT:], start=1))
    
    new_user = user[:history_start_idx] + new_history + user[next_step_idx:]
    
    x['conversations'][0]['value'] = new_user

with open(file.replace('.json', '_reduced.json'), 'w') as f:
    json.dump(data, f)