import os, json, numpy as np
from tqdm import tqdm
from task_prompt_lib import *
from copy import deepcopy

file = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/WAE_processed/WAE_7558k.json"
][0]

ORIGINAL_SCALE = 100
NEW_SCALE = 1000

ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"

def convert():
    data = json.load(open(file))
    
    new_samples = []
    
    for x in tqdm(data, total=len(data), desc=os.path.basename(file)):
        if 'gnd_' in x['id'] or 'ground' in x['id'] or 'ref_' in x['id']:
            unnorm_box = x['unnormalized_box']
            
            if 'wxh' in x:
                W, H = list(map(int, x['wxh'].split('x')))
            else: raise Exception()
            new_output = [(unnorm_box[0]+unnorm_box[2])/2/W, (unnorm_box[1]+unnorm_box[3])/2/H]
            new_output = list(map(lambda p: max(0, min(NEW_SCALE-1, round(p))), new_output))
            
            new_sample = deepcopy(x)
            
            if 'gnd_' in x['id'] or 'ground' in x['id']:
                new_sample['conversations'][1]['value'] = CLICK_TEMPLATE.format(target_x=new_output[0],target_y=new_output[1])
                
                if 'intent' in x['id']:
                    instruc = x['task_attr']
                else:
                    instruc = TURN_GND_INTO_PLANNING_PROMPT.format(insturc=x['task_attr'])
                
                new_sample['conversations'][0]['value'] = AGUVIS_PROMPT.format(global_task=instruc,history='None')

            elif 'ref_' in x['id']:
                user = new_sample['conversations'][0]['value']
                center_str_start = user.find(' (')
                center_str_end = user.find(') ', center_str_start)
                new_user = user[:center_str_start+1] + f'({new_output[0]},{new_output[1]})' + user[center_str_end+1:]
                new_user = new_user.replace(" (with point)","").replace(" (with bbox)","")
                
                new_sample['conversations'][0]['value'] = new_user
        elif 'webqa' in x['id']:
            