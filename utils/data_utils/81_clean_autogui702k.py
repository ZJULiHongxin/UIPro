import json, os, re
import cv2
from tqdm import tqdm
from copy import deepcopy

file = '/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/autogui625k_merged_android_77k_625000QAs.json'

data = json.load(open(file))

overlong = set()

new_samples = []
MAX_TOK = 520

for idx, x in tqdm(enumerate(data)):

    valid_convs = []

    for conv_idx in range(0,len(x['conversations']),2):
        user, gpt = x['conversations'][conv_idx]['value'], x['conversations'][conv_idx+1]['value']
        if len(user) >= MAX_TOK or len(gpt) >= MAX_TOK:
            overlong.add(idx)
        else:
            valid_convs.extend(x['conversations'][conv_idx:conv_idx+2])
    
    if len(valid_convs):
        sample = deepcopy(x)
        sample['conversations'] = valid_convs
    
    new_samples.append(sample)

with open(file, 'w') as f:
    json.dump(new_samples, f)
