"""
remove duplicate elements that have identical bboxes and annotations
"""

import random, json
from collections import defaultdict
from tqdm import tqdm
file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/MultiUI_processed/MultiUI_gnd_s1000_1920k.json"

data = json.load(open(file, 'r'))

task_box_attr = defaultdict(list)

new_data = []

unique_dict = {} 

# Datasets that also use img name as part of identifier
NEED_IMGNAME_IDENTIFIER = ["widgetcap", "ricosca"]
NEED_IMGNAME_IDENTIFIER = any(n in file.lower() for n in NEED_IMGNAME_IDENTIFIER)

for x in tqdm(data, total=len(data), desc=file):
    if any(k in x['id'] for k in ['gnd', 'ref', 'ground', 'textloc', 'ocr']):
        task = x['id'].split('_')[2]
        if task not in unique_dict: unique_dict[task] = defaultdict(list)

        if any(k in x['id'] for k in ['gnd', 'ground', 'textloc']):
            task_attr = x['task_attr']
        else: task_attr = x['conversations'][1]['value']
        
        if 'unnormalized_box' not in x:
            bbox = x['conversations'][1]['value']
        else:
            bbox = str(x['unnormalized_box'])
        identifier = f"{bbox}|{task_attr}" if not NEED_IMGNAME_IDENTIFIER else f"{x['image']}|{bbox}|{task_attr}"
        
        unique_dict[task][identifier].append(x)
    else:
        new_data.append(x)

for task, sets in unique_dict.items():
    for k, v in sets.items():
        new_data.append(random.choice(v))
        
        # if len(v) > 1:
        #     print(task, k, len(v))
        #     1+1

print(len(data), len(new_data))

save_to = file.replace('.json', '_dedup.json')
with open(save_to, 'w') as f:
    json.dump(new_data, f)