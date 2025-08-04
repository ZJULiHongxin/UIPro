import os, json, re, random
from tqdm import tqdm
from copy import deepcopy

# SMR + SV = 1141233

RESAMPLE = True

a=json.load(open("/mnt/nvme0n1p1/hongxin_li/highres_autogui/data/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"))

num_qas = sum(len(x['conversations'])//2 for x in a) # 4009078

new_samples_w_box = []
pattern = r'\[\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\]'

for x in tqdm(a, total=len(a)):
    new_convs_wo_box = []
    new_convs_w_box = []
    for i in range(0, len(x['conversations']), 2):
        # some samples have [x,x,x,x]-pattern substrings, which can be excluded by using `'image' in x`
        if re.search(pattern, x['conversations'][i]['value']) or re.search(pattern, x['conversations'][i+1]['value']) and 'image' in x:
        #if all(k in x['conversations'][i]['value'] for k in ['[',']']) or all(k in x['conversations'][i+1]['value'] for k in ['[',']']):
            new_convs_w_box.extend(x['conversations'][i:i+2])
        else:
            new_convs_wo_box.extend(x['conversations'][i:i+2])
    
    x_w_box = deepcopy(x)
    x_w_box['conversations'] = new_convs_w_box
    new_samples_w_box.append(x_w_box)

    x['conversations'] = new_convs_wo_box
    
    
# samples w/o boxes
new_samples = []
for x in a:
    if len(x['conversations']):
        new_samples.append(x)

new_samples_w_box_new = []
for x in new_samples_w_box:
    if len(x['conversations']):
        new_samples_w_box_new.append(x)
new_samples_w_box_new = new_samples_w_box_new * (354822 // len(new_samples_w_box_new)) + random.sample(new_samples_w_box_new, 354822 % len(new_samples_w_box_new))

with open("./data/onlyBox_to355k-v2.json", "w") as f:
    json.dump(new_samples_w_box_new, f)
    
new_samples.extend(deepcopy(random.sample(new_samples, len(a) - len(new_samples))))
new_num_qas = sum(len(x['conversations']) for x in new_samples)

temp = new_num_qas

idx = list(range(len(new_samples)))
random.shuffle(idx)

for i in tqdm(idx, total=len(idx)):
    if temp >= num_qas: break
    conv = new_samples[i]['conversations']
    if len(conv) < 4: continue

    replicate_i = random.randint(0, len(conv) // 2 - 1)
    copy_conv = deepcopy(conv[2*replicate_i:2*replicate_i+4])
    for x in copy_conv:
        x['value'] = x['value'].replace("<image>","").strip()
        
    new_samples[i]['conversations'].extend(copy_conv)
    temp += len(copy_conv)

newnew_num_qas = sum(len(x['conversations']) for x in new_samples)



with open("./data/naturalQA_noBox_355k.json", "w") as f:
    json.dump(new_samples, f)

cnt = 0
for x in new_samples:
    for i in range(len(x['conversations']), 2):
        if re.search(pattern, x['conversations'][i]['value']) or re.search(pattern, x['conversations'][i+1]['value']): cnt += 1