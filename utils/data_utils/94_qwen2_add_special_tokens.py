import os, json, re, random
from tqdm import tqdm

data = []

file = "/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/AutoGUI_FuncGnd_s1000_490k_ARoom_llamafac.json"
if file.endswith('.json'):
    with open(file, 'r', encoding='utf-8') as file:
        data = json.load(file)
else:
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

OBJ_REF_START, OBJ_REF_END = '<|object_ref_start|>', '<|object_ref_end|>'
BOX_START, BOX_END = '<|box_start|>', '<|box_end|>'

ADD_REF = False

SCALE = 1000

P_TAG, B_TAG = '(Output the center coordinates of the target)', '(Output the bounding box coordinates of the target)'
for x in tqdm(data, total=len(data)):
    for i in range(0,len(x['messages']),2):
        user, gpt = x['messages'][i]['content'].replace('.. ', '. '), x['messages'][i+1]['content']
        
        if gpt.startswith('('):
            if gpt.count(',') == 1:
                if P_TAG not in user:
                    x['messages'][i]['content'] = user + f" {P_TAG}"
                x['messages'][i+1]['content'] = f"{BOX_START}{gpt},{gpt}{BOX_END}"
            else:
                if B_TAG not in user:
                    user = user + f" {B_TAG}"
                
                # Special processing for MultiUI samples
                if 'MultiUI' in x['images'][0]:
                    if 'float' in user:
                        print(user)
                x['messages'][i]['content'] = re.sub(r'\s+', ' ', user)
                box = eval(gpt)
                x['messages'][i+1]['content'] = f"{BOX_START}({box[0]},{box[1]}),({box[2]},{box[3]}){BOX_END}"
            

        elif 'Please list' in user:
            elem_annos = gpt.split('\n')
            new_annos = []
            for anno in elem_annos:
                # 2 StaticText 'June 14, 2019' (40,8,60,10)
                first_blank_idx, last_blank_idx = anno.find(' '), anno.rfind(' ')
                refexp = anno[first_blank_idx+1:last_blank_idx]
                box = eval(anno[last_blank_idx+1:])
                new_anno = f"{anno[:first_blank_idx]} {OBJ_REF_START}{refexp}{OBJ_REF_END} {BOX_START}({box[0]},{box[1]}),({box[2]},{box[3]}){BOX_END}"
                new_annos.append(new_anno)
            x['messages'][i+1]['content'] = '\n'.join(new_annos)
        elif 'Classify the type' in user:
            pass
        else:
            raise Exception()
        
        1+1

file_path = '/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/AutoGUI_FuncGnd_s1000_490k_ARoom_llamafac_qwen2tag.json'

# Open the file in write mode
with open(file_path.replace(".json", "_sample.json"), 'w', encoding='utf-8') as f:
    json.dump(random.sample(data, 256),f, indent=2)

if file_path.endswith('.json'):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)
else:
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            # Convert each dictionary to a JSON string and write it to the file
            file.write(json.dumps(item) + '\n')
