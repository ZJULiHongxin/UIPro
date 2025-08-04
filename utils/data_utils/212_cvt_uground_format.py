import json, os
from tqdm import tqdm
file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/uipro_processed/Qwen_data/LlamaFactory_data/Web-Emu-Andcon-MobileViews_FuncGnd-IntentGndFormat-mixed_490k_qwen_llamafac.json"

data = json.load(open(file))

TEMPLATE = """Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {description}

Answer:"""
new_samples = []
for sample in tqdm(data, total=len(data)):
    for i in range(0, len(sample['messages']), 2):
        query = sample['messages'][i]['content']
        func_desc = query[query.find('This element'):]
    
        sample['messages'][i]['content'] = TEMPLATE.format(description=func_desc)
        sample['messages'][i+1]['content'] = sample['messages'][i+1]['content'].replace(',', ', ')

    new_samples.append(sample)
with open(file.replace('.json', '_UGround.json'), 'w') as f:
    json.dump(new_samples, f, indent=2)