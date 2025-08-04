import os, json
from tqdm import tqdm

file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_dedup_8MQAs_v3.jsonl"

format = ['llamafac'][0]

new_samples = []

if format == "llamafac":
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    
    for sample in tqdm(data, total=len(data)):
        splits = []
        for conv_i in range(0,len(sample['messages']),2):
            new_sample = {
                'messages': [
                    {"role": "user",
                    "content": sample['messages'][conv_i]['content']},
                    {"role": "assistant",
                     "content": sample['messages'][conv_i+1]['content']}
                ],
                'images': sample['images']
            }
            splits.append(new_sample)

        new_samples.extend(splits)

with open(file.replace(".jsonl", "_SingleTurn.jsonl"), "w") as f:
    for entry in new_samples:
        json_line = json.dumps(entry)
        f.write(json_line + '\n')