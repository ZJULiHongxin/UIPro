import os, json
from copy import deepcopy
from tqdm import tqdm
from pprint import pprint

file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/Mind2Web_processed/Mind2Web_train_7723.json"
image_dir = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/"

saveto = os.path.join(os.path.dirname(file), 'Qwen_data')
os.makedirs(saveto, exist_ok=True)
saveto_file = os.path.join(saveto, os.path.basename(file).replace('.json', '_qwen.json'))

data = json.load(open(file))

new_samples = []
USER_TAG, ASSIST_TAG, ROLE_TAG, CONTENT_TAG = "user", "assistant", "from", "value"
for sample in tqdm(data, total=len(data)):
    new_convs = []
    for conv_i in range(0,len(sample['conversations']),2):
        user = {
            ROLE_TAG: USER_TAG,
            CONTENT_TAG: sample['conversations'][conv_i]['value'].replace("<image>","").strip()
        }
        gpt = {
            ROLE_TAG: ASSIST_TAG,
            CONTENT_TAG: sample['conversations'][conv_i+1]['value'].replace("<image>","").strip()
        }

        new_convs.append(user)
        new_convs.append(gpt)
    
    new_convs[0][CONTENT_TAG] = "Picture 1: <img>" + os.path.join(image_dir, sample['image']) + "</img>" + new_convs[0][CONTENT_TAG]

    assert len(new_convs) == len(sample['conversations'])
    
    new_sample = deepcopy(sample)
    new_sample['conversations'] = new_convs
    new_samples.append(new_sample)

with open(saveto_file.replace('.json', '_sample.json'), 'w') as f:
    json.dump(new_samples[:160],f, indent=2)

with open(saveto_file, 'w') as f:
    json.dump(new_samples,f)