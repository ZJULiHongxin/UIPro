import os, json


from tqdm import tqdm
from pprint import pprint

file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/seeclick_train_1010k.json"

data = json.load(open(file))

new_samples = []
USER_TAG, ASSIST_TAG, ROLE_TAG, CONTENT_TAG = "user", "assistant", "role", "content"
for sample in tqdm(data, total=len(data)):
    new_convs, imgs = [], []
    for conv_i in range(0,len(sample['conversations']),2):
        query = sample['conversations'][conv_i]['value']
        if '</img>' in query:
            image_path, query = query.replace("<image>","").strip().split('</img>')
            image_path = image_path[16:]
            assert os.path.exists(image_path)
            imgs.append(image_path)

        user = {
            ROLE_TAG: USER_TAG,
            CONTENT_TAG: query.strip()
        }
        gpt = {
            ROLE_TAG: ASSIST_TAG,
            CONTENT_TAG: sample['conversations'][conv_i+1]['value'].replace("<image>","").strip()
        }

        new_convs.append(user)
        new_convs.append(gpt)
    
    new_convs[0][CONTENT_TAG] = "<image>\n" + new_convs[0][CONTENT_TAG]

    assert len(new_convs) == len(sample['conversations'])
    new_samples.append({'messages': new_convs, "images": imgs[:1]})

saveto = os.path.join(os.path.dirname(file), 'LlamaFactory_data')
os.makedirs(saveto, exist_ok=True)

saveto_file = os.path.join(saveto, os.path.basename(file).replace('.json', '_llamafac.json'))

with open(saveto_file.replace('.json', '_sample.json'), 'w') as f:
    json.dump(new_samples[:128],f, indent=2)

with open(saveto_file, 'w') as f:
    json.dump(new_samples,f)