import os, json

from tqdm import tqdm
from pprint import pprint

from task_prompt_lib import format_point_tag

file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/AndroidControl_processed/AndroidControl-train_IntengGnd_s1000_34340.json"
img_dir = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/"

saveto = os.path.dirname(file)
os.makedirs(saveto, exist_ok=True)
saveto_file = os.path.join(saveto, os.path.basename(file).replace('.json', '_florence.json'))

data = json.load(open(file))

new_samples = []; broken_imgs = []

PURE_TEXT = True


USER_TAG, ASSIST_TAG, ROLE_TAG, CONTENT_TAG = "user", "assistant", "role", "content"
for sample in tqdm(data, total=len(data)):
    new_convs = []
    for conv_i in range(0,len(sample['conversations']),2):
        user = {
            ROLE_TAG: USER_TAG,
            CONTENT_TAG: sample['conversations'][conv_i]['value'].replace("<image>","").strip()
        }
        gpt = {
            ROLE_TAG: ASSIST_TAG,
            CONTENT_TAG: format_point_tag(
                eval(sample['conversations'][conv_i+1]['value'].replace("<image>","").strip()), point_format='florence'
                )
        }

        new_convs.append(user)
        new_convs.append(gpt)

    assert len(new_convs) == len(sample['conversations'])
    
    if 'image' in sample:
        img_file = os.path.join(img_dir, sample['image'])#.replace(".jpg",".png")
        if not os.path.exists(img_file):
            print(img_file)
            broken_imgs.append(img_file)
            continue

        new_convs[0][CONTENT_TAG] = "<image>\n" + new_convs[0][CONTENT_TAG]
        
        new_samples.append({'messages': new_convs, "images": [
        img_file
        ]})
    else:
        new_samples.append({'messages': new_convs, "images": []})

print(f"Broken images: {len(broken_imgs)}")
with open(saveto_file.replace('.json', '_sample.json'), 'w') as f:
    json.dump(new_samples[:160],f, indent=2)

with open(saveto_file, 'w') as f:
    json.dump(new_samples,f)