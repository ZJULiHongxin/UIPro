import os, json

from tqdm import tqdm
from pprint import pprint

file = "/data/hongxin_li/scaling_exp/UIPro_processed/AutoGUI_FuncGnd_s1000_490k_ARoom.json"
img_dir = "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/"

saveto = os.path.join(os.path.dirname(file), 'LlamaFactory_data')
os.makedirs(saveto, exist_ok=True)
saveto_file = os.path.join(saveto, os.path.basename(file).replace('.json', '_llamafac.json'))

data = json.load(open(file))

new_samples = []; broken_imgs = []

PURE_TEXT = True

USER_TAG, ASSIST_TAG, ROLE_TAG, CONTENT_TAG = "user", "assistant", "role", "content"

READD_IMAGE_TAG = False

for sample in tqdm(data, total=len(data)):
    new_convs = []
    for conv_i in range(0,len(sample['conversations']),2):
        user = {
            ROLE_TAG: USER_TAG,
            CONTENT_TAG: sample['conversations'][conv_i]['value'].replace("<image>","").strip() if READD_IMAGE_TAG else sample['conversations'][conv_i]['value'].strip()
        }
        gpt = {
            ROLE_TAG: ASSIST_TAG,
            CONTENT_TAG: sample['conversations'][conv_i+1]['value'].replace("<image>","").strip() if READD_IMAGE_TAG else sample['conversations'][conv_i+1]['value'].strip()
        }

        new_convs.append(user)
        new_convs.append(gpt)

    assert len(new_convs) == len(sample['conversations'])
    
    if 'image' in sample or 'images' in sample:
        # Check if the image is JPG or PNG format
        img_name = sample['image'] if 'image' in sample else sample['images'][0]
        img_file = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_file):
            img_file = img_file.replace(".jpg",".png")

        if not os.path.exists(img_file):
            broken_imgs.append(img_file)
            continue

        if READD_IMAGE_TAG: 
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