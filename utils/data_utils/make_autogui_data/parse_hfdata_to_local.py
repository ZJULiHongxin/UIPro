import os, json, datasets
from tqdm import tqdm

data = datasets.load_dataset("AutoGUI/AutoGUI-v1-test", split="test")

samples = []

SAVE_DIR = "/mnt/jfs/copilot/lhx/ui_data/AutoGUI_v1"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)

for idx, sample in tqdm(enumerate(data), total=len(data)):
    img_save_path = f"images/{idx}.jpeg"
    
    img = sample['image']
    
    img_file = os.path.join(SAVE_DIR, img_save_path)
    if not os.path.exists(img_file):
        img.save(img_file)
    
    item = {
                'id': idx,
                'func': sample["func"],
                'elem_role': sample["elem_role"],
                'image_size': sample["image_size"],
                'device': sample["device"],
                'unnormalized_box': sample["unnormalized_box"],
                'img_filename': img_save_path,
            }
    
    samples.append(item)
with open(os.path.join(SAVE_DIR, "AutoGUI_v1_test.json"), "w") as f:
    json.dump(samples, f, indent=2)
    
