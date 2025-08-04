import json, os, datasets
from PIL import Image
from tqdm import tqdm

HF_PATH = "MMInstruction/OSWorld-G"

SAVE_TO = "/mnt/stepeval/datasets/VL_datasets/OSWorld-G"
SAVE_IMAGE_TO = "/mnt/stepeval/datasets/VL_datasets/OSWorld-G/images"
os.makedirs(SAVE_IMAGE_TO, exist_ok=True)
data = datasets.load_dataset(HF_PATH, split="test")

samples = []

for sample in tqdm(data, total=len(data), desc="Processing OSWorld-G data"):
    image_path = os.path.join(SAVE_IMAGE_TO, sample["image_path"])
    id, instruction, image, unnorm_bbox, gui_types = sample['id'], sample['instruction'], sample['image'], sample['mimo_bbox'], sample['GUI_types']
    
    if not os.path.exists(image_path):
        image.save(image_path)
    
    samples.append({
        "id": id,
        "instruction": instruction,
        "image_path": 'images/' + sample["image_path"],
        "image_size": image.size,
        "unnorm_bbox": unnorm_bbox,
        "gui_types": gui_types,
    })

with open(os.path.join(SAVE_TO, "OSWorld-G_test.json"), "w") as f:
    json.dump(samples, f, indent=2)