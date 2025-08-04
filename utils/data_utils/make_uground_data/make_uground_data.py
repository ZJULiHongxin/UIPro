import os, json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import io, random

from utils.data_utils.task_prompt_lib import format_point_tag

def image_from_bytes(image_bytes):
    """Creates a PIL Image object from bytes.

    Args:
        image_bytes: A byte string containing the image data.

    Returns:
        A PIL Image object, or None if an error occurs.
    """
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

DIR = "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/UGround-V1-Data"
IMAGE_SAVE_DIR = '/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/UGround'
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

data = load_dataset(DIR)['train']


new_samples = []

for i, x in tqdm(enumerate(data), total=len(data)):
    img_path = os.path.join(IMAGE_SAVE_DIR, f'{i:07d}.png')

    if not os.path.exists(img_path):
        img = image_from_bytes(x["image"]).convert("RGB")
        img.save(img_path)
    
    convs = eval(x["conversations"])
    
    user, gpt = convs[0]['value'], convs[1]['value']
    
    coords = eval(gpt)
    
    messages = [
        {
            'role': 'user',
            'content': user
        },
        {
            'role': 'assistant',
            'content': format_point_tag(coords, 'qwen2')
        }
    ]
    
    new_samples.append({
        'images': [img_path],
        'messages': messages
    })

SAVE_JSON_TO = "/data/hongxin_li/scaling_exp/UGround_processed/"
os.makedirs(SAVE_JSON_TO, exist_ok=True)

save_path = os.path.join(SAVE_JSON_TO, f'uground_{len(new_samples)}.json')


with open(save_path.replace(".json", "_sample.json"), 'w') as f:
    json.dump(random.sample(new_samples, 256), f)
    
with open(save_path, 'w') as f:
    json.dump(new_samples, f)