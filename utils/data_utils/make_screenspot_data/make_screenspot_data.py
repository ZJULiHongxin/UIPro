import os, json
from tqdm import tqdm
import datasets
from utils.data_utils.task_prompt_lib import elemgnd_prompt, format_point_tag, WITHPOINT_TAG_LONG

data = datasets.load_dataset("rootsautomation/ScreenSpot", split='test')

DATASET_NAME = 'ScreenSpot'
SAVE_ROOT = os.path.join('/data/hongxin_li/scaling_exp', DATASET_NAME + "_processed")
os.makedirs(SAVE_ROOT, exist_ok=True)
FORMAT = [
    'llava',
    'qwen2',
    'florence'
][0]

POINT_FORMAT = ['plain', 'qwen2', 'florence'][1]
idx = 0
CONV_TAG = ['conversations', 'messages'][idx]
ROLE_TAG = ['from', 'role'][idx]
USER_TAG = ['human', 'user'][idx]
ASSISTANT_TAG = ['gpt', 'assistant'][idx]
CONTENT_TAG = ['value', 'content'][idx]

SCALE = 1000

image_dir = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/ScreenSpot/screenspot_imgs"

samples = []


for sample in tqdm(data, total=len(data)):
    img_filename, normalized_box, instruction = sample['file_name'], sample['bbox'], sample['instruction']
    
    img_path = os.path.join(image_dir, img_filename)
    
    norm_x, norm_y = max(0, min(SCALE-1, round((normalized_box[0]+normalized_box[2])/2*SCALE))), max(0, min(SCALE-1, round((normalized_box[1]+normalized_box[3])/2*SCALE)))

    prompt = elemgnd_prompt[0].format(text=instruction)
    gnd_result = format_point_tag([norm_x, norm_y], POINT_FORMAT)

    sample = {
        'id': f'autogui_{DATASET_NAME}_elemgnd_{len(samples)}',
        CONV_TAG: [
            {
                ROLE_TAG: USER_TAG,
                CONTENT_TAG: '<image>\n' + prompt + f' {WITHPOINT_TAG_LONG}'
            },
            {
                ROLE_TAG: ASSISTANT_TAG,
                CONTENT_TAG: gnd_result
            }
        ]
    }
    
    if FORMAT == 'llava':
        sample['image'] = img_path
    else:
        sample['images'] = [img_path]

    samples.append(sample)

save_file = os.path.join(SAVE_ROOT, f'{DATASET_NAME}_s{SCALE}_{FORMAT}_{len(samples)}.json')

print(f'save {len(samples)} samples to {save_file}')
with open(save_file, 'w') as f:
    json.dump(samples, f, indent=2)