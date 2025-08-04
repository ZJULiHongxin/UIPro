import os, json, cv2, random
from tqdm import tqdm
from utils.data_utils.task_prompt_lib import SPATIAL_AWARE_FUNC_DESC_PROMPT,  SPATIAL_AWARE_FUNC_DESC_PROMPT_UNIQUE, QWEN_BOX_START, QWEN_BOX_END

# crop save dir
SAVE_CROP_DIR = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/spatial_aware_funcdesc_crop"
os.makedirs(SAVE_CROP_DIR, exist_ok=True)

new_samples = []
invalid_samples = []

DEBUG = False
SCALE = 1000
DILATION_SIZE = 5 # Leave some space for the text
DRAW = False
# web
data = json.load(open("/mnt/nvme0n1p1/hongxin_li/UI_training_data/raw/funcpred/20241127_diverse/funcgnd_349633.json"))

def make_desc_sample(sample_id, original_img_path, crop_img_path, desc, loc, unique=True):
    box_desc = f'{QWEN_BOX_START}({loc[0]},{loc[1]}),({loc[2]},{loc[3]}){QWEN_BOX_END}'
    sample = {
        'id': sample_id,
        'messages': [
            {'role': 'user', 'content': SPATIAL_AWARE_FUNC_DESC_PROMPT_UNIQUE.format(loc=box_desc) if unique else SPATIAL_AWARE_FUNC_DESC_PROMPT.format(loc=box_desc)},
            {'role': 'assistant', 'content': desc}
        ],
        'images': [original_img_path, crop_img_path]
    }
    return sample

# detect invalid func desc samples
def is_invalid_func_desc(desc):
    # no longer than 350
    return len(desc) > 350 or not desc.startswith('This element')

cnt = 0
for i, x in tqdm(enumerate(data), total=len(data), desc=f"Processing {len(data)} samples"):
    cnt += 1
    if DEBUG and i > 100:
        break

    func_desc = x['task_attr']
    if is_invalid_func_desc(func_desc):
        invalid_samples.append(x)
        continue
    
    crop_img_name = f"{cnt}.png"
    crop_img_path=os.path.join(SAVE_CROP_DIR, crop_img_name)

    if os.path.exists(crop_img_path):
        continue
    img_path = x['conversations'][0]['value'].split('</img>')[0][16:]
    img = cv2.imread(img_path)

    H, W = img.shape[:2]

    if W > H:
        img = cv2.resize(img, (W//2, H//2))
    else:
        img = cv2.resize(img, (W//3, H//3))
    H, W = img.shape[:2]

    x1, y1, x2, y2 = x['unnormalized_box']
    x1, y1, x2, y2 = int(max(0, x1 - DILATION_SIZE)), int(max(0, y1 - DILATION_SIZE)), int(min(x2 + DILATION_SIZE, W)), int(min(y2 + DILATION_SIZE, H))
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite(crop_img_path, crop_img)

    if DRAW:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite('test.png', img)

    sample = make_desc_sample(
        sample_id=f'autogui_funcdesc_{cnt}', 
        original_img_path=img_path, 
        crop_img_path=crop_img_path, 
        desc=func_desc,
        loc=[max(0, min(round(x1 / W * SCALE), SCALE)), max(0, min(round(y1 / H * SCALE), SCALE)), max(0, min(round(x2 / W * SCALE), SCALE)), max(0, min(round(y2 / H * SCALE), SCALE))],
        unique=False
    )
    new_samples.append(sample)

# android
file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/android_system_1021/android_system_FuncRef_38355.json"
data = json.load(open(file))

for i, x in tqdm(enumerate(data), total=len(data), desc=f"Processing {len(data)} samples"):
    cnt += 1
    if DEBUG and i > 100:
        break

    func_desc = x['conversations'][1]['value']
    if is_invalid_func_desc(func_desc):
        invalid_samples.append(x)
        continue

    crop_img_name = f"{cnt}.png"
    crop_img_path = os.path.join(SAVE_CROP_DIR, crop_img_name)
    if os.path.exists(crop_img_path):
        continue

    img_path = os.path.join("/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp", x['image'])
    img = cv2.imread(img_path)


    H, W = img.shape[:2]
    x1, y1, x2, y2 = x['unnormalized_box']
    x1, y1, x2, y2 = int(max(0, x1 - DILATION_SIZE)), int(max(0, y1 - DILATION_SIZE)), int(min(x2 + DILATION_SIZE, W)), int(min(y2 + DILATION_SIZE, H))
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite(crop_img_path, crop_img)

    if DRAW:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite('test.png', img)

    sample = make_desc_sample(
        sample_id=f'autogui_funcdesc_{cnt}', 
        original_img_path=img_path, 
        crop_img_path=crop_img_path, 
        desc=func_desc,
        loc=[max(0, min(round(x1 / W * SCALE), SCALE)), max(0, min(round(y1 / H * SCALE), SCALE)), max(0, min(round(x2 / W * SCALE), SCALE)), max(0, min(round(y2 / H * SCALE), SCALE))]
    )
    new_samples.append(sample)

# MobileViews
file = "/mnt/nvme0n1p1/hongxin_li/WebpageFunctionality/MobileViews_FuncPred1126/MobileViews_s1000_FuncGnd79151.json"

data = json.load(open(file))

for i, x in tqdm(enumerate(data), total=len(data), desc=f"Processing {len(data)} samples"):
    cnt += 1
    if DEBUG and i > 100:
        break

    func_desc = x['task_attr']
    if is_invalid_func_desc(func_desc):
        invalid_samples.append(x)
        continue

    crop_img_name = f"{cnt}.png"
    crop_img_path=os.path.join(SAVE_CROP_DIR, crop_img_name)

    if os.path.exists(crop_img_path):
        continue

    img_path = os.path.join("/mnt/nvme0n1p1/hongxin_li/WebpageFunctionality", x['image'])
    img = cv2.imread(img_path)


    H, W = img.shape[:2]
    x1, y1, x2, y2 = x['unnormalized_box']
    x1, y1, x2, y2 = int(max(0, x1 - DILATION_SIZE)), int(max(0, y1 - DILATION_SIZE)), int(min(x2 + DILATION_SIZE, W)), int(min(y2 + DILATION_SIZE, H))
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite(crop_img_path, crop_img)

    if DRAW:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite('test.png', img)

    sample = make_desc_sample(
        sample_id=f'autogui_funcdesc_{cnt}', 
        original_img_path=img_path, 
        crop_img_path=crop_img_path, 
        desc=func_desc,
        loc=[max(0, min(round(x1 / W * SCALE), SCALE)), max(0, min(round(y1 / H * SCALE), SCALE)), max(0, min(round(x2 / W * SCALE), SCALE)), max(0, min(round(y2 / H * SCALE), SCALE))]
    )
    new_samples.append(sample)

# AndroidControl
file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/AndroidControl_processed/AndroidControl_s1000_1091879.json"
data = json.load(open(file))

for i, x in tqdm(enumerate(data), total=len(data), desc=f"Processing {len(data)} samples"):
    if 'funcgnd' not in x['id']:
        continue

    if DEBUG and i > 100:
        break

    cnt += 1

    func_desc = x['task_attr']
    if is_invalid_func_desc(func_desc):
        invalid_samples.append(x)
        continue

    crop_img_name = f"{cnt}.png"
    crop_img_path=os.path.join(SAVE_CROP_DIR, crop_img_name)

    if os.path.exists(crop_img_path):
        continue

    img_path = os.path.join("/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/", x['image'])
    img = cv2.imread(img_path)

    H, W = img.shape[:2]
    x1, y1, x2, y2 = x['unnormalized_box']

    if (x2-x1)*(y2-y1) / (H*W) > 0.25: continue

    x1, y1, x2, y2 = int(max(0, x1 - DILATION_SIZE)), int(max(0, y1 - DILATION_SIZE)), int(min(x2 + DILATION_SIZE, W)), int(min(y2 + DILATION_SIZE, H))
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite(crop_img_path, crop_img)

    if DRAW:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite('test.png', img)

    sample = make_desc_sample(
        sample_id=f'autogui_funcdesc_{cnt}', 
        original_img_path=img_path, 
        crop_img_path=crop_img_path, 
        desc=func_desc,
        loc=[max(0, min(round(x1 / W * SCALE), SCALE)), max(0, min(round(y1 / H * SCALE), SCALE)), max(0, min(round(x2 / W * SCALE), SCALE)), max(0, min(round(y2 / H * SCALE), SCALE))]
    )
    new_samples.append(sample)

SAVE_ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed"

save_to = os.path.join(SAVE_ROOT, f'autogui_spatial_aware_funcdesc_{cnt}.json')

with open(save_to.replace('.json', '_sample.json'), 'w') as f:
    json.dump(random.sample(new_samples, 128), f, indent=2)

with open(save_to, 'w') as f:
    json.dump(new_samples, f)