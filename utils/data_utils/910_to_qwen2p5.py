import os, json, re, magic
from tqdm import tqdm
from collections import defaultdict

from utils.data_utils.misc import extract_integers
from utils.data_utils.task_prompt_lib import WITHBOX_TAG_LONG, WITHPOINT_TAG_LONG, WITHBOX_TAG, WITHPOINT_TAG, QWEN2P5_BBOX_TAG, QWEN2P5_POINT_TAG, QWEN2P5_BBOX_OUTPUT_TEMPLATE, QWEN2P5_POINT_OUTPUT_TEMPLATE, QWEN2P5_LIST, QWEN2P5_WIDLIST_TAG, QWEN2P5_WIDLIST_OUTPUT_TEMPLATE

file = "/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/GoClick_CoreSet-v2_3814k_florence.jsonl"

# Load data only once
if file.endswith('.jsonl'):
    with open(file) as f:
        data = [json.loads(line) for line in f]
else:
    with open(file) as f:
        data = json.load(f)

IMAGE_PATH_REPLACING_MAPPING = [
    ("/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/", "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/"),
    ("/mnt/nvme0n1p1/hongxin_li/UI_training_data/", "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/"),
    ("/mnt/nvme0n1p1/hongxin_li/", "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/")
]

ORIGINAL_SCALE = 1000

def scale_back(coord, original_scale, size):
    return max(0, min(size, round(coord * size / original_scale)))

# Cache for image dimensions to avoid repeated calls
image_info_cache = {}

def get_image_dimensions(image_path):
    if image_path in image_info_cache:
        return image_info_cache[image_path]
    
    img_info = magic.from_file(image_path)
    
    if 'precision' in img_info:
        W, H = list(map(int, re.search('precision 8, (\d+)x(\d+)', img_info).groups(1)))
    else:
        W, H = list(map(int, re.search('(\d+) x (\d+)', img_info).groups(1)))
    
    image_info_cache[image_path] = (W, H)
    return W, H

# Pre-process image path mapping for better performance
def update_image_path(original_path):
    new_path = original_path
    for k, v in IMAGE_PATH_REPLACING_MAPPING:
        new_path = new_path.replace(k, v)
    return new_path

# Pre-compute common replacements
TAG_REPLACEMENTS = {
    WITHPOINT_TAG: '',
    WITHBOX_TAG: '',
    WITHPOINT_TAG_LONG: '',
    WITHBOX_TAG_LONG: ''
}

# Check valid images once and store results
valid_images = set()
image_paths_to_check = set()

# First pass - collect unique image paths
for x in data:
    if 'images' in x and x['images']:
        image_path = update_image_path(x['images'][0])
        image_paths_to_check.add(image_path)

# Batch check image existence
for image_path in image_paths_to_check:
    if os.path.exists(image_path):
        valid_images.add(image_path)

new_samples, bad_samples = [], []
if 'florence' in file:
    for x in tqdm(data, total=len(data), desc=os.path.basename(file)):
        user, gpt = x['messages'][0]['content'], x['messages'][1]['content']
        
        # More efficient tag replacement
        for tag, replacement in TAG_REPLACEMENTS.items():
            gpt = gpt.replace(tag, replacement)

        # Update image path
        new_image = update_image_path(x['images'][0])
        
        # Skip invalid images
        if new_image not in valid_images:
            bad_samples.append(x)
            continue
            
        x['images'][0] = new_image
        
        # Get dimensions from cache
        W, H = get_image_dimensions(new_image)

        if 'list all ' in user:
            # Process list-type content
            elements = []
            lines = gpt.split('\n')
            if len(lines) < 3:
                bad_samples.append(x)
                continue
                
            for line in lines:
                elemrole_start = line.find(' ') + 1
                desc_start = line.find(' ', elemrole_start) + 2
                bbox_start = line.rfind(' ')

                elem_role = line[elemrole_start:desc_start].strip()
                elem_desc = line[desc_start:bbox_start].strip()
                elem_bbox = line[bbox_start:].strip()
                x1, y1, x2, y2 = extract_integers(elem_bbox)
                x1, y1, x2, y2 = scale_back(x1, ORIGINAL_SCALE, W), scale_back(y1, ORIGINAL_SCALE, H), scale_back(x2, ORIGINAL_SCALE, W), scale_back(y2, ORIGINAL_SCALE, H)
                output_str = QWEN2P5_WIDLIST_OUTPUT_TEMPLATE.format(x1=x1, y1=y1, x2=x2, y2=y2, refexp=elem_desc)

                elements.append(output_str)
            
            x['messages'][0]['content'] = f"{user} {QWEN2P5_WIDLIST_TAG}".replace('.. ', '. ')

            if len(elements) < 2:
                bad_samples.append(x)
                continue
                
            elem_str = '\n'.join(elements)
            if 'point_2d' in elem_str:
                bad_samples.append(x)
                continue
                
            x['messages'][1]['content'] = QWEN2P5_LIST.format(content=elem_str)
        else:
            # Process point/bbox-type content
            with_point, with_bbox = False, False
            old_tag = None
            
            if WITHPOINT_TAG in user:
                old_tag, with_point = WITHPOINT_TAG, True
            elif WITHPOINT_TAG_LONG in user:
                old_tag, with_point = WITHPOINT_TAG_LONG, True
            elif WITHBOX_TAG in user:
                old_tag, with_bbox = WITHBOX_TAG, True
            elif WITHBOX_TAG_LONG in user:
                old_tag, with_bbox = WITHBOX_TAG_LONG, True

            if not old_tag:
                bad_samples.append(x)
                continue

            new_tag = QWEN2P5_POINT_TAG if with_point else QWEN2P5_BBOX_TAG

            coords = extract_integers(gpt)
            if len(coords) == 2 and with_point: # "(<loc_872>,<loc_521>)"
                target_x, target_y = coords
                target_x, target_y = scale_back(int(target_x), ORIGINAL_SCALE, W), scale_back(int(target_y), ORIGINAL_SCALE, H)
                new_gpt = QWEN2P5_POINT_OUTPUT_TEMPLATE.format(target_x=target_x, target_y=target_y, refexp=x['messages'][0]['content'])
            elif len(coords) == 4 and with_bbox:
                x1, y1, x2, y2 = coords
                x1, y1, x2, y2 = scale_back(x1, ORIGINAL_SCALE, W), scale_back(y1, ORIGINAL_SCALE, H), scale_back(x2, ORIGINAL_SCALE, W), scale_back(y2, ORIGINAL_SCALE, H)
                new_gpt = QWEN2P5_BBOX_OUTPUT_TEMPLATE.format(x1=x1, y1=y1, x2=x2, y2=y2, refexp=x['messages'][0]['content'])
            else:
                bad_samples.append(x)
                continue

            if 'Please locate the target element I should interact with.' in user:
                idx = user.find('Please locate the target element I should interact with.')
                user = user.replace('Please locate the target element I should interact with.', new_tag)
                refexp = user[:idx].strip()

                x['messages'][0]['content'] = f"{refexp} {new_tag}".replace('.. ', '. ')
                x['messages'][1]['content'] = new_gpt
            else:
                x['messages'][0]['content'] = user.replace(old_tag, f"({new_tag.strip('.')})")
                x['messages'][1]['content'] = new_gpt

        new_samples.append(x)

# Process all samples at once instead of in a loop
for x in new_samples:
    image_content = x['messages'][0]['content']
    x['messages'][0]['content'] = '<image>\n' + image_content.replace('<image>', '').strip()

new_file = file.replace('_florence', '_qwen2p5')
print(f"Remove {len(bad_samples)} bad samples, save {len(new_samples)} samples to {new_file}")

# Write output files
bad_file = file.replace('.json', '_bad_samples.json')
with open(bad_file, 'w') as f:
    json.dump(bad_samples, f, indent=2, ensure_ascii=False)
    
with open(new_file, 'w') as f:
    json.dump(new_samples, f, indent=2, ensure_ascii=False)

