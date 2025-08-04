import os, json, cv2, re, magic
from tqdm import tqdm

from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import is_pure_color

SCALE = 1000
IMG_DIR = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/images/rico/"

DATASET_NAME = 'RefExp'

from datasets import load_dataset

ds = load_dataset("ivelin/ui_refexp_saved", split='train')

SAVE_IMG_DIR = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}"
SAVE_ROOT = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT, exist_ok=True)

USE_ACTION_PROMPT = True
DEBUG=False

SKIP_CHECKING = False

LONGEST = 1344

def make_refexp_data():
    unique_elems = {}
    elem_text_lengs = []

    samples = []    
    # record invalid samples
    invalid_elem_record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    
    if os.path.exists(invalid_elem_record_file):
        invalid_elem = json.load(open(invalid_elem_record_file))
        for k, v in invalid_elem.items():
            invalid_elem[k] = set(v)
    else:
        invalid_elem = {TOO_SMALL_ELEMENT: set(), INVALID_ELEM_BOX: set(), INVALID_ELEM_CONTENT: set(), BLANK_ELEM: set(), EMPTY_ELEM_TEXT: set(), OVERLY_LENGTHY_ELEM_TEXT: set(), DUPLICATE_ELEMEMNT: set()}

    ds.shuffle()

    for sample_idx, x in tqdm(enumerate(ds), total=len(ds), desc=DATASET_NAME):
        if DEBUG and sample_idx >= 200: break
        if x['image_id'] not in unique_elems: unique_elems[x['image_id']] = []

        img_path = os.path.join(IMG_DIR, x['image_id']+'.jpg')
        img = None

        W, H = list(map(int, re.search('precision 8, (\d+)x(\d+)', magic.from_file(img_path)).groups(1)))

        instruc = x['prompt'].strip()
        bbox_raw = eval(x['target_bounding_box'])
        bbox = bbox_raw['xmin'], bbox_raw['ymin'], bbox_raw['xmax'], bbox_raw['ymax']

        elem_text_lengs.append(instruc)
        sample_identifier = f"{x['image_id']}|{x['prompt']}"

        for v in invalid_elem.values():
            if sample_identifier in v:
                continue

        unnorm_boxes = list(map(round, [bbox[0]*W, bbox[1]*H, bbox[2]*W, bbox[3]*H]))
        
        if unnorm_boxes not in unique_elems[x['image_id']]:
            unique_elems[x['image_id']].append(unnorm_boxes)
            
        if not SKIP_CHECKING:
            # skip swipe actions
            if any([k in instruc for k in ['flip','swipe']]):
                invalid_elem[INVALID_ELEM_CONTENT].add(sample_identifier)
                continue

            if abs(bbox[0] - bbox[2]) <= 0.005 or abs(bbox[1] - bbox[3]) <= 0.005:
                invalid_elem[TOO_SMALL_ELEMENT].add(sample_identifier)
                continue

            if not (0<=bbox[0]<=1 and 0<=bbox[1]<=1 and 0<=bbox[2]<=1 and 0<=bbox[3]<=1 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                invalid_elem[INVALID_ELEM_BOX].add(sample_identifier)
                continue
                
            if len(instruc) == 0:
                invalid_elem[EMPTY_ELEM_TEXT].add(sample_identifier)
                continue

            if img is None:
                img = cv2.imread(img_path)

            if is_pure_color(img, unnorm_boxes):
                invalid_elem[BLANK_ELEM].add(sample_identifier)
                continue
        
        norm_center = [max(0, min(SCALE-1, round((bbox[0]+bbox[2])/2*SCALE))), max(0, min(SCALE-1, round((bbox[1]+bbox[3])/2*SCALE)))]
        center_str = f'({norm_center[0]},{norm_center[1]})'
        
        if USE_ACTION_PROMPT:
            action = CLICK_TEMPLATE.format(target_x=norm_center[0],target_y=norm_center[1])
            query = TURN_GND_INTO_PLANNING_PROMPT.format(instruc=instruc) if len(instruc.strip().split()) == 1 else instruc
            sample = make_actionplanning_sample(task_id=f'autogui_{DATASET_NAME}_intentgnd_{len(samples)}', global_task=query, gt_action=action, history='None', prompt_format_type='aguvis')
        else:
            sample = make_elemgnd_sample(task_id=f'autogui_{DATASET_NAME}_intentgnd_{len(samples)}', text=instruc, loc=center_str, output_tag='Please output its center coordinates.', foramt='action_json')

        sample['image'], sample['unnormalized_box'], sample['task_attr'] = 'rico/' + f'{x["image_id"]}.jpg', unnorm_boxes, instruc
        samples.append(sample)

    if sample_idx > 0 and sample_idx % 10000 == 0 or sample_idx == len(ds) - 1:
        with open(invalid_elem_record_file, 'w') as f:
            json.dump({k:list(v) for k,v in invalid_elem.items()}, f, indent=2)

    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k,v in unique_elems.items() if len(v)])

    report = f"#samples: {len(samples)}\n#Unique elements: {num_unique_elems}\n#Valid unique images: {num_valid_imgs}\n#All unique images: {len(unique_elems)}\nInvalid elem ratio: {num_invalid_elem} / {len(ds)} = {num_invalid_elem/len(ds):.2f}\ntext_loc_cnt: {len(samples)} | ocr_cnt: 0 | widgetlist_cnt: 0"
    print(report)

    file_name = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_{len(samples)// 1000}k{'_debug' if DEBUG else ''}{'_actformat' if USE_ACTION_PROMPT else ''}.json")
    print(f"save to {file_name}")
    with open(file_name.replace('.json', '_info.json'), "w") as f:
        json.dump({'num_samples': len(samples), '#num_unique_elems': num_unique_elems, '#all_elems': len(ds), '#valid_unique_images': num_valid_imgs, '#all_unique_images': len(unique_elems), 'text_loc_cnt': len(samples), 'ocr_cnt': 0, 'elemclass_cnt': 0, 'intentgnd_cnt': 0, 'widgetlist_cnt': 0, 'num_invalid_elements': num_invalid_elem}, f, indent=2)

    with open(file_name.replace(".json","_sample.json"), 'w') as f:
        json.dump(random.sample(samples,min(len(samples), 128)), f, indent=2)
        
    with open(file_name, 'w') as f:
        json.dump(samples, f, indent=2)

make_refexp_data()