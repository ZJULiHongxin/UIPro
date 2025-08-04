import os, json, cv2, re, magic
from tqdm import tqdm

from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import is_pure_color

SCALE = 1000
IMG_DIR = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/images/seeclick_web_imgs/"
ANNO_FILE = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/raw/seeclick_web.json"

DATASET_NAME = 'SeeClick-Web'
SAVE_ROOT = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"
os.makedirs(SAVE_ROOT, exist_ok=True)

USE_ACTION_PROMPT = False
DEBUG=False

SKIP_CHECKING = True

def make_seeclickweb_data():
    unique_elems = {}
    elem_text_lengs = []
    
    iterated_elem_cnt = 0

    samples = []

    data = json.load(open(ANNO_FILE))
    
    # record invalid samples
    invalid_elem_record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    
    if os.path.exists(invalid_elem_record_file):
        invalid_elem = json.load(open(invalid_elem_record_file))
        for k, v in invalid_elem.items():
            invalid_elem[k] = set(v)
    else:
        invalid_elem = {TOO_SMALL_ELEMENT: set(), INVALID_ELEM_BOX: set(), INVALID_ELEM_CONTENT: set(), BLANK_ELEM: set(), EMPTY_ELEM_TEXT: set(), OVERLY_LENGTHY_ELEM_TEXT: set(), DUPLICATE_ELEMEMNT: set()}

    if DEBUG: data = random.sample(data, 200)
    for group_idx, x in tqdm(enumerate(data), total=len(data), desc="SeeClick-Web"):
        if x['img_filename'] not in unique_elems: unique_elems[x['img_filename']] = []

        img_path = os.path.join(IMG_DIR, x['img_filename'])
        img = None

        W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img_path)).groups()))
        iterated_elem_cnt += len(x['elements'])

        used_instruc = []

        for elem_info in x['elements']:
            bbox, instruc = elem_info['bbox'], elem_info['instruction'].strip()
            elem_text_lengs.append(instruc)
            sample_identifier = f"{x['img_filename']}|{elem_info['instruction']}"

            if not SKIP_CHECKING and instruc in used_instruc:
                invalid_elem[DUPLICATE_ELEMEMNT].add(sample_identifier)
                continue
            
            used_instruc.append(instruc)

            is_invalid = False
            for v in invalid_elem.values():
                if sample_identifier in v:
                    is_invalid = True;break
            if is_invalid:
                continue

            unnorm_boxes = list(map(round, [bbox[0]*W, bbox[1]*H, bbox[2]*W, bbox[3]*H]))
            
            if unnorm_boxes not in unique_elems[x['img_filename']]:
                unique_elems[x['img_filename']].append(unnorm_boxes)
                
            if not SKIP_CHECKING:
                if abs(bbox[0] - bbox[2]) <= 0.005 or abs(bbox[1] - bbox[3]) <= 0.005:
                    invalid_elem[TOO_SMALL_ELEMENT].add(sample_identifier)
                    continue

                if not (0<=bbox[0]<=1 and 0<=bbox[1]<=1 and 0<=bbox[2]<=1 and 0<=bbox[3]<=1 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                    invalid_elem[INVALID_ELEM_BOX].add(sample_identifier)
                    continue

                if ('{' in instruc) or ('}' in instruc):
                    invalid_elem[INVALID_ELEM_CONTENT].add(sample_identifier)
                    continue
                    
                if len(instruc) == 0:
                    invalid_elem[EMPTY_ELEM_TEXT].add(sample_identifier)
                    continue
                
                if len(instruc) > 60:
                    invalid_elem[OVERLY_LENGTHY_ELEM_TEXT].add(sample_identifier)
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
                textloc_sample = make_actionplanning_sample(task_id=f'autogui_{DATASET_NAME}_textloc_{len(samples)}', global_task=instruc, gt_action=action, history='None', prompt_format_type='aguvis')
            else:
                textloc_sample = make_elemgnd_sample(task_id=f'autogui_{DATASET_NAME}_elemgnd_{len(samples)}', text=instruc, loc=center_str, output_tag='')#, foramt='action_json')

            textloc_sample['image'], textloc_sample['unnormalized_box'], textloc_sample['task_attr'], textloc_sample['elem_role'], textloc_sample['url'] = DATASET_NAME + '/' + x['img_filename'], unnorm_boxes, instruc, elem_info['data_type'], x['url']
            samples.append(textloc_sample)

        if group_idx > 0 and group_idx % 10000 == 0 or group_idx == len(data) - 1:
            with open(invalid_elem_record_file, 'w') as f:
                json.dump({k:list(v) for k,v in invalid_elem.items()}, f, indent=2)

    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k,v in unique_elems.items() if len(v)])

    report = f"#samples: {len(samples)}\n#Unique elements: {num_unique_elems}\n#Valid unique images: {num_valid_imgs}\n#All unique images: {len(unique_elems)}\nInvalid elem ratio: {num_invalid_elem} / {iterated_elem_cnt} = {num_invalid_elem/iterated_elem_cnt:.2f}\ntext_loc_cnt: {len(samples)} | ocr_cnt: 0 | widgetlist_cnt: 0"
    print(report)

    file_name = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_{len(samples)// 1000}k{'_debug' if DEBUG else ''}{'_actformat' if USE_ACTION_PROMPT else ''}.json")
    print(f"save to {file_name}")
    with open(file_name.replace('.json', '_info.json'), "w") as f:
        json.dump({'num_samples': len(samples), '#num_unique_elems': num_unique_elems, '#all_elems': iterated_elem_cnt, '#valid_unique_images': num_valid_imgs, '#all_unique_images': len(unique_elems), 'text_loc_cnt': len(samples), 'ocr_cnt': 0, 'elemclass_cnt': 0, 'intentgnd_cnt': 0, 'widgetlist_cnt': 0, 'num_invalid_elements': num_invalid_elem, 'invalid_elem_types': {k:len(v) for k,v in invalid_elem.items()}}, f, indent=2)

    with open(file_name.replace(".json","_sample.json"), 'w') as f:
        json.dump(random.sample(samples,min(len(samples), 128)), f, indent=2)
        
    with open(file_name, 'w') as f:
        json.dump(samples, f, indent=2)

make_seeclickweb_data()