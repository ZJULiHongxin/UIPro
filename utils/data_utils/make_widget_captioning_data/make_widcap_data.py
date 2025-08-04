import os, json, cv2, re, magic
from tqdm import tqdm

from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import is_pure_color, VerbExtactor

SCALE = 1000
IMG_DIR = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/images/rico/"
ANNO_FILE = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/raw/widget_captioning.json"

DATASET_NAME = 'WidgetCaptioning'
SAVE_ROOT = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"
os.makedirs(SAVE_ROOT, exist_ok=True)

USE_ACTION_PROMPT = False
DEBUG=False

SKIP_CHECKING = True

ELEMREF = False

CLEAN_DICT = {
    "shudown": "shutown",
    "closse": "close",
    "clik": "click",
    "locatoin": "location"
}

def clean_instruc(text):
    for k,v in CLEAN_DICT.items():
        text = text.replace(k, v)
    
    return text

def make_widcap_data():
    unique_elems = {}
    elem_text_lengs = []

    samples = []
    intentgnd_cnt = elemgnd_cnt = elemref_cnt = 0

    verb_extractor = VerbExtactor()

    data = json.load(open(ANNO_FILE))
    
    # record invalid samples
    invalid_elem_record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')
    
    if os.path.exists(invalid_elem_record_file):
        invalid_elem = json.load(open(invalid_elem_record_file))
        for k, v in invalid_elem.items():
            invalid_elem[k] = set(v)
    else:
        invalid_elem = {TOO_SMALL_ELEMENT: set(), INVALID_ELEM_BOX: set(), INVALID_ELEM_CONTENT: set(), BLANK_ELEM: set(), EMPTY_ELEM_TEXT: set(), OVERLY_LENGTHY_ELEM_TEXT: set(), DUPLICATE_ELEMEMNT: set()}

    # record the instruction type
    instruc_type_file = os.path.join(SAVE_ROOT, 'instruc_type_record.json')
    
    if os.path.exists(instruc_type_file):
        instruc_type_dict = json.load(open(instruc_type_file))
    else:
        instruc_type_dict = {}

    if DEBUG: data = random.sample(data, 200)
    for group_idx, x in tqdm(enumerate(data), total=len(data), desc=DATASET_NAME):
        if x['img_filename'] not in unique_elems: unique_elems[x['img_filename']] = []

        img_path = os.path.join(IMG_DIR, x['img_filename'])
        img = None

        W, H = list(map(int, re.search('precision 8, (\d+)x(\d+)', magic.from_file(img_path)).groups(1)))

        bbox, instruc = x['bbox'], x['instruction'].strip()
        elem_text_lengs.append(instruc)
        sample_identifier = f"{x['img_filename']}|{x['instruction']}"

        is_invalid=False
        for v in invalid_elem.values():
            if sample_identifier in v:
                is_invalid = True; break
        if is_invalid: continue

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

            if img is None:
                img = cv2.imread(img_path)

            if is_pure_color(img, unnorm_boxes):
                invalid_elem[BLANK_ELEM].add(sample_identifier)
                continue
        
        instruc = clean_instruc(instruc)
        norm_center = [max(0, min(SCALE-1, round((bbox[0]+bbox[2])/2*SCALE))), max(0, min(SCALE-1, round((bbox[1]+bbox[3])/2*SCALE)))]
        center_str = f'({norm_center[0]},{norm_center[1]})'
        
        if instruc not in instruc_type_dict:
            verb, verb_idx = verb_extractor.find_first_verb(instruc)
            instruc_type_dict[instruc] = verb

        if instruc_type_dict[instruc] is None:
            sample = make_elemgnd_sample(task_id=f'autogui_{DATASET_NAME}_elemgnd_{len(samples)}', text=instruc, loc=center_str, output_tag='')
            elemgnd_cnt += 1
        else:
            if USE_ACTION_PROMPT:
                action = CLICK_TEMPLATE.format(target_x=norm_center[0],target_y=norm_center[1])
                query = TURN_GND_INTO_PLANNING_PROMPT.format(instruc=instruc) if len(instruc.strip().split()) == 1 else instruc
                sample = make_actionplanning_sample(task_id=f'autogui_{DATASET_NAME}_intentgnd_{len(samples)}', global_task=query, gt_action=action, history='None', prompt_format_type='aguvis')
            else:
                sample = make_intentgnd_sample(task_id=f'autogui_{DATASET_NAME}_intentgnd_{len(samples)}', intent=instruc, loc=center_str, output_tag='')
            
            intentgnd_cnt += 1
        
        sample['image'], sample['unnormalized_box'], sample['task_attr'] = 'rico/' + x['img_filename'], unnorm_boxes, instruc
        samples.append(sample)

        if ELEMREF:
            elemref_sample = make_elemref_sample(task_id=f'autogui_{DATASET_NAME}_elemref_{len(samples)}', text=instruc, loc=center_str, output_tag=WITHPOINT_TAG)
            elemref_sample['image'], elemref_sample['unnormalized_box'], elemref_sample['task_attr'] = 'rico/' + x['img_filename'], unnorm_boxes, center_str
            samples.append(elemref_sample); elemref_cnt += 1

    if group_idx > 0 and group_idx % 10000 == 0 or group_idx == len(data) - 1:
        with open(invalid_elem_record_file, 'w') as f:
            json.dump({k:list(v) for k,v in invalid_elem.items()}, f, indent=2)
        with open(instruc_type_file, 'w') as f:
            json.dump(instruc_type_dict, f, indent=2)

    num_invalid_elem = sum(len(v) for v in invalid_elem.values())
    
    num_unique_elems = sum(len(v) for v in unique_elems.values())
    num_valid_imgs = len([k for k,v in unique_elems.items() if len(v)])

    report = f"#samples: {len(samples)}\n#Unique elements: {num_unique_elems}\n#Valid unique images: {num_valid_imgs}\n#All unique images: {len(unique_elems)}\nInvalid elem ratio: {num_invalid_elem} / {len(data)} = {num_invalid_elem/len(data):.2f}\nintentgnd: {intentgnd_cnt} | elemgnd: {elemgnd_cnt} | elemref: {elemref_cnt}"
    print(report)

    file_name = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_{len(samples)// 1000}k{'_debug' if DEBUG else ''}{'_actformat' if USE_ACTION_PROMPT else ''}.json")
    print(f"save to {file_name}")
    with open(file_name.replace('.json', '_info.json'), "w") as f:
        json.dump({'num_samples': len(samples), '#num_unique_elems': num_unique_elems, '#all_elems': len(data), '#valid_unique_images': num_valid_imgs, '#all_unique_images': len(unique_elems), 'intentgnd_cnt': intentgnd_cnt, 'elemgnd_cnt': elemgnd_cnt, 'elemref_cnt': elemref_cnt, 'num_invalid_elements': num_invalid_elem}, f, indent=2)

    with open(file_name.replace(".json","_sample.json"), 'w') as f:
        json.dump(random.sample(samples,min(len(samples), 128)), f, indent=2)
        
    with open(file_name, 'w') as f:
        json.dump(samples, f, indent=2)

make_widcap_data()