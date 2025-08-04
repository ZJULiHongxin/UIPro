import os, json, cv2, re, traceback, magic
import numpy as np
from tqdm import tqdm

from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import is_pure_color

# webqa的gpt response里有bbox
# 每个样本对话都是1QAs，qa样本也是
DATASET_NAME = "MultiUI"

MULTIUI_TASK_TYPES = {
    'UI understanding': ['meta_generate', 'embed_caption', 'action_prediction', 'webqa', 'embed_qa'],
    'OCR': ['long_text_OCR', 'title_identification'],
    'Grounding': ['element_ground', 'action_ground', 'element_ground_bbox', 'action_ground_bbox'],
    # 'v4_50K_10_curated_v2_action_ground_129979' gpt输出选项字母

    'Grounding_None': ['action_ground_{idx}_none_of_above', 'element_ground_{idx}_none_of_above', 'action_ground_bbox_{idx}_none_of_above', 'element_ground_bbox_{idx}_none_of_above']
    # GPT回答None
    }

SCALE = 1000
IMG_DIR = f"/mnt/vdb1/hongxin_li/{DATASET_NAME}"

SAVE_ROOT = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"
os.makedirs(SAVE_ROOT, exist_ok=True)
    
CAPTION = False
WEBQA = False
OCR = False
GND = True

SKIP_CHECKING = False
# 生成每种任务的示例
def generate_task_samples():
    multiui_sample_file = "/mnt/vdb1/hongxin_li/MultiUI/stage1_data.json"

    multiui_data = json.load(open(multiui_sample_file, 'r'))

    task_examples = {
        'meta_generate': [], # 整体UI描述
        'embed_caption': [],
        'action_prediction': [],
        'webqa': [],
        'embed_qa': [],
        'long_text_OCR': [],
        'title_identification': [],
        'element_ground': [],
        'action_ground': [],
        'element_ground_bbox': [],
        'action_ground_bbox': [],
    }
    remaining = []
    used_ids = []

    for x in tqdm(multiui_data, total=len(multiui_data)):
        added = False
        for task_type in task_examples.keys():
            if task_type in x['id']:
                if 'ground' in task_type and 'bbox' not in task_type:
                    if 'bbox' not in x['id']:
                        used_ids.append(x['id'])
                        task_examples[task_type].append(x)
                        added = True
                else:
                    used_ids.append(x['id'])
                    task_examples[task_type].append(x)
                    added = True
        
        if not added:
            remaining.append(x)

    assert len(remaining) == 0
    with open(os.path.join(os.path.dirname(__file__), 'multiui_samples.json'), 'w') as f:
        json.dump(task_examples, f, indent=2)

BBOX_PATTERN = re.compile(pattern = r'\[\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*,\s*(\d+\.\d+)\s*\]')

def to_real_box(box, img_h, img_w):
    x1, y1, x2, y2 = box
    x1 = round(x1 * img_w)
    y1 = round(y1 * img_h)
    x2 = round(x2 * img_w)
    y2 = round(y2 * img_h)
    return [x1, y1, x2, y2]

def check_sample_integrity():
    samples_each_task = json.load(open(os.path.join(os.path.dirname(__file__), 'multiui_samples_10k.json'), 'r'))
    
    samples = []
    for v in samples_each_task.values():
        samples.extend(v)

    for x in samples:
        sample_id = x['id']
        user, gpt = x['conversations'][0]['value'], x['conversations'][1]['value']

        print(f'User: {user}\nGPT: {gpt}')
        img = cv2.imread(os.path.join(IMG_DIR, x['image']))
        H, W = img.shape[:2]
        if 'webqa' in sample_id:
            # Find all matches in the text
            matches = BBOX_PATTERN.findall(gpt)
            for match in matches:
                # Convert the matched string to a list of floats
                bbox = to_real_box([float(num) for num in match], H, W)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
            cv2.imwrite('test.png', img)
            1+1
        elif 'element_ground_bbox' in sample_id:
            # Find all matches in the text
            matches = BBOX_PATTERN.findall(gpt)
            for match in matches:
                # Convert the matched string to a list of floats
                bbox = to_real_box([float(num) for num in match], H, W)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            cv2.imwrite('test.png', img)
            1+1
        elif 'action_ground_bbox' in sample_id:
            # Find all matches in the text
            matches = BBOX_PATTERN.findall(gpt)
            for match in matches:
                # Convert the matched string to a list of floats
                bbox = to_real_box([float(num) for num in match], H, W)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            cv2.imwrite('test.png', img)
            1+1

SKIPPED_TASKS = ['action_prediction', 'none_of_above']

def make_multiui_data():
    multiui_sample_file = "/mnt/vdb1/hongxin_li/MultiUI/stage1_data.json"

    multiui_data = json.load(open(multiui_sample_file, 'r'))
    
    uicaption_cnt = embedcaption_cnt = webqa_cnt = embed_qa_cnt = ocr_cnt = titleocr_cnt = elemgnd_cnt = intentgnd_cnt = processed_cnt = 0
    
    invalid_elem = []
    caption_samples = []
    exception_sample_idxs = []
    gnd_ref_qa_samples = []

    # load invalid samples
    invalid_ids = json.load(open("/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/MultiUI_processed/stats.json"))["invalid_elem"]

    # 将caption数据分离出来，作为预训练数据
    for sample_idx, x in tqdm(enumerate(multiui_data), total=len(multiui_data)):
        sample_id = x['id']
        if any(k in sample_id for k in SKIPPED_TASKS):
            continue
        
        processed_cnt += 1

        x['image'] = DATASET_NAME + '/' + x['image']
        user, gpt = x['conversations'][0]['value'], x['conversations'][1]['value']

        try:
            if CAPTION and ('meta_generate' in sample_id or 'caption' in sample_id):
                if 'meta_generate' in sample_id:
                    task_id = f'autogui_multiui_UIcaption_{uicaption_cnt}'
                    uicaption_cnt += 1
                elif 'caption' in sample_id:
                    task_id = f'autogui_multiui_embedcaption_{embedcaption_cnt}'
                    embedcaption_cnt += 1

                x['id'] = task_id
                
                caption_samples.append(x)
            elif WEBQA and 'webqa' in sample_id:
                matches = BBOX_PATTERN.findall(gpt)
                if len(matches):
                    for match in matches:
                        # Convert the matched string to a list of floats
                        original_box_str = '[{}]'.format(', '.join(match))
                        normalized_box = [str(round(float(num) * SCALE)) for num in match]
                        new_box_str = '({})'.format(','.join(normalized_box))
                        gpt = gpt.replace(original_box_str, new_box_str)
                
                    x['conversations'][1]['value'] = gpt

                task_id = f'autogui_multiui_webqa_{webqa_cnt}'
                x['id'] = task_id
                gnd_ref_qa_samples.append(x); webqa_cnt += 1
            elif WEBQA and 'embed_qa' in sample_id:
                task_id = f'autogui_multiui_embedqa_{embed_qa_cnt}'
                embed_qa_cnt += 1
                x['id'] = task_id
                gnd_ref_qa_samples.append(x)
            elif OCR and 'OCR' in sample_id:
                task_id = f'autogui_multiui_ocr_{ocr_cnt}'
                ocr_cnt += 1
                x['id'] = task_id
                gnd_ref_qa_samples.append(x)
            elif OCR and 'title_identification' in sample_id:
                task_id = f'autogui_multiui_titleidentification_{titleocr_cnt}'
                titleocr_cnt += 1
                x['id'] = task_id
                gnd_ref_qa_samples.append(x)
            elif GND and ('element_ground_bbox' in sample_id or 'action_ground_bbox' in sample_id):
                if sample_id in invalid_ids:
                    invalid_elem.append(sample_id)
                    continue
                match = BBOX_PATTERN.findall(gpt)[0]
                # Convert the matched string to a list of floats
                original_box_str = '[{}]'.format(', '.join(match))
        
                # detect whether the element is displayed on the screen
                W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(os.path.join(os.path.dirname(IMG_DIR), x['image']))).groups(1)))
                # img = cv2.imread(os.path.join(os.path.dirname(IMG_DIR), x['image']))
                # H, W = img.shape[:2]
                
                x1,y1,x2,y2 = to_real_box([float(num) for num in match], H, W)
                
                # if is_pure_color(image=img, roi=[x1,y1,x2,y2]):
                #     invalid_elem.append(sample_id)
                    
                #     # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #     # cv2.imwrite('test.png', img)
                #     continue

                normalized_box = [str(round(float(num) * SCALE)) for num in match]
                new_box_str = '({})'.format(','.join(normalized_box))
                x['conversations'][1]['value'] = new_box_str

                user = user.replace('\u201D', '"').replace('\u201C', '"').replace('\u2018',"'").replace('\u2019',"'")

                quote_idx = user.find("'")
                if quote_idx != -1:
                    quote = "'"
                else:
                    quote_idx = user.find('"')
                    if quote_idx != -1:
                        quote = '"'

                query_start = user.find(':')
                if query_start == -1 or (quote_idx != -1 and query_start > quote_idx):
                    query_start = user.find('description "')
                    if query_start != -1:
                        query_start += 10

                if query_start == -1:
                    continue
                
                single_quote_idx = user.find("'", query_start, query_start + 4)
                double_quote_idx = user.find('"', query_start, query_start + 4)
                
                if single_quote_idx != -1 and double_quote_idx != -1:
                    quote_idx = min(single_quote_idx, double_quote_idx)
                    if single_quote_idx < double_quote_idx:
                        quote = "'"
                    else: quote = '"'
                elif single_quote_idx != -1:
                    quote_idx = single_quote_idx; quote = "'"
                elif double_quote_idx != -1:
                    quote_idx = double_quote_idx; quote = '"'
                else:
                    quote_idx = -1

                if quote_idx != -1:
                    right_quote_idx = user.rfind(quote)
                    if right_quote_idx == quote_idx:
                        query = user[:user.rfind(quote)+1]
                    else:
                        query = user[quote_idx+1:right_quote_idx]
                else:
                    period_idx = user.find('.', query_start)
                    if period_idx != -1:
                        query = user[query_start+1:period_idx]
                    else: query = user[query_start+1:]

                if query in ['', '"', "'"] or '="' in query or "='" in query:
                    invalid_elem.append(sample_id)
                    continue

                if query[0] == '"' and query[-1] == '"' or query[0] == "'" and query[-1] == "'":
                    query = query.strip(' \'".')


                #print(query)
                if 'float' in query:
                    print(f"{sample_id}: {user}"); continue
                if 'element_ground_bbox' in sample_id:
                    x['conversations'][0]['value'] = re.sub(r'\s+', ' ', random.choice(web_loca_all_point_prompt).replace("with point", "with bbox") + f' {query.strip()}')
                    x['id'], x['task_attr'], x['unnormalized_box'] = f'autogui_multiui_elemgnd_{elemgnd_cnt}', query, [x1,y1,x2,y2]
                    gnd_ref_qa_samples.append(x)       
                    elemgnd_cnt += 1
                else:
                    intent = query[0].lower() + query[1:]
                    sample = make_intentgnd_sample(task_id = f'autogui_multiui_intentgnd_{intentgnd_cnt}', intent=intent.strip(' "'), loc=new_box_str, output_tag=WITHBOX_TAG)
                    sample['image'], sample['task_attr'], sample['unnormalized_box'] = x['image'], intent, [x1,y1,x2,y2]

                    intentgnd_cnt += 1

                    gnd_ref_qa_samples.append(sample)
        except Exception as e:
            exception_sample_idxs.append([sample_idx, traceback.format_exc()])
            print(e)

    
    with open(os.path.join(SAVE_ROOT, 'stats.json'), 'w') as f:
        json.dump({'total_samples': len(multiui_data), 'processed_samples': processed_cnt, 'invalid_sample_cnt': len(invalid_elem), 'collected_samples': len(caption_samples) + len(gnd_ref_qa_samples), 'uicaption_cnt': uicaption_cnt, 'embedcaption_cnt': embedcaption_cnt, 'webqa_cnt': webqa_cnt, 'embed_qa_cnt': embed_qa_cnt, 'ocr_cnt': ocr_cnt, 'titleocr_cnt': titleocr_cnt, 'elemgnd_cnt': elemgnd_cnt, 'intentgnd_cnt': intentgnd_cnt, 'invalid_elem': invalid_elem, 'exception_sample_idxs': exception_sample_idxs}, f, indent=2)

    gndrefqa_file_name = os.path.join(SAVE_ROOT, f'{DATASET_NAME}_gnd_ref_qa_scale{SCALE}_{len(gnd_ref_qa_samples)//1000}k.json')
    with open(gndrefqa_file_name.replace('.json', '_sample.json'), 'w') as f:
        json.dump(random.sample(gnd_ref_qa_samples,min(len(gnd_ref_qa_samples),160)), f, indent=2)
    with open(gndrefqa_file_name, 'w') as f:
        json.dump(gnd_ref_qa_samples, f)

    caption_file_name = os.path.join(SAVE_ROOT, f'{DATASET_NAME}_caption_{len(caption_samples)//1000}k.json')
    with open(caption_file_name.replace('.json', '_sample.json'), 'w') as f:
        json.dump(random.sample(caption_samples,min(len(caption_samples),160)), f, indent=2)
    with open(caption_file_name, 'w') as f:
        json.dump(caption_samples, f)



make_multiui_data()    