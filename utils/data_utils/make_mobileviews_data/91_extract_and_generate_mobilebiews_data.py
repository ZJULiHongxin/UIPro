import os, json, random, shutil
import pandas as pd
import cv2, pytesseract
from rapidfuzz import fuzz
from tqdm import tqdm
from PIL import Image
from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import classify_node, is_pure_color, is_valid_string, is_box_overlapping_np
from collections import defaultdict
# 噪声样本
INVALID = [100547, 100625]

DEBUG = False   

TEXTLOC = True
OCR = False
ELEMCLASS = True
INTENTGND = False
WIDGETLIST = True


text_loc_cnt = ocr_cnt = elemclass_cnt = intentgnd_cnt = widgetlist_cnt = 0

PROB_BOX = 0.0
SCALE=1000

mobileviews_dir = "/mnt/vdb1/hongxin_li/MobileViews/"
ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"

DATASET_NAME = 'mobileviews'
save_images_to = os.path.join(ROOT, f"{DATASET_NAME}{'_debug' if DEBUG else ''}")
# if os.path.exists(save_images_to):
#     shutil.rmtree(save_images_to)
os.makedirs(save_images_to, exist_ok=True)

samples = []

mv_screenshot_vh_mapping = {}

parts = ["MobileViews_0-150000", "MobileViews_150001-291197", "MobileViews_300000-400000", "MobileViews_400001-522301"]

for part in parts:
    df = pd.read_csv(mobileviews_dir+f'{part}.csv')
    mv_screenshot_vh_mapping = {}
    for k, v in zip(df.iloc[:, 0], df.iloc[:, 1]):
        mv_screenshot_vh_mapping[f'{part}/' + k] = f'{part}/' + v

processed_img_cnt = valid_img_cnt = iterated_elem_cnt = annotated_node_cnt = 0

node_invalid_types = defaultdict(int)

last_sample_cnt = 0

for sample_idx, (screenshot_file, vh_file) in tqdm(enumerate(mv_screenshot_vh_mapping.items()), total=len(mv_screenshot_vh_mapping)):
    if DEBUG and sample_idx % 5000 > 3: continue
    sample_id = int(os.path.basename(screenshot_file).split('.')[0])
    if sample_id in INVALID: continue

    screenshot_file = os.path.join(mobileviews_dir, screenshot_file)
    if not os.path.exists(screenshot_file): continue
    vh_file = os.path.join(mobileviews_dir, vh_file)
    if not os.path.exists(vh_file): continue
    
    processed_img_cnt += 1

    with open(vh_file, 'r') as f:
        vh = json.load(f)
    
    nodes = vh["views"]

    img = cv2.imread(screenshot_file)
    
    if img.shape[0] > 1920:
        img = img[:1920]
        cv2.imwrite(screenshot_file, img)
    H, W = img.shape[:2]

    all_valid_nodes = []
    for node in nodes:
        if len(node['children']) > 0: continue
        if not node['visible']: continue
        
        iterated_elem_cnt += 1

        #if 'desc' not in node: continue
        (x1,y1), (x2,y2) = node["bounds"]
        
        is_duplicate_box = False
        for x in all_valid_nodes:
            (x3, y3), (x4, y4) = x["bounds"]
            if (x1 == x3 and y1 == y3 and x2 == x4 and y2 == y4):
                is_duplicate_box = True
                break
        if is_duplicate_box:
            node_invalid_types['duplicate box'] += 1
            continue

        # detect invalid bounding boxes
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x1 >= x2 or y1 >= y2:
            node_invalid_types['invalid box coordinates'] += 1
            continue
        
        #  detect oversize nodes
        if (x2-x1) * (y2-y1) / (H*W) >= 0.65:
            node_invalid_types['oversize elemnent'] += 1
            continue
        
        # detect too small nodes
        if (y2-y1) / H <= 0.005 or (x2-x1) / W <= 0.005:
            node_invalid_types['too small elemnent'] += 1
            continue
                
        box = [round(x1/W*SCALE), round(y1/H*SCALE), round(x2/W*SCALE), round(y2/H*SCALE)]
        box = [min(max(0, i), SCALE-1) for i in box]
        box_str = f'[{box[0]},{box[1]},{box[2]},{box[3]}]'
        
        box_w, box_h = x2 - x1, y2 - y1
        center = [(x1+x2)/2, (y1+y2)/2]
        center_str = f"({min(max(0, round((x1+x2)/2/W*SCALE)), SCALE-1)},{min(max(0, round((y1+y2)/2/H*SCALE)), SCALE-1)})"

        # detect unsuccessfully displayed nodes
        if is_pure_color(img, [x1,y1,x2,y2]):
            node_invalid_types['element not displayed'] += 1
            continue
        
        all_valid_nodes.append(node)

    # 通过比对OCR结果和元素文本属性，以去除box覆盖在其他元素之上的无效元素
    if len(all_valid_nodes) > 1:
        all_valid_nodes_after_checking_overlap = []
        for node_idx, node in enumerate(all_valid_nodes):
            (x1,y1), (x2,y2) = node["bounds"]
            # check those icons and texts overlapping with others
            is_overlap = is_box_overlapping_np(target_box=[x1,y1,x2,y2], other_boxes=[[x['bounds'][0][0], x['bounds'][0][1], x['bounds'][1][0], x['bounds'][1][1]] for cur_idx, x in enumerate(all_valid_nodes) if cur_idx != node_idx], threshold=0.01)
            
            if is_overlap:
                elem_type = classify_node(node)
                if elem_type in ['Icon', 'Text']:
                    elem_text = node.get('text', None)
                    
                    if elem_text is not None:
                        elem_text = elem_text.strip()
                        lower_node_text = elem_text.lower()
                        elem_text_is_description = any(k in lower_node_text for k in ['icon', 'button', 'back'])
                        if not elem_text_is_description:
                            ocr_result = pytesseract.image_to_string(cv2.cvtColor(img[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)).strip()
                            similarity_ratio = fuzz.ratio(ocr_result, elem_text)
                            node['ocr'] = [ocr_result, similarity_ratio]
                            if similarity_ratio < 22:
                                node_invalid_types['overlapping element'] += 1
                                continue
                                # node['bad_ocr'] = True
                                #continue
            all_valid_nodes_after_checking_overlap.append(node)
        all_valid_nodes = all_valid_nodes_after_checking_overlap

    if len(all_valid_nodes) == 0: continue
    all_node_texts = [node['text'] for node in all_valid_nodes if 'text' in node]

    contain_invalid_characters = False
    for text in all_node_texts:
        if text is None: continue
        if not is_valid_string(text):
            contain_invalid_characters = True
            break

    if contain_invalid_characters:
        node_invalid_types['invalid text language'] += len(all_valid_nodes)
        continue

    # create image symlinks
    new_sc_file = os.path.join(save_images_to, os.path.basename(screenshot_file))
    
    if not os.path.exists(new_sc_file):
        os.symlink(screenshot_file, new_sc_file)
    
    short_img_path = new_sc_file[new_sc_file.find(DATASET_NAME):]
    if False:
        for idx, node in enumerate(all_valid_nodes):
            (x1,y1), (x2,y2) = node["bounds"]
            center = [round((x1+x2)/2), round((y1+y2)/2)]
            box_w, box_h = x2 - x1, y2 - y1

            elem_type = classify_node(node)
            if elem_type == 'Text': color = (255, 0, 0)
            elif elem_type == 'Icon': color = (0, 125, 125) # yellow
            elif elem_type == 'Image': color = (255, 0, 255) # purple
            else: color = (0, 255, 0)
            
            if node.get('bad_ocr', False):
                color = (0, 0, 255); thickness = 5
            else: thickness = 2
            cv2.rectangle(img, pt1=(x1,y1), pt2=(x2,y2), thickness=thickness, color=color)
            cv2.putText(img, f'{idx}', org=(center[0], center[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=color, thickness=2, lineType=cv2.LINE_AA)
            
        cv2.imwrite("test.png", img)
        1+1

    used_boxes, used_node_descs = [], []
    for node in all_valid_nodes:
        # Skip nodes with overlapped boxes
        if node["bound_box"] in used_boxes:
            node_invalid_types['overlapping element'] += 1
            continue
        else:
            used_boxes.append(node["bound_box"])

        x1,y1, x2,y2 = list(map(int, node["bound_box"].split(',')))
                
        box = [round(x1/W*SCALE), round(y1/H*SCALE), round(x2/W*SCALE), round(y2/H*SCALE)]
        box = [min(max(0, i), SCALE-1) for i in box]
        box_str = f'({box[0]},{box[1]},{box[2]},{box[3]})'
        
        box_w, box_h = x2 - x1, y2 - y1
        center = [(x1+x2)//2, (y1+y2)//2]
        normalized_center = [min(max(0, round(center[0]/W*SCALE)), SCALE-1), min(max(0, round(center[1]/H*SCALE)), SCALE-1)]
        
        center_str = f"({normalized_center[0]},{normalized_center[1]})"

        elem_type = classify_node(node)
        elem_text = node['text']
        elem_signature = node['signature']
        elem_desc = node.get('desc', '')
        elem_allowed_actions = node.get('allowed_actions', '')
        package = node['package']

        # 获得元素描述
        node_desc = None
        if elem_text:
            node_desc = elem_text
        if node.get('content_description', None) is not None:
            content_desc = node['content_description'].strip()
            if content_desc:
                # 如果content-desc更加详细，则将其和node_text共同作为目标描述，以避免node_text无法提供足够的信息定位元素
                # 举例：/data0/jingran/workspace/hongxin_li/digiagent/ckpts/0823_start_from_drawer/images/0823_start_from_drawer3/1724473143.7755919/6.png中，滑动条（seekbar）上的点的文本是"4.0", "5.0"，而content-desc为 "media volumn" "call volumne"，更加详细
                if elem_text is not None and len(elem_text) and len(content_desc) > len(elem_text):
                    node_desc = f"{content_desc}, {elem_text}"
                else:
                    node_desc = content_desc
        if node_desc is None and node.get('resource_id',None) is not None:
            raw_node_text = node['resource_id'].split('/')[-1].strip()
            if raw_node_text:
                node_desc = raw_node_text
        
        node['node_desc'] = node_desc
        
        if elem_text is not None and 0 < len(elem_text) <= 200:
            elem_text = elem_text.strip()
            with_box = random.random() < PROB_BOX
            # To avoid ambibuity, a text localization task is created if the elem text is unique.
            #if random.random() > 0.5 and 
            if TEXTLOC and all_node_texts.count(elem_text) <= 1:
                task_id = f'autogui_mobileviews_textloc_{text_loc_cnt}'
                sample = make_textloc_sample(task_id, text=elem_text, loc=box_str if with_box else center_str, output_tag=WITHBOX_TAG if with_box else WITHPOINT_TAG)
                sample['task_attr'], sample['unnormalized_box'], sample['image'], sample['sample_id'], sample['package'] = elem_text, [x1,y1, x2,y2], short_img_path, sample_id, package
                samples.append(sample)
                text_loc_cnt += 1

            if OCR:
                task_id = f'autogui_mobileviews_ocr_{ocr_cnt}'
                
                # IF the element text does not populate the whole box, we use the box as the reference as center refenrence is not accurate
                if is_pure_color(img, [center[0] - box_w // 10, center[1] - box_h // 10, center[0] + box_w // 10, center[1] + box_h // 10]): with_box = True

                loc = box_str if with_box else center_str
                sample = make_ocr_sample(task_id, text=elem_text, loc=loc, with_box=with_box)
                sample['task_attr'], sample['unnormalized_box'], sample['image'], sample['sample_id'], sample['package'] = loc, [x1,y1, x2,y2], short_img_path, sample_id, package
                samples.append(sample)
                ocr_cnt += 1

        if INTENTGND and node_desc and node_desc not in used_node_descs and 0 < len(node_desc) <= 200:
            used_node_descs.append(node_desc)
            
            # tag = elem_desc.split()[0][1:]
            # 虽然node里可能给出了元素的HTML tag，但这里还是使用安卓的标签类型
            tag = node['class'].split('.')[-1]
            with_box = random.random() < PROB_BOX
            task_id = f'autogui_mobileviews_intentgnd_{intentgnd_cnt}'

            if elem_text not in [None, 'None', 'none'] and all_node_texts.count(elem_text) <= 1:
                intent = gen_naive_action_gnd_anno(node_desc.strip(' ,.'), tag, normalized_center, scale=SCALE)
                
                sample = make_intentgnd_sample(task_id, intent=intent, loc=box_str if with_box else center_str, output_tag='')
                sample['task_attr'], sample['unnormalized_box'], sample['image'], sample['sample_id'], sample['package'] = intent, [x1,y1, x2,y2], short_img_path, sample_id, package
                samples.append(sample)
                intentgnd_cnt += 1

    if WIDGETLIST and len(all_valid_nodes) >= 2:
        task_id = f'autogui_mobileviews_widgetlist_{widgetlist_cnt}'
        
        node_texts_boxes = []
        for node in all_valid_nodes:
            node_desc = '' if node['node_desc'] is None else node['node_desc']

            unnormalized_box = list(map(int, node['bound_box'].split(',')))
            normalized_box = [round(unnormalized_box[0] / W * SCALE), round(unnormalized_box[1] / H * SCALE), round(unnormalized_box[2] / W * SCALE), round(unnormalized_box[3] / H * SCALE)]

            node_texts_boxes.append((node['class'].split('.')[-1], ' '.join(x for x in node_desc.strip(' ,.').split('\n') if x.strip()), unnormalized_box, normalized_box))


        node_texts_boxes.sort(key=lambda x: (x[2][1]+x[2][3], x[2][0]+x[2][2]))
        elem_list_str = '\n'.join(f"{i} {nodeclass} '{nodetext}' ({normalized_box[0]},{normalized_box[1]},{normalized_box[2]},{normalized_box[3]})" for i, (nodeclass, nodetext, unnormalized_box, normalized_box) in enumerate(node_texts_boxes))
        
        if False:
            img = cv2.imread(screenshot_file)
            print(elem_list_str)
            for i, (nodeclass, nodetext, box, _) in enumerate(node_texts_boxes):
                x1,y1, x2,y2 = box
                center = [round((x1+x2)/2), round((y1+y2)/2)]

                elem_type = classify_node(node)
                if elem_type == 'Text': color = (255, 0, 0)
                elif elem_type == 'Icon': color = (0, 125, 125)
                elif elem_type == 'Image': color = (255, 0, 255)
                else: color = (0, 255, 0)
                cv2.rectangle(img, pt1=(x1,y1), pt2=(x2,y2), thickness=2, color=color)
                cv2.putText(img, str(i), (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1)
            cv2.imwrite("test1.png", img)
            1+1
        
        
        sample = make_widgetlist_sample(task_id, elem_list=elem_list_str)
        sample['task_attr'], sample['image'], sample['sample_id'], sample['package'], sample['unnormalized_box'] = None, short_img_path, sample_id, package, [x[2] for x in node_texts_boxes]
        samples.append(sample)
        widgetlist_cnt += 1
    
    if len(samples) > last_sample_cnt:
        valid_img_cnt += 1
    last_sample_cnt = len(samples)
    # if valid_img_cnt >= 1000: break

invalid_elem = sum(cnt for cnt in node_invalid_types.values())
report = f"Valid image ratio: {valid_img_cnt} / {processed_img_cnt} = {valid_img_cnt/processed_img_cnt:.2f}\nInvalid elem ratio: {invalid_elem} / {iterated_elem_cnt} = {invalid_elem/iterated_elem_cnt:.2f}\ntext_loc_cnt: {text_loc_cnt} | ocr_cnt: {ocr_cnt} | elemclass_cnt: {elemclass_cnt} | intentgnd_cnt: {intentgnd_cnt} | widgetlist_cnt: {widgetlist_cnt}"
print(report)


file_name = os.path.join(ROOT, "mobileviews_processed", f"mobileviews_TextLoc_scale{SCALE}_{len(samples)// 1000}k{'_debug' if DEBUG else ''}.json")
print(f"save to {file_name}")
with open(file_name.replace('.json', '_info.json'), "w") as f:
    json.dump({'num_samples': len(samples), 'valid_img_cnt': valid_img_cnt, 'processed_img_cnt': processed_img_cnt, 'invalid_elem': invalid_elem, 'iterated_elem_cnt': iterated_elem_cnt, 'text_loc_cnt': text_loc_cnt, 'ocr_cnt': ocr_cnt, 'elemclass_cnt': elemclass_cnt, 'intentgnd_cnt': intentgnd_cnt, 'widgetlist_cnt': widgetlist_cnt, 'node_invalid_types': node_invalid_types}, f, indent=2)

with open(file_name.replace(".json","_sample.json"), 'w') as f:
    json.dump(random.sample(samples,min(len(samples), 128)), f, indent=2)
    
with open(file_name, 'w') as f:
    json.dump(samples, f, indent=2)
