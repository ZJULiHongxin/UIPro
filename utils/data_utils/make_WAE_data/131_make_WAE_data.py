import os, json, random, shutil, glob, re, magic
import pandas as pd
import cv2, pytesseract
from rapidfuzz import fuzz
from tqdm import tqdm
from PIL import Image
from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import classify_node, is_pure_color, is_valid_string, is_box_overlapping_np, find_all_elem_texts_boxes, detect_invalid_lang
from collections import defaultdict
import xml.etree.ElementTree as ET
from copy import deepcopy

DEBUG = False   

TEXTLOC = True
OCR = True
ICONGND = True
ICONREF = True
ELEMCLASS = True
INTENTGND = True
WIDGETLIST = True



PROB_BOX = 0.0
SCALE=1000
SKIP_CHECKING = False

# 由于该数据集UI分辨率变长的最大为1280，故不做resize
DATASET_NAME = 'WAE'

wae_dir = f"/mnt/vdb1/hongxin_li/{DATASET_NAME}/tmp"
ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"


save_images_to = os.path.join(ROOT, f"{DATASET_NAME}{'_debug' if DEBUG else ''}")
# if os.path.exists(save_images_to):
#     shutil.rmtree(save_images_to)
os.makedirs(save_images_to, exist_ok=True)


# Load ckpt
SAVE_ROOT = os.path.join(ROOT, f"{DATASET_NAME}_processed")
chunk_dir = os.path.join(SAVE_ROOT, 'chunks')
os.makedirs(chunk_dir, exist_ok=True)

ckpt_file = os.path.join(SAVE_ROOT, f'ckpt_s{SCALE}.json')
all_traj_names = sorted(os.listdir(wae_dir))
if DEBUG:
    all_traj_names = all_traj_names[:50]
if os.path.exists(ckpt_file):
    with open(ckpt_file, 'r') as f:
        ckpt = json.load(f)
        completely_proc_traj_names = ckpt['completely_proc_traj_names']
else:
    completely_proc_traj_names = []

processed_img_cnt = valid_img_cnt = iterated_elem_cnt = annotated_node_cnt = 0
text_loc_cnt = ocr_cnt = icongnd_cnt = iconref_cnt = elemclass_cnt = intentgnd_cnt = widgetlist_cnt = 0

last_sample_cnt = 0

CHUNK_SIZE = 25 if not DEBUG else 1
chunk_samples = []; chunk_traj_names = []; unique_elems = {}

invalid_elem_meta = {TOO_SMALL_ELEMENT: set(), INVALID_ELEM_BOX: set(), INVALID_ELEM_CONTENT: set(), BLANK_ELEM: set(), EMPTY_ELEM_TEXT: set(), OVERLY_LENGTHY_ELEM_TEXT: set(), DUPLICATE_ELEMEMNT: set(), OVERSIZED_ELEMENT: set(), EXTREME_ASPECT_RATIO: set(), GHOST_ELEMENT: set()}

# record invalid samples
invalid_elem_record_file = os.path.join(SAVE_ROOT, 'invalid_elem_record.json')

fitst_chunk_invalid_elem_file = os.path.join(chunk_dir, 'chunk_1_invalid_elems.json') # indexing from 1
if os.path.exists(invalid_elem_record_file):
    with open(f, 'r') as f:
        invalid_elem = json.load(f)
    for k, v in invalid_elem.items():
        invalid_elem[k] = set(v)
else:
    invalid_elem = deepcopy(invalid_elem_meta)

for traj_idx, traj_name in tqdm(enumerate(all_traj_names), total=len(all_traj_names), desc="Processing trajectories"):
    if traj_name in completely_proc_traj_names: continue
    
    chunk_traj_names.append(traj_name)

    screenshot_files = glob.glob(os.path.join(wae_dir, traj_name, '*', 'ui', '*.png'))

    if DEBUG: screenshot_files = screenshot_files[:70]
    for sample_idx, screenshot_file in enumerate(screenshot_files):
        #if DEBUG and sample_idx % 5000 > 3: continue

        screenshot_file = os.path.join(wae_dir, screenshot_file)
        if not os.path.exists(screenshot_file): continue

        vh_file = screenshot_file.replace(".png",".xml")
        if not os.path.exists(vh_file): continue
        
        if screenshot_file not in unique_elems: unique_elems[screenshot_file] = []

        processed_img_cnt += 1

        with open(vh_file, 'r') as f:
            xml = f.read()
            dom_tree = ET.fromstring(xml)
            
        nodes = find_all_elem_texts_boxes(dom_tree)

        # Detect illegal language types (except English and Chinese)
        contain_invalid_characters = False
        for node_idx, node in enumerate(nodes):
            text = node['text']
            node['idx'] = node_idx
            if text is None: continue
            if not is_valid_string(text):
                contain_invalid_characters = True
                break
        
        num_nodes = len([x for x in nodes if x['is_leaf']])
        if contain_invalid_characters:
            iterated_elem_cnt += num_nodes
            continue

        img = None # cv2.imread(screenshot_file)

        #if img is None: continue
        
        # detect landscape mode
        # if dom_tree.attrib['rotation'] == '1':

        #H, W = img.shape[:2]
        hw_info = re.search('(\d+) x (\d+)', magic.from_file(screenshot_file))
        if hw_info is None:
            iterated_elem_cnt += num_nodes
            continue
        W, H = list(map(int, hw_info.groups(1)))
        
        all_valid_nodes = []; broken_img = False
        for node in nodes:
            if not node['is_leaf'] or node.get("box", None) is None: continue
            
            iterated_elem_cnt += 1
            
            # Skip invalid samples
            sample_identifier = f"{screenshot_file}|{node['idx']}|{node['box']}"
            is_invalid=False
            for v in invalid_elem.values():
                if sample_identifier in v:
                    is_invalid = True; break
            if is_invalid: continue

            #if 'desc' not in node: continue
            x1,y1, x2,y2 = node["box"]
            if node["box"] not in unique_elems[screenshot_file]:
                unique_elems[screenshot_file].append(node["box"])
            
            if not SKIP_CHECKING:
                is_duplicate_box = False
                for x in all_valid_nodes:
                    x3, y3, x4, y4 = x["box"]
                    if (x1 == x3 and y1 == y3 and x2 == x4 and y2 == y4):
                        is_duplicate_box = True
                        break
                if is_duplicate_box:
                    invalid_elem[DUPLICATE_ELEMEMNT].add(sample_identifier)
                    continue

                # detect invalid bounding boxes
                if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x1 >= x2 or y1 >= y2:
                    invalid_elem[INVALID_ELEM_BOX].add(sample_identifier)
                    continue
            
                node_w, node_h = x2 - x1, y2 - y1

                #  detect oversize nodes
                if node_w * node_h / (H*W) >= 0.65:
                    invalid_elem[OVERSIZED_ELEMENT].add(sample_identifier)
                    continue

                #  detect extreme aspect ratio
                if node_w / node_h < 0.05 or node_w / node_h > 20:
                    invalid_elem[EXTREME_ASPECT_RATIO].add(sample_identifier)
                    continue
                
                # detect too small nodes
                if node_h / H <= 0.005 or node_w / W <= 0.005:
                    invalid_elem[TOO_SMALL_ELEMENT].add(sample_identifier)
                    continue
            
                # remove meaningless node
                if node['tag'] in ['LinearLayout', 'FrameLayout', 'GridView', 'View', 'ViewGroup', 'TextView'] and len(node.get('text', '')) == 0 and len(node.get('content-desc', '')) == 0:
                    invalid_elem[INVALID_ELEM_CONTENT].add(sample_identifier)
                    continue

                # detect unsuccessfully displayed nodes
                if img is None: img = cv2.imread(screenshot_file)
                if img is None:
                    broken_img = True; break
                if is_pure_color(img, [x1,y1,x2,y2]):
                    invalid_elem[BLANK_ELEM].add(sample_identifier)
                    continue
            
            # box = [round(x1/W*SCALE), round(y1/H*SCALE), round(x2/W*SCALE), round(y2/H*SCALE)]
            # box = [min(max(0, i), SCALE-1) for i in box]
            # box_str = f'[{box[0]},{box[1]},{box[2]},{box[3]}]'
            
            # box_w, box_h = x2 - x1, y2 - y1
            # center = [(x1+x2)/2, (y1+y2)/2]
            # center_str = f"({min(max(0, round((x1+x2)/2/W*SCALE)), SCALE-1)},{min(max(0, round((y1+y2)/2/H*SCALE)), SCALE-1)})"
            
            all_valid_nodes.append(node)

        if broken_img: continue

        # 通过比对OCR结果和元素文本属性，以去除box覆盖在其他元素之上的无效元素
        if len(all_valid_nodes) > 1:
            all_valid_nodes_after_checking_overlap = []
            for node_idx, node in enumerate(all_valid_nodes):
                x1,y1, x2,y2 = node["box"]
                
                # Skip invalid samples
                sample_identifier = f"{screenshot_file}|{node['idx']}|{node['box']}"
                is_invalid=False
                for v in invalid_elem.values():
                    if sample_identifier in v:
                        is_invalid = True; break
                if is_invalid: continue

                if not SKIP_CHECKING:
                    # check those icons and texts overlapping with others
                    is_overlap = is_box_overlapping_np(target_box=[x1,y1,x2,y2], other_boxes=[x['box'] for cur_idx, x in enumerate(all_valid_nodes) if cur_idx != node_idx], threshold=0.01)
                    
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
                                        invalid_elem[GHOST_ELEMENT].add(sample_identifier)
                                        continue
                                        # node['bad_ocr'] = True
                                        #continue
                all_valid_nodes_after_checking_overlap.append(node)
            all_valid_nodes = all_valid_nodes_after_checking_overlap

        if len(all_valid_nodes) == 0: continue
        all_node_texts = [node['text'] for node in all_valid_nodes if 'text' in node]
        
        # extract the app package name
        # example: '/mnt/vdb1/hongxin_li/WAE/tmp/com.twansoftware.pdfconverterpro_6000006-output/stoat_fsm_output/ui/S_1.png'
        traj_name = screenshot_file.split('/')[-4]
        package = traj_name.split('_')[0]
        # create image symlinks

        # save_traj_to = os.path.join(save_images_to, traj_name)
        # os.makedirs(save_traj_to, exist_ok=True)
        # new_sc_file = os.path.join(save_traj_to, os.path.basename(screenshot_file))
        
        # # WAE有重名图像
        # if not os.path.exists(new_sc_file):
        #     os.symlink(screenshot_file, new_sc_file)
        
        # short_img_path = new_sc_file[new_sc_file.find(DATASET_NAME):]
        
        short_img_path = f'{DATASET_NAME}/' + screenshot_file.split('WAE/tmp/')[1]
        
        if False:
            for idx, node in enumerate(all_valid_nodes):
                x1,y1, x2,y2 = node["box"]
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

        used_boxes = []
        for node in all_valid_nodes:
            # Skip invalid samples
            sample_identifier = f"{screenshot_file}|{node['idx']}|{node['box']}"
            is_invalid=False
            for v in invalid_elem.values():
                if sample_identifier in v:
                    is_invalid = True; break
            if is_invalid: continue
                
            # Skip nodes with overlapped boxes
            if not SKIP_CHECKING and node["box"] in used_boxes:
                invalid_elem[DUPLICATE_ELEMEMNT] += 1
                continue
            else:
                used_boxes.append(node["box"])
                
            elem_text = node['text']
            # 获得元素描述
            node_desc = None
            if elem_text:
                node_desc = elem_text
            if node.get('content-desc', None) is not None:
                content_desc = node['content-desc'].strip()
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
        
        all_node_descs = [node['node_desc'] for node in all_valid_nodes if node.get('node_desc', None) is not None]
        all_node_content_descs = [node['content-desc'] for node in all_valid_nodes if node.get('content-desc', None) is not None]

        used_node_descs = [], set()
        for node in all_valid_nodes:
            x1,y1, x2,y2 = node["box"]
                    
            box = [round(x1/W*SCALE), round(y1/H*SCALE), round(x2/W*SCALE), round(y2/H*SCALE)]
            box = [min(max(0, i), SCALE-1) for i in box]
            box_str = f'({box[0]},{box[1]},{box[2]},{box[3]})'
            
            box_w, box_h = x2 - x1, y2 - y1
            center = [(x1+x2)//2, (y1+y2)//2]
            normalized_center = [min(max(0, round(center[0]/W*SCALE)), SCALE-1), min(max(0, round(center[1]/H*SCALE)), SCALE-1)]
            
            center_str = f"({normalized_center[0]},{normalized_center[1]})"

            elem_type = classify_node(node)
            elem_text = node['text']
            
            if elem_text is not None and 0 < len(elem_text) <= 200:
                elem_text = elem_text.strip()
                with_box = random.random() < PROB_BOX

                # IF the element text does not populate the whole box, we use the box as the reference as the center refenrence is not accurate
                if img is None: img = cv2.imread(screenshot_file)
                if is_pure_color(img, [center[0] - box_w // 10, center[1] - box_h // 10, center[0] + box_w // 10, center[1] + box_h // 10]): with_box = True
                # To avoid ambibuity, a text localization task is created if the elem text is unique.
                #if random.random() > 0.5 and 
                if TEXTLOC and all_node_texts.count(elem_text) <= 1:
                    task_id = f'autogui_{DATASET_NAME}_textloc_{text_loc_cnt}'
                    sample = make_textloc_sample(task_id, text=elem_text, loc=box_str if with_box else center_str, output_tag='')
                    sample['task_attr'], sample['unnormalized_box'], sample['image'], sample['wxh'], sample['package'] = elem_text, [x1,y1, x2,y2], short_img_path, f"{W}x{H}", package
                    chunk_samples.append(sample)
                    text_loc_cnt += 1

                if OCR:
                    task_id = f'autogui_{DATASET_NAME}_ocr_{ocr_cnt}'

                    loc = box_str if with_box else center_str
                    sample = make_ocr_sample(task_id, text=elem_text, loc=loc, with_box=with_box)
                    sample['task_attr'], sample['unnormalized_box'], sample['image'], sample['wxh'], sample['package'] = loc, [x1,y1, x2,y2], short_img_path, f"{W}x{H}", package
                    chunk_samples.append(sample)
                    ocr_cnt += 1

            node_content_desc = node.get('content-desc', None)
            if (elem_text is None or len(elem_text)==0) and node_content_desc and 0 < len(node_content_desc) <= 200:            
                # Grounding任务要求目标独一无二，所以这里需要判断当前元素的node_content_desc是否重复出现了。若重复出现，则跳过
                if ICONGND and all_node_content_descs.count(node_content_desc) <= 1:
                    task_id = f'autogui_{DATASET_NAME}_icongnd_{icongnd_cnt}'
                    with_box = random.random() < PROB_BOX
                    sample = make_icongnd_sample(task_id, icon_desc=node_content_desc, loc=box_str if with_box else center_str, output_tag='')
                    sample['task_attr'], sample['unnormalized_box'], sample['image'], sample['wxh'], sample['package'] = node_content_desc, [x1,y1, x2,y2], short_img_path, f"{W}x{H}", package

                    chunk_samples.append(sample)
                    icongnd_cnt += 1
                
                if ICONREF:
                    task_id = f'autogui_{DATASET_NAME}_iconref_{iconref_cnt}'
                    with_box = random.random() < PROB_BOX
                    loc = box_str if with_box else center_str

                    sample = make_iconref_sample(task_id, icon_desc=node_content_desc, loc=box_str if with_box else center_str, with_box=with_box)
                    sample['task_attr'], sample['unnormalized_box'], sample['image'], sample['wxh'], sample['package'] = loc, [x1,y1, x2,y2], short_img_path, f"{W}x{H}", package

                    chunk_samples.append(sample)
                    iconref_cnt += 1
            
            node_desc = node.get('node_desc',None)
            if INTENTGND and node_desc and all_node_descs.count(node_desc) <= 1 and 0 < len(node_desc) <= 200:
                
                # tag = elem_desc.split()[0][1:]
                # 虽然node里可能给出了元素的HTML tag，但这里还是使用安卓的标签类型
                tag = node['tag']
                with_box = random.random() < PROB_BOX
                task_id = f'autogui_{DATASET_NAME}_intentgnd_{intentgnd_cnt}'

                if elem_text not in [None, 'None', 'none'] and all_node_texts.count(elem_text) <= 1:
                    intent = gen_naive_action_gnd_anno(node_desc.strip(' ,.'), tag, normalized_center, scale=SCALE)
                    
                    sample = make_intentgnd_sample(task_id, intent=intent, loc=box_str if with_box else center_str, output_tag='')
                    sample['task_attr'], sample['unnormalized_box'], sample['image'], sample['wxh'], sample['package'] = intent, [x1,y1, x2,y2], short_img_path, f"{W}x{H}", package
                    chunk_samples.append(sample)
                    intentgnd_cnt += 1

        if WIDGETLIST and len(all_valid_nodes) >= 2:
            task_id = f'autogui_{DATASET_NAME}_widgetlist_{widgetlist_cnt}'
            
            node_texts_boxes = []
            for node in all_valid_nodes:
                node_desc = '' if node['node_desc'] is None else node['node_desc']

                unnormalized_box = node['box']
                normalized_box = [round(unnormalized_box[0] / W * SCALE), round(unnormalized_box[1] / H * SCALE), round(unnormalized_box[2] / W * SCALE), round(unnormalized_box[3] / H * SCALE)]

                node_texts_boxes.append((node['tag'], ' '.join(x for x in node_desc.strip(' ,.').split('\n') if x.strip()), unnormalized_box, normalized_box))


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
            sample['task_attr'], sample['image'], sample['wxh'], sample['package'], sample['unnormalized_box'] = None, short_img_path, f"{W}x{H}", package, [x[2] for x in node_texts_boxes]
            chunk_samples.append(sample)
            widgetlist_cnt += 1
        
        if DEBUG:
            for i in range(last_sample_cnt, len(chunk_samples)):
                chunk_samples[i]['original_img_file'] = screenshot_file

        if len(chunk_samples) > last_sample_cnt:
            valid_img_cnt += 1
        last_sample_cnt = len(chunk_samples)
        # if valid_img_cnt >= 1000: break
    
    if traj_idx > 0 and (traj_idx % CHUNK_SIZE == 0 or traj_idx == len(all_traj_names)-1):
        chunk_idx = traj_idx // CHUNK_SIZE
        chunk_file = os.path.join(chunk_dir, f'chunk_{chunk_idx}_s{SCALE}.json')
        with open(chunk_file, 'w') as f:
            json.dump(chunk_samples, f)

        with open(os.path.join(chunk_dir, f'chunk_{chunk_idx}_invalid_elems.json'), 'w') as f:
            json.dump({k:list(v) for k,v in invalid_elem.items()}, f, indent=2)

        completely_proc_traj_names.extend(chunk_traj_names)
        
        with open(ckpt_file, 'w') as f:
            json.dump({'completely_proc_traj_names': completely_proc_traj_names}, f)
        
        num_invalid_elem = sum(len(v) for v in invalid_elem.values())

        report = f"Valid image ratio: {valid_img_cnt} / {processed_img_cnt} = {(valid_img_cnt/processed_img_cnt if processed_img_cnt > 0 else 0.0):.2f}\nInvalid elem ratio: {num_invalid_elem} / {iterated_elem_cnt} = {(num_invalid_elem/iterated_elem_cnt if iterated_elem_cnt > 0 else 0.0):.2f}\ntext_loc_cnt: {text_loc_cnt} | ocr_cnt: {ocr_cnt} | icongnd_cnt: {icongnd_cnt} | iconref_cnt: {iconref_cnt} | elemclass_cnt: {elemclass_cnt} | intentgnd_cnt: {intentgnd_cnt} | widgetlist_cnt: {widgetlist_cnt}"
        print(report)

        with open(chunk_file.replace('.json', f'_info.json'), "w") as f:
            json.dump({'num_samples': len(chunk_samples), 'valid_img_cnt': valid_img_cnt, 'processed_img_cnt': processed_img_cnt, 'num_invalid_elements': num_invalid_elem, 'iterated_elem_cnt': iterated_elem_cnt, 'text_loc_cnt': text_loc_cnt, 'ocr_cnt': ocr_cnt, 'icongnd_cnt': icongnd_cnt, 'iconref_cnt': iconref_cnt, 'elemclass_cnt': elemclass_cnt, 'intentgnd_cnt': intentgnd_cnt, 'widgetlist_cnt': widgetlist_cnt, 'report': report, 'invalid_elem_types': {k:len(v) for k,v in invalid_elem.items()}}, f, indent=2)

        processed_img_cnt = valid_img_cnt = iterated_elem_cnt = annotated_node_cnt = 0
        text_loc_cnt = ocr_cnt = icongnd_cnt = iconref_cnt = elemclass_cnt = intentgnd_cnt = widgetlist_cnt = 0
        last_sample_cnt = 0
        
        next_invalid_elem_record_file = os.path.join(chunk_dir, f'chunk_{chunk_idx+1}_invalid_elems.json')

        if os.path.exists(next_invalid_elem_record_file):
            with open(next_invalid_elem_record_file, 'r') as f:
                invalid_elem = json.load(f)
            for k, v in invalid_elem.items():
                invalid_elem[k] = set(v)
        else:
            invalid_elem = deepcopy(invalid_elem_meta)

        print(f"Saving chunk {chunk_idx} {len(chunk_samples)} samples to {chunk_file}")
        chunk_samples.clear(); chunk_traj_names.clear()

# aggregate all chunks
all_samples = []
processed_img_cnt = valid_img_cnt = iterated_elem_cnt = annotated_node_cnt = 0
text_loc_cnt = ocr_cnt = icongnd_cnt = iconref_cnt = elemclass_cnt = intentgnd_cnt = widgetlist_cnt = 0

all_chunks_num_invalid_elem = {}
num_unique_elems = sum(len(v) for v in unique_elems.values())

chunk_info_files = glob.glob(os.path.join(chunk_dir, f'*_s{SCALE}_info.json'))

for chunk_info_file in chunk_info_files:
    chunk_file = chunk_info_file.replace("_info","")
    with open(chunk_file, 'r') as f:
        samples = json.load(f)
        all_samples.extend(samples)

    with open(chunk_info_file, 'r') as f:
        info = json.load(f)
        processed_img_cnt += info['processed_img_cnt']
        valid_img_cnt += info['valid_img_cnt']
        iterated_elem_cnt += info['iterated_elem_cnt']
        text_loc_cnt += info['text_loc_cnt']
        ocr_cnt += info['ocr_cnt']
        icongnd_cnt += info['icongnd_cnt']
        iconref_cnt += info['iconref_cnt']
        elemclass_cnt += info['elemclass_cnt']
        intentgnd_cnt += info['intentgnd_cnt']
        widgetlist_cnt += info['widgetlist_cnt']
        for k, v in info['invalid_elem_types'].items():
            if k not in all_chunks_num_invalid_elem: all_chunks_num_invalid_elem[k] = 0
            all_chunks_num_invalid_elem[k] += v

all_invalid_elem = sum(cnt for cnt in all_chunks_num_invalid_elem.values())
num_valid_imgs = len([k for k,v in unique_elems.items() if len(v)])

report = {
        'num_samples': len(all_samples),
        '#valid_unique_images': num_valid_imgs,
        '#all_unique_images': len(unique_elems),
        'Valid image ratio': f'{valid_img_cnt} / {processed_img_cnt} = {valid_img_cnt/processed_img_cnt:.2f}',
        'processed_img_cnt': processed_img_cnt,
        '#num_unique_elems': num_unique_elems,
        'iterated_elem_cnt': iterated_elem_cnt,
        'num_invalid_elements': all_invalid_elem,
        'Invalid elem ratio': f'{all_invalid_elem} / {iterated_elem_cnt} = {all_invalid_elem/iterated_elem_cnt:.2f}',
        'text_loc_cnt': text_loc_cnt,
        'ocr_cnt': ocr_cnt,
        'icongnd_cnt': icongnd_cnt, 'iconref_cnt': iconref_cnt, 'elemclass_cnt': elemclass_cnt, 'intentgnd_cnt': intentgnd_cnt, 'widgetlist_cnt': widgetlist_cnt, 'invalid_elem_types': all_chunks_num_invalid_elem}


print(report)

os.makedirs(SAVE_ROOT, exist_ok=True)
file_name = os.path.join(SAVE_ROOT, f"{DATASET_NAME}_scale{SCALE}_{len(all_samples)// 1000}k{'_debug' if DEBUG else ''}.json")
print(f"save {len(all_samples)} samples to {file_name}")
with open(file_name.replace('.json', '_info.json'), "w") as f:
    json.dump(report, f, indent=2)

with open(file_name.replace(".json","_sample.json"), 'w') as f:
    json.dump(random.sample(all_samples,min(len(all_samples), 128)), f, indent=2)
    
with open(file_name, 'w') as f:
    json.dump(all_samples, f)
