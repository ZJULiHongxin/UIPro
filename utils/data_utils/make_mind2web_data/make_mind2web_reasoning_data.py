# 给AIW生成以下类型任务：1. 给定图像、任务、历史动作，预测下一动作的决策过程；2. 给定当前click意图，预测gnd坐标
import json, os, random, cv2, numpy as np
from tqdm import tqdm
import sys, shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from task_prompt_lib import *
from datasets import load_dataset, concatenate_datasets
import ast
from misc import is_pure_color, generate_negative_action_plans_for_web
from PIL import ImageDraw 
from datetime import datetime

random.seed(777)

DEBUG = False

DATASET_NAME = 'Mind2Web'

SCALE=1000
SPLIT = ['train', 'test', 'test_domain', 'test_task', 'test_website'][0]

PUSH2HUB = False
INTENTGND = False
REWARDMODEL_EVAL = True

MERGE_ACTION = False

MIND2WEB_ROOT = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/Mind2Web"

time_str = datetime.now().strftime("%Y-%m-%d")

mind2web_imgs_dir = os.path.join(MIND2WEB_ROOT, f"mind2web_rmeval_images_{time_str}")
os.makedirs(mind2web_imgs_dir, exist_ok=True)

split_name = f"mind2web_data_{SPLIT}"
if SPLIT == 'train':
    mind2web_data = json.load(open(f'{MIND2WEB_ROOT}/{split_name}.json', 'r'))
else:
    mind2web_data = json.load(open(f'{MIND2WEB_ROOT}/mind2web_data_test_task.json', 'r')) + json.load(open(f'{MIND2WEB_ROOT}/mind2web_data_test_website.json', 'r')) + json.load(open(f'{MIND2WEB_ROOT}/mind2web_data_test_domain.json', 'r'))

# Load HTML
ds = load_dataset("osunlp/Multimodal-Mind2Web")

# Load AXTree
AXTree_dir = "/mnt/vdb1/hongxin.li/Multimodal-Mind2Web/Mind2Web_AXTrees"

html_dict = {}
if SPLIT == 'train':
    split_data = ds['train']
elif SPLIT == 'test':
    split_data = concatenate_datasets([ds['test_domain'],ds['test_task'],ds['test_website']])
else:
    split_data = ds[SPLIT]
for sample in tqdm(split_data, total=len(split_data), desc="Extracting info..."):
    axtree_file = os.path.join(AXTree_dir, sample['annotation_id'], f"{sample['action_uid']}_before_clean.json")
    if not os.path.exists(axtree_file): continue

    html_dict[f"{sample['annotation_id']}-{sample['action_uid']}"] = {'axtree_file': axtree_file, 'screenshot': sample['screenshot'], 'neg_candidates': sample['neg_candidates']}
    
    if DEBUG and len(html_dict) >= 300: break

def make_mind2web_rmeval_data():
    samples, rewardmodel_eval_samples = [], []

    for ep_id, episode in tqdm(enumerate(mind2web_data), total=len(mind2web_data), desc=f'{MIND2WEB_ROOT}/{split_name}.json'):
        if DEBUG and ep_id % 20 > 1: continue
        prev_actions = []
        for action_repr in episode['action_reprs']:
            elem, act = action_repr.split('->')
            act = act.strip()
            elem = elem.replace('  ',' ').strip()
            if 'TYPE:' in act:
                split_id = act.find(':')
                act, text = act[:split_id], act[split_id+1:]
                text = text.strip(' \n\\').replace('"', '\\"').replace('\n', '\\n')
                prev_act_str = f"type \"{text}\" into the {elem}"
            elif act == 'ENTER':
                prev_act_str = f"press enter on {elem}"
            elif act == 'CLICK':
                prev_act_str = f"click on {elem}"
            elif act == 'HOVER':
                prev_act_str = f"hover over {elem}"
            elif 'SELECT:' in act:
                split_id = act.find(':')
                act, value = act[:split_id], act[split_id+1:]
                value = value.strip()
                prev_act_str = f"select {value} in the {elem}"
            else:
                raise Exception(f"unknown action: {act}")
            prev_actions.append(prev_act_str)

        for step_idx, step_info in enumerate(episode['actions']):
            #if DEBUG and step_idx % 2 == 0: continue
            identifier = f"{episode['annotation_id']}-{step_info['action_uid']}"
            if identifier not in html_dict: continue
                
            if "bbox" not in step_info:
                print("action not found")
                continue

            action_type = step_info['operation']['original_op']

            #if action_type != 'SELECT': continue
            # if len(img_filename) > 100:     # several image with long filename lead to error in linux, just jump it
            img = html_dict[identifier]['screenshot']

            W, OLD_H = img.size

            # 截取目标范围内的视窗
            H = round(W / 1280 * 720)
            point_y = step_info["bbox"]["y"] + (step_info["bbox"]["height"] / 2)

            if point_y - H//2 < 0:
                min_y = 0
            elif point_y + H//2 > OLD_H:
                min_y = OLD_H - H
            else:
                min_y = point_y - H//2

            step_info["bbox"]["y"] -= min_y
            point_x = step_info["bbox"]["x"] + step_info["bbox"]["width"] / 2
            point_y = step_info["bbox"]["y"] + step_info["bbox"]["height"] / 2
            x1, y1, x2, y2 = step_info["bbox"]["x"], step_info["bbox"]["y"], step_info["bbox"]["x"] + step_info["bbox"]["width"], step_info["bbox"]["y"] + step_info["bbox"]["height"]

            step_info['normalized_bbox'] = [x1 / W, y1 / H, x2 / W, y2 / H]
            
            if (step_info['normalized_bbox'][2]-step_info['normalized_bbox'][0]) * (step_info['normalized_bbox'][3]-step_info['normalized_bbox'][1]) >= 0.65:
                print('invalid bbox')
                continue

            x1, y1, x2, y2 = list(map(round, [x1, y1, x2, y2]))

            sc_crop = img.crop((0, min_y, W, min_y+H))
            
            img_name = f"{identifier}.png"
            sc_file = os.path.join(mind2web_imgs_dir, img_name)
            sc_crop.save(sc_file)
            
            short_img_name = sc_file[sc_file.find(DATASET_NAME)+len(DATASET_NAME)+1:]

            if step_idx < len(episode['actions']) - 1:
                next_img_name = os.path.join(mind2web_imgs_dir, f"{episode['annotation_id']}-{episode['actions'][step_idx+1]['action_uid']}.png")
                next_img_name = next_img_name[next_img_name.find(DATASET_NAME)+len(DATASET_NAME)+1:]
            else: next_img_name = short_img_name

            click_point = [point_x / W, point_y / H]
            
            # 读取AXTree
            with open(html_dict[identifier]['axtree_file'], 'r') as f:
                axtree_info = json.load(f)
            
            if len(axtree_info["obs_nodes_info"]) <= 5: continue
            nodes, axtree, reorder = axtree_info['obs_nodes_info'], axtree_info['content'], axtree_info['reorder']
            new_axtree_lines = []
            min_num_tabs = 999
            axtree_lines = axtree.split('\n')
            
            if DEBUG:
                img1 = ImageDraw.Draw(sc_crop)   
                img1.rectangle([(x1,y1), (x2,y2)], outline ="blue", width=3)
            
            boxes, texts, target_line_idx, smallest_area = [], [], -1, 1e8
            num_invalid_boxes = 0

            for line in axtree_lines[1:]:
                if any(k in line for k in ["status ''"]): continue

                node_order = line[line.find('[')+1:line.find(']')]
                node_text = line[line.find("'")+1:line.rfind("'")].strip()

                node_info = nodes[reorder[node_order]]
                x, y, width, height = node_info["union_bound"]
                y -= min_y
                if not ((0 <= (x + width//2) <= W) and (0 <= (y + height//2) <= H)):
                    continue
                
                box_x1,box_y1, box_x2,box_y2 = x,y, x+width,y+height

                area = (box_x2 - box_x1) * (box_y2 - box_y1)
                if not (0.0006 < area / (W*H) < 0.65):
                    num_invalid_boxes += 1; continue

                if DEBUG:
                    img1.rectangle([(box_x1,box_y1), (box_x2,box_y2)], outline ="red")      

                boxes.append([box_x1,box_y1, box_x2,box_y2])
                texts.append(node_text)

                num_tabs = len(line[:line.find('[')])
                if num_tabs < min_num_tabs: min_num_tabs = num_tabs

                if box_x1 <= point_x <= box_x2 and box_y1 <= point_y <= box_y2:
                    if area < smallest_area:
                        smallest_area = area
                        target_line_idx = len(new_axtree_lines)

                new_axtree_lines.append(line.replace(" required: False", "").replace(" disabled: True", ""))

            if num_invalid_boxes / len(axtree_lines) >= 0.25:
                print('too many invalid boxes')
                continue

            assert len(boxes) == len(new_axtree_lines), f"{len(boxes)} != {len(new_axtree_lines)}"

            if target_line_idx == -1:
                print('target line not found')
                continue
            
            boxes_with_valid_texts = [b for b,t in zip(boxes, texts) if t] # Avoid selecting empty elements as negative samples
            if len(boxes_with_valid_texts) < 7:
                print('not enough boxes')
                continue
            # else:
            #     new_axtree_lines[target_line_idx] = new_axtree_lines[target_line_idx] + TARGET_MARK

            new_axtree = '\n'.join([axtree_lines[0]] + [x[min_num_tabs:] for x in new_axtree_lines])
            
            if DEBUG:
                sc_crop.save("test.png")
                print(new_axtree)

            if action_type in ['HOVER', 'CLICK', 'ENTER']:
                if action_type == 'CLICK':
                    start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), click_point))

                    action_str = CLICK_TEMPLATE.format(target_x=start_x, target_y=start_y)
                elif action_type == 'HOVER':
                    start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), click_point))

                    action_str = HOVER_TEMPLATE.format(target_x=start_x, target_y=start_y)
                elif action_type == 'ENTER':
                    action_str = ENTER_TEMPLATE

                neg_actions = generate_negative_action_plans_for_web(gt_act_type=action_type.lower(), W=W, H=H, scale=SCALE, gt_box=[x1, y1, x2, y2], boxes=boxes_with_valid_texts)
            elif action_type == 'SELECT':                
                if is_pure_color(np.array(sc_crop), [x1, y1, x2, y2]):
                    print('blank selection element skipped')
                    continue
                text = step_info['operation']['value'].replace('"', '\\"')
                start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), click_point))
                action_str = SELECT_TEMPLATE.format(target_x=start_x, target_y=start_y, value=text)

                if False:
                    t = np.array(sc_crop)
                    cv2.rectangle(t, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    xx, yy = round(click_point[0]*W), round(click_point[1]*H)
                    cv2.circle(t, (xx, yy), 6, (0, 255, 0), 2)
                    cv2.imwrite("test.png", t)
                neg_actions = generate_negative_action_plans_for_web(gt_act_type='select', W=W, H=H, scale=SCALE, gt_box=[x1, y1, x2, y2], boxes=boxes_with_valid_texts, text=text)
            elif action_type == 'TYPE':
                text = step_info["operation"]["value"].strip()
                if text.count('"') % 2 != 0: text = text.strip('"')
                if text.count("'") % 2 != 0: text = text.strip("'")
                text = text.replace('"', '\\"')
                start_x, start_y = list(map(lambda x: max(0, min(SCALE-1, round(x*SCALE))), click_point))
                
                action_str = INPUT_TARGET_TEMPLATE.format(target_x=start_x, target_y=start_y, text=text)
                
                neg_actions = generate_negative_action_plans_for_web(gt_act_type='input_text', W=W, H=H, scale=SCALE, gt_box=[x1, y1, x2, y2], boxes=boxes_with_valid_texts, text=text)

            history = prev_actions[max(0, step_idx-MAX_PREV_ACT):step_idx]
            history_str = 'None' if step_idx < 1 or len(prev_actions) <= 1 else ' '.join(f'Step {i}. {step.strip(" .")}.' for i, step in enumerate(history, start=1))

            action_type = ast.literal_eval(action_str)['action_type']

            rmeval_sample = {'id': f'autogui_{DATASET_NAME}_rewardmodeleval_{len(rewardmodel_eval_samples)}', "image": short_img_name, "next_image": next_img_name, 'ep_id': episode['annotation_id'], 'step_id': step_info['action_uid'], 'next_step_id': episode['actions'][min(len(episode['actions'])-1, step_idx+1)]['action_uid'], 'task': episode['confirmed_task'], "step_instruction": prev_actions[step_idx], "action_type": action_type, "history": history, 'gt_action': action_str, 'neg_actions': neg_actions, "task_attr": step_info, "wxh": f"{W}x{H}", "axtree": new_axtree, "boxes": boxes}
            rewardmodel_eval_samples.append(rmeval_sample)


    save_to_dir = f"/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/{DATASET_NAME}_processed"
    os.makedirs(save_to_dir, exist_ok=True)
    save_to_file = os.path.join(save_to_dir, f"{DATASET_NAME}_{SPLIT}_s{SCALE}_{len(samples)}_RAW.json")

    save_to_file = os.path.join(save_to_dir, f"{DATASET_NAME}_{SPLIT}_rmeval_s{SCALE}_{len(rewardmodel_eval_samples)}.json")
    print(f"Save {len(rewardmodel_eval_samples)} samples to {save_to_file}")

    with open(save_to_file.replace(".json", "_sample.json"), "w") as f:
        json.dump(random.sample(rewardmodel_eval_samples,64), f, indent=2)
    
    backup_file = mind2web_imgs_dir+".json"
    with open(backup_file, "w") as f:
        json.dump(rewardmodel_eval_samples, f)
    
    if os.path.exists(save_to_file): os.remove(save_to_file)
    os.symlink(backup_file, save_to_file)
        
make_mind2web_rmeval_data()