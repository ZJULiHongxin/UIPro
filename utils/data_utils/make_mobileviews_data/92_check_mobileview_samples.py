import os, json, cv2, random, re
from misc import add_text
import sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from utils.data_utils.misc import add_text
RANDOM=True
SCALE = 100
pattern = r'\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\]'


data = json.load(open("/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/mobileviews_processed/mobileviews_149k_sample.json"))
data_dir = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp"

if RANDOM: random.shuffle(data)

for x in data:
    img = cv2.imread(os.path.join(data_dir, x['image']))
    H, W = img.shape[:2]

    if 'widgetlist' in x['id']:
        user, gpt = x['conversations'][0]['value'], x['conversations'][1]['value']
        node_texts_boxes = []
        for node_anno in x['conversations'][1]['value'].split('\n'):
            first_space, last_space = node_anno.find(' '), node_anno.rfind(' ')
            node_text = node_anno[first_space+1:last_space]
            node_box = list(map(int, node_anno[last_space+2:-1].split(',')))
            
            # scale back
            node_box = [int(node_box[0] * W / SCALE), int(node_box[1] * H / SCALE), int(node_box[2] * W / SCALE), int(node_box[3] * H / SCALE)]
            cv2.rectangle(img, (node_box[0], node_box[1]), (node_box[2], node_box[3]), color=(0,0,255), thickness=2)
            img = add_text(img, node_text)

    elif 'textloc' in x['id']:
        user, gpt = x['conversations'][0]['value'], x['conversations'][1]['value']
        if 'with point' in user:
            center = list(map(int, gpt[1:-1].split(',')))
            # scale back
            center = [int(center[0] * W / SCALE), int(center[1] * H / SCALE)]
            cv2.circle(img, center, radius=5, color=(0, 0, 255), thickness=2)
            cv2.circle(img, center, radius=2, color=(0, 0, 255), thickness=-1)
            img = add_text(img, f"{x['id']}: {x['text']}")
        else:
            box = list(map(int, gpt[1:-1].split(',')))
            box = [int(box[0] * W / SCALE), int(box[1] * H / SCALE), int(box[2] * W / SCALE), int(box[3] * H / SCALE)]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
            img = add_text(img, f"{x['id']}: {x['text']}")

    elif 'ocr' in x['id']:
        user, gpt, loc = x['conversations'][0]['value'], x['conversations'][1]['value'], x['loc']
        if 'with point' in user:
            center = list(map(int, loc[1:-1].split(',')))
            # scale back
            center = [int(center[0] * W / SCALE), int(center[1] * H / SCALE)]
            cv2.circle(img, center, radius=5, color=(0, 0, 255), thickness=2)
            cv2.circle(img, center, radius=2, color=(0, 0, 255), thickness=-1)
            img = add_text(img, f"{x['id']}: {gpt}")
        else:
            box = list(map(int, loc[1:-1].split(',')))
            box = [int(box[0] * W / SCALE), int(box[1] * H / SCALE), int(box[2] * W / SCALE), int(box[3] * H / SCALE)]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
            img = add_text(img, f"{x['id']}: {gpt}")
    elif 'intentgnd' in x['id']:
        user, gpt, intent = x['conversations'][0]['value'], x['conversations'][1]['value'], x['intent']
        if 'with point' in user:
            center = list(map(int, gpt[1:-1].split(',')))
            # scale back
            center = [int(center[0] * W / SCALE), int(center[1] * H / SCALE)]
            cv2.circle(img, center, radius=5, color=(0, 0, 255), thickness=2)
            cv2.circle(img, center, radius=2, color=(0, 0, 255), thickness=-1)
            img = add_text(img, f"{x['id']}: {intent}")
        else:
            box = list(map(int, gpt[1:-1].split(',')))
            box = [int(box[0] * W / SCALE), int(box[1] * H / SCALE), int(box[2] * W / SCALE), int(box[3] * H / SCALE)]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
            img = add_text(img, f"{x['id']}: {intent}")
    
    print(f"{x['id']} | user: {user} | gpt: {gpt}")
    cv2.imwrite("test.png", img)
    1+1