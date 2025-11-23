import os, json, cv2, random, re, ast

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_utils.misc import add_text
from utils.data_utils.misc import extract_integers

RANDOM=True
SCALE = 1000
pattern = r'\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\]'


data = json.load(open("/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/mobileviews_processed/mobileviews_TextLoc_OCR_IntentGnd_WidgetList_scale1000_6k_debug.json"))
data_dir = ["/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/"][-1]

if RANDOM: random.shuffle(data)

for idx, x in enumerate(data):
    
    
    #if 'planning'  in x['id']: continue
    #if 'funcpred' in x['image']:
    # if W > H: img = cv2.resize(img, dsize=(W//2, H//2))
    # else: img = cv2.resize(img, dsize=(W//3, H//3))
    convs = x.get('conversations', x.get('messages', []))
    for i in range(0,len(convs),2):
        img_path = x['image'] if 'image' in x else x['images'][0]
        if not os.path.exists(img_path):
            img_path = os.path.join(data_dir, img_path)

        if 's3' in img_path:
            img_path = img_path.replace("s3://guidata-lhx/", "")
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        user, gpt = convs[i].get('value', convs[i].get('content')), convs[i+1].get('value', convs[i+1].get('content'))
        color = (random.randint(0,200),random.randint(0,200),random.randint(0,200))
        
        if 'task_ids' in x:
            x['id'] = x['task_ids'][i//2]

        print(f"{idx}-{i//2}-{x.get('id', '')}: User: {user} \n GPT: {gpt}")
        if 'vg' in data or 'coco' in data:
            if user.endswith('].'):
                x1,y1,x2,y2 = list(map(int, user[user.rfind('[')+1:-2].split(',')))
                exp = gpt
            else:
                exp = user[user.find(':')+1:].strip()
                x1,y1,x2,y2 = list(map(int, gpt[1:-1].split(',')))
            
            if SCALE != -1:
                pt1 = (round(x1/SCALE*W), round(y1/SCALE*H))
                pt2 = (round(x2/SCALE*W), round(y2/SCALE*H))
            else:
                pt1 = (x1, y1)
                pt2 = (x2, y2)
            
            cv2.rectangle(img, pt1, pt2, color=color, thickness=2)
            center_x, center_y = (pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2
            
        else:
            x_id = x.get('id', '')
            if any(task_type in x_id for task_type in ['funcref', 'funcpred_ref', 'elemclass', 'elemref']):
                exp = gpt
                if 'with point' in user:
                    match = re.search(r'\((\d+),\s*(\d+)\)', user)
                    
                    center_x, center_y = match.groups()
                    if SCALE != -1:
                        center_x, center_y = round(int(center_x)/SCALE*W), round(int(center_y)/SCALE*H)
                        
                    cv2.circle(img, center=(center_x,center_y) , radius=8,  color=color, thickness=2)
                    cv2.circle(img, center=(center_x,center_y) , radius=2,  color=color, thickness=-1)
                else:
                    x1,y1,x2,y2 = list(map(round, x['unnormalized_box']))

                    cv2.rectangle(img, (x1,y1), (x2,y2), color=color, thickness=2)
            elif any(task_type in x_id for task_type in ['intentgnd', 'textloc', 'funcpred_ground', 'icongnd', 'elemgnd']):
                exp = user.split('.')[0].replace("<image>","").strip()
                if 'with point' in user or 'center coord' in user or gpt.count(',') == 1:
                    center_x, center_y = gpt[1:-1].split(',')
                    if SCALE != -1:
                        center_x, center_y = round(int(center_x)/SCALE*W), round(int(center_y)/SCALE*H)
                    cv2.circle(img, center=(center_x,center_y) , radius=8,  color=color, thickness=2)
                    cv2.circle(img, center=(center_x,center_y) , radius=2,  color=color, thickness=-1)
                else:
                    x1,y1,x2,y2 = list(map(int, gpt[1:-1].split(',')))
                    if SCALE != -1:
                        pt1 = (round(x1/SCALE*W), round(y1/SCALE*H))
                        pt2 = (round(x2/SCALE*W), round(y2/SCALE*H))
                    else:
                        pt1 = (x1, y1)
                        pt2 = (x2, y2)
                    center_x, center_y = (pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2
                    cv2.rectangle(img, pt1, pt2, color=color, thickness=2)
            elif 'ocr' in x_id or 'iconref' in x_id:
                exp = gpt
                if user.find('(') == -1: continue

                loc = user[user.find('(')+1:user.find(')')]
                if 'with point' in user:
                    center_x, center_y = loc.split(',')
                    if SCALE != -1:
                        center_x, center_y = round(int(center_x)/SCALE*W), round(int(center_y)/SCALE*H)
                    cv2.circle(img, center=(center_x,center_y) , radius=8,  color=color, thickness=2)
                    cv2.circle(img, center=(center_x,center_y) , radius=2,  color=color, thickness=-1)
                
                else:
                    x1,y1,x2,y2 = list(map(int, loc.split(',')))
                    if SCALE != -1:
                        pt1 = (round(x1/SCALE*W), round(y1/SCALE*H))
                        pt2 = (round(x2/SCALE*W), round(y2/SCALE*H))
                    else:
                        pt1 = (x1, y1)
                        pt2 = (x2, y2)
                    center_x, center_y = (pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2
                    cv2.rectangle(img, pt1, pt2, color=color, thickness=2)
            elif 'widgetlist' in x_id or 'list all' in user:
                exp = user
                gpt = gpt.strip(' []\n\t')
                for elem_idx, item in enumerate(gpt.split('\n')):
                    if 'bbox_2d' in gpt:
                        x1,y1,x2,y2 = extract_integers(item)[1:5]
                    else:
                        x1,y1,x2,y2 = list(map(int, item[item.rfind('(')+1:-1].split(',')))
                    if SCALE != -1:
                        pt1 = (round(x1/SCALE*W), round(y1/SCALE*H))
                        pt2 = (round(x2/SCALE*W), round(y2/SCALE*H))
                    else:
                        pt1 = (x1, y1)
                        pt2 = (x2, y2)

                    cv2.rectangle(img, pt1, pt2, color=color, thickness=2)
                    cv2.putText(img, text=str(elem_idx), org=(round((pt1[0]+pt2[0]) // 2), round((pt1[1]+pt2[1]) // 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.6, color=color, thickness=3)
                
            elif 'funcgnd' in x_id:
                exp = user.split('(with')[-1]
                if 'with bbox' in user:
                    x1, y1,x2, y2 = list(map(int, gpt[1:-1].split(',')))
                    if SCALE != -1:
                        pt1 = (round(x1/SCALE*W), round(y1/SCALE*H))
                        pt2 = (round(x2/SCALE*W), round(y2/SCALE*H))
                    else:
                        pt1 = (x1, y1)
                        pt2 = (x2, y2)
                    cv2.rectangle(img, pt1, pt2, color=color, thickness=2)
                else:
                    center_x, center_y = gpt[1:-1].split(',')

                    if SCALE != -1:
                        center_x, center_y = round(int(center_x)/SCALE*W), round(int(center_y)/SCALE*H)
                    
                    cv2.circle(img, center=(center_x,center_y) , radius=7,  color=color, thickness=3)
                    cv2.circle(img, center=(center_x,center_y) , radius=2,  color=color, thickness=-1)
            elif 'planning' in x_id:
                exp = gpt.split('Action:')[-1].strip()
                action_info = ast.literal_eval(exp)
                act_type = action_info.get('action_type', action_info['action'])
                if act_type == 'swipe': # {"action_type": "swipe", "start": (x,y), "direction": <"up", "down", "left", "right">, "distance: <"short", "medium", "long">}
                    start = action_info.get("start", action_info["coordinate"])
                    if SCALE != -1:
                        center_x, center_y = round(start[0]/SCALE*W), round(start[1]/SCALE*H)
                    else:
                        center_x, center_y = start[0], start[1]
                    cv2.circle(img, center=(center_x, center_y) , radius=7,  color=color, thickness=3)
                    cv2.circle(img, center=(center_x, center_y) , radius=2,  color=color, thickness=-1)
                elif act_type in ['click', 'long-press', 'long_press', 'type', 'input_text']: # {"action_type": "long_press", "target": (x,y)}
                    target = action_info.get("target", action_info.get("coordinate", None))
                    if target is not None:
                        if SCALE != -1:
                            center_x, center_y = round(target[0]/SCALE*W), round(target[1]/SCALE*H)
                        else:
                            center_x, center_y = target[0], target[1]
                        cv2.circle(img, center=(center_x, center_y), radius=7, color=color, thickness=3)
                        cv2.circle(img, center=(center_x,center_y) , radius=2,  color=color, thickness=-1)
            else: # '{"bbox_2d": [960, 382, 1009, 396]}'
                coords = extract_integers(gpt)
                if len(coords) % 2 != 0:
                    coords = coords[1:]
                if coords[0] == coords[2] and coords[1] == coords[3]:
                    coords = coords[:2]
                
                if len(coords):
                    if len(coords) == 2:
                        x1, y1 = coords
                        if SCALE != -1:
                            x1, y1 = round(x1/SCALE*W), round(y1/SCALE*H)
                        cv2.circle(img, center=(x1, y1), radius=3, color=color, thickness=-1)
                        cv2.circle(img, center=(x1, y1), radius=8, color=color, thickness=3)
                    elif len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        if SCALE != -1:
                            x1, y1, x2, y2 = round(x1/SCALE*W), round(y1/SCALE*H), round(x2/SCALE*W), round(y2/SCALE*H)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
                    exp = user
                else:
                    exp = ''
        

        img = add_text(img,f"{i//2}.{exp}", font_color=color)
        cv2.imwrite("test.png", img)
        1+1
