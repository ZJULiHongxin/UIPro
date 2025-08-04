# 用于从原始VG 5.4M 标注生成训练样本
import os, json, random
import cv2
import numpy as np
from tqdm import tqdm
from utils.data_utils.misc import add_text

SCALE=100

gnd_template = "Please provide the bounding box coordinate of the region this sentence describes"
ref_template = "Please provide a short description for this region"

vg = json.load(open("/mnt/nvme0n1p1/hongxin_li/highres_autogui/data/region_descriptions.json"))



vg_path_map = {}
for path in os.listdir('./data/vg/VG_100K'):
    vg_path_map[path] = 'VG_100K'
for path in os.listdir('./data/vg/VG_100K_2'):
    vg_path_map[path] = 'VG_100K_2'

# load vg image sizes
vg_img_sizes_cache = os.path.join(os.path.dirname(__file__), "vg_img_sizes.json")
if os.path.exists(vg_img_sizes_cache):
    print("loading vg image sizes from a local cache")
    vg_img_sizes = json.load(open(vg_img_sizes_cache))
else:
    print("Parsing vg image sizes")
    vg_img_sizes = {}
    for x in tqdm(vg, total=len(vg)):
        img_name = f"{x['regions'][0]['image_id']}.jpg"
        img = cv2.imread(f"./data/vg/{vg_path_map[img_name]}/{img_name}")
        vg_img_sizes[img_name] = {'width': img.shape[1], 'height': img.shape[0]}
    with open(vg_img_sizes_cache, "w") as f:
        json.dump(vg_img_sizes,f)

samples = []
for x in tqdm(vg, total=len(vg)):
    img_name = f"{x['regions'][0]['image_id']}.jpg"
    W, H = vg_img_sizes[img_name]['width'], vg_img_sizes[img_name]['height']

    for region in x['regions']:
        # img = cv2.imread(f"./data/vg/{vg_path_map[img_name]}/{img_name}")
        # cv2.rectangle(img, (region['x'], region['y']), (region['x']+region['width'], region['y']+region['height']), color=(0,255,0), thickness=2)
        # img = add_text(img, text=region['phrase'])
        # cv2.imwrite('t.png', img)
        #1+1
        x1, y1, x2 , y2 = region['x'] / W, region['y'] / H, (region['x']+region['width']) / W, (region['y']+region['height']) / H
        
        if SCALE == 1.0:
            box_str = f"[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]"
        else:
            box_str = f"[{int(x1*SCALE):d}, {int(y1*SCALE):d}, {int(x2*SCALE):d}, {int(y2*SCALE):d}]"

        if random.random() >= 0.5:
            conv = [
                {
                    "from": "human",
                    "value": f"<image>\n{gnd_template}: {region['phrase'].strip(' .')}."
                },
                {
                    "from": "gpt",
                    "value": box_str
                }]
        else:
            conv = [{
                    "from": "human",
                    "value": f"{ref_template}: {box_str}."
                },
                {
                    "from": "gpt",
                    "value": f"{region['phrase'].strip(' .')}."
                }
                ]

        samples.append(
            {
                'id': str(len(samples)),
                'image': f"vg/{vg_path_map[img_name]}/{img_name}",
                'conversations': conv
            }
        )

# load sharegpt4v coco
with open(f"./data/entire_vg_{len(samples)}.json", "w") as f:
    json.dump(samples, f)

