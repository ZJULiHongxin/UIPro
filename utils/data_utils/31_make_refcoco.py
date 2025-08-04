# 用于从RefCOCO, RefCOCOg, RefCOCO+ 316370标注生成训练样本
# 需要先在这里（https://github.com/lichengunc/refer/issues/14）下载三个数据集的压缩包，然后运行refer.py获得原始标注
import os, json, random, pickle
import cv2
import numpy as np
from tqdm import tqdm

SCALE = 100
def add_text(img, text, font_scale=0.6, font_color=(0,0,0)):
    height, width = img.shape[:2]
    # Define the size of the new region to add
    new_region_height = 60  # Adjust as needed

    # Create a new blank image with extra space
    new_image = np.zeros((height + new_region_height, width, 3), dtype=np.uint8)

    # Copy the original image to the top of the new image
    new_image[:height, :] = img

    # Fill the new region with a color (e.g., white)
    new_image[height:, :] = [255, 255, 255]  # White color

    # Now you can add text to the new region
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height + (new_region_height + text_size[1]) // 2
    cv2.putText(new_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
    
    return new_image

gnd_template = "Please provide the bounding box coordinate of the region this sentence describes"
ref_template = "Please provide a short description for this region"

refcoco = json.load(open("/mnt/nvme0n1p1/hongxin_li/highres_autogui/data/refcoco_g_plus.json"))

samples = []
for x in tqdm(refcoco, total=len(refcoco)):
    img_name = x['image']
    
    for anno_i, anno in enumerate(x['annotations']):
        exp, box = anno['exp'], anno['box']
        
        x1, y1, w, h = box
        x2, y2 = x1+w, y1+h
        W, H = x['width'], x['height']

        # color = (random.randint(0,200),random.randint(0,200),random.randint(0,200))
        # img = cv2.imread(os.path.join("./data", img_name))
        # cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
        # img = add_text(img, text=f"{anno_i}. {exp}", font_color=color)
        # cv2.imwrite('t.png', img)
        # 1+1

        if SCALE == 1:
            box_str = f"[{x1/W}, {y1/H}, {x2/W}, {y2/H}]"
        else:
            box_str = f"[{int(x1/W*SCALE)}, {int(y1/H*SCALE)}, {int(x2/W*SCALE)}, {int(y2/H*SCALE)}]"

        if random.random() >= 0.5:
            conv = [
                {
                    "from": "human",
                    "value": f"<image>\n{gnd_template}: {exp}."
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
                    "value": f"{exp}."
                }
                ]

        samples.append(
            {
                'id': str(len(samples)),
                'image': img_name,
                'wxh': f"{x['width']}x{x['height']}",
                'conversations': conv
            }
        )

# load sharegpt4v coco
with open(f"./data/entire_refcoco_g_plus_{len(samples)}.json", "w") as f:
    json.dump(samples, f)

