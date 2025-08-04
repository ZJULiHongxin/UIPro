import json, os, datasets, cv2
import numpy as np
from tqdm import tqdm

HF_PATH = "visualwebbench/VisualWebBench"

SAVE_TO = "/mnt/stepeval/datasets/VL_datasets/VisualWebBench"

ANNO_MAP = {
    "action_ground": "instruction",
    "element_ground": "elem_desc",
}

for task_split in ["action_ground", "element_ground"][::-1]:
    data = datasets.load_dataset(HF_PATH, task_split, split='test')
    SAVE_IMAGE_TO = f"/mnt/stepeval/datasets/VL_datasets/VisualWebBench/{task_split}/images"
    os.makedirs(SAVE_IMAGE_TO, exist_ok=True)
    samples = []

    for sample in tqdm(data, total=len(data), desc=f"Processing VWB {task_split} data"):
        image_path = os.path.join(SAVE_IMAGE_TO, sample["id"] + '.png')
        id, instruction, image, norm_bbox = sample['id'], sample[ANNO_MAP[task_split]], sample['raw_image'].convert('RGB'), sample['options'][sample['answer']]
        
        if not os.path.exists(image_path):
            image.save(image_path)

        # if True:
        #     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        #     H, W = image.shape[:2]
        #     cv2.rectangle(image, (int(norm_bbox[0] * W), int(norm_bbox[1] * H)), (int(norm_bbox[2] * W), int(norm_bbox[3] * H)), (0, 255, 255), 2)
        #     cv2.imwrite('test.png', image)
        #     print(instruction)

        samples.append({
            "id": id,
            "instruction": instruction,
            "image_path": '/'.join(image_path.split('/')[-3:]),
            "image_size": image.size,
            "norm_bbox": norm_bbox,
        })

    with open(os.path.join(SAVE_TO, f"VisualWebBench_{task_split}.json"), "w") as f:
        json.dump(samples, f, indent=2)