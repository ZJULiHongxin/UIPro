import os, json, magic, re
from tqdm import tqdm
from collections import defaultdict
import multiprocessing

num_workers = 8  # Set number of workers for multiprocessing

file = [
    "/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/GoClick_CoreSet-v2_3814k_qwen2p5_sample.json",
    "/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_dedup_4336kQAs_v4_qwen2_14ksubset_A.jsonl",
    "/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_dedup_4336kQAs_v4_qwen2_1489ksubset_A.jsonl",
    "/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_dedup_4336kQAs_v4_A.jsonl"
][-1]

if file.endswith(".json"):
    with open(file, "r") as f:
        data = json.load(f)
elif file.endswith(".jsonl"):
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} items from {file}")

showui_data = []

# Aggregate data by image_path
collections = defaultdict(list)

for i, x in tqdm(enumerate(data), total=len(data), desc="Aggregating data by image_path"):
    collections[x["images"][0] + f"@{i}"].append(x)

def process_image_group(args):
    image_path, items = args
    image_path = image_path.rsplit("@", 1)[0]
    img_info = magic.from_file(image_path)
    if 'precision 8' in img_info:
        W, H = list(map(int, re.search('precision 8, (\d+)x(\d+)', img_info).groups(1)))
    elif 'PNG' in img_info:
        W, H = list(map(int, re.search('(\d+) x (\d+)', img_info).groups(1)))
    else:
        W = H = None
    samples = []
    
    for x in items:
        for conv_i in range(0, len(x["messages"]), 2):
            samples.append({
                "instruction": x["messages"][conv_i]["content"].replace("<image>", "").strip(),
                "response": x["messages"][conv_i + 1]["content"].replace("<image>", "").strip()
            })
    return {
        "img_url": image_path,
        "img_size": [W, H],
        "element": samples
    }, len(samples)

ROOT_DIR = "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/ShowUI"
SAVE_DIR = os.path.join(ROOT_DIR, "UIProShowUI/metadata")

os.makedirs(SAVE_DIR, exist_ok=True)

# Multiprocessing for per-image processing
groups = list(collections.items())
showui_data = []
num_qas = 0
max_qas = -1
for group in tqdm(groups, total=len(groups), desc="Processing data by image_path"):
    item, nqas = process_image_group(group)
    if nqas > max_qas:
        max_qas = nqas
        print(f"Max qas: {max_qas}")
    showui_data.append(item)
    num_qas += nqas
num_samples = len(showui_data)

save_file = os.path.join(SAVE_DIR, os.path.basename(file).replace(".jsonl", ".json"))
print(f"Saving {num_samples} samples and {num_qas} qas to {save_file}")
with open(save_file, "w") as f:
    json.dump(showui_data, f, indent=2)
