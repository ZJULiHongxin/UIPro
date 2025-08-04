import re, json, magic, random, cv2
from tqdm import tqdm
from utils.data_utils.task_prompt_lib import FUNCREF_PROMPTS

file = "/data/hongxin_li/scaling_exp/UIPro_processed/AutoGUI_FuncRef_s1000_490k.json"

data = json.load(open(file))


new_samples = []

for sample in tqdm(data, total=len(data)):
    user, gpt = sample['messages'][0]['content'], sample['messages'][1]['content']

    if 'login' in gpt: continue

    target = user[user.find('('):user.rfind(')')+1]
    x, y = eval(target)
    img_path = sample['images'][0].replace("/mnt/nvme0n1p1/hongxin_li/WebpageFunctionality", "/mnt/jfs/copilot/lhx/ui_data/WebpageFunctionality").replace('/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp', '/mnt/jfs/copilot/lhx/ui_data')

    img_info = magic.from_file(img_path)

    if 'precision' in img_info:
        W, H = list(map(int, re.search('precision 8, (\d+)x(\d+)', img_info).groups(1)))
    else:
        W, H = list(map(int, re.search('(\d+) x (\d+)', img_info).groups(1)))

    unnorm_x, unnorm_y = round(x * W / 1000), round(y * H / 1000)
    
    prompt = random.choice(FUNCREF_PROMPTS).format(coordinate=f"[{unnorm_x}, {unnorm_y}]")
    new_samples.append({
        "messages": [
            {
                "role": "user",
                "content": f"<image>\n{prompt}"
            },
            {
                "role": "assistant",
                "content": gpt
            }
        ],
        "images": [img_path]
    })
    
    if False:
        img = cv2.imread(img_path)
        cv2.circle(img, (unnorm_x, unnorm_y), 5, (0, 0, 255), -1)
        cv2.imwrite("test.png", img)
        1+1

print(f"Sample size: {len(new_samples)}")
with open(file.replace(".json", "_qwen2p5.json"), "w") as f:
    json.dump(new_samples, f)

save_to = file.replace(".json", "_qwen2p5_sample.json")
print(f"Save to {save_to}")
with open(save_to, "w") as f:
    json.dump(random.sample(new_samples, 256), f, indent=2)
        

    