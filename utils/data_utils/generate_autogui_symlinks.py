import os, json, shutil, random
from tqdm import tqdm

index = 0
server = ["L20", "118", "A6000"][index]

data_path = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data",
    "/data0/jingran/workspace/UI_training_data/Ours-Pretrain/scaling_exp",
    "/data2/hongxin_li/UI_training_data"
][index]

traj_dir = [
    "/mnt/nvme0n1p1/hongxin_li/WebpageFunctionality/test_samples",
    "/data0/jingran/workspace/hongxin_li/WebFun/test_samples",
    "/data2/hongxin_li/WebpageFunctionality/test_samples"
][index]

coco_dir = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/images/coco/train2017",
    "",
    "/data2/hongxin_li/UI_training_data/images/coco/train2017"
][index]

cauldron_dir = [
    "/data0/jingran/workspace/UI_training_data/Ours-Pretrain/images/the_cauldron",
    "",
    "/data2/hongxin_li/UI_training_data/images/the_cauldron"
][index]

llava_150k_file = os.path.join(f"{data_path}/LLaVA-Instruct-150K/llava_instruct_150k.json")

#scale_exp_file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/raw/funcpred/20241019_fourSimpleTasks/autogui_foursimpletasks654k.json"
data_file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/raw/funcpred/20241019_fourSimpleTasks/autogui_1283623QAs.json" # os.path.join(data_path, "scaling_exp", scale_exp_file)
dataset_name = os.path.basename(data_file)
img_dir = os.path.join(data_path, "scaling_exp", dataset_name.replace(".json", "_llava"))

if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.makedirs(img_dir)

with open(data_file, "r") as f:
    autogui_data = json.load(f)

collections = {}
mapping = {}

DEBUG = False
for sample in tqdm(autogui_data):
    if 'llava' in sample['id']: continue 

    img_path = sample["conversations"][0]["value"].split("</img>")[0][16:] # example: /data0/jingran/workspace/hongxin_li/WebFun/test_samples/0531/0900_1199_of_2021/traj/24-05-31-14-28-13_bca01d32-63d4-4304-9227-c853edaddcaf/step008_actionJ_domainen.help.roblox.com.png
    collections.setdefault(img_path, []).append(sample)

new_samples = []
cnt = 0
for i, (img_path, samples) in tqdm(enumerate((collections.items())), "make"):
    if DEBUG and i == 1e4: break

    new_img_path = os.path.join(img_dir, f"{i}.png")
    
    if not os.path.exists(new_img_path):
        os.symlink(img_path, new_img_path)
        cnt += 1

    mapping[img_path] = new_img_path

    for sample in samples:
        new_conversations = []
        for conv_i, conv in enumerate(sample['conversations']):
            if conv_i % 2 == 0:
                if len(new_conversations) == 0:
                    new_conversations.append({ "from": "human", "value": f"<image>\n{conv['value'].split('</img>')[1].strip().replace('<ref>', '')}" })
                else:
                    new_conversations.append({ "from": "human", "value": conv['value']})
            else:
                new_conversations.append({ "from": "gpt", "value": conv['value'] })

        new_samples.append(
            {
                'id': sample['id'],
                "image": os.path.basename(new_img_path),
                "conversations": new_conversations
            }
        )

# skip llava as it is directly used
# llava_instruct150k = json.load(open(llava_150k_file, "r"))
# random.shuffle(llava_instruct150k)
# for i, sample in tqdm(enumerate(llava_instruct150k), total=len(llava_instruct150k), desc="llava"):
#     if DEBUG and i == 1e4: break

#     new_img_path = os.path.join(img_dir, os.path.basename(sample["image"]))
#     if not os.path.exists(new_img_path):
#         os.symlink(os.path.join(coco_dir, sample["image"]), new_img_path)
#         cnt += 1

print(f"Create {cnt} symlinks...")

# new_samples.extend(llava_instruct150k)
# random.shuffle(new_samples)

# with open(os.path.join(img_dir, "mapping.json"), "w") as f:
#     json.dump(mapping, f)

save_to = os.path.join(data_path, "scaling_exp", dataset_name.replace(".json", "_llava.json"))

with open(save_to.replace(".json", "_sample.json"), "w") as f:
    json.dump(random.sample(new_samples, 80), f, indent=2)
    
with open(save_to, "w") as f:
    json.dump(new_samples, f)
    