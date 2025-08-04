import datasets, json, magic, random, re, os
from tqdm import tqdm
from PIL import Image

import datasets
from datasets import Dataset, Value, Sequence
from datasets import DatasetDict

import hashlib

def generate_hash(path_string):
    # Encode the path string to bytes
    path_bytes = path_string.encode('utf-8')
    
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the bytes
    sha256_hash.update(path_bytes)
    
    # Get the hexadecimal digest of the hash
    hash_code = sha256_hash.hexdigest()
    
    return hash_code

idx = 0
CONV_TAG = ['conversations', 'messages'][idx]
CONTENT_TAG = ['value', 'content'][idx]

train_data = []

def load_json_or_jsonl(file_path):
    if file_path.endswith('.json'):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

# MobileViews
file = "/mnt/nvme0n1p1/hongxin_li/WebpageFunctionality/MobileViews_FuncPred1126/MobileViews_s1000_FuncGnd79151.json"
movileviews_train_data = load_json_or_jsonl(file)

#movileviews_train_data = movileviews_train_data[:256]
for x in tqdm(movileviews_train_data, total=len(movileviews_train_data)):
    img = x['image'] = os.path.join("/mnt/nvme0n1p1/hongxin_li/WebpageFunctionality/", x['image'])

    x['image_id'] = generate_hash(img)

    x[CONV_TAG][0][CONTENT_TAG] = x[CONV_TAG][0][CONTENT_TAG].replace("<image>","").strip()

    if img.endswith('.png'):
        W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img)).groups(1)))
    else:
        W, H = list(map(int, re.search('precision 8, (\d+)x(\d+)', magic.from_file(img)).groups(1)))

    if 'test_samples' in img:
        x['source'] = 'Common Crawl'
        if W > H:
            x['device'] = 'web'
            W //= 2; H //= 2
        else:
            x['device'] = 'mobile'
            W //= 3; H //= 3
    elif 'AndroidControl' in img:
        x['source'] = 'Android Control'
        x['device'] = 'mobile'
    elif 'MobileViews' in img:
        x['source'] = 'Mobile Views'
        x['device'] = 'mobile'
    elif 'android_system' in img:
        x['source'] = 'Android Emulator'
        x['device'] = 'mobile'
    else:
        raise ValueError(f"Unknown source: {img}")

    x['image_size(wxh)'] = [W,H]


# AndroidControl
file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/android_system_1021/android_system_FuncGndRef_s1000.json"
androidcontrol_train_data = load_json_or_jsonl(file)
androidcontrol_train_data = [x for x in androidcontrol_train_data if 'Locate the element' in x['conversations'][0]['value']]
#androidcontrol_train_data = androidcontrol_train_data[:256]
for x in tqdm(androidcontrol_train_data, total=len(androidcontrol_train_data)):
    img = x['image'] = os.path.join("/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp", x['image'])

    x['image_id'] = generate_hash(img)

    x[CONV_TAG][0][CONTENT_TAG] = x[CONV_TAG][0][CONTENT_TAG].replace("<image>","").strip()

    if img.endswith('.png'):
        W, H = list(map(int, re.search('(\d+) x (\d+)', magic.from_file(img)).groups(1)))
    else:
        W, H = list(map(int, re.search('precision 8, (\d+)x(\d+)', magic.from_file(img)).groups(1)))

    x['source'] = 'Android Control'
    x['device'] = 'mobile'

    x['image_size(wxh)'] = [W,H]

train_data = movileviews_train_data + androidcontrol_train_data

random.shuffle(train_data)

train_dataset = Dataset.from_generator(
    lambda: ({
        "image": (Image.open(sample['image']).resize((1280,720) if sample['device'] == 'web' else (428,746)) if sample['source'] == 'Common Crawl' else Image.open(sample['image'])),
        'image_id': sample['image_id'],
        "instruction": sample[CONV_TAG][0][CONTENT_TAG],
        "answer": sample[CONV_TAG][1][CONTENT_TAG],
        "unnormalized_box": sample["unnormalized_box"],
        # "elem_text": sample["elem_text"],
        # "elem_role": sample['elem_role'],
        "func": sample["task_attr"],
        "image_size": sample['image_size(wxh)'],
        'device': sample['device'],
        'source': sample['source']} for sample  in tqdm(train_data)),
    features=datasets.Features({
        "image": datasets.Image(),
        "image_id": datasets.Value("string"),
        "instruction": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "unnormalized_box": Sequence(Value("float32")),
        #"elem_text": datasets.Value("string"),
        #"elem_role": datasets.Value("string"),
        "func": datasets.Value("string"),
        "image_size": datasets.Value("string"),
        'device': datasets.Value("string"),
        'source': datasets.Value("string"),
        }),
)

# funcpred_test_file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/func_pred/func_pred.json"
# with open(funcpred_test_file, "r") as f:
#     test_data = json.load(f)
    
# test_dataset = Dataset.from_generator(
#     lambda: ({"image": Image.open(os.path.join(os.path.dirname(funcpred_test_file), "imgs", sample['image'])).resize((1280,720) if sample['device'] == 'web' else (428,746)), "func": sample["func"], "point": f"({sample['point'][0]},{sample['point'][1]})".format(), "unnormalized_box": sample["unnormalized_box"], "elem_text": sample["elem_text"], "elem_role": sample['elem_tag'], "image_size": '1280x720' if sample['device'] == 'web' else '428x746', 'device': sample['device']} for sample in test_data),
#     features=datasets.Features({
#         "image": datasets.Image(),
#         "func": datasets.Value("string"),
#         "point": datasets.Value("string"),
#         "unnormalized_box": Sequence(Value("float32")),
#         "elem_text": datasets.Value("string"),
#         "elem_role": datasets.Value("string"),
#         "image_size": datasets.Value("string"),
#         'device': datasets.Value("string")
#         }),
# )

dataset_dict = DatasetDict({
    "train": train_dataset,
    # "test": test_dataset,
})

# Save the dataset locally before pushing (optional)

# Push to Hugging Face
# dataset_dict.push_to_hub("my_username/my_dataset")

# Make a HF dataset
dataset_dict.push_to_hub("AutoGUI/AutoGUI-v2", private=False, token='')
        