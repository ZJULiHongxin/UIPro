import json, random, os
from tqdm import tqdm
import transformers, json
from copy import deepcopy

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

FORMAT = ['llava', 'qwen', 'llama_fac'][-1]
CONV_TAG = {
    'llava': 'conversations',
    'qwen': 'conversations',
    'llama_fac': 'messages'
}[FORMAT]

CONTENT_TAG = {
    'llava': 'value',
    'qwen': 'value',
    'llama_fac': 'content'
}[FORMAT]

path = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_dedup_8MQAs_v3_SingleTurn_woWAE_1FourthSeeClickWeb_1FourthTextLoc_florence_4236k.jsonl"

if path.endswith('.jsonl'):
    with open(path, 'r') as f:
        a = [json.loads(line) for line in f]
else:
    a = json.load(open(path))

img_collections = {}
other_attr_collection = {}
for x in a:
    if FORMAT == 'llava':
        img_file = x['image']
    elif FORMAT == 'llama_fac':
        img_file = x['images'][0]
    elif FORMAT == 'qwen':
        img_file, query = x[CONV_TAG][0][CONTENT_TAG].split('</img>')
        img_file = img_file[16:]
        x[CONV_TAG][0][CONTENT_TAG] = query
        assert os.path.exists(img_file)

    if len(x[CONV_TAG]) == 0:
        continue

    img_collections.setdefault(img_file, []).extend(x[CONV_TAG])
    other_attr_collection.setdefault(img_file, []).extend([{k:v for k,v in x.items() if k not in ['image', CONV_TAG]} for _ in range(len(x[CONV_TAG]) // 2)])

num_qas_before_merging = sum(len(x[CONV_TAG]) // 2 for x in a)

merged_samples = []
NUM = 99999
MAX_TOKEN = 3200 # GLOBAL是577tokens，local是每个144tokens，最大token数=2048 - 577 - 144*6*0.95=650


if MAX_TOKEN == -1:
    for img, convs in tqdm(img_collections.items(), total=len(img_collections)):
        other_attrs = other_attr_collection[img]

        conv_splits = [i for i in range(0, len(convs) // 2, NUM)]
        conv_splits.append(len(convs) // 2)
        
        for j in range(len(conv_splits)-1):
            conv_split = convs[2*conv_splits[j]:2*conv_splits[j+1]]
            if len(conv_split) == 0: continue
            for turn in conv_split:
                turn[CONTENT_TAG] = turn[CONTENT_TAG].replace('<image>','').strip()
            
            conv_split[0][CONTENT_TAG] = f'<image>\n{conv_split[0][CONTENT_TAG]}'
            sample = {
                'id': str(len(merged_samples)),
                'image': img,
                CONV_TAG: conv_split
            }
            merged_samples.append(sample)
else:
    for img_idx, (img, convs) in tqdm(enumerate(img_collections.items()), total=len(img_collections)):
        other_attrs = other_attr_collection[img]

        for i, turn in enumerate(convs):
            convs[i][CONTENT_TAG] = convs[i][CONTENT_TAG].replace('<image>','').strip()

        # new version: 超过650前截断
        num_added = 0
        idx, token_cnt, conv_split, sample_attr_split = 0, 0, [], []
        while idx < len(convs):
            this_conv = convs[idx:idx+2]; this_conv_attr = other_attrs[idx//2:idx//2+1]
            this_conv_token_cnt = len(tokenizer.apply_chat_template([{'role':'user','content':this_conv[0][CONTENT_TAG]}, {'role':'assistant','content':this_conv[1][CONTENT_TAG]}]))
            
            # 如果加入下一对话导致超过MAX_TOK，或当前已经是最后一个对话
            if token_cnt + this_conv_token_cnt >= MAX_TOKEN or idx + 2 == len(convs):
                # 如果当前没有对话，则直接加入
                if len(conv_split) == 0:
                    conv_split = this_conv
                    sample_attr_split = this_conv_attr
                    idx += 2

                if FORMAT in ['llava', 'llama_fac']:
                    conv_split[0][CONTENT_TAG] = f'<image>\n{conv_split[0][CONTENT_TAG]}'
                elif FORMAT == 'qwen':
                    conv_split[0][CONTENT_TAG] = f'Picture 1: <img>{img}</img>{conv_split[0][CONTENT_TAG]}'

                sample = {
                    'id': str(len(merged_samples)),
                    CONV_TAG: deepcopy(conv_split),
                }
                
                if FORMAT == 'llama_fac':
                    sample['images'] = [img]
                else:
                    sample['image'] = img
                    sample['sample_attrs'] = deepcopy(sample_attr_split)
                merged_samples.append(sample)
                num_added += 1
            
                token_cnt, conv_split, sample_attr_split = 0, [], []
                assert idx == sum(len(x[CONV_TAG]) for x in merged_samples[-num_added:])

            else:
                idx += 2
                conv_split.extend(this_conv)
                sample_attr_split.extend(this_conv_attr)
                token_cnt += this_conv_token_cnt

            # old version: 超过了650 才截断     
        #assert len(convs) == sum(len(x[CONV_TAG]) for x in merged_samples[-num_added:])
        # old version: 超过了650 才截断     
        # idx, token_cnt, conv_split = 0, 0, []
        # while idx < len(convs):
        #     this_conv = convs[idx:idx+2]
            
        #     conv_split.extend(this_conv)
        #     token_cnt += len(tokenizer.apply_chat_template([{'role':'user','content':this_conv[0][CONTENT_TAG]}, {'role':'assistant','content':this_conv[1][CONTENT_TAG]}]))
            
        #     idx += 2
        #     if token_cnt >= MAX_TOKEN or idx == len(convs):
        #         conv_split[0][CONTENT_TAG] = f'<image>\n{conv_split[0][CONTENT_TAG]}'
        #         sample = {
        #             'id': str(len(merged_samples)),
        #             'image': img,
        #             CONV_TAG: deepcopy(conv_split)
        #         }
        #         merged_samples.append(sample)
            
        #         token_cnt, conv_split = 0, []

num_qas_after_merging = sum(len(x[CONV_TAG]) // 2 for x in merged_samples)

assert num_qas_before_merging == num_qas_after_merging

for x in tqdm(merged_samples, total=len(merged_samples), desc='checking...'):
    if FORMAT == 'llava' and '<image>' not in x[CONV_TAG][0][CONTENT_TAG]:
        raise False

random.shuffle(merged_samples)
path = path.split('.json')[0]
with open(path + f"_merged_{len(merged_samples)}_MAXTOK{MAX_TOKEN}_sample.json",'w') as f:
    json.dump(merged_samples[:256],f, indent=2)

with open(path + f"_merged_{len(merged_samples)}_MAXTOK{MAX_TOKEN}.jsonl",'w') as f:
    for x in merged_samples:
        f.write(json.dumps(x) + '\n')