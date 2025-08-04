import os, json
from tqdm import tqdm

file = '/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_dedup_8MQAs_v3_SingleTurn_woWAE_1FourthSeeClickWeb_1FourthTextLoc_florence_4236k_merged_1674120_MAXTOK3200.jsonl'

if file.endswith('.jsonl'):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
elif file.endswith('.json'):
    with open(file, 'r') as f:
        data = json.load(f)

bad_cnt = 0

new_samples = []

for x in tqdm(data, total=len(data)):
    new_conv = []
    for turn_i in range(0, len(x['messages']), 2):
        if '<loc_' in x['messages'][turn_i]['content']:
            bad_cnt += 1
            continue

        temp = x['messages'][turn_i+1]['content'].replace("<loc_", "").replace(">,", ",").replace(">)", ")").strip(' <>')

        if temp.startswith('(') and not temp.endswith(')'):
            temp = temp + ')'

        x['messages'][turn_i]['from'] = 'human'; x['messages'][turn_i].pop('role')
        x['messages'][turn_i]['value'] = x['messages'][turn_i].pop('content').replace(".. ", ". ").replace(" . ", ". ")

        x['messages'][turn_i+1]['from'] = 'gpt'; x['messages'][turn_i+1].pop('role')
        x['messages'][turn_i+1]['value'] = temp; x['messages'][turn_i+1].pop('content')
        new_conv.append(x['messages'][turn_i])
        new_conv.append(x['messages'][turn_i+1])

    x['conversations'] = new_conv
    x.pop('messages')
    x['image'] = x.pop('images')[0]

    if len(x['conversations']) > 0:
        new_samples.append(x)

    assert os.path.exists(x['image'])

file = file.split('.json')[0]

print(f"#original: {len(data)} | #bad: {bad_cnt} | #good: {len(data) - bad_cnt}")
with open(file.replace("_florence", "") + '_sample.json', 'w') as f:
    json.dump(new_samples[:256], f, indent=4)

with open(file.replace("_florence", "") + '.jsonl', 'w') as f:
    for x in new_samples:
        f.write(json.dumps(x) + '\n')



