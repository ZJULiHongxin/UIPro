import json, random, os

file = "./data/multi_grained_text_localization.jsonl"

data = []
with open(file, encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

new_samples = []

{'image': ['./imgs/DUE_Benchmark/TabFact/pages/1-23316034-23_page0.png'], 'messages': [{'role': 'user', 'content': '<|image|>Detect the text in the bounding box <bbox>26,315,902,360</bbox>'}, {'role': 'assistant', 'content': '<ocr> 3 96 Adam Gilchrist 89 7 20 40 </ocr>'}], 'task_name': 'bbox2t_sft', 'dataset_name': 'TabFact'}

{'image': ['./imgs/DUE_Benchmark/DocVQA/pngs/ftcn0000_1.png'], 'messages': [{'role': 'user', 'content': '<|image|>Locate the postion of the text <ocr> Display/POS Kit includes Floor Base Display with Riser Card and detachable </ocr>'}, {'role': 'assistant', 'content': '<bbox>291,411,810,427</bbox>'}], 'task_name': 't2bbox_sft', 'dataset_name': 'DocVQA'}

num_qas = 0
for i, x in enumerate(data):
    convs = []
    for turn in x['messages']:
        if turn['role'] == 'user':
            convs.append({'from': 'human', 'value': turn['content'].replace('<ocr> ', '').replace(' </ocr>', '').replace('<|image|>', '').replace('<bbox>', '[').replace('</bbox>', ']').strip()})
        elif turn['role'] == 'assistant':
            convs.append({'from': 'gpt', 'value': turn['content'].replace('<ocr> ', '').replace(' </ocr>', '').replace('<|image|>', '').replace('<bbox>', '[').replace('</bbox>', ']').strip()})

    num_qas += len(convs)//2
    new_samples.append({
        'id': f'{x["task_name"]}_{i}',
        'image': 'docowl' + x['image'][0][6:],
        'conversations': convs
    })

save_to = os.path.join(os.path.dirname(file), f"docowl_text_gndref_{num_qas}QAs.json")
with open(save_to.replace('.json', '_sample.json'),"w") as f:
    json.dump(random.sample(new_samples, 256),f)

with open(save_to.replace('.json', '_355k.json'),"w") as f:
    json.dump(random.sample(new_samples, 354822),f)

with open(save_to,"w") as f:
    json.dump(new_samples,f)
