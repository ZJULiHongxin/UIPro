import transformers, json
from tqdm import tqdm
tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")

file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/L20_seeclick_PtBoxGndRef_widcap_ricosca_refexp_motif_web_869k_llava_merged_459936_MAXTOK650.json"

with open(file) as f:
    data = json.load(f)

token_cnts = []

overlength = []

for x in tqdm(data, total=len(data)):
    for i, turn in enumerate(x['conversations']):
        x['conversations'][i]['role'] = 'user' if i % 2 == 0 else 'assistant'
        x['conversations'][i]['content'] = turn['value']

    token_len = len(tokenizer.apply_chat_template(x['conversations']))
    token_cnts.append(token_len)
    
    if token_len > 2048:
        overlength.append(x)
        
print(f"{len(data)} samples")
print(f"Max #token: {max(token_cnts)} | Min: {min(token_cnts)} | Avg: {sum(token_cnts)/len(token_cnts)} | Total: {sum(token_cnts)}")

t = [len(x['conversations'])//2 for x in data]
print(f"MAX #turn: {max(t)} | MIN: {min(t)} | Avg: {sum(t)/len(t)} | Total: {sum(t)}")