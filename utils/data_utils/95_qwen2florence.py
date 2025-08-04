import re, json
from tqdm import tqdm

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
tokenizer = processor.tokenizer

SCALE = 1000
MAX_LEN = 1000


POINT_PATTERN = re.compile(r'\(\d+,\d+\)')


# Function to replace integers with the format <loc_xxx>
def replace_integers_with_loc(input_text):
    # Match any integer (including those inside parentheses or elsewhere)
    def replacement(match):
        num = max(0, min(SCALE-1, int(match.group())))  # Extract the integer
        return f"<loc_{num:d}>"  # Format as <loc_xxx> with zero padding
    
    # Replace integers using a regex
    return re.sub(r'\b\d+\b', replacement, input_text)

file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_dedup_8MQAs_v3_SingleTurn.jsonl"

if file.endswith('.json'):
    with open(file, "r") as f:
        data = json.load(f)
else:
    data = []
    with open(file, "r") as f:
        for line in f:
            data.append(json.loads(line))


new_samples, num_skipped = [], 0
for sample in tqdm(data, total=len(data)):
    new_query, new_answer = sample['messages'][0]['content'].replace("<image>","").strip(), sample['messages'][1]['content']
    
    if 'WAE/' in sample['images'][0]: continue

    if POINT_PATTERN.findall(new_query):
        new_query = replace_integers_with_loc(new_query)
        new_query = new_query.replace(" (with point)","").replace(" (with bbox)","")
        sample['messages'][0]['content'] = new_query
    else:
        new_query = new_query.replace("with point", "Output the center coordinates of the target").replace("with bbox", "Output the bounding box coordinates of the target")
        if 'Please list all' in new_query:
            new_lines = []
            for line in new_answer.split('\n'):
                box_str = line[line.rfind('<|box_start|>'):]
                new_box_str = replace_integers_with_loc(box_str.replace("),(", ","))
                
                new_line = line[:line.rfind('<|box_start|>')].replace("<|object_ref_start|>", "").replace("<|object_ref_end|>", "") + new_box_str
                new_lines.append(new_line)
            new_answer = '\n'.join(new_lines)
            
            
        else: # "<|box_start|>(149,455),(245,467)<|box_end|>"
            if 'Output the center coordinates' in new_query:
                new_answer = replace_integers_with_loc(new_answer[:new_answer.find(',(')]).replace("),(", ",")
            else:
                new_answer = replace_integers_with_loc(new_answer).replace("),(", ",")

        if 'Classify the type' in new_query: continue

        new_answer = new_answer.replace("<|box_start|>", "").replace("<|box_end|>", "").replace(".. ", ". ")

        assert not 'with point' in new_query
        assert '<loc' in new_answer, f"{new_query} | {new_answer}"
    
    sample['messages'][0]['content'] = new_query
    sample['messages'][1]['content'] = new_answer

    if 576 + len(tokenizer.encode(f"{new_query} {new_answer}", add_special_tokens=True)) > MAX_LEN:
        num_skipped += 1
        continue
        
    new_samples.append(sample)

save_file = file.split('.json')[0] + f'_florence_{len(new_samples)//1000}k.jsonl'

print(f"Save {len(new_samples)} and skip {num_skipped} samples to {save_file}")
with open(save_file, 'w') as f:
    for sample in new_samples:
        f.write(json.dumps(sample) + "\n")

with open(save_file.replace(".jsonl", "_sample.jsonl"), 'w') as f:
    for sample in new_samples[:128]:
        f.write(json.dumps(sample) + "\n")