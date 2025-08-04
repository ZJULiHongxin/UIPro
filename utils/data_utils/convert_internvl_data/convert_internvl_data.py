import re, json, spacy
from tqdm import tqdm
from typing import Dict, List
from utils.data_utils.task_prompt_lib import WITHBOX_TAG_LONG, WITHPOINT_TAG_LONG

nlp = spacy.load("en_core_web_md")
def extract_obj_ref(query: str) -> str:
    """Extract the object reference from the query."""
    # IntentGnd
    if 'I want to' in query:
        # Capture only the substring after "I want to" and before "Please"
        sub_str = query[query.find('I want to') + len('I want to') : query.find('Please')].strip(' .')
        
        first_word = sub_str.split(' ')[0].lower()
        if first_word in ['click', 'view', 'open', 'select', 'go', 'get', 'check', 'learn', 'read', 'fill', 'jump', 'launch', 'choose', 'turn', 'browse', 'visit', 'scroll', 'explore', 'navigate', 'investigate']:
            # Parse sub_str with spaCy
            doc = nlp(sub_str)
            
            # Initialize ref as the original sub_str (fallback)
            ref = sub_str
            
            # Find the first noun or proper noun and return the substring from that token onward
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN", "DET"] or token.text.lower() in ["the", "a", "an", 'what', 'which', 'where', 'when', 'why', 'how', 'who', 'whom', 'whose', 'which', 'that', 'this', 'these', 'those', 'some', 'any', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']:
                    # spaCy gives the character offset in token.idx,
                    # so use that to slice the sub_str
                    ref = sub_str[token.idx - doc[0].idx:].strip()
                    break
        else:
            ref = sub_str
    # IconGnd
    elif 'icon?' in query: # "Where is the \"Home\" icon?"
        ref = query[query.find('Where is') + len('Where is') : query.find('icon?') + len('icon')]
    # ElemGnd
    elif 'element?' in query: # "Where is the \"Seil Marschall\" element?"
        ref = query[query.find('Where is') + len('Where is') : query.find('element?') + len('element')]
    # TextGnd
    elif 'Locate the text' in query: # "Locate the text \"Settings\""
        ref = query[query.find('Locate ') + len('Locate ') : ].strip(' .')
    # FuncGnd
    elif 'Locate the element according to its detailed functionality description.' in query: # "Locate the element according to its detailed functionality description. (Output the bounding box coordinates of the target)"
        ref = query[query.find('description') + len('description') + 1 : ].replace("This element", "the element that").split('. ')[0] # only keep the first sentence of the functionality description.
    else:
        ref = None
        if '?' in query:
            ref = query[query.find('?') + 1 : ].strip(' .')
        else:
            parts = query.split('. ')
            last_part = parts[-1].strip()
            if not last_part.endswith('.'):
                ref = last_part

    return ref.strip(' .') if ref is not None else None


def process_grounding_sample(sample: Dict) -> List[Dict]:
    """Process a single grounding sample and convert it to planning tasks."""
    bad_qas = 0

    messages = sample['conversations']
    new_messages = []
    for idx in range(0, len(messages), 2):
        user, gpt = messages[idx]['value'].replace('<image>', '').strip(), messages[idx+1]['value'].strip()
        
        if 'list all' not in user.lower():
            match = re.search(r'<ref>(.*?)</ref>', user)

            if match:
                ref = match.group(1)

                if WITHBOX_TAG_LONG in user:
                    messages[idx]['value'] = f"Please provide the bounding box coordinate of the region this sentence describes: <ref>{ref}</ref>"
                else:
                    messages[idx]['value'] = f"Please provide the center coordinate of the region this sentence describes: <ref>{ref}</ref>"
                
                new_messages.extend(messages[idx:idx+2])

            else:
                bad_qas += 1
    
    if len(new_messages) == 0:
        return None, bad_qas

    else:
        # remove all <image> tags
        for i in range(len(new_messages)):
            new_messages[i]['value'] = new_messages[i]['value'].replace('<image>', '').strip()
        
        new_messages[0]['value'] = '<image>\n' + new_messages[0]['value']
        # check whether the <image> tag is in the first message and not in the remaining turns
        assert '<image>' in new_messages[0]['value'] and '<image>' not in str(new_messages[1:])

        return sample, bad_qas

file = "/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_v3_woWAE_1D4SeeClickWeb_1D4TextLoc+AITWAndConM2W-IntentGnd_4288k_merged_1726178_MAXTOK3200_internvl_B.jsonl"

with open(file) as f:
    data = [json.loads(x) for x in f]

new_samples = []
total_bad_qas = 0
for x in tqdm(data, total=len(data)):
    new_x, bad_qas = process_grounding_sample(x)
    total_bad_qas += bad_qas
    if new_x is not None:
        new_samples.append(new_x)

new_file = file.replace(".json", "_OfficialPrompt.json")
print(f"Bad qas: {total_bad_qas}. Save to {new_file}")

with open(new_file, "w") as f:
    for x in new_samples:
        f.write(json.dumps(x)+'\n')
    