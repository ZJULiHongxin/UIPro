import json, re
import random
import spacy
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from utils.data_utils.task_prompt_lib import (
    CLICK_TEMPLATE,
    HOVER_TEMPLATE,
    LONG_PRESS_TEMPLATE,
    DOUBLECLICK_TEMPLATE,
    MOVETO_TEMPLATE,
    RIGHTCLICK_TEMPLATE,
    DRAG_TEMPLATE,
    ACTION_PREFIXES,
    KEYCOMB_TEMPLATE,
    INPUT_TEMPLATE,
    INPUT_TARGET_TEMPLATE,
    SEARCH_QUERIES,
    INPUT_ACTION_PREFIXES_WITH_TEXT,
    get_gnd2planning_prompt
)

OBJ_REF_START, OBJ_REF_END = '<|object_ref_start|>', '<|object_ref_end|>'

# Load a small English model. You could load another model if desired.
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

def parse_box_coordinates(box_str: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Parse box coordinates from string format '(x1,y1),(x2,y2)'."""
    start_str, end_str = box_str.split('),(')
    start_x, start_y = map(int, start_str.strip('(').split(','))
    end_x, end_y = map(int, end_str.strip(')').split(','))
    return (start_x, start_y), (end_x, end_y)

def extract_box_coordinates(content: str) -> List[Tuple[str, Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Extract element references and their box coordinates from the assistant's response."""
    elements = []
    lines = content.strip().split('\n')
    
    for line in lines:
        # Use regex to find element references and box coordinates
        element_ref_match = re.search(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>', line)
        box_coords_match = re.search(r'<\|box_start\|>(.*?)<\|box_end\|>', line)
        
        element_ref = None
        if element_ref_match := re.search(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>', line):
            element_ref = element_ref_match.group(1).strip()
        
        if box_coords_match := re.search(r'<\|box_start\|>(.*?)<\|box_end\|>', line):
            box_coords = box_coords_match.group(1).strip()
            coords = parse_box_coordinates(box_coords)
            
        elements.append((element_ref, coords))
    
    return elements

def convert_to_point_action(p1, p2, only_click=False) -> Dict[str, str]:
    """Convert box coordinates to a point action template."""
    (start_x, start_y), (end_x, end_y) = p1, p2
    # Calculate center point
    target_x = (start_x + end_x) // 2
    target_y = (start_y + end_y) // 2
    
    # List of possible point-based action templates
    template = CLICK_TEMPLATE if only_click else random.choice([CLICK_TEMPLATE, HOVER_TEMPLATE, LONG_PRESS_TEMPLATE, DOUBLECLICK_TEMPLATE, RIGHTCLICK_TEMPLATE])
    
    # Randomly select one action type
    return template.format(target_x=target_x, target_y=target_y)

def convert_to_drag_action(p1, p2) -> str:
    """Convert box coordinates to a drag action template."""
    (start_x, start_y), (end_x, end_y) = p1, p2
    return DRAG_TEMPLATE.format(
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y
    )

def convert_to_input_text_action(p1, p2, query: str, ref: str) -> str:
    """Convert box coordinates and query to a realistic input text action."""
    lower_query = query.lower()
    
    # Skip search icon/elements as they may not be the input field
    if ref is None:
        return False, None, None, None
    if not ('search' in lower_query and 'want to' in lower_query and 'locate the text' not in lower_query and 'icon' not in lower_query):
        return False, None, None, None
            
    (start_x, start_y), (end_x, end_y) = p1, p2
    target_x = (start_x + end_x) // 2
    target_y = (start_y + end_y) // 2

    # Generate realistic search text
    text = random.choice(SEARCH_QUERIES)
    
    # randomly determine whether to generate click + input_text or just input_text
    if random.random() < 0.5:
        action = INPUT_TARGET_TEMPLATE.format(target_x=target_x, target_y=target_y, text=text)
        multi_action = False
    else:
        action = convert_to_point_action(p1, p2, only_click=True) + '\n' + INPUT_TEMPLATE.format(text=text)
        multi_action = True
    
    vague_intent = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['vague']).format(text=text, target=ref)
    specific_intent = random.choice(INPUT_ACTION_PREFIXES_WITH_TEXT['specific']).format(text=text, target=ref)
    return multi_action, vague_intent, specific_intent, action

def process_grounding_sample(sample: Dict) -> List[Dict]:
    """Process a single grounding sample and convert it to planning tasks."""
    
    messages = sample['messages']
    new_messages = []
    for idx in range(1, len(messages), 2):
        user, gpt = messages[idx-1]['content'].replace('<image>', '').strip(), messages[idx]['content'].strip()
        
        if 'list all' in user.lower():
            new_conv = messages[idx-1:idx+1]
        else:
            p1, p2 = extract_box_coordinates(gpt)[0][1]

            if p1 == p2: # point output
                action = convert_to_point_action(p1, p2, only_click= any(k in user.lower() for k in ['click', 'press', 'tap', 'touch']))
                user = user.replace(" (Output the center coordinates of the target)", "")
            else:
                action = convert_to_drag_action(p1, p2)
                user = user.replace(" (Output the bounding box coordinates of the target)", "")

            action_type = eval(action)['action_type']

            ref = extract_obj_ref(user)

            multi_action = False; additonoal_actions = ''

            # generate the input_text action
            multi_action, vague_intent, specific_intent, input_text_action = convert_to_input_text_action(p1, p2, user, ref)
            
            if specific_intent is not None and random.random() < 0.8:
                action = input_text_action
                user = vague_intent
            else:
                if ref is None:
                    ref_exp = 'the target element'
                else:
                    ref_exp = ref

                if not user.endswith(('.', '!', '?')): # Where is the "Seil Marschall" element?. Click the target element.
                    user += '.'

                if p1 == p2:
                    vague_intent = f' {random.choice(ACTION_PREFIXES[action_type]["vague"])} the target element.'
                    specific_intent = f'{random.choice(ACTION_PREFIXES[action_type]["specific"])} {ref_exp}'
                    user += vague_intent
                else:
                    vague_intent = f' {random.choice(ACTION_PREFIXES["drag"]["vague"])} the target region.'
                    specific_intent = f'{random.choice(ACTION_PREFIXES["drag"]["specific"])} "{ref_exp}"'
                    user += vague_intent

                    if 'copy' in user.lower():
                        additonoal_actions = '\n' + KEYCOMB_TEMPLATE.format(key_combination='Ctrl+C')
                        multi_action = True

            user = get_gnd2planning_prompt(user.strip(), cot=ref is not None, multi_action=multi_action).replace('.. ','. ')

            new_conv = [
                    {
                        'role': 'user',
                        'content': f"<image>\n{user}"
                    },
                    {
                        'role': 'assistant',
                        'content': f"{OBJ_REF_START}{specific_intent}{OBJ_REF_END}\n{action}{additonoal_actions}"
                    }]

        new_messages.extend(new_conv)
    
    # remove all <image> tags
    for i in range(len(new_messages)):
        new_messages[i]['content'] = new_messages[i]['content'].replace('<image>', '').strip()
    
    new_messages[0]['content'] = '<image>\n' + new_messages[0]['content']
    # check whether the <image> tag is in the first message and not in the remaining turns
    assert '<image>' in new_messages[0]['content'] and '<image>' not in str(new_messages[1:])

    sample['messages'] = new_messages
    return sample

def main(input_file: str):
    """Convert grounding samples to planning samples."""
    # Read input JSON/JSONL file
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r') as f:
            grounding_samples = [json.loads(line) for line in f]
    elif input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            grounding_samples = json.load(f)
    else:
        raise ValueError(f'Unsupported file format: {input_file}')
    
    print(f'Processing {len(grounding_samples)} grounding samples in {input_file}')
    # Process all samples
    new_samples = []
    for sample in tqdm(grounding_samples, total=len(grounding_samples), desc='Processing grounding samples'):
        if len(sample['messages']) > 2:
            new_sample = process_grounding_sample(sample)
            new_samples.append(new_sample)
    
    # Write output JSON file
    output_file = input_file.replace('.json', '_planning.json')

    with open(output_file.replace('.json', '_sample.json'), 'w') as f:
        json.dump(random.sample(new_samples, 128), f, indent=2)

    with open(output_file, 'w') as f:
        for sample in new_samples:
            f.write(json.dumps(sample) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert grounding samples to planning samples')
    
    parser.add_argument('--input_file', help='Input JSON file containing grounding samples', default='/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_dedup_4336kQAs_v4.jsonl')
    
    args = parser.parse_args()
    main(args.input_file)
