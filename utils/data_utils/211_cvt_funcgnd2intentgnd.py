import os, json, random
from tqdm import tqdm

file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/android_system_1021/Qwen_data/android_system_FuncGnd_38195_s1000_qwen.json"

with open(file, "r") as f:
    data = json.load(f)

# convert funcgnd to intentgnd
# funcgnd example: {'from': 'user', 'value': "Picture 1: <img>/mnt/nvme0n1p1/hongxin_li/WebpageFunctionality/test_samples/0531/1200_1499_of_2021/traj/24-05-31-14-33-15_d27ba951-0838-416a-aed0-0c3d7079a931/step000_actionNA_domainbusiness.linkedin.com.png</img>\nLocate the element according to its detailed functionality description (with point). This element redirects the user to a sales enablement platform, providing access to modern selling techniques and LinkedIn's sales solutions for finding and generating leads."}, {'from': 'assistant', 'value': '(203,43)'}

# intentgnd example: Open the colors and style settings and change the theme to light blue

# A list of action phrases
action_phrases = [
    "click on", "navigate to", "go to", "open", "access", "visit", "launch",
    "select", "choose", "pick", "check",
    "activate",
    "view", "find", "locate", "search for",
]

input_action_phrases = [
        "enter", "type in", "fill in", "input texts to",
]

prompt_template = "I want to {action} {target}. Please locate the target element I should interact with. (Output the center coordinates of the target)"

def reformat_func_desc(func_desc, elem_role, norm_x, norm_y):
    # randomly add position information to the func_desc
    pos_desc = ""
    if random.random() < 0.3:
        if norm_x < 100 and norm_y < 100:
            pos_desc = " at the top left corner"
        elif norm_x > 900 and norm_y < 100:
            pos_desc = " at the top right corner"
        elif norm_x < 100 and norm_y > 900:
            pos_desc = " in the bottom left corner" 
        elif norm_x > 900 and norm_y > 900:
            pos_desc = " at the bottom right corner"
        elif 400 < norm_x < 600 and 400 < norm_y < 600:
            pos_desc = " in the center of the screen"

    new_func_desc = f"the element{pos_desc} that {func_desc.split('his element ')[1].strip(' .')}"

    if elem_role == "textbox":
        action = random.choice(input_action_phrases)
    else:
        action = random.choice(action_phrases)
    query = prompt_template.format(action=action, target=new_func_desc).replace("the element that serves as ","")
    return query

MIXED = True

new_samples = []
for sample_idx, sample in tqdm(enumerate(data), total=len(data)):
    new_convs = []
    func_desc = sample['task_attr']

    if 'his element ' not in func_desc or any(len(sent) > 400 for sent in func_desc.split('. ')):
        continue
    
    if not (sample_idx % 2 == 0 and MIXED):
        # Extract image path and query if present
        image_path, query = sample['conversations'][0]['value'].split('</img>')
        image_path = image_path[16:]

        for i in range(0, len(sample['conversations']), 2):
            query = sample['conversations'][i]['value'].split('</img>')[-1]
            norm_x, norm_y = eval(sample['conversations'][i+1]['value'])
            # Convert funcgnd to intentgnd
            intentgnd = reformat_func_desc(func_desc, sample.get('elem_role',''), norm_x, norm_y)
            sample['conversations'][i]['value'] = intentgnd
        # Reconstruct the value with image if present
        sample['conversations'][0]['value'] = f"Picture 1: <img>{image_path}</img>\n{sample['conversations'][0]['value']}"

    new_samples.append(sample)

# Save the converted data
output_file = file.replace('.json', f"_IntentGndFormat{'-mixed' if MIXED else ''}.json")
with open(output_file.replace('.json','_sample.json'), 'w') as f:
    json.dump(random.sample(new_samples,256), f, indent=2)

with open(output_file, 'w') as f:
    json.dump(new_samples, f)

print(f"{len(new_samples)} samples saved to {output_file}")