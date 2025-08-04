import  os, json, ast
from tqdm import tqdm

SCALE = 1000
# 将统一动作空间转换回AndroidControl原本的空间
def convert_androidcontrol():
    file = "/mnt/vdb1/hongxin_li/AndroidControl/AndroidControl-train_105834.json"
    
    data = json.load(open(file))
    
    new_samples = []
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        sample['conversations'][0]['value'] = sample['conversations'][0]['value'].replace("Given the UI", "[AndroidControl] Given the UI")
        action_type = sample['action_type']
        
        if action_type == 'swipe':
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "scroll", "direction": "{sample["task_attr"]["direction"]}"}}'
            sample['action_type'] = 'scroll'
        elif action_type == 'click':
            target  = ast.literal_eval(sample['conversations'][1]['value'][7:])["target"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "click", "x": {target[0]}, "y": {target[1]}}}'
        elif action_type == 'long_press':
            target  = ast.literal_eval(sample['conversations'][1]['value'][7:])["target"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "long_press", "x": {target[0]}, "y": {target[1]}}}'
        elif action_type == 'input_text':
            text  = ast.literal_eval(sample['conversations'][1]['value'][7:])["text"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "type", "text": "{text}"}}'
        elif action_type == 'open_app':
            text  = ast.literal_eval(sample['conversations'][1]['value'][7:])["text"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "open_app", "app_name": "{text}"}}'
        elif action_type == 'status':
            goal_status  = ast.literal_eval(sample['conversations'][1]['value'][7:])["goal_status"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "status", "goal_status": {goal_status}}}'
        else:
            1+1
        new_samples.append(sample)
    
    with open(file.replace(".json","_OriActionSpace.json"), "w") as f:
        json.dump(new_samples, f)

def convert_aitw():
    file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/AITW_processed/AITW_trainval_37642.json"
    
    data = json.load(open(file))
    
    new_samples = []
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        sample['conversations'][0]['value'] = sample['conversations'][0]['value'].replace("Given the UI", "[AITW] Given the UI")
        action_type = sample['action_type']
        step_info = sample['step_info']
        if action_type == 'swipe':
            touch, lift = list(map(lambda p: max(0, min(SCALE-1, round(p*SCALE))), step_info['touch'])), list(map(lambda p: max(0, min(SCALE-1, round(p*SCALE))), step_info['lift']))
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "dual_point_gesture", "start": ({touch[0]},{touch[1]}), "end": ({lift[0]},{lift[1]})}}'
            sample['action_type'] = 'dual_point_gesture'
        elif action_type == 'click':
            target  = ast.literal_eval(sample['conversations'][1]['value'][7:])["target"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "dual_point_gesture", "start": ({target[0]},{target[1]}), "end": ({target[0]},{target[1]})}}'
            sample['action_type'] = 'dual_point_gesture'
        elif action_type == 'long_press':
            target  = ast.literal_eval(sample['conversations'][1]['value'][7:])["target"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "long_press", "x": {target[0]}, "y": {target[1]}}}'
        elif action_type == 'input_text':
            text  = ast.literal_eval(sample['conversations'][1]['value'][7:])["text"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "type", "typed_text": "{text}"}}'
            sample['action_type'] = 'type'
        elif action_type == 'navigate_back':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "press_back"}'
            sample['action_type'] = 'press_back'
        elif action_type == 'navigate_home':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "press_home"}'
            sample['action_type'] = 'press_home'
        elif action_type == 'status':
            goal_status  = ast.literal_eval(sample['conversations'][1]['value'][7:])["goal_status"]
            if goal_status == 'successful':
                sample['conversations'][1]['value'] = 'Action: {"action_type": "status_complete"}'
                sample['action_type'] = 'status_complete'
            else:
                sample['conversations'][1]['value'] = 'Action: {"action_type": "status_impossible"}'
                sample['action_type'] = 'status_impossible'
        elif action_type == 'enter':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "press_enter"}'
            sample['action_type'] = 'press_enter'
        else:
            1+1
        new_samples.append(sample)
    
    with open(file.replace(".json","_OriActionSpace.json"), "w") as f:
        json.dump(new_samples, f)

def convert_aitz():
    file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/AITZ_processed/AITZ_train_25878.json"
    
    data = json.load(open(file))
    
    new_samples = []
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        if 'planning' not in sample['id']: continue
        sample['conversations'][0]['value'] = sample['conversations'][0]['value'].replace("Given the UI", "[AITZ] Given the UI")
        
        gpt_cot, gpt_action = sample['conversations'][1]['value'].split('Action: ')
        action_info = ast.literal_eval(gpt_action)
        action_type = action_info['action_type']
        step_info = sample['step_info']
        if action_type == 'swipe':
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace(gpt_action, f'{{"action_type": "swipe", "direction": "{action_info["direction"]}"}}') # aitz里scroll的作用和swipe一样
        elif action_type == 'click':
            target  = action_info["target"]
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace(gpt_action, f'{{"action_type": "click", "coord_y": {target[1]}, "coord_x": {target[0]}}}')
        elif action_type == 'input_text':
            text  = action_info["text"]
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace(gpt_action, f'{{"action_type": "type", "text": "{text}"}}')
            sample['action_type'] = 'type'
        elif action_type == 'navigate_back':
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace(gpt_action, '{"action_type": "press", "key": "back"}')
            sample['action_type'] = 'press'
        elif action_type == 'navigate_home':
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace(gpt_action, '{"action_type": "press", "key": "home"}')
            sample['action_type'] = 'press'
        elif action_type == 'enter':
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace(gpt_action, '{"action_type": "press", "key": "enter"}')
            sample['action_type'] = 'press'
        elif action_type == 'status':
            goal_status  = action_info["goal_status"]
            if goal_status == 'successful':
                sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace(gpt_action, '{"action_type": "stop", "task_state": "successful"}')
                sample['action_type'] = 'stop'
            else:
                sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace(gpt_action, '{"action_type": "stop", "task_state": "infeasible"}')
                sample['action_type'] = 'stop'

        new_samples.append(sample)
    
    with open(file.replace(".json","_OriActionSpace.json"), "w") as f:
        json.dump(new_samples, f)

def convert_amex():
    file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/AMEX_processed/AMEX_38709.json"
    
    data = json.load(open(file))
    
    new_samples = []
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        sample['conversations'][0]['value'] = sample['conversations'][0]['value'].replace("Given the UI", "[AMEX] Given the UI")
        gpt_cot, gpt_action = sample['conversations'][1]['value'].split('Action: ')
        action_info = ast.literal_eval(gpt_action)
        action_type = action_info['action_type']
        step_info = sample['step_info']
        if action_type == 'swipe':
            touch = [step_info['touch_coord'][0] / step_info['device_dim'][0], step_info['touch_coord'][1] / step_info['device_dim'][1]]
            lift = [step_info['lift_coord'][0] / step_info['device_dim'][0], step_info['lift_coord'][1] / step_info['device_dim'][1]]
            touch, lift = list(map(lambda p: max(0, min(SCALE-1, round(p*SCALE))), touch)), list(map(lambda p: max(0, min(SCALE-1, round(p*SCALE))), lift))
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "scroll", "touch": ({touch[0]},{touch[1]}), "lift": ({lift[0]},{lift[1]})}}'
            sample['action_type'] = 'scroll'
        elif action_type == 'click':
            target  = action_info["target"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "tap", "target": ({target[0]},{target[1]})}}'
            sample['action_type'] = 'dual_point_gesture'
        elif action_type == 'long_press':
            target  = ast.literal_eval(sample['conversations'][1]['value'][7:])["target"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "long_press", "x": {target[0]}, "y": {target[1]}}}'
        elif action_type == 'input_text':
            text  = ast.literal_eval(sample['conversations'][1]['value'][7:])["text"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "type", "text": "{text}"}}'
            sample['action_type'] = 'type'
        elif action_type == 'navigate_back':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "press_back"}'
            sample['action_type'] = 'press_back'
        elif action_type == 'navigate_home':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "press_home"}'
            sample['action_type'] = 'press_home'
        elif action_type == 'status':
            goal_status  = action_info["goal_status"]
            if goal_status == 'successful':
                sample['conversations'][1]['value'] = 'Action: {"action_type": "status_complete"}'
                sample['action_type'] = 'status_complete'
            else:
                sample['conversations'][1]['value'] = 'Action: {"action_type": "status_impossible"}'
                sample['action_type'] = 'status_impossible'
        elif action_type == 'enter':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "press_enter"}'
            sample['action_type'] = 'press_enter'
        else:
            1+1
        new_samples.append(sample)
    
    with open(file.replace(".json","_OriActionSpace.json"), "w") as f:
        json.dump(new_samples, f)


def convert_guiact():
    file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse_processed/guiact-smartphone-train_llava_64299.json"
    
    data = json.load(open(file))
    
    new_samples = []
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        sample['conversations'][0]['value'] = sample['conversations'][0]['value'].replace("Given the UI", "[GUIAct] Given the UI")
        gpt_cot, gpt_action = sample['conversations'][1]['value'].split('Action: ')
        action_info = ast.literal_eval(gpt_action)
        action_type = action_info['action_type']
        step_info = sample['step_info']
        if action_type == 'swipe':
            dual_points = step_info['dual_point']['related']
            from_point, to_point = list(map(float, dual_points['from'][7:-8].split(','))), list(map(float, dual_points['to'][7:-8].split(',')))
            touch, lift = list(map(lambda p: max(0, min(SCALE-1, round(p*SCALE))), from_point)), list(map(lambda p: max(0, min(SCALE-1, round(p*SCALE))), to_point))
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "swipe", "from": ({touch[0]},{touch[1]}), "to": ({lift[0]},{lift[1]})}}'
        elif action_type == 'click':
            target  = action_info["target"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "tap", "target": ({target[0]},{target[1]})}}'
            sample['action_type'] = 'tap'
        elif action_type == 'input_text':
            text  = ast.literal_eval(sample['conversations'][1]['value'][7:])["text"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "input", "text": "{text}"}}'
            sample['action_type'] = 'input'
        elif action_type == 'status':
            goal_status  = action_info["goal_status"]
            answer = action_info['answer']
            if len(answer) == 0:
                if goal_status == 'successful':
                    sample['conversations'][1]['value'] = 'Action: {"action_type": "answer", "text": "task complete"}'
                    sample['action_type'] = 'answer'
                else:
                    sample['conversations'][1]['value'] = 'Action: {"action_type": "answer", "text": "task impossible"}'
                    sample['action_type'] = 'answer'
            else:
                sample['conversations'][1]['value'] = f'Action: {{"action_type": "answer", "text": "{answer}"}}'
                sample['action_type'] = 'answer'
        elif action_type == 'enter':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "press_enter"}'
            sample['action_type'] = 'press_enter'
        new_samples.append(sample)
    
    with open(file.replace(".json","_OriActionSpace.json"), "w") as f:
        json.dump(new_samples, f)

def convert_guiodyssey():
    file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUIOdyssey_processed/GUIOdyssey_107662.json"
    
    data = json.load(open(file))
    
    new_samples = []
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        sample['conversations'][0]['value'] = sample['conversations'][0]['value'].replace("Given the UI", "[GUIOdyssey] Given the UI")
        gpt_cot, gpt_action = sample['conversations'][1]['value'].split('Action: ')
        action_info = ast.literal_eval(gpt_action)
        action_type = action_info['action_type']
        step_info = sample['step_info']
        if action_type == 'swipe':
            norm_from = [step_info["info"][0][0], step_info["info"][0][1]]
            norm_to = [step_info["info"][1][0], step_info["info"][1][1]]
            norm_from = list(map(lambda p: max(0, min(SCALE-1, round(p/1000*SCALE))), norm_from))
            norm_to = list(map(lambda p: max(0, min(SCALE-1, round(p/1000*SCALE))), norm_to))
            
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "scroll", "pos1": ({norm_from[0]},{norm_from[1]}), "pos2": ({norm_to[0]},{norm_to[1]})}}'
            sample['action_type'] = 'scroll'
        elif action_type == 'drag':
            norm_from, norm_to = action_info['start'], action_info['end']
            
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "scroll", "pos1": ({norm_from[0]},{norm_from[1]}), "pos2": ({norm_to[0]},{norm_to[1]})}}'
            sample['action_type'] = 'scroll'
        elif action_type == 'click':
            target  = action_info["target"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "click", "pos": ({target[0]},{target[1]})}}'
            sample['action_type'] = 'click'
        elif action_type == 'input_text':
            text  = ast.literal_eval(sample['conversations'][1]['value'][7:])["text"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "input", "text": "{text}"}}'
            sample['action_type'] = 'input'
        elif action_type == 'long_press':
            target  = action_info["target"]
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "long_press", "pos": ({target[0]},{target[1]})}}'
            sample['action_type'] = 'long_press'
        elif action_type == 'status':
            goal_status  = action_info["goal_status"]
            if goal_status == 'successful':
                sample['conversations'][1]['value'] = 'Action: {"action_type": "complete"}'
                sample['action_type'] = 'complete'
            else:
                sample['conversations'][1]['value'] = 'Action: {"action_type": "impossible"}'
                sample['action_type'] = 'impossible'
        elif action_type == 'navigate_home':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "home"}'
            sample['action_type'] = 'home'
        elif action_type == 'navigate_back':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "back"}'
            sample['action_type'] = 'back'
        elif action_type == 'navigate_recent':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "recent"}'
            sample['action_type'] = 'recent'
        elif action_type == 'enter':
            sample['conversations'][1]['value'] = 'Action: {"action_type": "press_enter"}'
            sample['action_type'] = 'press_enter'
        else:
            1+1
            
        new_samples.append(sample)
    
    with open(file.replace(".json","_OriActionSpace.json"), "w") as f:
        json.dump(new_samples, f)

def convert_mind2web():
    file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/Mind2Web_processed/Mind2Web_train_7723.json"
    
    data = json.load(open(file))
    
    new_samples = []
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        sample['conversations'][0]['value'] = sample['conversations'][0]['value'].replace("Given the UI", "[Mind2Web] Given the UI")
        gpt_cot, gpt_action = sample['conversations'][1]['value'].split('Action: ')
        action_info = ast.literal_eval(gpt_action)
        action_type = action_info['action_type']
        step_info = sample['step_info']


        if action_type == 'input_text':
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace('"input_text"','"type"')
            sample['action_type'] = 'type'

        new_samples.append(sample)
    
    with open(file.replace(".json","_OriActionSpace.json"), "w") as f:
        json.dump(new_samples, f)

def convert_guiactweb():
    file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse_processed/guiact-web-train_llava_109286.json"
    
    data = json.load(open(file))
    
    new_samples = []
    for idx, sample in tqdm(enumerate(data), total=len(data)):
        sample['conversations'][0]['value'] = sample['conversations'][0]['value'].replace("Given the UI", "[GUIAct-Web] Given the UI")
        gpt_cot, gpt_action = sample['conversations'][1]['value'].split('Action: ')
        action_info = ast.literal_eval(gpt_action)
        action_type = action_info['action_type']
        step_info = sample['step_info']

        if action_type == 'scroll':
            relative_down, relative_right = round(float(step_info["scroll"]["related"]["down"])*SCALE), round(float(step_info["scroll"]["related"]["right"])*SCALE)
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "scroll", "relative_down": {relative_down}, "relative_right": {relative_right}}}'
        elif action_type == 'input_text':
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace('"input_text"','"input"')
            sample['action_type'] = 'input'
        elif action_type == 'status':
            answer = step_info['text'].replace("\n", "\\n").replace('"', "'")
            if step_info['text'] == 'task complete':
                sample['conversations'][1]['value'] = 'Action: {"action_type": "answer", "text": "task complete"}'
            elif step_info['text'] == 'task impossible':
                sample['conversations'][1]['value'] = 'Action: {"action_type": "answer", "text": "task complete"}'
            else:
                sample['conversations'][1]['value'] = f'Action: {{"action_type": "answer", "text": "{answer}"}}'
            sample['action_type'] = 'answer'
        elif action_type == 'drag':
            act_dict = eval(sample['conversations'][1]['value'][7:])
            from_point = f"({act_dict['start'][0]},{act_dict['start'][1]})"
            to_point = f"({act_dict['end'][0]},{act_dict['end'][1]})"
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "select_text", "from": {from_point}, "to": {to_point}}}'
        elif action_type in ['click', 'hover', 'enter']:
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace('"target"', '"element"')
        elif action_type == 'select':
            sample['conversations'][1]['value'] = sample['conversations'][1]['value'].replace('"target"', '"element"').replace('"value"', '"text"')
        elif action_type == 'hotkey':
            sample['conversations'][1]['value'] = f'Action: {{"action_type": "copy", "text": "{step_info["text"]}"}}'
        else:raise Exception()

        new_samples.append(sample)
    
    with open(file.replace(".json","_OriActionSpace.json"), "w") as f:
        json.dump(new_samples, f)

convert_mind2web()
convert_guiactweb()
# convert_aitw()
# convert_aitz()
# convert_amex()
# convert_guiact()
# convert_guiodyssey()