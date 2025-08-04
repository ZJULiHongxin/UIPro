import time
import ast
from typing import List
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from peft import PeftModel
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

import transformers
if transformers.__version__ >= '4.48.0':
    from transformers import Qwen2_5_VLForConditionalGeneration

if transformers.__version__ >= '4.45.0':
    from transformers import Qwen2VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import scroll2swipe

_GND_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."

_NAV_SYSTEM = """You are an assistant trained to navigate the {_APP} screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
{_ACTION_SPACE}
"""

_NAV_FORMAT = """
Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

action_map = {
'web': """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required. 
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
""",

'phone': """
1. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
2. `SWIPE`: Swipe the screen, value is not applicable and the position [[x1,y1], [x2,y2]] is the start and end position of the swipe operation.
3. `TAP`: Tap on an element, value is not applicable and the position [x,y] is required.
4. `ANSWER`: Answer the question, value is the status (e.g., 'task complete') and the position is not applicable.
5. `ENTER`: Enter operation, value and position are not applicable.
""" + \
"""6. `LONG_PRESS`: Long press on an element, value is not applicable and the position [x,y] is required.
7. `NAVIGATE_BACK`: Navigate back, value and position are not applicable.
8. `NAVIGATE_HOME`: Navigate to home, value and position are not applicable.
9. `OPEN_APP`: Open an app, value is the app name and the position is not applicable.
10. `WAIT`: Wait for a while, value and position are not applicable.
"""
}

def to_showui_action(action: dict, W: int, H: int, scenario: str, scale: int = 1000):
    # UIPro action format:
    # {'action': 'CLICK', 'value': None, 'position': [0.49, 0.42]},
    # ShowUI action format:
    # {'action': 'CLICK', 'value': None, 'position': [0.49, 0.42]},
    # {'action': 'INPUT', 'value': 'weather for New York city', 'position': [0.49, 0.42]},
    # {'action': 'ENTER', 'value': None, 'position': None}
    if isinstance(action, str): action = ast.literal_eval(action)
    act_type = action['action_type']
    
    if 'target' in action:
        x, y = action['target']
        x, y = x / 1000 * scale, y / 1000 * scale
        
    if act_type == 'click':
        if scenario == 'web':
            return f"{{'action': 'CLICK', 'value': None, 'position': [{x:.2f}, {y:.2f}]}}"
        else:
            return f"{{'action': 'TAP', 'value': None, 'position': [{x:.2f}, {y:.2f}]}}"
    elif act_type == 'hover':
        return f"{{'action': 'HOVER', 'value': None, 'position': [{x:.2f}, {y:.2f}]}}"
    elif act_type == 'long_press':
        return f"{{'action': 'LONG_PRESS', 'value': None, 'position': [{x:.2f}, {y:.2f}]}}"
    elif act_type == 'swipe':
        original_dir = action['direction']
        if scenario == 'web':
            direction = scroll2swipe(original_dir)
            return f"{{'action': 'SCROLL', 'value': '{direction}', 'position': None}}"
        else:
            distance = action['distance']
            direction, start, end = format_swiping_dual_points(original_dir, scale=1, scroll2swipe=False, distance=distance)
            return f"{{'action': 'SWIPE', 'value': None, 'position': [[{start[0]:.2f}, {start[1]:.2f}], [{end[0]:.2f}, {end[1]:.2f}]]}}"
    elif act_type == 'scroll':
        direction = action['direction']
        return f"{{'action': 'SCROLL', 'value': '{direction}', 'position': None}}"
    elif act_type == 'drag':
        start_x, start_y = action['start']
        end_x, end_y = action['end']
        start_x, start_y = start_x / 1000 * scale, start_y / 1000 * scale
        end_x, end_y = end_x / 1000 * scale, end_y / 1000 * scale
        return f"{{'action': 'SELECT_TEXT', 'value': None, 'position': [[{start_x:.2f}, {start_y:.2f}], [{end_x:.2f}, {end_y:.2f}]]}}"
    elif act_type == 'hotkey':
        key = action.get('key', action.get('key_comb'))
        if key.replace('-','').replace('_','').replace(' ','').replace('+','').lower() == 'ctrlc':
            return f"{{'action': 'COPY', 'value': None, 'position': None}}"
        elif key.replace('-','').replace('_','').replace(' ','').replace('+','').lower() == 'ctrlv':
            return f"{{'action': 'PASTE', 'value': None, 'position': None}}"
        else:
            raise ValueError(f"Unsupported hotkey: {key}")
    elif act_type == 'input_text':
        return f"{{'action': 'INPUT', 'value': '{action['text']}'}}"
    elif act_type == 'enter':
        return "{'action': 'ENTER', 'value': None, 'position': None}"
    elif act_type == 'answer':
        return f"{{'action': 'ANSWER', 'value': '{action['text']}'}}"
    elif act_type == 'navigate_back':
        return "{'action': 'NAVIGATE_BACK', 'value': None, 'position': None}"
    elif act_type == 'navigate_home':
        return "{'action': 'NAVIGATE_HOME', 'value': None, 'position': None}"
    elif act_type == 'open_app':
        return f"{{'action': 'OPEN_APP', 'value': '{action['app_name']}'}}"
    elif act_type == 'wait':
        return "{'action': 'WAIT', 'value': None, 'position': None}"
    elif act_type == 'status':
        return f"{{'action': 'ANSWER', 'value': '{action['answer']}'}}"
    else:
        raise ValueError(f"Unsupported action type: {act_type}")

def showui_to_original_action(action: dict, scale: int = 1, scenario: str = 'phone'):
    # The reverse function of to_showui_action
    if isinstance(action, str): action = ast.literal_eval(action)

    action_type = action['action']
    target = action['position']
    if action_type in ['CLICK', 'TAP']:
        return CLICK_TEMPLATE.format(target_x=target[0], target_y=target[1])
    elif action_type == 'HOVER':
        return HOVER_TEMPLATE.format(target_x=target[0], target_y=target[1])
    elif action_type == 'LONG_PRESS':
        return LONG_PRESS_TEMPLATE.format(target_x=target[0], target_y=target[1])
    elif action_type == 'SWIPE':
        start, end = target
        norm_from_x, norm_from_y = start
        norm_to_x, norm_to_y = end
        vertical_shift, horizontal_shift = norm_to_y - norm_from_y, norm_to_x - norm_from_x

        # judged the scrolling direction
        if abs(vertical_shift) > abs(horizontal_shift):
            direction = 'down' if vertical_shift > 0 else 'up'
            distance = discretize_dist(abs(vertical_shift)/scale)
        else:
            direction = 'right' if horizontal_shift > 0 else 'left'
            distance = discretize_dist(abs(horizontal_shift)/scale)

        return SWIPE_TEMPLATE.format(start_x=norm_from_x, start_y=norm_from_y, direction=direction, distance=distance)
    elif action_type == 'SCROLL':
        if scenario.lower() == 'mobile':
            direction = scroll2swipe(action['value'])
            return SWIPE_TEMPLATE.format(start_x=None, start_y=None, direction=direction, distance=None)
        else:
            return SCROLL_TEMPLATE.format(direction=action['value'], distance=None)
    elif action_type == 'SELECT_TEXT':
        start, end = target
        norm_from_x, norm_from_y = start
        norm_to_x, norm_to_y = end
        return DRAG_TEMPLATE.format(start_x=norm_from_x, start_y=norm_from_y, end_x=norm_to_x, end_y=norm_to_y)
    
    elif action_type == 'COPY':
        return KEYCOMB_TEMPLATE.format(keycomb='ctrl-c')
    elif action_type == 'PASTE':
        return KEYCOMB_TEMPLATE.format(keycomb='ctrl-v')
    elif action_type == 'INPUT':
        if target is not None:
            return CLICK_TEMPLATE.format(target_x=target[0], target_y=target[1]) + '<&>' + INPUT_TEMPLATE.format(text=action['value'])
        else:
            return INPUT_TEMPLATE.format(text=action['value'])
    elif action_type == 'ENTER':
        return ENTER_TEMPLATE
    elif action_type == 'ANSWER':
        return ANSWER_TEMPLATE.format(text=action['value'])
    elif action_type == 'NAVIGATE_BACK':
        return NAVIGATE_BACK_TEMPLATE
    elif action_type == 'NAVIGATE_HOME':
        return NAVIGATE_HOME_TEMPLATE

class SHOWUI:
    def __init__(self, device, model_name: str = 'showlab/ShowUI-2B'):
        self.model_name = model_name
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, 
                attn_implementation="eager",
            )
     
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1344*28*28, size={"shortest_edge": 256*28*28, "longest_edge": 1344*28*28})

    def test_speed(self, prompt: str, image: str = None, max_new_tokens: int = 512, sys_prompt: str = ''):
        prompt = prompt.replace('<image>','').strip()

        _SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _SYSTEM},
                    {"type": "image", "image": image, "min_pixels": 256*28*28, "max_pixels": 1344*28*28},
                    {"type": "text", "text": prompt}
                ],
            }
        ]


        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(conversation)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device) # input_ids: B x L | attention_mask: B x L | pixel_values: (B*X) x Y | image_grid_thw: B x num_grids
        
        start = time.time()
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0, do_sample=False)
        duration = time.time() - start

        generated_ids_lst = []
        for input_ids, output_ids in zip(inputs.input_ids, output_ids):
            generated_ids = output_ids[len(input_ids):]
            if isinstance(generated_ids, torch.Tensor):
                generated_ids = generated_ids.tolist()

            # if 151647 in generated_ids: # '<|object_ref_end|>'
            #     generated_ids = generated_ids[generated_ids.index(151647)+1:]
            generated_ids_lst.append(generated_ids)

        return duration, generated_ids_lst[0] # assume bs = 1

        
    def get_model_response(self, prompt: str, image: str = None, temperature: float = 0.0, max_new_tokens: int = 512, history: List[str] = None, task_type: str = 'gnd', scenario: str = 'phone'):
        prompt = prompt.replace('<image>','').strip()

        if task_type == 'gnd':
            conversation = [
                {
                    "role": "user",
                    "content": [
                    {"type": "text", "text": _GND_SYSTEM},
                    {"type": "image", "image": image, "min_pixels": 256*28*28, "max_pixels": 1344*28*28},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        elif task_type == 'nav':
            sys_prompt = _NAV_SYSTEM.format(_APP=scenario, _ACTION_SPACE=action_map[scenario]) + _NAV_FORMAT
            
            history_acts = [{"type": "text", "text": act} for act in history]
            conversation = [
                    {
                        "role": "user",
                    "content": [
                        {"type": "text", "text": sys_prompt},
                        {"type": "text", "text": f'Task: {prompt}'}]\
                            + history_acts + \
                        # {"type": "text", "text": PAST_ACTION},
                        [{"type": "image", "image": image, "min_pixels": 256*28*28, "max_pixels": 1344*28*28}]
                    ,
                }
            ]
        elif task_type == 'nav2':
            sys_prompt = _NAV_SYSTEM.format(_APP=scenario, _ACTION_SPACE=action_map[scenario]) + _NAV_FORMAT
            
            prompt += '\nThe action history is: ' + ('\n'.join(history) if isinstance(history, list) else history)

            conversation = [
                {
                    "role": "user",
                "content": [
                    {"type": "text", "text": sys_prompt},
                    {"type": "text", "text": f'Task: {prompt}'}] + \
                    [{"type": "image", "image": image, "min_pixels": 256*28*28, "max_pixels": 1344*28*28}]
                ,
            }
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(conversation)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device) # input_ids: B x L | attention_mask: B x L | pixel_values: (B*X) x Y | image_grid_thw: B x num_grids

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False if temperature == 0.0 else True, temperature=temperature)

        generated_ids_lst = []
        for input_ids, output_ids in zip(inputs.input_ids, output_ids):
            generated_ids = output_ids[len(input_ids):]
            if isinstance(generated_ids, torch.Tensor):
                generated_ids = generated_ids.tolist()

            # if 151647 in generated_ids: # '<|object_ref_end|>'
            #     generated_ids = generated_ids[generated_ids.index(151647)+1:]
            generated_ids_lst.append(generated_ids)

        response = self.processor.batch_decode(generated_ids_lst, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        return response