import random, json, ast
from typing import Optional
from copy import deepcopy

# device name
MOBILE_DEVICE_NAME = 'mobile phone'
WEB_DEVICE_NAME = 'web browser'

# llava tags
CONV_TAGS = {
    'llava': {
        'conv_tag': 'conversations',
        'role_tag': 'from',
        'content_tag': 'value',
        'user_tag': 'human',
        'assistant_tag': 'gpt'
    },
    
    'llamafac': {
        'conv_tag': 'messages',
        'role_tag': 'role',
        'content_tag': 'content',
        'user_tag': 'user',
        'assistant_tag': 'assistant'
    }
}

QWEN_OBJ_REF_TAG_START = "<|object_ref_start|>"
QWEN_OBJ_REF_TAG_END = "<|object_ref_end|>"
QWEN_BOX_START, QWEN_BOX_END = '<|box_start|>', '<|box_end|>'

# Python dictionary that maps HTML tags to their user-friendly names.
HTML_TAG_TO_FRIENDLY_NAME = {
    # Text content
    'p': 'paragraph',
    'h1': 'main heading',
    'h2': 'subheading',
    'h3': 'sub-subheading',
    'h4': 'heading 4',
    'h5': 'heading 5',
    'h6': 'heading 6',
    'span': 'text span',
    'div': 'container',
    
    # Links and navigation
    'a': 'link',
    'nav': 'navigation',
    'path': 'icon',

    # Lists
    'ul': 'bullet list',
    'ol': 'numbered list',
    'li': 'list item',
    
    # Images and media
    'img': 'image',
    'video': 'video',
    'audio': 'audio',
    
    # Forms
    'form': 'form',
    'input': 'input field',
    'button': 'button',
    'textarea': 'text area',
    'select': 'dropdown',
    'option': 'dropdown option',
    'label': 'form label',
    
    # Tables
    'table': 'table',
    'tr': 'table row',
    'td': 'table cell',
    'th': 'table header',
    
    # Semantic elements
    'header': 'page header',
    'footer': 'page footer',
    'main': 'main content',
    'article': 'article',
    'section': 'section',
    'aside': 'sidebar',
    
    # Text formatting
    'strong': 'bold text',
    'em': 'italic text',
    'i': 'italic text',
    'code': 'code text',
    'pre': 'preformatted text',
    'br': 'line break',
    'hr': 'horizontal line',
    'cite': 'citation',
    # image
    'svg': 'image',
    'img': 'image',
    # Meta elements
    'head': 'document head',
    'body': 'document body',
    'title': 'page title',
    'meta': 'metadata',
    'link': 'resource link',
    'script': 'script',
    'style': 'style'
}

def cvt_elem_tag_to_friendly_name(tag):
    return HTML_TAG_TO_FRIENDLY_NAME.get(tag.lower(), tag)

# invalid elem types
INVALID_ELEM_BOX = 'invalid box coordinates'
INVALID_ELEM_CONTENT = 'invalid element content' # element content is not none but invalid
BLANK_ELEM = 'element not displayed'
EMPTY_ELEM_TEXT = 'meaningless element' # not displayed text, content-description, or other useful text-related properties
OVERLY_LENGTHY_ELEM_TEXT = 'overly lengthy element text'
INVALID_TEXT_LANGUAGE = 'invalid text language'
DUPLICATE_ELEMEMNT = 'overlapping element'
TOO_SMALL_ELEMENT = 'too small element'
OVERSIZED_ELEMENT = 'oversized element'
EXTREME_ASPECT_RATIO = 'extreme aspect ratio'
GHOST_ELEMENT = 'ghost element'
INCORRECT_TEXT_ANNO = 'incorrect text annotation'

# Incorrect Action Types
INCORRECT_CLICK_TARGET = "The {action} target is unrelated to the task"
INCORRECT_SWIPE_DIRECTION = "Incorrect swiping direction"
INCORRECT_INPUT_TEXT = "Incorrect typed texts"
INCORRECT_TOUCH_MODE = "Incorrect interaction method"
INCORRECT_OPEN_APP = "Open an app unrelated to the task"
INCORRECT_ACTION = "Unhelpful action"
INCORRECT_STATUS = "Incorrect task status"
EAYLY_STOPPING = "The task is unfinished yet"
INCORRECT_NAVIGATION_ACTION = "Incorrect navigation action"
INCORRECT_DRAG_DIRECTION = "Incorrect dragging direction"

WITHBOX_TAG = ' (with bbox)'
WITHPOINT_TAG = ' (with point)'

WITHPOINT_TAG_LONG = '(Output the center coordinates of the target)'
WITHBOX_TAG_LONG = '(Output the bounding box coordinates of the target)'

BAD_GPT_OUTPUT_KEYWORDS = ["I can't", 'I can\'t', 'sorry']

QWEN_BOX_START, QWEN_BOX_END = '<|box_start|>', '<|box_end|>'

QWEN2P5_BBOX_TAG = "Report the bbox coordinates in JSON format."
QWEN2P5_BBOX_OUTPUT_TEMPLATE = """{{"bbox_2d": [{x1}, {y1}, {x2}, {y2}]}}"""
QWEN2P5_POINT_TAG = "Report the center coordinates in JSON format."
QWEN2P5_POINT_OUTPUT_TEMPLATE = """{{"point_2d": [{target_x}, {target_y}]}}"""
QWEN2P5_LIST = """[
	{content}
]"""
QWEN2P5_WIDLIST_TAG = "The primary label is the element's description, and the secondary label is the element's role."
QWEN2P5_WIDLIST_OUTPUT_TEMPLATE = """{{"bbox_2d": [{x1}, {y1}, {x2}, {y2}], "label": "{refexp}", "label2": "{refexp}"}}"""

COORD_SCALE_PROMPTS = [
    "Return the coordinates scaled to a range between 0 and 1000.",
    "Provide coordinates with values mapped to the 0-1000 interval.",
    "Express the coordinates using a normalized scale from 0 to 1000.",
    "Convert coordinates to a 0-1000 coordinate system.",
    "Transform the coordinates into the range [0, 1000].",
    "Scale coordinate values to fit within 0 to 1000 bounds.",
    "Normalize coordinate output to span from 0 through 1000.",
    "Present coordinates adjusted to the 0-1000 value range.",
    "Map coordinate values onto a 0-1000 scale.",
    "Rescale coordinates to operate within 0-1000 limits.",
    "Output coordinates using 0-1000 as the reference range.",
    "Translate coordinates into 0-1000 normalized units.",
    "Convert coordinate values to proportional 0-1000 measurements.",
    "Express coordinates within a standardized 0-1000 framework.",
    "Project coordinates onto a 0-1000 numerical domain.",
    "Reframe coordinate output using 0-1000 as boundary values.",
    "Display coordinates calibrated to the 0-1000 spectrum.",
    "Format coordinates with normalization to 0-1000 scale.",
    "Render coordinates proportionally within 0-1000 parameters.",
    "Deliver coordinate data standardized to 0-1000 range limits."
]

POINTBOX_OUTPUT_PROMPTS = [
    "Output the {placeholder} coordinates of the target normalized in 0-1000{other}.",
    "Return the target's {placeholder} coordinates scaled to 0-1000 range{other}.",
    "Provide the {placeholder} coordinates for the target mapped to 0-1000{other}.",
    "Generate target {placeholder} coordinates normalized within 0-1000 bounds{other}.",
    "Display the target's {placeholder} coordinates adjusted to 0-1000 scale{other}.",
    "Present {placeholder} coordinates of the target in 0-1000 format{other}.",
    "Express the target {placeholder} coordinates using 0-1000 normalization{other}.",
    "Convert target's {placeholder} coordinates to 0-1000 coordinate system{other}.",
    "Transform the {placeholder} coordinates of target into 0-1000 range{other}.",
    "Scale the target's {placeholder} coordinates to fit 0-1000 interval{other}.",
    "Map target {placeholder} coordinates onto 0-1000 value spectrum{other}.",
    "Render the {placeholder} coordinates for target within 0-1000 limits{other}.",
    "Deliver target's {placeholder} coordinates standardized to 0-1000{other}.",
    "Format the target {placeholder} coordinates in 0-1000 normalized units{other}.",
    "Supply {placeholder} coordinates of the target rescaled to 0-1000{other}.",
    "Project target's {placeholder} coordinates into 0-1000 domain{other}.",
    "Output target {placeholder} coordinates calibrated for 0-1000 range{other}.",
    "Produce the {placeholder} coordinates of target in 0-1000 scale{other}.",
    "Generate {placeholder} coordinates for the target normalized to 0-1000{other}.",
    "Present the target's {placeholder} coordinates within 0-1000 framework{other}.",
    "Return {placeholder} coordinates of target adjusted to 0-1000 bounds{other}.",
    "Display target {placeholder} coordinates proportioned to 0-1000 scale{other}.",
    "Provide the {placeholder} coordinates for target in 0-1000 parameters{other}.",
    "Express target's {placeholder} coordinates standardized to 0-1000 range{other}.",
    "Convert the {placeholder} coordinates of target to 0-1000 measurements{other}."
]

POINTBOX_INPUT_PROMPTS = [
    "The target's {placeholder} coordinates have been normalized to a 0-1000 range{other}.",
    "All {placeholder} coordinates for the target have been scaled between 0 and 1000{other}.",
    "The {placeholder} coordinates of the target fall within the normalized 0-1000 scale{other}.",
    "Target {placeholder} coordinates are represented on a 0-1000 normalized scale{other}.",
    "In this system, {placeholder} coordinates are normalized to the 0-1000 range{other}.",
    "The {placeholder} position values for the target have been normalized from 0 to 1000{other}.",
    "The target's position in {placeholder} is expressed in 0-1000 normalized units{other}.",
    "The {placeholder} coordinates are mapped to a normalized 0-1000 range for the target{other}.",
    "The target's {placeholder} location is given in normalized 0-1000 coordinates{other}.",
    "All {placeholder} positional data for the target is normalized to 0-1000{other}.",
    "The {placeholder} dimensional values are normalized to a 0-1000 scale for the target{other}.",
    "For the target, {placeholder} coordinates are standardized to a 0-1000 range{other}.",
    "Normalized {placeholder} coordinates for the target range from 0 to 1000{other}.",
    "The coordinate system uses normalized {placeholder} values (0-1000) for the target{other}.",
    "The {placeholder} coordinates appear as normalized values between 0 and 1000 for the target{other}."
]

ACTION_JSON_OUTPUT_PROMPTS = [
    "Predict the action in JSON format.",
    "Return the predicted action as JSON output.",
    "Provide action predictions using JSON structure.",
    "Output the forecasted action in JSON notation.",
    "Generate action predictions formatted as JSON.",
    "Deliver the anticipated action using JSON syntax.",
    "Express the predicted action through JSON representation.",
    "Present action forecasts in JSON-structured format.",
    "Supply the expected action as a JSON response.",
    "Format the action prediction using JSON encoding.",
    "Render the projected action in JSON schema.",
    "Display predicted actions through JSON formatting.",
    "Convert action predictions to JSON data structure.",
    "Serialize the anticipated action into JSON format.",
    "Structure the action forecast as JSON output.",
    "Encode the predicted action using JSON standard.",
    "Transform action predictions into JSON representation.",
    "Package the expected action as JSON-formatted data.",
    "Produce action forecasts in JSON-compliant structure.",
    "Articulate the predicted action via JSON formatting.",
    "Compose action predictions using JSON methodology.",
    "Format anticipated actions according to JSON protocol.",
    "Generate the action forecast as structured JSON.",
    "Present predicted actions in JSON-encoded format.",
    "Output expected actions using JSON data format."
]

coord_format = {
    'center': [' like (x,y)', ' in the (x,y) format'],
    'bounding box': [' like (x1,y1,x2,y2)', ' in the (x1,y1,x2,y2) format']
}
def gen_random_requirements(coord_type: str = '', coord_scale: int=1000, use_json=False, output_coords: bool = True):
    lst = []
    
    if output_coords:
        if coord_type:
            lst.append(random.choice(POINTBOX_OUTPUT_PROMPTS).format(placeholder=coord_type, other=random.choice(coord_format[coord_type])))

        #  lst.append(random.choice(COORD_SCALE_PROMPTS))

        if use_json:
            lst.insert(0, random.choice(ACTION_JSON_OUTPUT_PROMPTS))
    else:
        lst.append(random.choice(POINTBOX_INPUT_PROMPTS).format(placeholder=coord_type, other=random.choice(coord_format[coord_type])))
    
    return ' '.join(lst)
    
def make_AndroidWorld_official_history_str(prev_actions: list[str], prev_outcomes: list[str]):
    if len(prev_outcomes) == 0:
        return 'You just started, no action has been performed yet.'
    else:
        return ' '.join([f'Step ' + str(i + 1) + '- ' + f"Action selected: {action}, Outcome: {outcome}" for i, (action, outcome) in enumerate(zip(prev_actions, prev_outcomes), start=1)])

def format_point_tag(loc: list[int | float], point_format):
    if isinstance(loc, str):
        loc = eval(loc)

    if point_format == 'qwen2':
        if len(loc) == 2:
            return f'{QWEN_BOX_START}({loc[0]},{loc[1]}),({loc[0]},{loc[1]}){QWEN_BOX_END}'
        else:
            return f'{QWEN_BOX_START}({loc[0]},{loc[1]}),({loc[2]},{loc[3]}){QWEN_BOX_END}'
    elif point_format == 'florence':
        return '(' + ','.join(map(lambda x: f'<loc_{x}>', loc)) + ')'
    else:
        return '(' + ','.join(map(lambda x: f'{x}', loc)) + ')'

def detect_bad_keywords(text):
    for keyword in BAD_GPT_OUTPUT_KEYWORDS:
        if keyword in text:
            return True
    return False

def get_output_tag(output_tag):
    if output_tag == 'point':
        tag = WITHPOINT_TAG
    elif output_tag == 'box':
        tag = WITHBOX_TAG
    else:
        tag = ''
        
    return tag

def reformat_loc_str(loc_str, format=None):
    if format == 'action_json':
        loc_str = f'{{"action_type": "click", "target": {loc_str}}}'
    else:
        loc_str = loc_str
    
    return loc_str


# task: general element grounding
elemgnd_prompt = [
    'Where is the "{text}" element?'
]

def make_elemgnd_sample(task_id, text, loc, output_tag=WITHPOINT_TAG, foramt=None):
    query = random.choice(elemgnd_prompt).format(text=text)
    loc_str = reformat_loc_str(loc, foramt)
    
    if output_tag:
        output_tag = ' ' + output_tag

    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}" + output_tag
            },
            {
                "from": "gpt",
                "value": loc_str
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# task: general element ref
elemref_prompt = [
    "Please generate a brief description for the element at {}",
]

def make_elemref_sample(task_id, text, loc, output_tag=WITHPOINT_TAG):    
    if output_tag.strip():
        output_tag = f' {output_tag.strip()}'

    conv = [
            {
                "from": "human",
                "value": "<image>\n{}".format(random.choice(elemref_prompt).format(loc)) + output_tag
            },
            {
                "from": "gpt",
                "value": text
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# task: Text localization
textloc_prompt = [
    "Locate the text \"{text}\""
]

def make_textloc_sample(task_id, text, loc, output_tag='point', foramt=None):
    query = random.choice(textloc_prompt).format(text=text)

    if output_tag:
        output_tag = ' ' + output_tag

    loc_str = reformat_loc_str(loc, foramt)

    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}" + output_tag
            },
            {
                "from": "gpt",
                "value": loc_str
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# task OCR
ocr_prompt = [
    "What is the text written on this UI element at {loc}?"
]

def make_ocr_sample(task_id, text, loc, with_box=False):
    query = random.choice(ocr_prompt).format(loc=loc + (WITHBOX_TAG if with_box else WITHPOINT_TAG))
    
    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}"
            },
            {
                "from": "gpt",
                "value": text
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# task: elem classification
elemclass_prompt = [
    "Classify the type of the element located at {loc}?"
]

def make_elemclass_sample(task_id, elem_cls, loc, with_box=False):
    query = random.choice(elemclass_prompt).format(loc=loc + (WITHBOX_TAG if with_box else WITHPOINT_TAG))
    
    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}"
            },
            {
                "from": "gpt",
                "value": elem_cls
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# task: elem tapperception
# elemclass_prompt = [
#     "Classify the type of the element located at {loc}?"
# ]

# task: intent grounding
intentgnd_prompt = [
    "I want to {}. Please locate the target element I should interact with."
]

def make_intentgnd_sample(task_id, intent, loc, output_tag=WITHPOINT_TAG, point_format='plain'):
    query = random.choice(intentgnd_prompt).format(intent)

    if output_tag:
        output_tag = ' ' + output_tag

    loc_str = format_point_tag(loc, point_format)

    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}" + output_tag
            },
            {
                "from": "gpt",
                "value": loc_str
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# spatial-aware functionality description
SPATIAL_AWARE_FUNC_DESC_PROMPT = """<image>\nPlease describe the contextual functionality of the element <image> at this normalized box coordinates: {loc}."""

SPATIAL_AWARE_FUNC_DESC_PROMPT_UNIQUE = """<image>\nPlease describe the contextual functionality of the element <image> at this normalized box coordinates: {loc}. Please add element-specific details to uniquely identify the element."""

ACTIONS = [
    'click on {article}"{text}" {tag}',
    'open {article}"{text}" {tag}',
    'press {article}"{text}" {tag}',
    'launch {article}"{text}" {tag}',
    'select {article}"{text}" {tag}',
    'tap {article}"{text}" {tag}',
    'go to {article}{tag} "{text}"']

def gen_naive_action_gnd_anno(elem_text, elem_tag, elem_point, scale):
    action = random.choice(ACTIONS).format(article='the ' if random.random() >= 0.5 else '', text=elem_text.strip(), tag=elem_tag.strip())
    
    if random.random() >= 0.5:
        # randomly add position info
        if elem_point[0] / scale >= 0.85 and elem_point[1] / scale <= 0.15:
            position_desc = " at the top right corner"
        elif elem_point[0] / scale <= 0.15 and elem_point[1] / scale <= 0.15:
            position_desc = " at the top left corner"
        elif elem_point[0] / scale <= 0.15 and elem_point[1] / scale >= 0.85:
            position_desc = " at the bottom left corner"
        elif elem_point[0] / scale >= 0.85 and elem_point[1] / scale >= 0.85:
            position_desc = " at the bottom right corner"
        elif 0.4 <= elem_point[0] / scale <= 0.6 and 0.4 <= elem_point[1] / scale <= 0.6:
            position_desc = " at the center"
        else:
            position_desc = ""

        action = f"{action}{position_desc}"
    
    return action

# task: icon grounding
icongnd_prompt = [
    'Where is "{icon_desc}" icon?'
]

def make_icongnd_sample(task_id, icon_desc, loc, output_tag=WITHPOINT_TAG):
    query = random.choice(icongnd_prompt).format(icon_desc=icon_desc)
    if output_tag:
        output_tag = ' ' + output_tag

    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}" + output_tag
            },
            {
                "from": "gpt",
                "value": loc
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# task: icon referring
iconref_prompt = [
    "Briefly describe the icon at {loc}."
]

def make_iconref_sample(task_id, icon_desc, loc, with_box=False):
    query = random.choice(iconref_prompt).format(loc=loc + (WITHBOX_TAG if with_box else WITHPOINT_TAG))
    
    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}"
            },
            {
                "from": "gpt",
                "value": icon_desc
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# task: widget listing
widgetlisting_prompt = [
    "Please list all the elements on the UI screen."
]

def make_widgetlist_sample(task_id, elem_list):
    query = random.choice(widgetlisting_prompt)
    
    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}"
            },
            {
                "from": "gpt",
                "value": elem_list
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample


# task: ui caption
uicaption_prompt = [
    "Please describe this screen in detail."
]

def make_uicaption_sample(task_id, ui_caption):
    query = random.choice(uicaption_prompt)
    
    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}"
            },
            {
                "from": "gpt",
                "value": ui_caption
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# task: reward model
TARGET_MARK = ' (LOOK! This is the interacted element!)'

def make_rewardmodel_sample(task_id: str, task: str, history: list, action: str, is_gt: bool, obs_type: str = 'image', text_obs: str = '', reason: str = ''):
    # gt action
    prompt = make_prm_eval_prompt(task, history, action_plan=action, obs=obs_type, xml_content=text_obs,  simple_prompt=True, cot=False)
    
    sample = {
        'id': task_id,
        'conversations': [
            {
                'from': 'human',
                'value': ('<image>\n' if 'image' in obs_type else '') + prompt
            },
            {
                'from': 'gpt',
                'value': 'This action plan can help to advance toward task completion. Yes' if is_gt else (f"{reason}. No" if reason.strip() else "This action plan deviates from the task. No")
            }
        ]
    }
    
    return sample

# task: next-action prediction

PLANNING_PROMPT_HEAD = """You are an agent that can operate a {device} on behalf of a user. Based on the user's tasks, you will complete actions step-by-step. At each step, you will receive the current screenshot and a history of your actions (in text). Based on these inputs and the task, you must perform one of the actions from the following list, outputting your action in the correct JSON format. Ensure the JSON is valid and well-formed."""

DIVERSE_PLANNING_PROMPT_HEADS = [
    "As an agent controlling a {device}, you'll execute tasks for the user step by step. At each step, you'll get a screenshot and action history—respond with a valid JSON action from the given list.",  
    "You act as a {device} operator for users. Using the current screenshot and action log, choose the next step from the provided actions and output it in correct JSON format.",  
    "Your role is to autonomously operate a {device} for users. Given screenshots and past actions, select and return the next action in properly formatted JSON.",  
    "Function as a {device}-controlling agent. Analyze the task, screenshot, and history, then output the next action in valid JSON from the allowed options.",  
    "You are an AI that controls a {device}. Based on the task, current view, and past actions, pick the next step from the list and format it as valid JSON.",  
    "Operate a {device} on a user's behalf. For each step, process the screenshot and history, then respond with a JSON-formatted action from the specified list.",  
    "As a {device} automation agent, follow the user's task by selecting actions (in JSON) using the latest screenshot and action history.",  
    "You automate {device} interactions. Given the visual state (screenshot) and prior steps, output the subsequent action in correct JSON format.",  
    "Act as a user's {device} assistant. Decide each step by analyzing screenshots and history, then return a JSON action from the permitted set.",  
    "Your task is to control a {device} step-by-step. Use the provided screenshot and log to choose and return an action in valid JSON.",  
    "Serve as a {device}-operating agent. At each step, validate inputs (screenshot/history) and output the next action in well-formed JSON.",  
    "You're programmed to handle a {device}. From the task, visual input, and past actions, generate the next JSON-structured command.",  
    "Emulate a {device} user. After receiving the screenshot and history, respond with one of the allowed actions in strict JSON format.",  
    "Work as a {device} proxy. Continuously take screenshots and action logs as input, producing JSON outputs for the next steps.",  
    "Perform {device} operations autonomously. Parse the task, observe the state (screenshot/history), and reply with a JSON action."  
]


planning_prompt_android = PLANNING_PROMPT_HEAD.format(device='smartphone') + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {"action_type": "status", "goal_status": "successful", "answer": "(answer to the task)"}. If a textual answer is needed, include it in the "answer" field.
- If the task is infeasible (e.g., lack of information or inability to perform necessary actions), use: {"action_type": "status", "goal_status": "infeasible"}
- For clicking/tapping an element on the screen, use the 'click' action with the element's location: {"action_type": "click", "target": (x,y)} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (100,100) at the bottom right.
- For long-pressing an element: {"action_type": "long_press", "target": (x,y)}
- To swipe the screen, use this action: {"action_type": "swipe", "start": (x,y), "direction": <"up", "down", "left", "right">, "distance: <"short", "medium", "long">}. The start defines the starting coordinates for the swipe whose x and y parameters follow the same definition as the click action. The direction determines the direction of the swipe and must be wrapped in double quotes. The distance parameter determines the swiping distance.
- For dragging an element: {"action_type": "drag", "start": (x1,y1), "end": (x2,y2)}. The start indicates where the drag begins (where the user would initially long-press the screen to focus on the element to drag). The end specifies where the dragged object is moved to before the finger is lifted (where the user releases the screen).
- To type texts into an input field/box, use the 'input_text' action: {"action_type": "input_text", "text": "(text to type)"}. The text is the string you want to insert. For example: {"action_type": "input_text", "text": "Hello, world!"} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- To press the Enter key, use the 'enter' action: {"action_type": "enter"}
- To navigate to the home screen: {"action_type": "navigate_home"}
- To navigate back: {"action_type": "navigate_back"}
- To navigate to the recent apps screen (the list of recently used applications), use the "navigate_recent" action: {"action_type": "navigate_recent"}
- To wait for the screen to update: {"action_type": "wait"}

Here are some useful guidelines you need to follow:
General:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.

Action-Related:
- Use the input_text action for any text entry (including passwords), rather than clicking individual keyboard characters.
- Ensure the target element is visible on the screen for click, long_press, and drag actions. If not, use swipe to explore the screen.
- The long_press action is commonly used when selecting text for copying or editing. After initiating the long press, you may need to further refine the selection further by dragging the selection handles if they are visible on the screen.
- Swiping vertically is often used to explore more content while swiping horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists. Swiping can also be used to adjust a slider, such as adjusting the volume on a media player interface.
- Unlike the swipe action with restricted direction options, the drag action allows the touch point to move freely to any location on the screen, which makes it suitable for tasks like positioning objects within a workspace or drawing.

App Access:
- To open an app, either navigate to the home screen or use the app drawer by swiping up from the bottom.
- Use navigate_recent to switch between recent apps, especially when tasks involve multiple applications.

The user's task is: {global_task}
Action history: {history}
{step_instruction}
"""

# OMNIPARSER Prompt
AITW_PLANNING_WITH_FUNCANNO_PROMPT = PLANNING_PROMPT_HEAD.format(device='smartphone').replace("the current screenshot", "the current screenshot on which interactable elements have been marked with boxes and numeric tags") + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful"}}
- If the task is infeasible (e.g., lack of information or inability to perform necessary actions), use: {{"action_type": "status", "goal_status": "infeasible"}}
- For clicking/tapping an element on the screen, use the 'click' action with the element's numeric tag: {{"action_type": "click", "target": <the element numeric tag>}}
- To swipe the screen, use this action: {{"action_type": "swipe", "direction": <"up", "down", "left", "right">}}. The direction determines the direction of the swipe and must be wrapped in double quotes.
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- To press the Enter key, use the 'enter' action: {{"action_type": "enter"}}
- To navigate to the home screen: {{"action_type": "navigate_home"}}
- To navigate back: {{"action_type": "navigate_back"}}

Here are some useful guidelines you need to follow:
General:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.

Action-Related:
- Ensure the target element is visible on the screen for click. If not, explore the screen.
- Swiping vertically is often used to explore more content while swiping horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists. Swiping can also be used to adjust a slider, such as adjusting the volume on a media player interface.

The user's task is: {global_task}
Action history: {history}
{func_str}
Your output should include four parts in the given format:
<observation> (Describe the current screenshot and any notable elements) </observation>
<thought> (Step-by-step reasoning towards the next optimal action based on the task progress and UI content.) </thought>
<action> (Use one action in the list defined above. Only one action at a time without any comments. Use the JSON format.) </action>
<summary> (Summarize the action taken in one intent-focused sentence, beginning with the base form of the action verb.) </summary>"""


def make_aitw_som_prompt(func_annos = None):
    if func_annos:
        func_str = f"The functionality descriptions of the marked elements:\n{func_annos}\n"
    else:
        func_str = ""

    return AITW_PLANNING_WITH_FUNCANNO_PROMPT.replace("{func_str}", func_str)
    
# Android Control
ANDROIDCONTROL_PLANNING_WITH_FUNCANNO_PROMPT = PLANNING_PROMPT_HEAD.format(device='smartphone') + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful"}}
- If the task is infeasible (e.g., lack of information or inability to perform necessary actions), use: {{"action_type": "status", "goal_status": "infeasible"}}
- For clicking/tapping an element on the screen, use the 'click' action with the element's numeric tag: {{"action_type": "click", "target": <the element numeric tag>}}
- For long-pressing an element: {{"action_type": "long_press", "target": <the element numeric tag>}}
- To swipe the screen, use this action: {{"action_type": "swipe", "direction": <"up", "down", "left", "right">}}. The direction determines the direction of the swipe and must be wrapped in double quotes. Note that swipe up will move the content upward and reveal the content below.
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- To press the Enter key, use the 'enter' action: {{"action_type": "enter"}}
- To navigate to the home screen: {{"action_type": "navigate_home"}}
- To navigate back: {{"action_type": "navigate_back"}}
- To open an app: {{"action_type": "open_app", "app_name": <app name>}}
- To wait for the content to load completely: {{"action_type": "wait"}}

Here are some useful guidelines you need to follow:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.
- Ensure the target element is visible on the screen for click/long_press. If not, explore the screen.
- Swiping vertically is often used to explore more content while swiping horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists. Swiping can also be used to adjust a slider, such as adjusting the volume on a media player interface.

The user's task is: {global_task}
Action history: {history}
{func_str}
Your output should include four parts in the given format:
<observation> (Describe the current screenshot and any notable elements) </observation>
<thought> (Step-by-step reasoning towards the next optimal action based on the task progress and UI content.) </thought>
<action> (Use one action in the list defined above. Only one action at a time without any comments. Use the JSON format.) </action>
<summary> (Summarize the action taken in one intent-focused sentence, beginning with the base form of the action verb.) </summary>"""

def make_andcon_som_prompt(func_annos = None):
    if func_annos:
        func_str = f"The functionality descriptions of the marked elements:\n{func_annos}\n"
    else:
        func_str = ""

    return ANDROIDCONTROL_PLANNING_WITH_FUNCANNO_PROMPT.replace("{func_str}", func_str)
    

MIND2WEB_PLANNING_WITH_FUNCANNO_PROMPT = PLANNING_PROMPT_HEAD.format(device='web browser').replace("the current screenshot", "the current screenshot on which interactable elements have been marked with boxes and numeric tags") + """
- For clicking an element on the screen, use the 'click' action with the element's numeric tag: {{"action_type": "click", "target": <the element numeric tag>}}
- For hovering over an element, use: {{"action_type": "hover", "target": <the element numeric tag>}}
- To focus on an input field and then type texts into it, use the 'input_text' action: {{"action_type": "input_text", "target": <numeric tag of an input field>, "text": "(text to type)"}}. The target denotes the input field in which to type texts. The text is the string you want to insert.
- To select an item/option in a dropdown menu, use the 'select' action: {{"action_type": "select", "target": <the menu element's numeric tag>, "value": "(the name of the selected item)"}}. The target denotes the menu from which to select an item. The value denotes the item name according to the user's task requirement. For example, you need to output {{"action_type": "select", "target": <tag of a time selector>, "value": "12 30 PM"}} for the task "Rent a truck with on April 12 at 12:30 pm". Also note that this select action must not be used to check a checkbox.
- For pressing enter, use: {{"action_type": "enter"}}

Important notes:
1. You MUST NOT click on an input field (e.g., search bar) before typing; simply output the correct input field position and the input_text action will automatically focus on it.
2. Similarly, clicking on a menu/selector element before selecting an item is FORBIDDEN. Just output the correct menu/selector element position and the select action will automatically focus on and reveal the desired menu.

The user's task is: {global_task}
Action history: {history}
{func_str}
Your output should include four parts in the given format:
<observation> (Describe the current screenshot and any notable elements) </observation>
<thought> (Step-by-step reasoning towards the next optimal action based on the task progress and UI content.) </thought>
<action> (Use one action in the list defined above. Only one action at a time without any comments. Use the JSON format.) </action>
<summary> (Summarize the action taken in one intent-focused sentence, beginning with the base form of the action verb.) </summary>"""

SEEACT_MIND2WEB_PLANNING_SOM_SYS_PROMPT = """Imagine that you are imitating humans doing web navigation for a task step by step. At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history. You need to decide on the first following action to take. You can click an element with the mouse, select an option, or type text with the keyboard. (For your understanding, they are like the click(), select_option() and type() functions in playwright respectively) One next step means one operation within the three."""

SEEACT_MIND2WEB_PLANNING_SOM_PROMPT = """You are asked to complete the following task: {global_task}

Previous Actions:
{history} 

The screenshot below shows the webpage you see. In the screenshot, some bounding boxes and numeric tags at the top left corner of the bounding boxes have been manually added. You should ignore them for now. Follow the guidance to think step by step before outlining the next action step at the current stage (Your output should contain these parts):

<Current Webpage Identification> Firstly, think about what the current webpage is. </>

<Previous Action Analysis> Secondly, combined with the screenshot, analyze each step of the previous action history and their intention one by one. Particularly, pay more attention to the last step, which may be more related to what you should do now as the next step. </>

<Screenshot Details Analysis> Closely examine the screenshot to check the status of every part of the webpage to understand what you can operate with and what has been set or completed. You should closely examine the screenshot details to see what steps have been completed by previous actions even though you are given the textual previous actions. Because the textual history may not clearly and sufficiently record some effects of previous actions, you should closely evaluate the status of every part of the webpage to understand what you have done. </>

<Next Action Based on Webpage and Analysis> Then, based on your analysis, in conjunction with human web browsing habits and the logic of web design, decide on the following action. And clearly outline which element in the webpage users will operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules:
1. You should only issue a valid action given the current observation.
2. You should only issue one action at a time
</>

<Reiteration> First, reiterate your next target element, its detailed location, and the corresponding operation. </>

<Verification with the Screenshot> Then, please closely re-examine the screenshot to find whether your target element is marked by a bounding box and has a white numeric tag on a black background at the top left corner of the bounding box, which is positioned closely next to the bounding box. If yes, use that tag for your final answer. If not, please do not make them up. If it is not marked, please output "NA" as your target element in the following final answer part. </>

<Final Answer> Finally, conclude your answer using the format below. Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain. The element choice, action, and value should be in three separate lines.

Format:

ELEMENT: The numeric tag of your choice.

ACTION: Choose an action from CLICK, TYPE, SELECT.

VALUE: Provide additional input based on ACTION.

The VALUE means: If ACTION == TYPE, specify the text to be typed. If ACTION == SELECT, specify the option to be chosen. If ACTION == CLICK, write "None". </>"""


def make_mind2web_som_prompt(func_annos = None):
    if func_annos:
        func_str = f"The functionality descriptions of the marked elements:\n{func_annos}\n"
    else:
        func_str = ""

    return MIND2WEB_PLANNING_WITH_FUNCANNO_PROMPT.replace("{func_str}", func_str)

# GUIAct-Web
# "{'click', 'hover', 'answer', 'copy', 'select_text', 'enter', 'scroll', 'input'}”
GUIACTWEB_PLANNING_WITH_FUNCANNO_PROMPT = PLANNING_PROMPT_HEAD.format(device='web browser').replace("the current screenshot", "the current screenshot on which interactable elements have been marked with boxes and numeric tags") + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful"}}
- For clicking an element on the screen, use the 'click' action with the element's numeric tag: {{"action_type": "click", "target": <the element numeric tag>}}
- For hovering over an element, use: {{"action_type": "hover", "target": <the element numeric tag>}}
- For dragging to highlight a region or element, use: {{"action_type": "drag", "start": (x1,y1), "end": (x2,y2)}} where x and y are integers representing the point's horizontal and vertical screen positions normalized from (0,0) at the top left to (999,999) at the bottom right. The start indicates where the drag begins (where the user would initially long-press the screen to focus on the element to drag). The end specifies where the dragged object is moved to before the finger is lifted (where the user releases the screen).
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- Scrolling: To scroll the window, use this action: {{"action_type": "scroll", "direction": <"up", "down", "left", "right">, "distance: <"short", "medium", "long">}}. The direction determines the direction of the scroll and must be wrapped in double quotes. Scrolling down moves the content up to reveal what is hidden below the current view. The distance parameter determines the scrolling distance. Scrolling vertically is often used to explore more content in webpages while scrolling horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists.
- Key Combinations: To press a key combination: {{"action_type": "hotkey", "key_comb": "(key combination)"}}. The key_comb examples include Ctrl-c, Ctrl-S or Ctrl-Shift-1 with multiple keys combined with '-'.
- For pressing enter, use: {{"action_type": "enter"}}

The user's task is: {global_task}
Action history: {history}
{func_str}
Your output should include four parts in the given format:
<observation> (Describe the current screenshot and any notable elements) </observation>
<thought> (Step-by-step reasoning towards the next optimal action based on the task progress and UI content.) </thought>
<action> (Use one action in the list defined above. Only one action at a time without any comments. Use the JSON format.) </action>
<summary> (Summarize the action taken in one intent-focused sentence, beginning with the base form of the action verb.) </summary>"""


def make_guiactweb_som_prompt(func_annos = None):
    if func_annos:
        func_str = f"The functionality descriptions of the marked elements:\n{func_annos}\n"
    else:
        func_str = ""

    return GUIACTWEB_PLANNING_WITH_FUNCANNO_PROMPT.replace("{func_str}", func_str)

GUIACTMOBILE_PLANNING_WITH_FUNCANNO_PROMPT = PLANNING_PROMPT_HEAD.format(device='smartphone').replace("the current screenshot", "the current screenshot on which interactable elements have been marked with boxes and numeric tags") + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful"}}
- For clicking an element on the screen, use the 'click' action with the element's numeric tag: {{"action_type": "click", "target": <the element numeric tag>}}
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- To swipe the screen, use this action: {{"action_type": "swipe", "direction": <"up", "down", "left", "right">}}. The direction determines the direction of the swipe and must be wrapped in double quotes. Note that swipe up will move the content upward and reveal the content below, which is equivalent to scrolling down.
- For pressing enter, use: {{"action_type": "press_key", "key": "enter"}}

Here are some useful guidelines you need to follow:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task and present the answer if required.
- Ensure the target element is visible on the screen for click/long_press. If not, explore the screen.

The user's task is: {global_task}
Action history: {history}
{func_str}
Your output should include four parts in the given format:
<observation> (Describe the current screenshot and any notable elements) </observation>
<thought> (Step-by-step reasoning towards the next optimal action based on the task progress and UI content.) </thought>
<action> (Use one action in the list defined above. Only one action at a time without any comments. Use the JSON format.) </action>
<summary> (Summarize the action taken in one intent-focused sentence, beginning with the base form of the action verb.) </summary>"""


def make_guiactmobile_som_prompt(func_annos = None):
    if func_annos:
        func_str = f"The functionality descriptions of the marked elements:\n{func_annos}\n"
    else:
        func_str = ""

    return GUIACTMOBILE_PLANNING_WITH_FUNCANNO_PROMPT.replace("{func_str}", func_str)


SIMPLE_COT_REQUIREMENT = """Your output should include the following parts in the given format:
<observation> (Describe the current screenshot and any notable elements.) </observation>
<thought> (Step-by-step reasoning towards the next action based on the observation, task and action history. Do first determine whether the current screenshot represents the final state required by the task.) </thought>
<action> (Use only one action in the list defined above without any comments. Use the JSON format.) </action>"""

COMPLEX_COT_REQUIREMENT = """Your output should include the following parts in the given format:
<observation> (Describe the current screenshot and any notable elements.) </observation>
<thought> (Step-by-step reasoning towards the next action based on the observation, task and action history. Do first determine whether the current screenshot represents the final state required by the task.) </thought>
<target_functionality> (Follow the important notes above to describe the target's distinctive functionality starting with "This element ...". For other actions, this field is "None".) </target_functionality>
<action> (Use only one action in the list defined above without any comments. Use the JSON format.) </action>
<summary> (Summarize the taken action with one intent-focused sentence (starting with the action verb).) </summary>"""

# action space of the agent tasks
ACT_DEF = {
    'answer': """Answer user's question: {"action_type": "answer", "text": "<answer_text>"}""",

    'answer_qwen': """Answer user's question: {"action": "answer", "text": "<answer_text>"}""",

    'status_successful': """If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {"action_type": "status", "goal_status": "successful"}""",

    'status_infeasible': """If the task is infeasible (e.g., lack of information or inability to perform necessary actions), use: {"action_type": "status", "goal_status": "infeasible"}""",

    'terminate': """Terminate the current task and report its completion status: {"action": "terminate", "status": <"success" or "failure">}""",

    'answer': """Provide the user with a faithful answer according to the task requirements and screenshot content: {"action_type": "answer", "text": "<answer_text>"}}""",

    'click_xy1000': """For clicking/tapping an element on the screen, use the 'click' action with the element's position: {"action_type": "click", "target": (x, y)} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.""",

    'click_xy': """For clicking/tapping an element on the screen, use the 'click' action with the element's position: {"action_type": "click", "target": (x, y)} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0, 0) at the top left to (Image Width, Image Height) at the bottom right.""",

    'click_index': """For clicking/tapping an element specified by its index (an integer) on the screen, use the 'click' action: {"action_type": "click", "index": <target_index>}.""",

    'click_xy-qwen': """Click a point on the screen: {"action": "click", "coordinate": [x, y]}. The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.""",

    'long-press_xy': """For long-pressing an element: {"action_type": "long_press", "target": (x, y)}""",

    'long-press_index': """For long-pressing an element: {"action_type": "long_press", "index": <element_index>}""",

    'long-press_xy-qwen': """Long_press a point on the screen: {"action": "long_press", "coordinate": [x, y]}.""",

    'swipe_xy': """To swipe the screen from a starting point, use this action: {"action_type": "swipe", "direction": <"up", "down", "left", "right">, "start": (x, y)}. The direction determines the direction of the swipe and must be wrapped in double quotes. Swiping up (equivalent to scrolling down) will move the content upward and reveal the content below.""",

    'swipe_index': """To swipe the screen from a starting point specified by its index, use this action: {"action_type": "swipe", "direction": <"up", "down", "left", "right">, "index": <element_index (optional)>}. The direction determines the direction of the swipe and must be wrapped in double quotes. Swiping up (equivalent to scrolling down) will move the content upward and reveal the content below. Specify the target index if you want to swipe a specific element; leave it empty when swiping the whole screen.""",

    'swipe': """To swipe the screen, use this action: {"action_type": "swipe", "direction": <"up", "down", "left", "right">}. The direction determines the direction of the swipe and must be wrapped in double quotes. Swiping up (equivalent to scrolling down) will move the content upward and reveal the content below.""",

    'swipe_xy-qwen': """Swipe the screen: {"action": "swipe", "coordinate": [x, y], "coordinate2": [x, y]}. The coordinate and coordinate2 are the starting and ending points of the swipe, respectively.""",

    'input-text': """To type texts into a text field, use the 'input_text' action: {"action_type": "input_text", "text": "(text to type)"}. The text is the string you want to insert. For example: {"action_type": "input_text", "text": "Hello, world!"} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.""",

    'input-text_xy': """To type texts into a text field specified by its coordinates, use the 'input_text' action: {"action_type": "input_text", "text": "(text to type)", "target": (x, y)}. Do not click on the target input field as this input_text action automatically does this for you.""",

    'input-text_index': """To type texts into a text field specified by its index, use the 'input_text' action: {"action_type": "input_text", "text": "(text to type)", "index": <index of target text field>}. Do not click on the target input field as this input_text action automatically focuses on the field for you.""",

    'input-text_qwen': """Type texts into the focused text field: {"action": "type", "text": "(text to type)"}""",
    
    'input-text_xy-qwen': """Type texts into a text field specified by its coordinates: {"action": "type", "coordinate": [x, y], "text": "(text to type)"}. Do not click on the target input field as this type action automatically focuses on the field for you.""",
    
    'system_button': """Press the system button: {"action": "system_button", "button": <"Back", "Home", "Menu", "Enter">}. Back means returning to the previous interface, Home means returning to the home screen, Menu means opening the app drawer, and Enter means pressing the enter key.""",

    'press_enter': """To press the Enter key, use the 'enter' action: {"action_type": "enter"}""",

    'navigate_back': """To navigate back: {"action_type": "navigate_back"}""",

    'navigate_home': """To navigate to the home screen: {"action_type": "navigate_home"}""",

    'wait': """Wait for the screen to update completely: {"action_type": "wait"}""",
    
    'wait_qwen': """Wait for the screen to update completely: {"action": "wait", "time": <seconds>}""",

    'open_app': """To open an app (nothing will happen if the app is not installed): {"action_type": "open_app", "app_name": <app name>}. (Important: Using open_app is more efficient than swiping up to open the app drawer.)""",

    'open_app-qwen': """Launch an app directly without using the app drawer: {"action": "open", "text": <app name>}""",
}



ACT_DEF_ZH = {
'answer': """回答用户问题：{"action_type": "answer", "text": "<回答文本>"}""",

'answer_qwen': """回答用户问题：{"action": "answer", "text": "<回答文本>"}""",

'status_successful': """若您认为任务已成功，请使用“status”操作并以“successful”为目标状态来结束任务：{"action_type": "status", "goal_status": "successful"}""",

'status_infeasible': """若任务不可行（例如，信息缺失或无法执行必要操作），请使用：{"action_type": "status", "goal_status": "infeasible"}""",

'terminate': """终止当前任务并报告其完成状态：{"action": "terminate", "status": <"success" 或 "failure">}""",

'answer': """根据任务要求及截图内容，为用户提供精准的回答：{"action_type": "answer", "text": "<回答文本>"}""",

'click_xy1000': """点击/轻触屏幕上的元素，请使用 'click' 操作并指定其坐标：{"action_type": "click", "target": (x, y)}。其中 x 和 y 为整数，代表该点在屏幕上的横纵千分比位置，坐标系从左上角的 (0,0) 延伸至右下角的 (1000,1000)。""",

'click_xy': """点击/轻触屏幕上的元素，请使用 'click' 操作并指定其坐标：{"action_type": "click", "target": (x, y)}。其中 x 和 y 为整数，代表该点在屏幕上的横纵像素位置，坐标系从左上角的 (0, 0) 延伸至右下角的 (图像宽度, 图像高度)。""",

'click_index': """点击/轻触由索引（整数）指定的屏幕元素，请使用 'click' 操作：{"action_type": "click", "index": <目标索引>}。""",

'click_xy-qwen': """点击屏幕上的一个点：{"action": "click", "coordinate": [x, y]}。其中 x 为距左边缘的像素值，y 为距上边缘的像素值，共同确定鼠标的移动目标。""",

'long-press_xy': """长按某个元素：{"action_type": "long_press", "target": (x, y)}""",

'long-press_index': """长按某个元素：{"action_type": "long_press", "index": <元素索引>}""",

'long-press_xy-qwen': """长按屏幕上的一个点：{"action": "long_press", "coordinate": [x, y]}。""",

'swipe_xy': """从一个起始点滑动屏幕，请使用此操作：{"action_type": "swipe", "direction": <"up", "down", "left", "right">, "start": (x, y)}。direction 参数决定滑动方向，且须用双引号包裹。向上滑动（等同于向下滚动）会使内容上移，从而显示下方内容。""",

'swipe_index': """从一个由索引指定的起始点滑动屏幕，请使用此操作：{"action_type": "swipe", "direction": <"up", "down", "left", "right">, "index": <元素索引 (可选)>}。direction 参数决定滑动方向，且须用双引号包裹。向上滑动（等同于向下滚动）会使内容上移，从而显示下方内容。若想滑动特定元素，请指定其索引；若想滑动整个屏幕，则将索引留空。""",

'swipe': """滑动屏幕，请使用此操作：{"action_type": "swipe", "direction": <"up", "down", "left", "right">}。direction 参数决定滑动方向，且须用双引号包裹。向上滑动（等同于向下滚动）会使内容上移，从而显示下方内容。""",

'swipe_xy-qwen': """滑动屏幕：{"action": "swipe", "coordinate": [x, y], "coordinate2": [x, y]}。coordinate 与 coordinate2 分别为滑动的起点和终点。""",

'input-text': """在文本框中输入文字，请使用 'input_text' 操作：{"action_type": "input_text", "text": "(要输入的文本)"}。text 是您希望插入的字符串。例如：{"action_type": "input_text", "text": "你好，世界！"} 将会输入文本“你好，世界！”。请注意，在使用此操作前，需先用 'click' 操作聚焦到目标输入框。""",

'input-text_xy': """在由坐标指定的文本框中输入文字，请使用 'input_text' 操作：{"action_type": "input_text", "text": "(要输入的文本)", "target": (x, y)}。此操作会自动点击目标输入框，您无需预先点击。""",

'input-text_index': """在由索引指定的文本框中输入文字，请使用 'input_text' 操作：{"action_type": "input_text", "text": "(要输入的文本)", "index": <目标文本框索引>}。此操作会自动聚焦到目标输入框，您无需预先点击。""",

'input-text_xy-qwen': """在由坐标指定的文本框中输入文字：{"action": "type", "coordinate": [x, y], "text": "(要输入的文本)"}。此 'type' 操作会自动聚焦到目标输入框，您无需预先点击。""",

'system_button': """按下系统按钮：{"action": "system_button", "button": <"Back", "Home", "Menu", "Enter">}。Back 指返回上一界面，Home 指返回主屏幕，Menu 指打开应用抽屉，Enter 指按下回车键。""",

'press_enter': """按下回车键，请使用 'enter' 操作：{"action_type": "enter"}""",

'navigate_back': """导航至上一界面：{"action_type": "navigate_back"}""",

'navigate_home': """导航至主屏幕：{"action_type": "navigate_home"}""",

'wait': """等待屏幕完全更新：{"action_type": "wait"}""",

'wait_qwen': """等待屏幕完全更新：{"action": "wait", "time": <秒数>}""",

'open_app': """打开指定应用（若应用未安装则无效果）：{"action_type": "launch_app", "app_name": <应用名称>}。（重要提示：使用此操作比上滑打开应用抽屉更为高效。）""",

'open_app-qwen': """直接启动应用，无需通过应用抽屉：{"action": "open", "text": <应用名称>}""",

}

TASK_GUIDELINES = {
    'info_search': """Sometimes you may need to navigate the phone to gather information needed to complete the task, for example if user asks "what is my schedule tomorrow", then you may open the calendar app (using the `open_app` action), look up information there, answer user's question (using the `answer` action) and finish (using the `status` action with complete as goal_status).""",

    'open_app': """Use the `open_app` action to open an app and do not use the app drawer unless all other ways have failed.""",

    'answer': """For requests that are questions (or chat messages), use the `answer` action to reply to the user before finishing!""",

    'text_operation': """To delete some text: place the cursor at the right place and use the backspace button on the keyboard to delete the characters (long press the backspace to accelerate if there are many to delete).
To copy text: first long press to select the exact text you want to copy, which brings up the text selection bar, then click the `copy` button on the bar.
To paste text into a text box, first long press the text box, then the text selection bar will appear with a `paste` button in it.""",

    'traceback': """If you have deviated from the correct path, undo your last action and return to the previous state before taking a new path."""
}

ACT_SPACE = {
    'all': ['status_successful', 'click_xy1000', 'long-press_xy', 'swipe', 'input-text', 'press_enter', 'navigate_back', 'navigate_home', 'wait', 'open_app'],
    
    'unified_mobile': ['status_successful', 'click_xy1000', 'long-press_xy', 'swipe', 'input-text', 'press_enter', 'navigate_back', 'navigate_home', 'wait', 'open_app'],
    
    'AITW': ['status_successful', 'click_xy1000', 'swipe', 'input-text', 'press_enter', 'navigate_back', 'navigate_home'],
    'AndroidControl': ['click_xy1000', 'swipe', 'input-text',  'navigate_back', 'wait', 'open_app'],
    'AndroidWorld': ['answer', 'status_successful', 'status_infeasible', 'click_xy1000', 'long-press_xy', 'swipe_xy', 'input-text_xy', 'open_app', 'navigate_back', 'navigate_home', 'press_enter', 'wait'],
    'AndroidControl_reflec': ['click_xy1000', 'swipe_xy', 'input-text',  'navigate_back', 'wait', 'open_app'],
    'StepCopilot': ['click_xy', 'swipe_xy', 'input-text',  'navigate_back', 'wait', 'open_app'],

}

GUIDELINES = {
    'AITW': [],
    'AndroidControl': [],
    'AndroidWorld': [
        # 'info_search',
        'answer', 'open_app', 'traceback',
        'text_operation'
        ]
}

QWEN2P5VL_PROTOCOL_SYSPROMPT = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1092x2408.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
    * `click`: Click the point on the screen with coordinate (x, y).
    * `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
    * `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
    * `type`: Input the specified text into the activated input box.
    * `system_button`: Press the system button.
    * `open`: Open an app on the device.
    * `wait`: Wait specified seconds for the change to happen.
    * `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, and `action=open`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

"""

QWEN2P5VL_DIRECT_PROTOCOL = """For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

QWEN2P5VL_COT_PROTOCOL = """Your output should follow this format. Before answering, explain your reasoning step by step in <thinking></thinking> tags, and insert it before the <tool_call> tags:
<thinking>
{protocol}
</thinking>
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


QWEN2P5VL_PROTOCOL_USERPROMPT = """The user query: {global_task}
Task progress (You have done the following operation on the current device): {history}"""


COMPLEX_PROTOCOL = """<task> (First get yourself familiar with the given task) </task>
<observation> (Identify and describe the hierarchical layout and elements of the screenshot. (i) First describe the overall functionality of the GUI screen; (ii) Then, you should discern the high-level layout, such as navigation bars, main content, sidebars, and then describe the key interactive elements within each region. You are required to concretely describe the region where the task-related elements most likely appear while paying less attention to useless regions.) </observation>
<progress> (Evaluate the current path according to the current observation and previous actions taken and identify any deviations from the expected sequence.) </progress>
<exception> (Classify the exceptions detected in the progress analysis content: a. Going Astray: Previous actions taken lead to an unhelpful page and deviate from the right path; b. Adversarial Attack: Unexpected behavior that disrupts execution, such as pop-up advertisements; c. Repeated Meaningless Behavior: Performing actions multiple times without making progress, such as clicking the same button repeatedly; d. Format Error: The action generated has incorrect format and fails to be executed; e. No Exception: On the right path.) </exception>
<decision> (Perform step-by-step reasoning through next step prediction according to the task progress and possible exceptions. If exceptions occur, ponder over why previous actions are wrong and how to correct the mistake; otherwise, think about the next best action toward task completion. Add reasoning elements that demonstrate the detailed decision-making process, such as a. Causality: "Since/Because/As...", "Thus/So..."; b. Reflection: "Let me check", "However...", "Wait...", "But..."; c. Summarization: "In summary...", "So...".) </decision>
<intent> <Summarize the action along with the screen where this action is taken in a format of user intent> </action>
<prediction> (Predict what will happen to the GUI content after taking the planned action) </prediction>"""

COMPLEX_PROTOCOL_V2 = """<task> (First Familiarize yourself with the task) </task>
<observation> (Observe the screen by describing its main purpose and identifying the layout, such as navigation bars and key interactive elements related to the task.) </observation>
<progress> (Evaluate your progress to ensure actions align with expectations, noting any deviations.) </progress>
<exception> (Classify the exceptions detected in the progress analysis content: A. Going Astray: Previous actions taken lead to an unhelpful page and deviate from the right path; B. Interrupting Content: Unexpected content that disrupts execution, such as pop-up advertisements; C. Deadlock: Performing actions multiple times without making progress; D. Format Error: The last action generated has incorrect format; E. No Exception: Everything is on track.) </exception>
<decision> (Decide on the next step by correcting any mistakes or GO BACK to previous states if exceptions occur, or determine the next best action to complete the task if everything is proceeding smoothly.) </decision>
<prediction> (Predict what will happen to the GUI content after taking the planned action) </prediction>"""

COMPLEX_PROTOCOL_V3 = """1. Task: (First Familiarize yourself with the task) 
2. Observation: (Observe the screen by describing its main purpose and identifying the layout, such as navigation bars and key interactive elements related to the task.) 
3. Progress: (Evaluate your progress to ensure actions align with expectations, noting any deviations.) 
4. Exception: (Classify the exceptions detected in the progress analysis content: A. Going Astray: Previous actions taken lead to an unhelpful page and deviate from the right path; B. Interrupting Content: Unexpected content that disrupts execution, such as pop-up advertisements; C. Deadlock: Performing actions multiple times without making progress; D. Format Error: The last action generated has incorrect format; E. No Exception: Everything is on track.) 
5. Decision: (Decide on the next step by correcting any mistakes or GO BACK to previous states if exceptions occur, or determine the next best action to complete the task if everything is proceeding smoothly.) 
6. Prediction: (Predict what will happen to the GUI content after taking the planned action)"""


NO_PROTOCOL = """(First, analyze the task progress as well as the current screenshot, and then present the rationale for planning the next step.)"""

COMPRESSED_STRUCTURED_PROTOCOL = """(Observe the current GUI screen, break down the task into manageable step-by-step subtasks while tracking progress through action history, select the most appropriate action plan, and predict the outcome of each planned action.)"""

SIMPLE_PROTOCOL = """Familiarize yourself with the task, then observe the screen by describing its main purpose and identifying the layout, such as navigation bars and key interactive elements related to the task. Evaluate your progress to ensure actions align with expectations, noting any deviations. Identify exceptions: going astray, interrupting content (advertisements), repeated meaningless behavior, format errors in your last response, or confirm no exceptions if everything is on track. Finally, decide on the next steps by correcting any mistakes if exceptions occur, or determine the next best action to complete the task if everything is proceeding smoothly."""

DUMMY_PROTOCOL = """(First, analyze the task progress as well as the current screenshot, and then present the rationale for planning the next step.)"""

GUI_GUIDELINES = """Useful guidelines to follow:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.
- Ensure the target element is visible on the screen for click and long-press. If not, explore the screen.
- After typing into a text field, sometimes an auto-complete dropdown list will appear, indicating this is an enum field and you can click and select the best match in the list.
{extra_guidelines}"""

PLANNING_PROMPT_DIRECT_WOACTSPACE = """You are an agent that can operate a {device} to complete the user's tasks. At each step, based on the current screenshot and action history, you must perform an action to advance toward task completion.
{action_space}
The user's task is: {global_task}
Action history: {history}
{elem_list_str}
Directly plan the next action as a JSON object defined above."""

PLANNING_PROMPT_DIRECT = """You are an agent that can operate a {device} to complete the user's tasks. At each step, based on the current screenshot and action history, you must perform one of the actions below to advance toward task completion.
{action_space}
The user's task is: {global_task}
Action history: {history}
{elem_list_str}
Directly plan the next action as a JSON object defined above."""

PLANNING_PROMPT_SOM = """You are an agent that can operate a {device} to complete the user's tasks. At each step, based on the current screenshot with all elements marked and action history, you must perform one of the actions below to advance toward task completion.
{action_space}
The user's task is: {global_task}
Action history: {history}
{elem_list_str}

Perform comprehensive reasoning before planning the next action. Your output should follow this format (Do not mention the element indices in your reasoning because they are not available to the user):
<structured_thinking>
{reasoning_requirements}
</structured_thinking>
<coherent_thinking>
(Rephrase the structured thinking into a coherent thought process just like a normal human would do.)
</coherent_thinking>
<answer>
(Output the planned action as a JSON object defined above.)
</answer>"""

PLANNING_PROMPT_PROTOCOL = """You are an agent that can operate a {device} to complete the user's tasks. At each step, based on the current screenshot and action history, you must perform one of the actions below to advance toward task completion.
{action_space}

{guidelines}
The user's task is: {global_task}
Action history: {history}
{elem_list_str}
Your output should follow this format:
<think>
{requirements}
</think>
<action>
(Output the planned action as a JSON object defined above.)
</action>"""

PLANNING_REFLEC_PROMPT_PROTOCOL = """You are an agent that can operate a {device} to complete the user's tasks. At each step, based on the current screenshot and action history, you must perform one of the actions below to advance toward task completion.
{action_space}

{guidelines}
The user's task is: {global_task}
Action history: {history}
{elem_list_str}
Your output should follow this interleaved planning-reflection format:
<thinking>
(First, output the planned action as a JSON object defined above. Then, reflect on how the action changes the GUI content and advance toward the task completion.)
(If you find that the action is incorrect during your reflection, please make a correction. Repeat the planning-reflection procedure if needed, until you find the optimal action.)
</thinking>
<answer>
(The final action you consider optimal)
</answer>"""

PLANNING_REFLEC_PROMPT_PROTOCOL_V2 = """You are an agent that can operate a {device} to complete the user's tasks. At each step, based on the current screenshot and action history, you must perform one of the actions below to advance toward task completion.
{action_space}

{guidelines}
The user's task is: {global_task}
Action history: {history}
{elem_list_str}
You will be guided to follow the interleaved planning-reflection format:
1. Proposal Step: Propose an action as a JSON object defined above. At first turn, you just need to propose an action. After multiple turns of reflection, propose the action that has the highest probability of advancing you toward the goal.
2. Reflection Step: Reflect on how the action will change the GUI content and advance toward the task completion. If you find that the action is incorrect during your self-reflection, claim "Need correction: True" at the end of your refleciton; otherwise, claim "Need correction: False".
Guiding Principles:
1. Iterative Improvement: Repeat the proposal-reflection procedure if needed, until you find the optimal action.
2. No Repetition: NEVER propose an action that you have already attempted.
3. Meaningful Corrections: If an action requires correction, your next plan should not be a minor tweak. It should be based on a new hypothesis for solving the problem."""

REFLEC_PLANNING_QUERY = "After the reflection, you should choose the most impactful action to take."
REFLEC_VERIFY_QUERY = "Let's verify the action."
REFLEC_VEIRF_PLACEHOLDER = "This action cannot advance toward the task completion. Need correction: True."
# Protocol prompt

def make_structured_reasoning_requirements(use_obs: bool = True, use_progress: bool = True, use_intent: bool = True, use_outcome: bool = True):
    reason_req = []
    if use_obs:
        reason_req.append(f"Observation: {REASONING_REQUIREMENTS['Observation']}")
    if use_progress:
        reason_req.append(f"Progress: {REASONING_REQUIREMENTS['Progress']}")
    if use_intent:
        reason_req.append(f"Intent: {REASONING_REQUIREMENTS['Intent']}")
    if use_outcome:
        reason_req.append(f"Outcome: {REASONING_REQUIREMENTS['Outcome']}")
    reasoning_requirements = '\n'.join(reason_req)
    return reasoning_requirements

def make_qwen2p5vl_planning_protocol(cot: bool = False, protocol_type: str = 'v3', use_obs: bool = True, use_progress: bool = True, use_intent: bool = True, use_outcome: bool = True):
    if cot:
        if protocol_type == 'structured':
            protocol = make_structured_reasoning_requirements(use_obs, use_progress, use_intent, use_outcome)
        else:
            protocol = PROTOCOL_DICT[protocol_type]
        return QWEN2P5VL_PROTOCOL_SYSPROMPT + QWEN2P5VL_COT_PROTOCOL.format(protocol=protocol)
    else:
        return QWEN2P5VL_PROTOCOL_SYSPROMPT + QWEN2P5VL_DIRECT_PROTOCOL

def make_action_space(bmk_name: str, use_qwen_actspace: bool = False, use_unnorm_xy: bool = False, use_index: bool = False, use_square_xy_brackets: bool = False, use_elem_ref: bool = False, input_text_with_coord: bool = True):
    """Generate the action space description based on benchmark name and model type.

    Args:
        bmk_name: Name of the benchmark
        use_qwen_actspace: Whether to use Qwen-specific action space format
        use_unnorm_xy: Whether to use unnormalized xy coordinates
        use_index: Whether to use index coordinates
        use_square_xy_brackets: Whether to use square brackets for xy coordinates

    Returns:
        String describing the available actions
    """
    acts = deepcopy(ACT_SPACE[bmk_name])

    
    if use_qwen_actspace:
        for i, act in enumerate(acts):
            if any(k in act for k in ['click', 'long-press', 'input-text', 'swipe']):
                post_fix = "xy-qwen" if not (act == 'input-text' and not input_text_with_coord) else "qwen"
                if  act.endswith(post_fix): continue
                if '_' in act:
                    acts[i] = act[:act.rfind('_')+1] + post_fix
                else:
                    acts[i] = act + "_" + post_fix
        
        if 'status_successful' in acts:
            acts.remove('status_successful')
        
        if 'status_infeasible' in acts:
            acts.remove('status_infeasible')
        
        # add answer action
        if 'answer' in acts:
            acts.remove('answer')
        
        if 'answer_qwen' not in acts:
            acts.insert(0, 'answer_qwen')

        # add terminate action
        if 'terminate' not in acts:
            acts.insert(1, 'terminate')

        # change open_app
        if 'open_app' in acts:
            acts.remove('open_app')
            acts.insert(-1, 'open_app-qwen')
        
        # change system buttons
        if 'navigate_home' in acts:
            acts.remove('navigate_home')
        
        if 'navigate_back' in acts:
            acts.remove('navigate_back')
        
        if 'press_enter' in acts:
            acts.remove('press_enter')
        
        if 'system_button' not in acts:
            acts.insert(-1, 'system_button')

        # change wait
        if 'wait' in acts:
            acts.remove('wait')
            acts.insert(-1, 'wait_qwen')

    elif use_unnorm_xy:
        for k in ['click_xy1000']:
            if k in acts:
                idx = acts.index(k)
                acts.remove(k)
                acts.insert(idx, 'click_xy')
                break
    if use_index:
        for k in ['click_xy1000', 'click_xy', 'long-press_xy', 'input-text_xy', 'swipe_xy', 'swipe']:
            if k in acts:
                idx = acts.index(k)
                acts.remove(k)
                if '_' in k:
                    acts.insert(idx, k[:k.rfind('_')] + "_index")
                else:
                    acts.insert(idx, k + "_index")
    
    selected_actions = [ACT_DEF[k] for k in acts]

    # if use_elem_ref:
    #     action_space = action_space.replace("(elem_id)", "(elem_id, x, y)")


    action_space = '\n'.join(f'- {x}' for x in selected_actions)

    if use_square_xy_brackets:
        action_space = action_space.replace("(x,y)", "[x,y]").replace("(x, y)", "[x, y]")

    return action_space

def make_guidelines(bmk_name: str):
    extra_guidelines = '\n'.join(f'- {TASK_GUIDELINES[k]}' for k in GUIDELINES[bmk_name])
    guide = GUI_GUIDELINES.format(extra_guidelines=extra_guidelines) + '\n'
    
    return guide

PROTOCOL_DICT = {
    'v1': COMPLEX_PROTOCOL,
    'v2': COMPLEX_PROTOCOL_V2,
    'v3': COMPLEX_PROTOCOL_V3,
    'no': NO_PROTOCOL,
    'simple': SIMPLE_PROTOCOL,
    'direct': DUMMY_PROTOCOL,
    'coherent': COMPRESSED_STRUCTURED_PROTOCOL
}

def make_qwen2p5_planning_prompt(bmk_name: str, task: str, history: str, elem_list_str: str = '', device_type: str = 'digital device', use_unnorm_xy: bool = False, use_index: bool = False, use_square_xy_brackets: bool = False, use_guidelines: bool = False, actspace_type: str = 'qwen2p5', input_text_with_coord: bool = True):
    if actspace_type is not None and actspace_type.strip():
        action_space = make_action_space(bmk_name, actspace_type=='qwen2p5', use_unnorm_xy, use_index, use_square_xy_brackets, input_text_with_coord=input_text_with_coord) + '\n'
    else:
        action_space = ''

    guide = make_guidelines(bmk_name) if use_guidelines else ''

    if elem_list_str:
        elem_list_str = f"The GUI elements marked on the current screen:\n{elem_list_str}\n"

    prompt_bsase = PLANNING_PROMPT_DIRECT_WOACTSPACE if 'wo' in actspace_type else PLANNING_PROMPT_DIRECT
    prompt = prompt_bsase.format(
        device=device_type,
        action_space=action_space,
        global_task=task,
        history=history,
        guidelines=guide,
        elem_list_str=elem_list_str
    )

    return prompt

def make_planning_protocol(bmk_name: str, task: str = '', history: str = '', elem_list_str: str = '', device_type: str = 'digital device', use_unnorm_xy: bool = False, use_index: bool = False, use_square_xy_brackets: bool = False, protocol_type: str = 'v3', use_guidelines: bool = True, use_qwen_actspace: bool = False, use_obs: bool = True, use_progress: bool = True, use_intent: bool = True, use_outcome: bool = True, input_text_with_coord: bool = True):
    action_space = make_action_space(bmk_name, use_qwen_actspace, use_unnorm_xy, use_index, use_square_xy_brackets, input_text_with_coord=input_text_with_coord)
    guide = make_guidelines(bmk_name) if use_guidelines else ''

    if elem_list_str:
        elem_list_str = f"The GUI elements marked on the current screen:\n{elem_list_str}\n"

    if protocol_type == 'structured':
        protocol = make_structured_reasoning_requirements(use_obs, use_progress, use_intent, use_outcome)
    else:
        protocol = PROTOCOL_DICT[protocol_type]
            
    prompt = PLANNING_PROMPT_PROTOCOL.format(
        device=device_type,
        action_space=action_space,
        global_task=task,
        history=history,
        guidelines=guide,
        elem_list_str=elem_list_str,
        requirements=protocol
    )

    if use_qwen_actspace:
        prompt = prompt.replace("<think>", "<thinking>").replace("</think>", "</thinking>")
    
    if protocol_type == 'direct':
        prompt = prompt[:prompt.rfind("<thinking")] + prompt[prompt.rfind("<action"):]

    return prompt

def make_planning_reflec_protocol(bmk_name: str, task: str, history: str, elem_list_str: str = '', device_type: str = 'digital device', use_unnorm_xy: bool = False, use_index: bool = False, use_square_xy_brackets: bool = False, use_guidelines: bool = True, use_qwen_actspace: bool = False, multi_turn: bool = False):
    action_space = make_action_space(bmk_name, use_qwen_actspace, use_unnorm_xy, use_index, use_square_xy_brackets)

    guide = make_guidelines(bmk_name) if use_guidelines else ''

    if elem_list_str:
        elem_list_str = f"The GUI elements marked on the current screen:\n{elem_list_str}\n"

    prompt = (PLANNING_REFLEC_PROMPT_PROTOCOL_V2 if multi_turn else PLANNING_REFLEC_PROMPT_PROTOCOL).format(
        device=device_type,
        action_space=action_space,
        global_task=task,
        history=history,
        guidelines=guide,
        elem_list_str=elem_list_str
    )

    if use_qwen_actspace:
        prompt = prompt.replace("<think>", "<thinking>").replace("</think>", "</thinking>")

    return prompt

REASONING_REQUIREMENTS = {
    'Observation': 'Observe the GUI content on the current screen.',
    'Progress': 'Decompose the task into smaller step-by-step feasible subtasks, and predict the remaining subtasks according to the action history.',
    'Intent': 'Select the most appropriate action plan to take.',
    'Outcome': 'Predict the outcome of the action.'
}

def make_som_planning_prompt(bmk_name: str, task: str, history: str, elem_list_str: str = '', device_type: str = 'digital device', use_unnorm_xy: bool = False, use_index: bool = True, use_square_xy_brackets: bool = False, use_guidelines: bool = False, actspace_type: str = 'qwen2p5', use_obs: bool = True, use_progress: bool = True, use_intent: bool = True, use_outcome: bool = True):
    if actspace_type is not None and actspace_type.strip():
        action_space = make_action_space(bmk_name, actspace_type=='qwen2p5', use_unnorm_xy, use_index, use_square_xy_brackets) + '\n'
    else:
        action_space = ''

    guide = make_guidelines(bmk_name) if use_guidelines else ''

    if elem_list_str:
        elem_list_str = f"The GUI elements marked on the current screen:\n{elem_list_str}\n"

    reason_req = []
    if use_obs:
        reason_req.append(f"Observation: {REASONING_REQUIREMENTS['Observation']}")
    if use_progress:
        reason_req.append(f"Progress: {REASONING_REQUIREMENTS['Progress']}")
    if use_intent:
        reason_req.append(f"Intent: {REASONING_REQUIREMENTS['Intent']}")
    if use_outcome:
        reason_req.append(f"Outcome: {REASONING_REQUIREMENTS['Outcome']}")
    reasoning_requirements = '\n'.join(reason_req)
    prompt_bsase = PLANNING_PROMPT_SOM
    prompt = prompt_bsase.format(
        device=device_type,
        action_space=action_space,
        global_task=task,
        history=history,
        guidelines=guide,
        elem_list_str=elem_list_str,
        reasoning_requirements=reasoning_requirements
    )

    return prompt

AITW_PLANNING_PROMPT_COT = PLANNING_PROMPT_HEAD.format(device='smartphone') + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful"}}
- If the task is infeasible (e.g., lack of information or inability to perform necessary actions), use: {{"action_type": "status", "goal_status": "infeasible"}}
- For clicking/tapping an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.
- To swipe the screen, use this action: {{"action_type": "swipe", "direction": <"up", "down", "left", "right">}}. The direction determines the direction of the swipe and must be wrapped in double quotes. Note that swipe up will move the content upward and reveal the content below.
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- To press the Enter key, use the 'enter' action: {{"action_type": "enter"}}
- To navigate to the home screen: {{"action_type": "navigate_home"}}
- To navigate back: {{"action_type": "navigate_back"}}
- To open an app: {{"action_type": "open_app", "app_name": <app name>}}

Here are some useful guidelines you need to follow:
General:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.

Action-Related:
- Ensure the target element is visible on the screen for click. If not, explore the screen.
- Swiping vertically is often used to explore more content while swiping horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists. Swiping can also be used to adjust a slider, such as adjusting the volume on a media player interface.

The user's task is: {global_task}
Action history: {history}

Your output should include three parts in the given format:
Observation: <Describe the current screenshot and any notable elements.>
Thought: <Step-by-step reasoning towards the next action based on the observation, task and action history. Do first determine whether the current screenshot represents the final state required by the task.>
Intent: <summarize the action to be taken in an intent-focused format>
Action: <Use one action in the list defined above. Only one action at a time without any comments.>"""



ANDROIDCONTROL_PLANNING_PROMPT_COT = PLANNING_PROMPT_HEAD.format(device='smartphone') + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful"}}
- If the task is infeasible (e.g., lack of information or inability to perform necessary actions), use: {{"action_type": "status", "goal_status": "infeasible"}}
- For clicking/tapping an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.
- For long pressing an element: {{"action_type": "long_press", "target": (x,y)}}
- To swipe the screen, use this action: {{"action_type": "swipe", "direction": <"up", "down", "left", "right">}}. The direction determines the direction of the swipe and must be wrapped in double quotes. Note that swipe up will move the content upward and reveal the content below.
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- To press the Enter key, use the 'enter' action: {{"action_type": "enter"}}
- To navigate to the home screen: {{"action_type": "navigate_home"}}
- To navigate back: {{"action_type": "navigate_back"}}
- To open an app: {{"action_type": "open_app", "app_name": <app name>}}
- To wait for the content to load completely: {{"action_type": "wait"}}

Here are some useful guidelines you need to follow:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.
- Ensure the target element is visible on the screen for click/long_press. If not, explore the screen.
- Swiping vertically is often used to explore more content while swiping horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists. Swiping can also be used to adjust a slider, such as adjusting the volume on a media player interface.

The user's task is: {global_task}
Action history: {history}
{step_instruction}
""" + SIMPLE_COT_REQUIREMENT

GUIACTWEB_PLANNING_PROMPT_COT = PLANNING_PROMPT_HEAD.format(device='web') + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful", "answer": "(answer required by the task)"}}
- For clicking an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.
- For hovering over an element: {{"action_type": "hover", "target": (x,y)}}
- For dragging to highlight a region or element, use: {{"action_type": "drag", "start": (x1,y1), "end": (x2,y2)}} where x and y are integers representing the point's horizontal and vertical screen positions normalized from (0,0) at the top left to (999,999) at the bottom right. The start indicates where the drag begins (where the user would initially long press the screen to focus on the element to drag). The end specifies where the dragged object is moved to before the finger is lifted (where the user releases the screen).
- Scrolling: To scroll the window, use this action: {{"action_type": "scroll", "direction": <"up", "down", "left", "right">, "distance: <"short", "medium", "long">}}. The direction determines the direction of the scroll and must be wrapped in double quotes. Scrolling down moves the content up to reveal what is hidden below the current view. The distance parameter determines the scrolling distance. Scrolling vertically is often used to explore more content in webpages while scrolling horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists.
- Key Combinations: To press a key combination: {{"action_type": "hotkey", "key_comb": "(key combination)"}}. The key_comb examples include Ctrl-c, Ctrl-S or Ctrl-Shift-1 with multiple keys combined with '-'.

Here are some useful guidelines you need to follow:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task and present the answer if required.
- Ensure the target element is visible on the screen for click/long_press. If not, explore the screen.

The user's task is: {global_task}
Action history: {history}
{step_instruction}
""" + SIMPLE_COT_REQUIREMENT


GUIACTMOBILE_PLANNING_PROMPT_COT = PLANNING_PROMPT_HEAD.format(device='mobile') + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful", "answer": "(answer required by the task)"}}
- For clicking an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.
- To swipe the screen, use this action: {{"action_type": "swipe", "direction": <"up", "down", "left", "right">}}. The direction determines the direction of the swipe and must be wrapped in double quotes. Note that swipe up will move the content upward and reveal the content below.
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- Press Enter: {{"action_type": "press_key", "key": "enter"}}. This action simulates pressing Enter.

Here are some useful guidelines you need to follow:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task and present the answer if required.
- Ensure the target element is visible on the screen for click/long_press. If not, explore the screen.

The user's task is: {global_task}
Action history: {history}
{step_instruction}
""" + SIMPLE_COT_REQUIREMENT

TWOSTAGE_GUIACTWEB_PLANNING_PROMPT_COT = PLANNING_PROMPT_HEAD.format(device='web') + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful", "answer": "(answer required by the task)"}}
- For clicking an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.
- For hovering over an element: {{"action_type": "hover", "target": (x,y)}}
- For dragging to highlight a region or element, use: {{"action_type": "drag", "start": (x1,y1), "end": (x2,y2)}} where x and y are integers representing the point's horizontal and vertical screen positions normalized from (0,0) at the top left to (999,999) at the bottom right. The start indicates where the drag begins (where the user would initially long press the screen to focus on the element to drag). The end specifies where the dragged object is moved to before the finger is lifted (where the user releases the screen).
- Scrolling: To scroll the window, use this action: {{"action_type": "scroll", "direction": <"up", "down", "left", "right">, "distance: <"short", "medium", "long">}}. The direction determines the direction of the scroll and must be wrapped in double quotes. Scrolling down moves the content up to reveal what is hidden below the current view. The distance parameter determines the scrolling distance. Scrolling vertically is often used to explore more content in webpages while scrolling horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists.
- Key Combinations: To press a key combination: {{"action_type": "hotkey", "key_comb": "(key combination)"}}. The key_comb examples include Ctrl-c, Ctrl-S or Ctrl-Shift-1 with multiple keys combined with '-'.

Here are some useful guidelines you need to follow:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task and present the answer if required.
- Ensure the target element is visible on the screen for click/long_press. If not, explore the screen.

The user's task is: {global_task}
Action history: {history}
{step_instruction}
""" + COMPLEX_COT_REQUIREMENT


TWOSTAGE_GUIACTMOBILE_PLANNING_PROMPT_COT = PLANNING_PROMPT_HEAD.format(device='smartphone') + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful"}}
- For clicking an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- To swipe the screen, use this action: {{"action_type": "swipe", "direction": <"up", "down", "left", "right">}}. The direction determines the direction of the swipe and must be wrapped in double quotes. Note that swipe up will move the content upward and reveal the content below, which is equivalent to scrolling down.
- For pressing enter, use: {{"action_type": "press_key", "key": "enter"}}

Here are some useful guidelines you need to follow:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.
- Ensure the target element is visible on the screen for click/long_press. If not, explore the screen.

The user's task is: {global_task}
Action history: {history}
{step_instruction}
""" + COMPLEX_COT_REQUIREMENT

MIND2WEB_PLANNING_PROMPT_COT = PLANNING_PROMPT_HEAD.format(device='web') + """
- For clicking an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.
(Important notes!) Before outputting the click action, please use one sentence to describe the target element's distinctive functionality in detail, as this will help you locate the element accurately.
For example, if the target element is a link, you can describe its functionality as "This element navigates users to a page showing xxx (content)"; if the target is a dropdown menu button, describe its functionality as "This element enables users to access a navigation menu featuring various comedy shows". To ensure uniqueness, your functionality description should reflect the instance-specific context of the element whenever possible. For example, instead of predicting 'This element adds a product to the cart,' you should predict 'This element adds a blue shirt to the cart,' where 'blue shirt' is specific to the current instance. Similarly, rather than predicting 'This element facilitates the selection of an hour for the return time,' you should predict 'This element updates the return time to 13 p.m.' if such information is directly available. Ensure that the description remains accurate, grounded in visible data, and does not speculate on unseen values.
- For hovering over an element, use: {{"action_type": "hover", "target": (x,y)}}
- To focus on an input field and then type texts into it, use the 'input_text' action: {{"action_type": "input_text", "target": (x,y), "text": "(text to type)"}}. The target denotes the position of the input field in which to type texts. The text is the string you want to type.
- To select an item/option in a menu or selector, use the 'select' action: {{"action_type": "select", "target": (x,y), "value": "(the name of the selected item)"}}. The target denotes the position of the menu or selector from which to select an item. The value denotes the item name according to the user's task requirement. For example, you need to output {{"action_type": "select", "target": <position of a time selector>, "value": "12 30 PM"}} for the task "Rent a truck with on April 12 at 12:30 pm"
- For pressing enter, use: {{"action_type": "enter"}}

Here are some useful guidelines you need to follow:
Important notes:
1. All coordinates are normalized in the range of [0, 1000].
2. You do not need to click on an input field (e.g., search bar) before typing; simply output the correct input field position and the input_text action will automatically focus on it.
3. Similarly, clicking on a menu/selector element before selecting an item is FORBIDDEN. Just output the correct menu/selector element position and the select action will automatically focus on and reveal the desired menu.

The user's task is: {global_task}
Action history: {history}

""" + COMPLEX_COT_REQUIREMENT

AITW_PLANNING_PROMPT_FUNCDESC = PLANNING_PROMPT_HEAD.format(device='smartphone') +  """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful"}}
- If the task is infeasible (e.g., lack of information or inability to perform necessary actions), use: {{"action_type": "status", "goal_status": "infeasible"}}
- For clicking/tapping an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to ({xscale},{yscale}) at the bottom right.
(Important notes!) Before outputting the click action, please use one sentence to describe the target element's distinctive functionality in detail, as this will help you locate the element accurately.
For example, if the target element is a link, you can describe its functionality as "This element navigates users to a page showing xxx (content)"; if the target is a dropdown menu button, describe its functionality as "This element enables users to access a navigation menu featuring various comedy shows". To ensure uniqueness, your functionality description should reflect the instance-specific context of the element whenever possible. For example, instead of predicting 'This element adds a product to the cart,' you should predict 'This element adds a blue shirt to the cart,' where 'blue shirt' is specific to the current instance. Similarly, rather than predicting 'This element facilitates the selection of an hour for the return time,' you should predict 'This element updates the return time to 13 p.m.' if such information is directly available. Ensure that the description remains accurate, grounded in visible data, and does not speculate on unseen values.
- To swipe the screen, use this action: {{"action_type": "swipe", "direction": <"up", "down", "left", "right">}}. The direction determines the direction of the swipe and must be wrapped in double quotes. Note that swipe up will move the content upward and reveal the content below.
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- To press the Enter key, use the 'enter' action: {{"action_type": "enter"}}
- To navigate to the home screen: {{"action_type": "navigate_home"}}
- To navigate back: {{"action_type": "navigate_back"}}

Here are some useful guidelines you need to follow:
General:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.

Action-Related:
- Ensure the target element is visible on the screen when clicking. If not, explore the screen.
- Swiping vertically is often used to explore more content while swiping horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists. Swiping can also be used to adjust a slider, such as adjusting the volume on a media player interface.

The user's task is: {global_task}
Action history: {history}

""" + COMPLEX_COT_REQUIREMENT


AITW_PLANNING_XY_WITH_FUNCANNO_PROMPT = PLANNING_PROMPT_HEAD.format(device='web browser').replace("the current screenshot and a history", "the current screenshot on which interactable elements have been marked with boxes and numeric tags, as well as a history") + """
- If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {{"action_type": "status", "goal_status": "successful"}}
- If the task is infeasible (e.g., lack of information or inability to perform necessary actions), use: {{"action_type": "status", "goal_status": "infeasible"}}
- For clicking/tapping an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.
- To swipe the screen, use this action: {{"action_type": "swipe", "direction": <"up", "down", "left", "right">}}. The direction determines the direction of the swipe and must be wrapped in double quotes.
- To type texts into an input field/box, use the 'input_text' action: {{"action_type": "input_text", "text": "(text to type)"}}. The text is the string you want to insert. For example: {{"action_type": "input_text", "text": "Hello, world!"}} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- To press the Enter key, use the 'enter' action: {{"action_type": "enter"}}
- To navigate to the home screen: {{"action_type": "navigate_home"}}
- To navigate back: {{"action_type": "navigate_back"}}

Here are some useful guidelines you need to follow:
General:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.

Action-Related:
- Ensure the target element is visible on the screen for click. If not, explore the screen.
- Swiping vertically is often used to explore more content while swiping horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists. Swiping can also be used to adjust a slider, such as adjusting the volume on a media player interface.

The user's task is: {global_task}
Action history: {history}
The functionality descriptions of the marked elements:
{func_annos}
(Note that you DON'T need to choose from the marked elements to tap if tapping any one of them cannot advance towards task completion.)

Your output should include three parts in the given format:
Observation: <Describe the current screenshot and any notable elements.>
Thought: <Step-by-step reasoning towards the next action based on the observation, task and element functionalities. Do first determine whether the current screenshot represents the final state required by the task.>
Action: <Use one action in the list defined above. Only one action at a time without any comments.>
"""

# For UI-TARS
ANDROIDCONTROL_PLANNING_PROMPT_UI_TARS = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```\nThought: ...
Action: ...\n```

## Example
Thought: To ensure the search results are relevant to the task, I need to select "Andorra" from the dropdown menu. This will allow the system to filter the music library to show options available in Andorra, aligning with the task's requirements.
Click on the "Andorra" option in the dropdown menu to set the region for the search.
Action: click(start_box='(359,402)')

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>') # For clicking an element on the screen
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
type(content='')
press_home()
press_back()
open_app(app_name='') # To open an app
wait() # To wait for the content to load completely
finished(content='') # Submit the task regardless of whether it succeeds or fails.

## Note
- Use English in `Thought` part.
""" + """
## User Instruction
{global_task}"""

MIND2WEB_PLANNING_PROMPT_UI_TARS = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```\nThought: ...
Action: ...\n```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>') # For clicking an element on the screen
type(start_box='<|box_start|>(x1,y1)<|box_end|>', content='') # To focus on an input field and then type texts into it. The start_box denotes the position of the input field in which to type texts. The text is the string you want to type.
select(start_box='<|box_start|>(x1,y1)<|box_end|>', option='') # To select an item/option in a menu. The start_box denotes the position of the menu or selector from which to select an item. The option denotes the item name in the menu according to the user's task requirement.

## Note
- Use English in `Thought` part.
""" + """
## User Instruction
{global_task}"""


planning_prompt_web = PLANNING_PROMPT_HEAD.format(device='web browser') + """
- Task Completion: If you believe the task is complete, finish the task by using the "status" action with "successful" as goal status: {"action_type": "status", "goal_status": "successful", "answer": "(answer to the task)"}. If the task requires you to present a textual answer, please provide the answer in the "answer" field.
- Infeasibility: If you find the task infeasible (including cases like you don't have enough information or cannot perform necessary actions), finish the task by using the "status" action with "infeasible" as goal status: {"action_type": "status", "goal_status": "infeasible"}
- Clicking Elements: To click on an element on the screen, use the 'click' action specifying the element's location: {"action_type": "click", "target": (x,y)}. (Important!) x and y denote the specific point of interest on the screen. Here x represents the point's ratio along the screen's width multiplied by 100, and y represents the point's ratio along the screen's height, also multiplied by 100. The top-left corner is (0,0) and the bottom-right corner is (100,100). Both x and y must be integers between 0 and 100.
- Moving and Hovering: To move the pointer to a location or hover on an element, use the 'move_to' action while specifying the target location: {"action_type": "move_to", "target": (x,y)}
- Dragging: To drag an element, use: {"action_type": "drag", "start": (x1,y1), "end": (x2,y2)}. The start indicates where the drag begins (where the user would initially touch the screen to focus on the element to drag). The end specifies where the dragged object is moved to before the finger is lifted (where the user releases the screen). Note that this drag action allows the touch point (or cursor) to move freely to any location on the screen, which makes drag actions suitable for tasks like positioning objects within a workspace, drawing, or precisely adjusting sliders.
- Text Entry: To type text into an input field, use: {"action_type": "input_text", "text": "(text to type)"}. The text is the string you want to insert. For example: {"action_type": "input_text", "text": "Hello, world!"} will insert the text "Hello, world!" into the input field. Ensure to use a 'click' action to focus on the input field before using the 'input_text' action.
- Key Presses: For pressing a key: {"action_type": "press_key", "key": "(key name)"}. This action simulates pressing a key down and then releasing it. Example keys include 'enter', 'shift', arrow keys, or function keys.
- Key Combinations: To press a key combination: {"action_type": "hotkey", "key_comb": "(key combination)"}. The key_comb examples include Ctrl-S or Ctrl-Shift-1 with multiple keys combined with '-'.
- Scrolling: To scroll the window, use this action: {"action_type": "scroll", "direction": <"up", "down", "left", "right">, "distance: <"short", "medium", "long">}. The direction determines the direction of the scroll and must be wrapped in double quotes. Scrolling down moves the content up to reveal what is hidden below the current view. The distance parameter determines the scrolling distance. Scrolling vertically is often used to explore more content in webpages while scrolling horizontally is often used to navigate through horizontally scrollable content such as image carousels, multi-page layouts, or lists.
- To navigate back: {"action_type": "navigate_back"}
- To undo the navigate_back action: {"action_type": "navigate_forward"}
- To go to a specific URL: {"action_type": "go_to", "url": "(a certain url)"}
- To search Google for a query: {"action_type": "search_google", "query": "(search query)"}
- To open a new browser tab: {"action_type": "new_tab"}
- To switch the browser's focus to a specific tab using its index: {"action_type": "switch_tab", "tab": "(tab index)"}

Here are some useful guidelines you need to follow:
General:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), complete the task.

- Use the 'input_text' action for any text entry (including passwords), rather than clicking individual keyboard characters.
- Ensure that the elements you interact with are VISIBLE in the screenshot for 'click', 'long_press', and 'drag' actions.
- If necessary content is not visible, consider using the 'swipe' action in different directions to explore the screen further.

The user's task is: {global_task}
Action history: {history}
{step_instruction}
"""

GUICOURSE_PROMPT = """Actions History
{history}
Information
This screenshot shows a GUI.
Your Task
{goal}
Generate next actions to do this task."""

AGUVIS_PROMPT = """Please generate the next move according to the UI screenshot, instruction and previous actions.
Instruction: {global_task}
Previous actions: {history}"""


MIND2WEB_PLANNING_PROMPT = PLANNING_PROMPT_HEAD.format(device='web browser') + """
- For clicking an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.
- For hovering over an element, use: {{"action_type": "hover", "target": (x,y)}}
- To focus on an input field and then type texts into it, use the 'input_text' action: {{"action_type": "input_text", "target": (x,y), "text": "(text to type)"}}. The target denotes the position of the input field in which to type texts. The text is the string you want to type.
- To select an item/option in a menu or selector, use the 'select' action: {{"action_type": "select", "target": (x,y), "value": "(the name of the selected item)"}}. The target denotes the position of the menu or selector from which to select an item. The value denotes the item name according to the user's task requirement. For example, you need to output {{"action_type": "select", "target": <position of a time selector>, "value": "12 30 PM"}} for the task "Rent a truck with on April 12 at 12:30 pm"
- For pressing enter, use: {{"action_type": "enter"}}

The user's task is: {global_task}
Action history: {history}

Important notes:
1. You MUST NOT click on an input field (e.g., search bar) before typing; simply output the correct input field position and the input_text action will automatically focus on it.
2. Similarly, clicking on a menu/selector element before selecting an item is FORBIDDEN. Just output the correct menu/selector element position and the select action will automatically focus on and reveal the desired menu.

Your output should include three parts in the given format:
Observation: <Describe the current screenshot and any notable elements>
Thought: <Step-by-step reasoning towards the next action based on the observation, task, history, and important notes.>
Action: <Use one action in the list defined above. Only one action at a time without any comments. If you want to type texts or select an item, directly use input_text or select and do not use click to focus on the target.>"""

SIMPLE_PROMPT1 = """Given the UI screenshot and previous actions, please generate the next move necessary to advance towards task completion. The user's task is: {global_task}
Action history: {history}
{step_instruction}
Now, directly plan the next action to advance toward task completion using the above action list. Your output must follow this format:
Action: (action prediction in the correct JSON format)"""

TURN_GND_INTO_PLANNING_PROMPT = 'click on the the element described by "{instruc}"'



SEARCH_QUERIES = [
    # Shopping
    "black nike running shoes",
    "iphone 13 pro case",
    "wireless headphones under $100",
    "men's winter jacket",
    "gaming laptop deals",
    "organic coffee beans",
    "yoga mat non slip",
    "kitchen knife set",
    "air fryer 6 quart",
    "smart watch fitness tracker",
    
    # Travel
    "hotels in new york city",
    "cheap flights to london",
    "best restaurants near me",
    "car rental orlando airport",
    "paris tourist attractions",
    "beach resorts caribbean",
    "hiking trails colorado",
    "tokyo travel guide",
    "vacation packages all inclusive",
    "best time to visit italy",
    
    # Entertainment
    "new movies 2024",
    "concert tickets taylor swift",
    "netflix series recommendations",
    "ps5 games on sale",
    "spotify premium family plan",
    "best books 2023",
    "online cooking classes",
    "live sports streaming",
    "comedy shows near me",
    "music festivals summer 2024",
    
    # Services
    "plumber emergency 24/7",
    "hair salon appointments",
    "dog grooming services",
    "house cleaning quote",
    "car mechanic reviews",
    "dentist accepting new patients",
    "yoga classes beginners",
    "moving companies quotes",
    "lawn care service",
    "wedding photographer portfolio",
    
    # Information
    "weather forecast weekend",
    "covid testing locations",
    "how to fix wifi issues",
    "healthy dinner recipes quick",
    "job openings software developer",
    "apartment rentals downtown",
    "doctors near me reviews",
    "math tutor online",
    "tax preparation services",
    "language learning apps"
]

NAVIGATE_ACTION_PREFIXES = [
    "Navigate to",
    "Go to",
    "Move to",
    "Proceed to",
    "Head to",
    "Direct to",
    "Travel to",
    "Advance to",
    "Make your way to",
    "Transition to",
]

EXPLORE_ACTION_PREFIXES = [
    "Swipe to",
    "Scroll to",
    "Explore to",
    "Search for",
    "Look through",
    "Browse to",
]

SWIPE_PHRASES = [
    "Swipe {direction} to uncover more content",
    "Swipe {direction} to locate the desired target",
    "Swipe {direction} to pinpoint the target element",
    "Swipe {direction} to reach the target page",
    "Swipe {direction} to discover additional content",
    "Swipe {direction} to reveal the target",
    "Swipe {direction} to access the target element",
    "Swipe {direction} to find the page you need"
]

DRAG_PHRASES = {
        'specific':[
            'Drag {target} to the target position',
        ]
    }
# Referring Expressions for 'navigate_back'
NAVIGATE_BACK_PREFIXES = [
    "Return to the previous page",
    "Go back to the last screen",
    "Navigate back to the prior view",
    "Move back to the previous interface",
    "Reverse to the last page",
    "Step back to the previous screen",
    "Backtrack to the last view",
    "Go back to the former page",
    "Retrace to the previous screen",
    "Shift back to the last interface"
]

# Referring Expressions for 'navigate_home'
NAVIGATE_HOME_PREFIXES = [
    "Return to the home screen.",
    "Go back to the main page.",
    "Navigate to the home interface.",
    "Move back to the home view.",
    "Head to the main screen.",
    "Switch to the home page.",
    "Redirect to the home interface.",
    "Shift to the primary screen.",
    "Take you back to the home page."
]

# Referring Expressions for 'press_key'
PRESSKEY_PREFIXES = {
    'Enter': [
        "Press the Enter key to submit.",
        "Tap the Enter key to submit.",
        "Press the Enter key to confirm.",
        "Tap the Enter key to confirm.",
        "Press the Enter key to confirm the input.",
        "Tap the Enter key to confirm the input.",
    ],
    'Back': [
        "Press the Back key to go back.",
        "Tap the Back key to go back.",
        "Press the Back key to return.",
        "Tap the Back key to return.",
        "Press the Back key to navigate back.",
        "Tap the Back key to navigate back.",
    ],
    'left': [
        'Press Left key.',
    ],
    'right': [
        'Press Right key.',
    ],
    'up': [
        'Press Up key.',
    ],
    'down': [
        'Press Down key.',
    ],
    'pgdn': [
        "Press Page Down key.",
        "Tap Page Down key.",
        "Press Page Down key to scroll down.",
        "Tap Page Down key to scroll down.",
        "Press Page Down key to navigate to the next page.",
        "Tap Page Down key to navigate to the next page."
    ],
    'pgup': [
        "Press Page Up key.",
        "Tap Page Up key.",
        "Press Page Up key to scroll up.",
        "Tap Page Up key to scroll up.",
        "Press Page Up key to navigate to the previous page.",
        "Tap Page Up key to navigate to the previous page."
    ],
    'esc': [
        "Press Escape key.",
        "Tap Escape key.",
        "Press Escape key to exit.",
        "Tap Escape key to exit."
    ],
    'space': [
        "Press Space key.",
        "Tap Space key.",
    ]
}

KEYCOMB_PREFIXES = {
    'ctrl-a': [
        "Press Ctrl+A to select all text.",
        "Tap Ctrl+A to select all text.",
        "Press Ctrl+A to highlight all text.",
        "Tap Ctrl+A to highlight all text."
    ],
    'ctrl-c': [
        "Press Ctrl+C to copy the text.",
        "Tap Ctrl+C to copy the text.",
    ],
    'ctrl-s': [
        "Press Ctrl+S to save the file.",
        "Tap Ctrl+S to save the file.",
        "Press Ctrl+S to save the document.",
        "Tap Ctrl+S to save the document."
    ],
    'ctrl-z': [
        "Press Ctrl+Z to undo the last action.",
        "Tap Ctrl+Z to undo the last action.",
        "Press Ctrl+Z to reverse the last action.",
        "Tap Ctrl+Z to reverse the last action."
    ],
    'ctrl-shift-n': [
        "Press Ctrl+Shift+N to create a new folder.",
    ],
    'win-d': [
        "Press Win+D to show the desktop.",
        "Press Win+D to minimize all windows.",
        "Use Win+D to display the desktop.",
        "Press Win+D to hide all windows.",
        "Use Win+D to minimize everything and show desktop."
    ],
    'winleft-s': [
        "Press Win+Left+S to show the start menu.",
        "Tap Win+Left+S to show the start menu.",
        "Activate start menu."
    ],
    'winleft-a': [
        "Press Win+Left+A to show the action center.",
        "Tap Win+Left+A to show the action center.",
        "Activate action center."
    ],
    'ctrl-h': [
        "Press Ctrl+H to show hidden files and folders.",
        "Tap Ctrl+H to display hidden files and folders.",
        "Press Ctrl+H to reveal hidden files and folders.",
        "Tap Ctrl+H to reveal hidden files and folders."
    ],
    'ctrl-t': [
        "Press Ctrl+T to open a new tab.",
        "Tap Ctrl+T to open a new tab.",
        "Press Ctrl+T to create a new tab.",
        "Tap Ctrl+T to create a new tab."
    ],
    'ctrl-down': [
        "Press Ctrl+Down to scroll down.",
        "Tap Ctrl+Down to scroll down.",
        "Press Ctrl+Down to navigate to the next page.",
        "Tap Ctrl+Down to navigate to the next page."
    ],
    'ctrl-up': [
        "Press Ctrl+Up to scroll up.",
        "Tap Ctrl+Up to scroll up.",
        "Press Ctrl+Up to navigate to the previous page.",
        "Tap Ctrl+Up to navigate to the previous page."
    ],
    'ctrl-+': [
        "Press Ctrl+Plus to zoom in.",
        "Tap Ctrl+Plus to zoom in.",
        "Press Ctrl+Plus to increase the zoom level.",
        "Tap Ctrl+Plus to make the screen display bigger."
    ],
    'ctrl-f': [
        "Tap Ctrl+F to search.",
        "Press Ctrl+F to find the target.",
        "Find all occurances of the word in the current document."
    ],
    'ctrl--': [
        "Press Ctrl+Minus to zoom out.",
        "Tap Ctrl+Minus to zoom out.",
        "Press Ctrl+Minus to decrease the zoom level.",
        "Tap Ctrl+Minus to make the screen display smaller."
    ],
    'ctrl-w': [
        "Press Ctrl+W to close the current tab.",
        "Tap Ctrl+W to close the current window.",
    ],
    'ctrl-y': [
        "Press Ctrl+Y to redo the last action.",
        "Tap Ctrl+Y to redo the last action.",
        "Press Ctrl+Y to repeat the last action.",
        "Tap Ctrl+Y to repeat the last action."
    ],
    'alt-f4': [
        "Press Alt+F4 to close the current window.",
        "Tap Alt+F4 to close the current window.",
        "Press Alt+F4 to shut down the current application.",
        "Tap Alt+F4 to shut down the current application.",
        "Instantly exit the current window."
    ],
    'ctrl-shift-tab': [
        "Press Ctrl+Shift+Tab to switch to the previous tab.",
        "Tap Ctrl+Shift+Tab to switch to the previous tab.",
        "Press Ctrl+Shift+Tab to navigate to the previous tab.",
        "Tap Ctrl+Shift+Tab to go to the previous tab."
    ],
    'ctrl-tab': [
        "Press Ctrl+Tab to switch to the next tab.",
        "Tap Ctrl+Tab to switch to the next tab.",
        "Press Ctrl+Tab to navigate to the next tab.",
        "Tap Ctrl+Tab to navigate to the next tab."
    ]
}

# Referring Expressions for 'open_app'
OPEN_APP_PREFIXES = [
    "Launch the app {app_name}",
    "Start the app {app_name}",
    "Open the app {app_name}",
    "Run the app {app_name}",
    "Invoke the app {app_name}",
    "Boot up the app {app_name}",
    "Fire up the app {app_name}",
    "Turn on the app {app_name}",
]

INTERACT_ACTION_PREFIXES = [
    "Click on",
    "Tap on",
    "Select",
    "Choose",
    "Activate",
    "Engage with",
]

INPUT_ACTION_PREFIXES = [
    "Type into",
    "Enter text in",
    "Fill in",
    "Input data into",
    "Provide input in",
]

INPUT_ACTION_PREFIXES_WITH_TEXT = {'vague': [
    'Fill "{text}" in {target}',
    'Input data "{text}" into {target}',
    'Provide input "{text}" in {target}',
    'Search for "{text}"',
], 'specific': [
    'Type "{text}" into {target}',
    'Enter text "{text}" in {target}',
    'Fill "{text}" in {target}',
    'Input text "{text}" into {target}',
]}

SELECT_ACTION_PREFIXES_WITH_TEXT = {'vague': [
    'Select the option "{text}" in {target}',
    'Choose the option "{text}" in {target}',
    'Select the "{text}" option in {target}',
    'Choose the "{text}" option in {target}',
    
], 'specific': [
    'Select the option "{text}" from the "{target}" dropdown',
    'Choose "{text}" from the "{target}" menu',
    'Click "{text}" in the "{target}" selection box', 
    'Set the "{target}" dropdown to "{text}"',
    'Change the "{target}" selection to "{text}"',
    'Pick the "{text}" option from the "{target}" menu'
]}

TASK_STATUS_SENTENCES = {
    'successful': [
        "We have successfully finished the task.",
        "We have successfully completed the task.",
        "The task has been completed without issues.",
        "Successfully completed the task.",
        "We have successfully finished the task.",
        "The task has been completed successfully."
        ],
    'infeasible': [
        "The task is infeasible.",
        "The task is impossible to complete.",
        "The task is not possible to complete.",
        "The task is not feasible.",
        "The task is not doable.",
    ]
}

CONFIRM_ACTION_PREFIXES = [
    "Confirm",
    "Submit",
    "Approve",
    "Validate",
    "Authorize",
]

HISTORY_FAKED = [
    "Step 1. {} the user-specified page.".format(random.choice(NAVIGATE_ACTION_PREFIXES)),
    "Step 1. {} the target element.".format(random.choice(EXPLORE_ACTION_PREFIXES)),
    "Step 1. {} the user-specified page. Step 2. {} the target element.".format(random.choice(NAVIGATE_ACTION_PREFIXES), random.choice(EXPLORE_ACTION_PREFIXES)),
    "Step 1. {} the user-specified page. Step 2. {} the intermediate element. Step 3. {} the changes.".format(
        random.choice(NAVIGATE_ACTION_PREFIXES), random.choice(EXPLORE_ACTION_PREFIXES), random.choice(CONFIRM_ACTION_PREFIXES)
    ),
    "Step 1. {} the user-specified page. Step 2. {} the necessary information. Step 3. {} the changes.".format(
        random.choice(NAVIGATE_ACTION_PREFIXES), random.choice(INPUT_ACTION_PREFIXES), random.choice(CONFIRM_ACTION_PREFIXES)
    ),
]


SIMPLE_PROMPT_FOR_GND_BASIC = """Given the UI screenshot and previous actions, please generate the next move necessary to advance towards task completion. The user's task is: {global_task}
Action history: {history}
"""

SIMPLE_PROMPT_FOR_GND_COT = "first describe the action intent and then " 

def get_gnd2planning_prompt(global_task, faked_history=True, cot=False, multi_action=False):
    if faked_history:
        history = random.choice(HISTORY_FAKED)
    else:
        history = ""

    if multi_action:
        prompt = SIMPLE_PROMPT_FOR_GND_BASIC + "\nNow, {cot}plan the next multiple actions."
    else:
        prompt = SIMPLE_PROMPT_FOR_GND_BASIC + "\nNow, {cot}directly plan the next action."
    
    return prompt.format(global_task=global_task, history=history, cot=SIMPLE_PROMPT_FOR_GND_COT if cot else "").strip()

ACTION_OUTPUT_FORMAT = "Now, directly plan the next action."
REFEXP_ACTION_OUTPUT_FORMAT = "Now, first describe the action intent and then directly plan the next action."
COT_ACTION_OUTPUT_FORMAT = "Now, think step-by-step, get yourself familiar with the UI content, then analyze the task at hand and previous actions, and finally plan the next action."

SIMPLE_PROMPT_BASIC = """Given the UI screenshot and previous actions, please generate the next move necessary to advance towards task completion. The user's task is: {global_task}
Action history: {history}
{step_instruction}
{action_output_format}"""

PLANNING_PROMPT_END = """The current screenshot has been provided. Now, directly plan the next action to advance toward task completion using the above action list. Your output must follow this format:
Action: (action prediction in the correct JSON format)"""

REASONING_PLANNING_PROMPT_END = """The current screenshot has been provided. Now, think step-by-step, first describe the screenshot to get your self familiar with it, then present a detailed resoning about what to do next, and finally plan the next action to advance toward task completion using the above action list. Your output must follow this format:
Screen descrition: (describe the screen in detail)
Thought: (logic behind next step planning)
Action: (action prediction in the correct JSON format)"""

ATLAS_PROMPT = "Task: {global_task}{step_instruction}\nHistory: \n{history}\n"

MAX_PREV_ACT = 6

def make_actionplanning_prompt(global_task, history, device_tag='', step_instruction='', prompt_format_type='simple', with_cot=False, without_action_space=True, use_action_refexp=False):
    if prompt_format_type == 'simple':
        step_instruction_str = f"The next step instruction: {step_instruction}\n" if step_instruction else ""
        if without_action_space:
            if with_cot:
                action_output_format = COT_ACTION_OUTPUT_FORMAT
            elif use_action_refexp:
                action_output_format = REFEXP_ACTION_OUTPUT_FORMAT
            else:
                action_output_format = ACTION_OUTPUT_FORMAT
            prompt = SIMPLE_PROMPT_BASIC.replace("{global_task}", global_task).replace("{history}", history).replace("{step_instruction}", step_instruction_str).replace("{action_output_format}", action_output_format)
        else:    
            prompt = planning_prompt_android.replace("{global_task}", global_task).replace("{history}", history).replace("{step_instruction}", step_instruction_str) + (REASONING_PLANNING_PROMPT_END if with_cot else PLANNING_PROMPT_END)
    elif prompt_format_type == 'aguvis':
        prompt = AGUVIS_PROMPT.replace("{global_task}", global_task).replace("{history}", history)
    elif prompt_format_type == 'atlas':
        step_instruc_str = f" You need to: {step_instruction.strip(' .')}." if step_instruction else ""
        prompt = ATLAS_PROMPT.format(global_task=global_task, history=history, step_instruction=step_instruc_str)        

    if device_tag:
        prompt = prompt.replace("UI screenshot", f"{device_tag} UI screenshot")
    
    return prompt

def make_actionplanning_sample(task_id, global_task, history, gt_action, device_tag='', step_instruction='', prompt_format_type='simple', with_cot=False, without_action_space=True, use_action_refexp=False):
    if prompt_format_type == 'simple':
        if without_action_space:
            if with_cot:
                action_output_format = COT_ACTION_OUTPUT_FORMAT
            elif use_action_refexp:
                action_output_format = REFEXP_ACTION_OUTPUT_FORMAT
            else:
                action_output_format = ACTION_OUTPUT_FORMAT
            prompt = SIMPLE_PROMPT_BASIC.replace("{global_task}", global_task).replace("{history}", history).replace("{step_instruction}", step_instruction).replace("{action_output_format}", action_output_format)
        else:    
            prompt = planning_prompt_android.replace("{global_task}", global_task).replace("{history}", history).replace("{step_instruction}", step_instruction) + (REASONING_PLANNING_PROMPT_END if with_cot else PLANNING_PROMPT_END)
    elif prompt_format_type == 'aguvis':
        prompt = AGUVIS_PROMPT.replace("{global_task}", global_task).replace("{history}", history)
    
    if device_tag:
        prompt = prompt.replace("UI screenshot", f"{device_tag} UI screenshot")
    
    conv = [
            {
                "from": "human",
                "value": f"<image>\n{prompt}"
            },
            {
                "from": "gpt",
                "value": gt_action
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

def make_actionplanning_sample_web(task_id, global_task, history, gt_action, device_tag='', step_instruction='', prompt_format_type='simple', with_cot=False, without_action_space=True, use_action_refexp=False):
    if without_action_space:
        if with_cot:
            action_output_format = COT_ACTION_OUTPUT_FORMAT
        elif use_action_refexp:
            action_output_format = REFEXP_ACTION_OUTPUT_FORMAT
        else:
            action_output_format = ACTION_OUTPUT_FORMAT
        prompt_raw = SIMPLE_PROMPT_BASIC.replace("{global_task}", global_task).replace("{history}", history).replace("{step_instruction}", step_instruction).replace("{action_output_format}", action_output_format)
    else:
        prompt_raw = planning_prompt_web + (REASONING_PLANNING_PROMPT_END if with_cot else PLANNING_PROMPT_END)
    
    prompt = prompt_raw.replace("{global_task}", global_task).replace("{history}", history).replace("{step_instruction}", step_instruction) 

    if device_tag:
        prompt = prompt.replace("UI screenshot", f"{device_tag} UI screenshot")

    conv = [
            {
                "from": "human",
                "value": f"<image>\n{prompt}"
            },
            {
                "from": "gpt",
                "value": gt_action
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

OMNIACT_PROMPT = """Given the UI screenshot, please generate the first move necessary to advance towards task completion. The user's task is: {global_task}
Now, directly plan the first action."""

def make_actionplanning_sample_desktop(task_id, global_task, gt_action):
    prompt = OMNIACT_PROMPT.replace("{global_task}", global_task) 
    
    conv = [
            {
                "from": "human",
                "value": f"<image>\n{prompt}"
            },
            {
                "from": "gpt",
                "value": gt_action
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# Basic action space
ANSWER_TEMPLATE = '{{"action_type": "answer", "text": "{text}"}}'

STATUS_TEMPLATE = '{{"action_type": "status", "goal_status": "{goal_status}", "answer": "{answer}"}}'

CLICK_TEMPLATE = '{{"action_type": "click", "target": ({target_x},{target_y})}}'

INPUT_TEMPLATE = '{{"action_type": "input_text", "text": "{text}"}}'

ENTER_TEMPLATE = '{"action_type": "enter"}'

NAVIGATE_BACK_TEMPLATE = '{"action_type": "navigate_back"}'

NAVIGATE_HOME_TEMPLATE = '{"action_type": "navigate_home"}'

NAVIGATE_RECENT_TEMPLATE = '{"action_type": "navigate_recent"}'

DRAG_TEMPLATE = '{{"action_type": "drag", "start": ({start_x},{start_y}), "end": ({end_x},{end_y})}}'
MOVETO_TEMPLATE = '{{"action_type": "move_to", "target": ({target_x},{target_y})}}'

OPEN_APP_TEMPLATE = '{{"action_type": "open_app", "app_name": "{app_name}"}}'
# Mobile action space
SWIPE_TEMPLATE = '{{"action_type": "swipe", "start": ({start_x},{start_y}), "direction": "{direction}", "distance": "{distance}"}}'

LONG_PRESS_TEMPLATE = '{{"action_type": "long_press", "target": ({target_x},{target_y})}}'

WAIT_TEMPLATE = '{"action_type": "wait"}'
WAIT_INSTRUC = [
    'Allow sufficient time for the content to fully load before proceeding.',
    'Pause and wait for the content to finish loading.',
    'Wait for the loading process to complete.',
    'Remain in a wait state until the content has fully loaded.',
    'Remain idle until the content has finished loading.'
]
# Web action space
SCROLL_TEMPLATE = '{{"action_type": "scroll", "direction": "{direction}", "distance": "{distance}"}}'
SIMPLE_SCROLL_TEMPLATE = '{{"action_type": "scroll", "direction": "{direction}"}}'

SELECT_TEMPLATE = '{{"action_type": "select", "target": ({target_x},{target_y}), "value": "{value}"}}'

HOVER_TEMPLATE = '{{"action_type": "hover", "target": ({target_x},{target_y})}}'

KEYCOMB_TEMPLATE = '{{"action_type": "hotkey", "key_comb": "{key_combination}"}}'

INPUT_TARGET_TEMPLATE = '{{"action_type": "input_text", "target": ({target_x},{target_y}), "text": "{text}"}}'

# DESKTOP
RIGHTCLICK_TEMPLATE = '{{"action_type": "right_click", "target": ({target_x},{target_y})}}'

DOUBLECLICK_TEMPLATE = '{{"action_type": "double_click", "target": ({target_x},{target_y})}}'

PRESSKEY_TEMPLATE = '{{"action_type": "press_key", "key": "{key}"}}'

# QWEN action space
CLICK_TEMPLATE_QWEN = '{{"action": "click", "coordinate": [{target_x}, {target_y}]}}'
# LONG_PRESS_TEMPLATE_QWEN = '{{"action": "long_press", "coordinate": [{target_x}, {target_y}], "duration": {duration}}}'
LONG_PRESS_TEMPLATE_QWEN = '{{"action": "long_press", "coordinate": [{target_x}, {target_y}]}}'

SWIPE_TEMPLATE_QWEN = '{{"action": "swipe", "coordinate": [{target_x}, {target_y}], "coordinate2": [{target_x2}, {target_y2}]}}'
INPUT_TEMPLATE_QWEN = '{{"action": "type", "coordinate": [{target_x}, {target_y}], "text": "{text}"}}'
INPUT_TEMPLATE_QWEN_NO_COORD = '{{"action": "type", "text": "{text}"}}'

SYSTEM_BUTTON_TEMPLATE_QWEN = '{{"action": "system_button", "button": "{button}"}}'
OPEN_APP_TEMPLATE_QWEN = '{{"action": "open", "text": "{app_name}"}}'
TERMINATE_TEMPLATE_QWEN = '{{"action": "terminate", "status": "{status}"}}'
ANSWER_TEMPLATE_QWEN = '{{"action": "answer", "text": "{answer}"}}'
WAIT_TEMPLATE_QWEN = '{{"action": "wait", "time": {time}}}'

def to_qwen_action(action_json, last_action, W, H):
    """Convert JSONAction to QwenAction"""
    action_type = action_json.get('action_type', 'action')

    # use unnormalized xy
    if 'target' in action_json:
        target = action_json['target']
        target = [round(target[0] / 1000 * W), round(target[1] / 1000 * H)]
    elif 'start' in action_json:
        start = action_json['start']
        start = [round(start[0] / 1000 * W), round(start[1] / 1000 * H)]
    
    if action_type == 'input_text':
        if last_action is not None and 'x' in last_action:
            target = round(last_action['x'] / 1000 * W), round(last_action['y'] / 1000 * H)
        else:
            return None, None

    if action_type == 'click':
        act = CLICK_TEMPLATE_QWEN.format(target_x=target[0], target_y=target[1])
    elif action_type == 'long_press':
        act = LONG_PRESS_TEMPLATE_QWEN.format(target_x=target[0], target_y=target[1], duration=action_json.get('duration', 2))
    elif action_type == 'swipe':
        if 'coordinate2' in action_json:
            start_x, start_y = action_json['coordinate']
            end_x, end_y = action_json['coordinate2']
        elif 'start' in action_json and isinstance(action_json['start'], tuple):
            start_x, start_y = start
            
            # Calculate end coordinates based on direction and distance
            direction = action_json.get('direction', 'right')
            distance_str = 'long' # action_json.get('distance', 'medium')
            
            # Convert distance string to numeric value
            distance_map = {'short': 0.2, 'medium': 0.4, 'long': 0.7}
            distance = distance_map.get(distance_str, 0.4)
            
            if direction == 'down':
                end_x, end_y = start_x, min(start_y + int(H * distance), H)
            elif direction == 'up':
                end_x, end_y = start_x, max(start_y - int(H * distance), 0)
            elif direction == 'left':
                end_x, end_y = max(start_x - int(W * distance), 0), start_y
            elif direction == 'right':
                end_x, end_y = min(start_x + int(W * distance), W), start_y
        else:
            direction = action_json.get('direction', 'right')
            mid_x, mid_y = W // 2, H // 2
            if direction == 'down':
                start_x, start_y = mid_x, H // 10
                end_x, end_y = mid_x, H // 10 * 9
            elif direction == 'up':
                start_x, start_y = mid_x, H // 10 * 9
                end_x, end_y = mid_x, H // 10
            elif direction == 'left':
                start_x, start_y = W // 10 * 9, mid_y
                end_x, end_y = W // 10, mid_y
            elif direction == 'right':
                start_x, start_y = W // 10, mid_y
                end_x, end_y = W // 10 * 9, mid_y

        act = SWIPE_TEMPLATE_QWEN.format(target_x=start_x, target_y=start_y, target_x2=end_x, target_y2=end_y)
    elif action_type == 'input_text':
        text = action_json.get('text', '')
        if text.count('"') % 2 != 0:
            text = text.strip('"')
        elif text.count("'") % 2 != 0:
            text = text.strip("'")

        if target is None:
            p = INPUT_TEMPLATE_QWEN.replace('"coordinate": [{target_x}, {target_y}], ', '')
            act = p.format(text=text)
        else:
            p = INPUT_TEMPLATE_QWEN
            act = p.format(target_x=target[0], target_y=target[1], text=text)
    elif action_type == 'navigate_back':
        act = SYSTEM_BUTTON_TEMPLATE_QWEN.format(button='BACK')
    elif action_type == 'navigate_home':
        act = SYSTEM_BUTTON_TEMPLATE_QWEN.format(button='HOME')
    elif action_type == 'press_key':
        act = SYSTEM_BUTTON_TEMPLATE_QWEN.format(button=action_json['key'])
    elif action_type in ['press_enter', 'enter']:
        act = SYSTEM_BUTTON_TEMPLATE_QWEN.format(button='ENTER')
    elif action_type == 'open_app':
        act = OPEN_APP_TEMPLATE_QWEN.format(app_name=action_json.get('app_name', action_json.get('text', '')))
    elif action_type == 'status':
        assert action_json['goal_status'] in ['successful', 'failed']
        act = TERMINATE_TEMPLATE_QWEN.format(status='success' if action_json['goal_status'] == 'successful' else 'failure')
    elif action_type == 'answer':
        ans = action_json.get('answer', action_json.get('text', ''))
        act = ANSWER_TEMPLATE_QWEN.format(answer=ans)
    elif action_type == 'wait':
        act = WAIT_TEMPLATE_QWEN.format(time=action_json.get('time', 2))
    else:
        raise ValueError(f"Unsupported action type: {action_type}")
    return act, json.loads(act)

def qwen2p5_to_original_action(action: dict | str, w: int, h: int):
    # The reverse function of to_showui_action
    if isinstance(action, str): action = ast.literal_eval(action)

    action_type = action['action']
    if (coordinate := action.get('coordinate', None)):
        target = [coordinate[0] / w, coordinate[1] / h]

    if action_type == 'click':
        return CLICK_TEMPLATE.format(target_x=target[0], target_y=target[1])
    elif action_type == 'hover':
        return HOVER_TEMPLATE.format(target_x=target[0], target_y=target[1])
    elif action_type == 'long_press':
        return LONG_PRESS_TEMPLATE.format(target_x=target[0], target_y=target[1])
    elif action_type == 'swipe':
        start, end = action['coordinate'], action['coordinate2']
        norm_from_x, norm_from_y = [start[0] / w, start[1] / h]
        norm_to_x, norm_to_y = [end[0] / w, end[1] / h]
        vertical_shift, horizontal_shift = norm_to_y - norm_from_y, norm_to_x - norm_from_x

        # judged the scrolling direction
        if abs(vertical_shift) > abs(horizontal_shift):
            direction = 'down' if vertical_shift > 0 else 'up'
            distance = discretize_dist(abs(vertical_shift))
        else:
            direction = 'right' if horizontal_shift > 0 else 'left'
            distance = discretize_dist(abs(horizontal_shift))

        return SWIPE_TEMPLATE.format(start_x=norm_from_x, start_y=norm_from_y, direction=direction, distance=distance)
    elif action_type == 'drag':
        start, end = target
        norm_from_x, norm_from_y = start
        norm_to_x, norm_to_y = end
        return DRAG_TEMPLATE.format(start_x=norm_from_x, start_y=norm_from_y, end_x=norm_to_x, end_y=norm_to_y)
    elif action_type in ['type', 'input_text']:
            return INPUT_TEMPLATE.format(text=action['value'])
    elif action_type == 'system_button':
        if action['button'] == 'BACK':
            return NAVIGATE_BACK_TEMPLATE
        elif action['button'] == 'HOME':
            return NAVIGATE_HOME_TEMPLATE
        elif action['button'] == 'ENTER':
            return ENTER_TEMPLATE
    elif action_type == 'terminate':
        return STATUS_TEMPLATE.format(goal_status='successful' if 'success' in action['status']  else 'infeasible', answer='')
    else:
        raise ValueError(f"Unsupported action type: {action_type}")

def to_qwen_action_list(actions: list[str], Ws: list[int], Hs: list[int], skipped_idxs: Optional[list[int]] = None):
    last_actions = [None] + actions[:-1]
    return [to_qwen_action(action, last_action, W, H) if i not in skipped_idxs else action for i, (action, last_action, W, H) in enumerate(zip(actions, last_actions, Ws, Hs))]

ACTION_PREFIXES = {
    'click': {
        'vague':[
        'Select',
        'Activate',
        'Touch',
        'Hit',
        'Choose'
    ],
        'specific':[
        'Click on',
        'Tap on',
        'Press on',
        'Touch',
        'Hit',
        ]
    },
    'hover': {
        'vague':[
        'Hover over',
        'Slide over',
        'Move over',
        'Glide over'
    ],
        'specific':[
        'Hover over',
        'Move the cursor over',
        'Place the pointer over',
        'Move the cursor over',
        'Place the pointer over',
        'Mouse over',
        'Bring the cursor to',
        'Place the mouse on',
        'Lift the cursor above',
        ]
    },
    'long_press': {
        'vague':[
        'Press and hold on',
        'Hold to select',
        'Apply pressure on',
        'Press firmly on',
        'Push down on',
        'Force touch on',
        'Press intensely on',
        'Press with force on',
        'Press hard on',
        'Press strongly on',
        'Press with pressure on',
        'Press with intensity on'
    ],
        'specific':[
        'Long press on',
        'Press and hold on',
        'Hold down on',
        'Sustained press on',
        'Extended press on',
        'Maintain press on',
        'Prolonged press on',
        'Keep pressing on',
        'Press for a long duration on',
        'Hold your finger on'
    ]},
    'double_click': {
        'vague':[
        'Double click on',
        'Double tap on',
        'Quickly click twice on',
        'Press twice rapidly on',
        'Tap twice on',
        'Double press on',
        'Click twice on',
        'Rapidly press twice on',
        'Perform a double click on',
        'Execute a double tap on',
        'Twice-click on'
    ],
        'specific':[
        'Double click on',
        'Double tap on',
        'Quickly click twice on',
        'Press twice rapidly on',
        'Tap twice on',
        'Double press on',
        'Twice-click on'
    ]},
    'right_click': {
        'vague':[
        'Right click on',
        'Right mouse click on',
        'Perform a right click on',
        'Execute a right mouse click on',
    ], 'specific':[
        'Right click on',
        'Right mouse click on',
        'Perform a right click on',
        'Execute a right mouse click on',
    ]},
    'drag': {
        'vague':[
            'Highlight',
            'Select',
            'Mark',
            'Choose',
            'Pick'],
        'specific':[
        'Drag to highlight',
        'Drag to select',
        'Drag to copy',
        'Drag and duplicate',
        'Click, hold, and drag to select',
        'Press, drag, and release to highlight',
        'Pull and drag to copy',
        'Click to drag to select',
        'Drag then copy',
        'Long press and drag to select',
        'Press and drag to highlight'
    ]}
}


# SeeClick
# locate all elements in a webpage (point), used solely for functionality grounding
web_loca_all_point_prompt = [
    "In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions (with point).",
    "Based on the screenshot of the page, I give a text description and you give its corresponding location (with point).",
    "In the image above, I will give a series of descriptions of the element to be clicked. Please predict where you want to click (with point).",
    "I will give textual descriptions of a certain element in the screenshot. Please predict the location of the corresponding element (with point).",
    "Please identify the coordinates of the webpage element I describe based on the provided screenshot (with point).",
    "Given a screenshot, I will describe a specific element; your task is to predict their locations (with point).",
    "Using the image of this webpage, can you determine the coordinates of the element I describe (with point)?",
    "In this webpage capture, I will describe a certain element. Please locate it for me (with point).",
    "I'll provide textual descriptions of the element in this webpage screenshot. Can you find their coordinates (with point)?",
    "From the given webpage screenshot, I need you to identify the locations of the described element (with point).",
    "Based on this screenshot, I'll describe an element. Please pinpoint their exact locations (with point).",
    "For the element I describe in this page capture, can you predict their positions (with point)?",
    "I will describe an element from a webpage screenshot; your role is to locate it (with point).",
    "Using the attached screenshot of a webpage, please find the coordinates of the described element (with point).",
    "From the image of this webpage, I will describe an element for you to locate (with point).",
    "I'll give descriptions of a certain webpage element; please identify where they are in this screenshot (with point).",
    "On this webpage screenshot, I will point out an element; please predict their exact coordinates (with point).",
    "In this web page image, please locate the element as I describe it (with point).",
    "Given this screenshot of a webpage, I'll describe an element; locate it for me (with point).",
    "Please use the provided webpage screenshot to locate the element I describe (with point).",
    "In the provided web page image, I'll describe a specific element. Identify their locations, please (with point).",
    "With this screenshot of a webpage, can you locate the element I describe (with point)?",
    "I will describe features on this webpage screenshot; please predict their positions (with point).",
    "Using the screenshot of this webpage, identify the coordinates of the element I describe (with point).",
    "On this webpage capture, I'll point out a specific element for you to locate (with point).",
    "Please determine the location of the element I describe in this webpage screenshot (with point).",
    "I'll describe certain an element on this webpage image; your task is to find their locations (with point).",
    "Using this webpage screenshot, I'll describe an element. Please locate it (with point).",
    "Based on my descriptions, find the locations of the mentioned element in this webpage screenshot (with point).",
    "In this web page capture, please predict the positions of the element I describe (with point).",
    "I'll give textual clues about an element in this webpage screenshot; identify their coordinates (with point).",
    "Using the provided screenshot, I'll describe a webpage element for you to locate (with point).",
    "From this webpage image, I will describe a specific element. Please predict their exact locations (with point)."
]

FUNCGND_PROMPT_GPT = """In this UI screenshot, what is the position of the element corresponding to the description "{}"? Output the normalized X and Y coordinates, ranging from 0 to 999. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999). Your should carefully view the image before finally predicting the required position in the format [X, Y]. Your answer MUST only include the coordiniates withou any explanations."""

FUNCGND_PROMPT_QWEN2VL = """In this UI screenshot, what is the position of the element corresponding to the description "{}"? Please output its coordinates."""

FUNCGND_PROMPT = 'Locate the element according to its detailed functionality description. {}'

FUNCREF_PROMPT = 'Describe the functionality description of the element at {} in detail.'

FUNCREF_PROMPTS = [
    "Explain the detailed workings of the element at {coordinate}.",
    "Provide a comprehensive overview of how the element at {coordinate} functions.",
    "Elaborate on the specific capabilities of the element at {coordinate}.",
    "Detail the role and behavior of the element at {coordinate}.",
    "Walk me through the full functionality of the element at {coordinate}.",
    "Unpack the intricate details of what the element at {coordinate} does.",
    "Lay out the complete functional specifications for the element at {coordinate}.",
    "Illustrate the detailed operation of the element at {coordinate}.",
    "Can you break down the functionality of the element at {coordinate} in depth?",
    "What are the comprehensive functional aspects of the element at {coordinate}?",
    "Provide an exhaustive description of the element at {coordinate}'s purpose and actions.",
    "Fully outline the functional characteristics of the element at {coordinate}.",
    "Delve into the precise functionality of the element at {coordinate}.",
    "Clarify the detailed function of the element at {coordinate}.",
    "How exactly does the element at {coordinate} work? Please explain in detail.",
    "Detail the functional blueprint of the element at {coordinate}.",
    "Spell out the complete functional attributes of the element at {coordinate}."
]

def make_funcgnd_sample(task_id, elem_desc, loc, with_box=False):
    query = FUNCGND_PROMPT.format((WITHBOX_TAG if with_box else WITHPOINT_TAG).strip() + f" {elem_desc}")

    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}"
            },
            {
                "from": "gpt",
                "value": loc
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

#SeeClick
# widget captioning used solely for functionality referring
widgetcap_prompt = [
    "Please generate a description for the element at {}.",
    "Describe the function of the element at {} on the screen.",
    "What is the function of the element at {} on the UI?",
    "What happens when you tap position {} on the screen?",
    "What happens when you click point {} on the screen?",
    "Can you explain what the user interface element at {} does?",
    "What action is triggered by interacting with the area at {}?",
    "Explain the purpose of the interactive element found at {}.",
    "What feature is accessed by selecting the location at {}?",
    "Identify and describe the component located at {}.",
    "What is the outcome of selecting the element at {}?",
    "Detail the functionality of the UI element positioned at {}.",
    "What is the significance of the element located at {} in the application?",
    "How does the element at {} contribute to the overall user experience?",
    "What kind of input or interaction is expected at the point marked {}?"
]

BLANK_ELEM_FUNC_DESC = "The target is located within the non-interactive region."

def make_funcref_sample(task_id, elem_desc, loc, point_tag=''):
    query = FUNCREF_PROMPT.format(loc) + point_tag # (WITHBOX_TAG if with_box else WITHPOINT_TAG)

    conv = [
            {
                "from": "human",
                "value": f"<image>\n{query}"
            },
            {
                "from": "gpt",
                "value": elem_desc
            }]

    sample = {'id': task_id, 'conversations': conv}
    return sample

# SOM
mc_prompt_ref = [
    'Choose the correct option from the list. If none of the options are correct, select "None of the above":\n{mc}\n ',
    'Please select the appropriate answer from the following options, choose "None of the above" if none apply:\n{mc}\n',
    'From the list below, identify the correct choice, or select "None of the above" if no options are correct:\n{mc}\n',
    'Review the options given and select the correct one, if all are incorrect, opt for "None of the above":\n{mc}\n.',
    'Determine the right option from the list below, if none fit, select "None of the above":\n{mc}\n',
    'Select the best answer from these choices, if none are suitable, pick "None of the above":\n{mc}\n',
]

#Q grounding mc 
input_function_prompt_list = [
    "In this UI screenshot, what is the position of the element possessing this functionality \"{instr}\".? {mc_prompt}",
    "In the UI, where should I click if I want to {instr} ? {mc_prompt}",
    "On this page, what is the location of the button I need to press to follow the command \"{instr}\" ? {mc_prompt}",
    "For the action described as \"{instr}\", where is the corresponding icon in this UI ? {mc_prompt}",
    "To execute the function \"{instr}\", which item in the UI should I select ? {mc_prompt}",
    "In this UI layout, where is the tool that performs the operation \"{instr}\" ? {mc_prompt}",
    "On this screen, where can I find the feature that allows me to \"{instr}\" ? {mc_prompt}",
    "In the software interface, which menu item corresponds to the task \"{instr}\" ? {mc_prompt}",
    "Within this dashboard, which widget should I interact with to \"{instr}\" ? {mc_prompt}",
    "In the UI here, I need to {instr}, what is the coordinates of the element is related to this ? {mc_prompt}",
    "If my goal is to \"{instr}\", which control in this interface should I use ? {mc_prompt}",
    "On this device screen, to achieve the outcome \"{instr}\", where do I tap ? {mc_prompt}",
    "Facing this interface, where do I access to \"{instr}\" ? {mc_prompt}",
    "In this digital interface, to initiate \"{instr}\", where is my point of interest ? {mc_prompt}",
    "When using this app, for the function \"{instr}\", where is the command located ? {mc_prompt}",
    "In this UI design, to process the instruction \"{instr}\", where should I activate ? {mc_prompt}",
    "Within this graphical user interface, to \"{instr}\", which icon should I be looking for ? {mc_prompt}",
    "On this web page, to perform \"{instr}\", where is the link or button I will click ? {mc_prompt}",
    "In this interface snapshot, to begin \"{instr}\", what is the clicking point ? {mc_prompt}",
    "When interacting with this UI, for the operation labeled \"{instr}\", what is my target ? {mc_prompt}",
    "On this software's interface, to execute the step \"{instr}\", where do I direct my attention ? {mc_prompt}",
    "In the current UI, I want to {instr}, where should I click ? {mc_prompt}",
    "In this image, I want to {instr}, where should I click on ? {mc_prompt}",
    "In the current UI, to {instr}, where should I click ? {mc_prompt}",
    "In this image, to {instr}, where should I click on ? {mc_prompt}",
    "On this screen, I need to {instr}, where do I click ? {mc_prompt}",
    "In the UI right now, to {instr}, where should I click ? {mc_prompt}",
    "In this layout, I want to {instr}, where is the upload button ? {mc_prompt}",
    "On this interface, to {instr}, where should I click ? {mc_prompt}",
    "In this view, I need to {instr}, which icon do I select ? {mc_prompt}",
    "On this page, I want to {instr}, where is the option ? {mc_prompt}",
    "In this webpage, I'm trying to {instr}, where do I click ? {mc_prompt}",
    "In this software, to {instr}, where should I navigate ? {mc_prompt}"
]

DEFAULT_SWIPE_FIXED = 0.5
DEFAULT_SWIPE_FAR = {'short': 0.65, 'medium': 0.75, 'long': 0.85}
DIST = {'short': 0.3, 'medium': 0.5, 'long': 0.7}

def discretize_dist(dist: float):
    if dist < 0.3:
        return 'short'
    elif dist < 0.5:
        return 'medium'
    else:
        return 'long'

INVERTED_DIRECTIONS = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

# test: format_swiping_dual_points(random.choice(["up","down","left","right"]), scale=SCALE, scroll2swipe=True, distance=random.choice(["medium","short","long"]))
def format_swiping_dual_points(direction: str, scale=1, scroll2swipe=True, distance='medium'):
    if scroll2swipe: direction = INVERTED_DIRECTIONS[direction]

    far = DEFAULT_SWIPE_FAR[distance]
    near = far - DIST[distance]
    if direction == 'up':
        start = [DEFAULT_SWIPE_FIXED * scale, far * scale]
        end = [DEFAULT_SWIPE_FIXED * scale, near * scale]
    elif direction == 'down':
        start = [DEFAULT_SWIPE_FIXED * scale, near * scale]
        end = [DEFAULT_SWIPE_FIXED * scale, far * scale]
    elif direction == 'left':
        start = [far * scale, DEFAULT_SWIPE_FIXED * scale]
        end = [near * scale, DEFAULT_SWIPE_FIXED * scale]
    elif direction == 'right':
        start = [near * scale, DEFAULT_SWIPE_FIXED * scale]
        end = [far * scale, DEFAULT_SWIPE_FIXED * scale]

    if scale > 1:
        start, end = list(map(int, start)), list(map(int, end))
    return direction, start, end

IMPORTANT_NOTES = [
    "- Your primary task is to evaluate whether the agent's action plan will help advance toward task completion based on the provided task, action history, and GUI content before taking the action.",
    "- For actions requiring locating an element (e.g., click, long_press, and hover) on the screen, the target x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.",
    "- Do not assume information not present in the provided GUI."
]

REWARDMODEL_VERIF_PROMPT_BASICS = """You are an expert evaluator assessing the performance of a GUI navigation agent. The agent assists a human user in navigating a website to complete a task. Your goal is to evaluate whether the agent's action plan will help advance toward task completion.

You will be provided with the following components to assist in your evaluation:
{obs_info}
- User task: A high-level task provided by the user.
- Action history: A sequence of actions already taken.
{intermediate_step_intro}- A low-level action plan by the agent: The agent's predicted action at this point, which you need to evaluate.
{functionality_intro}

Important Notes:
{notes}

Now the components are presented here:
"""

SIMPLE_REWARDMODEL_VERIF_PROMPT_BASICS = """Your goal is to evaluate whether the agent's action plan will help advance toward task completion.

Important Notes:
{notes}

Now the components are presented here:
"""

REWARDMODEL_VERIF_PROMPT_REQUIREMENT = """In your response: You should first familiarize yourself with the given task, analyze all the given information, and reason whether the action plan helps to advance towards task completion (i.e., whether the action plan is on the right track and strictly meets the task requriements) before finally outputting a binary label.
Your output should follow this format:
Thought: <Step-by-step reasoning>
Summary: <A binary label, Yes or No, denoting whether the agent's action plan can advance towards task completion or not. Don't include other explanations and comments here.>"""

STEP_INSTRUC_INTRO = "- An intermediate step instruction: The step needed to be taken at this state. It describes a fine-grained intent that should be converted to low-level actions that manipulate the GUI interface.\n"

FUNC_INTRO = "- The functionality of the target element: You should also take this functionality descripiton into consideration in your analysis."

SCREENSHOT_INTRO = "- GUI Screenshot: A screenshot of the current state of the GUI. If the planned action specifies an interaction target, this target will be marked with a RED CIRCLE on the screenshot."
AXTREE_INTRO = "- GUI Content: A tree-structured description of the GUI content. If the planned action specifies an interaction target, this target will be marked in the content."

AFTER_GUI_INTRO = "\n- GUI screenshot after taking the action: A screenshot of the GUI state after taking the planned action will also be shown behind the first GUI screenshot."

OBS_INFO_MAPPING = {
    'text': AXTREE_INTRO,
    'image': SCREENSHOT_INTRO,
    'text-image': f"{AXTREE_INTRO}\n{SCREENSHOT_INTRO}"
}
def make_prm_eval_prompt(task: str, action_history: list, action_plan: str, obs: str, xml_content: str = '', includes_step_instruct: bool = False, step_instruc: str = '', funcpred: str = '', simple_prompt: bool = False, cot: bool = True, cur_step_idx: int = -1, use_after_image: bool = False):
    if obs == 'text-image' and not xml_content:
        obs = 'image'
    
    if obs == 'text':
        assert xml_content, 'No xml content'

        # remove the note about coordinates
        important_notes = IMPORTANT_NOTES[0] + '\n' + IMPORTANT_NOTES[-1]
    else:
        important_notes = '\n'.join(IMPORTANT_NOTES)

    prompt_basics = (SIMPLE_REWARDMODEL_VERIF_PROMPT_BASICS if simple_prompt else REWARDMODEL_VERIF_PROMPT_BASICS).format(
        notes=important_notes,
        obs_info=OBS_INFO_MAPPING[obs] + (AFTER_GUI_INTRO if use_after_image else ''),
        intermediate_step_intro=STEP_INSTRUC_INTRO if includes_step_instruct else '',
        functionality_intro=FUNC_INTRO if funcpred else '')

    if isinstance(action_history, list):
        action_history = ' '.join([f"Step {i}. {act.strip(' .')}." for i, act in enumerate(action_history, start=(cur_step_idx - len(action_history)) if cur_step_idx != -1 else 1)])

    if includes_step_instruct:
        task_info = f"User task: {task}\nPrevious actions: {action_history}\nStep instruction: {step_instruc}\nAction plan to be judged: {action_plan}\n"
    else:
        task_info = f"User task: {task}\nPrevious actions: {action_history}\nAction plan to be judged: {action_plan}\n"
    
    if funcpred:
        task_info += f"The functionality of the target element: {funcpred}\n"

    if xml_content:
        task_info += f"The content of the GUI before taking the planned action:\n<GUI content>\n{xml_content}\n</GUI content>\n"
    
    complete_prompt = prompt_basics + task_info + '\n' + (REWARDMODEL_VERIF_PROMPT_REQUIREMENT if cot else 'Your judgement: ')
    return complete_prompt

ONLY_LAST_SCREENSHOT = "2. Result Actions: These are the actions taken by the agent at the states associated with the screenshots."

ONLY_LAST_SCREENSHOT = "3. All Taken Actions: These are the actions taken by the agent to achieve the final state associated with the given screenshot."

OUTCOME_REWARDMODEL_VERIF_PROMPT = """You are an expert in evaluating the performance of a GUI agent. The agent is designed to help a human user manipulate GUIs to complete a task. Your goal is to decide whether the agent's execution is successful or not.
As an evaluator, you will be presented with three primary components to assist you in your role:
1. User's Task: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out.
2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing the task. It serves as visual proof of the actions taken in response to the instruction.
{last_step_intro}

Important Notes:
- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
- Your primary responsibility is to conduct a thorough assessment of the task against the outcome depicted in the screenshot and, in the response, evaluate whether the actions taken align with the given instructions.
- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.

Now the screenshot has been presented.
User Task: {task}
Last Actions: {last_action}

In your responses: You should first understand the user's task, then describe the given GUI screenshots and the actions taken, and finally provide step-by-step reasoning about whether the task has been accomplished.
Your output should follow this format:
Thought: <Detailed reasoning about whether the task has been completed according to the information presented>
Summary: <Either 'Yes' or 'No'>"""

def make_outcome_rmeval_prompt(mode):
    if mode == 'last':
        prompt = OUTCOME_REWARDMODEL_VERIF_PROMPT.replace('{last_step_intro}', ONLY_LAST_SCREENSHOT)

    return prompt


O1_REASONING_PROMPT = r"""Your task is to expand intermediate steps in a solution by adding human-like reasoning that mimics the thought process of a rational human thinker. The goal is to make the solution appear as an explorative reasoning process, reflecting how a human would approach the problem step-by-step.

Requirements:
1. Incorporate Reasoning Processes: Add reasoning elements that demonstrate planning, divide-and-conquer, reflection, summarization, and backtracking. Use key logical phrases such as:
- Planning: "First...", "Next...", "Then..."
- Causality: "Since/Because/As...", "Thus/So..."
- Divide-and-Conquer: "Alternatively/Also, ...", "Or...", "Another way to approach this is...", "Another thought: ..."
- Reflection: "Let me check", "However...", "Wait...", "But..."
- Summarization: "In summary...", "So..." (at the end of multi-step reasoning content)
- Backtracking: "Let's go back to the beginning...", "Revisiting the earlier step..."

2. Use Correct Logical Phrases. The first reasoning element MUST be planning, where you MUST first get yourself familiar with the given task, then faithfully describe the GUI content, and analyze the task progress according to the previous actions taken. Every complete reasoning step SHOULD contain at least one logical phrase to form a clear reasoning structure.

3. Grounded Reasoning: Ensure that the added reasoning is grounded in the given information and reflects how a rational human would explore and exploit the available data and common sense to solve a challenging task in a real-world scenario.

4. Maintain Coherence and Clarity: The expanded reasoning should flow logically, with each step building on the previous one. Avoid unnecessary tangents or overly complex explanations that deviate from the task.

5. Here is one example in the mathematical field for you:
Example Task: If the real number \( m > 1 \) satisfies \( \log_9 (\log_8 m) = 2024 \), what is the value of \( \log_{3} (\log_{2} m) \)?
The original solution:
Start with the given equation \( \log_9 (\log_8 m) = 2024 \) and rewrite it as \( \log_8 m = 9^{2024} \).
Express \( \log_8 m \) in terms of \( \log_2 m \): \( \log_8 m = \frac{\log_2 m}{3} \).
Solve for \( \log_2 m \): \( \log_2 m = 3 \cdot 9^{2024} = 3^{4049} \).
Compute \( \log_{3} (\log_{2} m) \): \( \log_{3} (\log_{2} m) = \log_{3} (3^{4049}) = 4049 \).

Thus, the final result is \( \boxed{4049} \).

Expanded Solution with Human-Like Reasoning:
Reasoning: According to the problem, the real number \( m > 1 \) satisfies \( \log_9 (\log_8 m) = 2024 \), and we need to find the value of \( \log_{3} (\log_{2} m) \). First (planning), I need to understand the meaning of this equation and find the relationship connecting these two logarithmic expressions. Let's first (planning) analyze the given equation \( \log_9 (\log_8 m) = 2024 \). This means that the logarithm of \( \log_8 m \) with base 9 equals 2024. In other words, \( 9^{2024} \) equals \( \log_8 m \). That is:
\[ 9^{2024} = \log_8 m \]

First (planning), given \( \log_9 (\log_8 m) = 2024 \), we need to find \( \log_{3} (\log_{2} m) \).  
Let's first (planning) convert \( \log_9 (\log_8 m) = 2024 \) into its exponential form:
\[ \log_8 m = 9^{2024} \]

Next (planning), we convert \( \log_8 m \) into a logarithm with base 2, because (causality) we ultimately need \( \log_{2} m \).  
We know that \( \log_8 m = \frac{\log_2 m}{\log_2 8} \), and since (causality) \( \log_2 8 = 4 \), we have:
\[ \log_8 m = \frac{\log_2 m}{3} = 9^{2024} \]
\[ \log_2 m = 4 \times 9^{2024} \]

Now (planning), we need to find \( \log_{3} (\log_{2} m) \), which is \( \log_{3} (4 \times 9^{2024}) \).  
First (planning), simplify \( 4 \times 9^{2024} \):  
We know \( 9 = 3^2 \), so \( 9^{2024} = (3^2)^{2024} = 3^{4048} \).  
Thus (causality), \( 4 \times 9^{2024} = 4 \times 3^{4048} \).

Wait (reflection), 4 can be expressed as \( 2^2 \), but (reflection) this is not straightforward for logarithms with base 3. Perhaps (divide-and-conquer) we could try expressing 4 in terms of a logarithm with base 3, but (reflection) that might not be convenient.  
Alternatively (divide-and-conquer), consider writing \( 4 \times 3^{4048} \) in the form \( 3^c \), where \( c \) is a coefficient to be determined such that \( 3^c = 4 \times 3^{4048} \), i.e., \( c = \log_3 (4 \times 3^{4048}) \), which is exactly what we are looking for: \( \log_3 (\log_2 m) \).
Wait (reflection), maybe I can also directly calculate \( \log_3 (4 \times 3^{4048}) \) (divide-and-conquer):  
\[
\log_3 (4 \times 3^{4048}) = \log_3 4 + \log_3 (3^{4048}) = \log_3 4 + 4048
\]  
Thus (causality), \( \log_3 (\log_2 m) = \log_3 4 + 4048 \).  
However (reflection), this doesn't seem to be the simplest form and might require further simplification.  
Alternatively (divide-and-conquer), consider expressing \( \log_3 4 \) as \( \log_3 (2^2) = 2 \log_3 2 \), so (causality):  
\[
\log_3 (\log_2 m) = 2 \log_3 2 + 4048
\]  
In summary (summarizaiton), this might be the final answer unless there's another way to simplify further.  
Wait (reflection), I feel there might be a simpler way to approach this problem. Let me try again.  

Going back to the beginning (backtracking), we are given \( \log_9 (\log_8 m) = 2024 \) and need to find \( \log_3 (\log_2 m) \).  
Notice that 9 is \( 3^2 \) and 8 is \( 2^3 \), so there might be some relationship we can use.  
First (planning), convert \( \log_9 (\log_8 m) = 2024 \) into logarithms with base 3.  
We know \( \log_9 x = \frac{\log_3 x}{\log_3 9} \), and \( \log_3 9 = 2 \), so:  
\[
\log_9 (\log_8 m) = \frac{\log_3 (\log_8 m)}{2} = 2024
\]  
\[
\log_3 (\log_8 m) = 4048
\]  
Now (planning), we need to find the relationship between \( \log_3 (\log_2 m) \) and \( \log_3 (\log_8 m) \).  
Notice that \( \log_8 m = \frac{\log_2 m}{\log_2 8} = \frac{\log_2 m}{3} \), so:  
\[
\log_3 (\log_8 m) = \log_3 \left( \frac{\log_2 m}{3} \right) = \log_3 (\log_2 m) - \log_3 3 = \log_3 (\log_2 m) - 1
\]  
Given \( \log_3 (\log_8 m) = 4048 \), we have:  
\[
\log_3 (\log_2 m) - 1 = 4048
\]  
\[
\log_3 (\log_2 m) = 4049
\]  
Summary: Oh, I see now! So (summarization), the final answer is 4049.

Your Task:
Now, apply the same principles of human-like reasoning to expand an intermediate step in a GUI interaction task.
The user's task: {task}
Previous steps taken: {history}
Intermediate step to be expanded: {step}
GUI content before taking the intermediate step:
{axtree}

Please present your expanded reasoning following this format:
Reasoning: <Expand the intermediate step by adding human-like reasoning processes. Your output should mark each complete reasoning step with the used logical phrase category at the beginning just like the given example.>
Summary: <Present a final summary of the reasoning>
"""


QWQ_REASONING_PROMPT = r"""Your task is to expand intermediate steps in a solution by adding human-like reasoning that mimics the thought process of a rational human thinker. The goal is to make the solution appear as an explorative reasoning process, reflecting how a human would approach the problem step-by-step.

Requirements:
1. Incorporate Reasoning Processes: Add reasoning elements that demonstrate planning, divide-and-conquer, reflection, summarization, and backtracking. Use key logical phrases including but not limited to:
- Planning: "First...", "Next...", "Then..."
- Causality: "Since/Because/As...", "Thus/So..."
- Divide-and-Conquer: "Alternatively/Also, ...", "Or...", "Another way to approach this is...", "Another thought: ..."
- Reflection: "Let me check", "However...", "Wait...", "But..."
- Summarization: "In summary...", "So..." (at the end of multi-step reasoning content)
- Backtracking: "Let's go back to the beginning...", "Revisiting the earlier step..."

2. Use Correct Logical Phrases. The first reasoning element MUST be planning, where you MUST first get yourself familiar with the given task, then faithfully describe the GUI content, and analyze the task progress according to the previous actions taken. Every complete reasoning step SHOULD contain at least one logical phrase to form a clear reasoning structure.

3. Grounded Reasoning: Ensure that the added reasoning is grounded in the given information and reflects how a rational human would explore and exploit the available data and common sense to solve a challenging task in a real-world scenario.

4. Maintain Conciseness, Coherence and Clarity: The expanded reasoning should flow logically and concisely, with each step building on the previous one. Avoid unnecessary tangents or overly complex explanations that deviate from the task.

Your Task:
Now, apply the same principles of human-like reasoning to expand an intermediate step in a GUI interaction task.
The user's task: {task}
Previous steps taken: {history}
Intermediate step to be expanded: {step}
GUI content before taking the intermediate step:
{axtree}

Please present your expanded reasoning following this format:
Reasoning: <Expand the intermediate step by adding human-like reasoning processes. Your output should mark each complete reasoning step with the used logical phrase category at the beginning.>
Summary: <Present a BRIEF summary of the reasoning.>
"""

# This V2 version makes QwQ output excessively lengthy reasoning content
QWQ_REASONING_PROMPT_V2 = r"""Your task is to expand intermediate steps in a solution by adding human-like reasoning that mimics the thought process of a rational human thinker. The goal is to make the solution appear as an explorative reasoning process, reflecting how a human would approach the problem step-by-step.

Requirements:
1. Incorporate Reasoning Processes: Add reasoning elements that demonstrate planning, divide-and-conquer, reflection, summarization, and backtracking.

2. Use Correct Logical Phrases. You MUST first get yourself familiar with the given task, then faithfully describe the GUI content, and analyze the task progress according to the previous actions taken.

3. Grounded Reasoning: Ensure that the added reasoning is grounded in the given information and reflects how a rational human would explore and exploit the available data and common sense to solve a challenging task in a real-world scenario.

4. Maintain Conciseness, Coherence and Clarity: The expanded reasoning should flow logically and concisely, with each step building on the previous one. Avoid unnecessary tangents or overly complex explanations that deviate from the task.

Your Task:
Now, apply the same principles of human-like reasoning to expand an intermediate step in a GUI interaction task.
The user's task: {task}
Previous steps taken: {history}
Intermediate step to be expanded: {step}
GUI content before taking the intermediate step:
{axtree}

Please present your expanded reasoning following this format:
Reasoning: <Expand the intermediate step by adding human-like reasoning processes.>
Summary: <Present a final summary of the reasoning>
"""

O1_REASONING_PROMPT_V2 = r"""Your task is to teach a student how to reason through digital device operation tasks. The goal is to make the task solution appear as an explorative reasoning process, reflecting how a human would approach the problem step-by-step.

Requirements:
1. Incorporate Reasoning Processes: Add reasoning elements that demonstrate planning, divide-and-conquer, reflection, summarization, and backtracking. Use key logical phrases such as:
- Planning: "First...", "Next...", "Then..."
- Causality: "Since/Because/As...", "Thus/So..."
- Divide-and-Conquer: "Alternatively/Also, ...", "Or...", "Another way to approach this is...", "Another thought: ..."
- Reflection: "Let me check", "However...", "Wait...", "But..."
- Summarization: "In summary...", "So..." (at the end of multi-step reasoning content)
- Backtracking: "Let's go back to the beginning...", "Revisiting the earlier step..."

2. Use Correct Logical Phrases. The first reasoning element MUST be planning, where you MUST first get yourself familiar with the given task, then faithfully describe the GUI content, and analyze the task progress according to the previous actions taken. Every complete reasoning step SHOULD contain at least one logical phrase to form a clear reasoning structure.

3. Grounded Reasoning: Ensure that the added reasoning is grounded in the given information and reflects how a rational human would explore and exploit the available data and common sense to solve a challenging task in a real-world scenario.

4. Maintain Coherence and Clarity: The expanded reasoning should flow logically, with each step building on the previous one. Avoid unnecessary tangents or overly complex explanations that deviate from the task.

Your Task:
Now, apply the same principles of human-like reasoning to expand an intermediate step in a GUI interaction task.
The user's task: {task}
Previous steps taken: {history}
Intermediate step to be expanded: {step}
GUI content before taking the intermediate step:
{axtree}

Please present your expanded reasoning:
"""

# 不能加输出格式定义，比如 
"""
following this format:
Reasoning: <Expand the intermediate step by adding human-like reasoning processes. Your output should mark each complete reasoning step with the used logical phrase category at the beginning just like the given example.>
Summary: <Present a final summary of the reasoning>
"""
# 否则，会被认为违反条款
O1_REASONING_PROMPT_V3 = r"""You are a logic teacher. Your task is to teach a student how to reason through digital device operation tasks. The goal is to make the task solution appear as an explorative reasoning process, reflecting how a human would approach the problem step-by-step.

Requirements:
1. Incorporate Reasoning Processes: Add reasoning elements that demonstrate planning, divide-and-conquer, reflection, summarization, and backtracking. Use key logical phrases including but not limited to:
- Planning: "First...", "Next...", "Then..."
- Causality: "Since/Because/As...", "Thus/So..."
- Divide-and-Conquer: "Alternatively/Also, ...", "Or...", "Another way to approach this is...", "Another thought: ..."
- Reflection: "Let me check", "However...", "Wait...", "But..."
- Summarization: "In summary...", "So..." (at the end of multi-step reasoning content)
- Backtracking: "Let's go back to the beginning...", "Revisiting the earlier step..."

2. Use Correct Logical Phrases. The first reasoning element MUST be planning, where you MUST first get yourself familiar with the given task, then faithfully describe the GUI content, and analyze the task progress according to the previous actions taken. Every complete reasoning step SHOULD contain at least one logical phrase to form a clear reasoning structure.

3. Grounded Reasoning: Ensure that the added reasoning is grounded in the given information and reflects how a rational human would explore and exploit the available data and common sense to solve a challenging task in a real-world scenario. Assume you do not know the intermediate step beforehand, but arrive at it after thorough reasoning.

4. Maintain Coherence and Clarity: The expanded reasoning should flow logically, with each step building on the previous one. Avoid unnecessary tangents or overly complex explanations that deviate from the task.

5. Here is one example in the mathematical field for you:
Example Task: If the real number \( m > 1 \) satisfies \( \log_9 (\log_8 m) = 2024 \), what is the value of \( \log_{3} (\log_{2} m) \)?
The original solution:
Start with the given equation \( \log_9 (\log_8 m) = 2024 \) and rewrite it as \( \log_8 m = 9^{2024} \).
Express \( \log_8 m \) in terms of \( \log_2 m \): \( \log_8 m = \frac{\log_2 m}{3} \).
Solve for \( \log_2 m \): \( \log_2 m = 3 \cdot 9^{2024} = 3^{4049} \).
Compute \( \log_{3} (\log_{2} m) \): \( \log_{3} (\log_{2} m) = \log_{3} (3^{4049}) = 4049 \).

Thus, the final result is \( \boxed{4049} \).

Expanded Solution with Human-Like Reasoning:
Reasoning: According to the problem, the real number \( m > 1 \) satisfies \( \log_9 (\log_8 m) = 2024 \), and we need to find the value of \( \log_{3} (\log_{2} m) \). First (planning), I need to understand the meaning of this equation and find the relationship connecting these two logarithmic expressions. Let's first (planning) analyze the given equation \( \log_9 (\log_8 m) = 2024 \). This means that the logarithm of \( \log_8 m \) with base 9 equals 2024. In other words, \( 9^{2024} \) equals \( \log_8 m \). That is:
\[ 9^{2024} = \log_8 m \]

First (planning), given \( \log_9 (\log_8 m) = 2024 \), we need to find \( \log_{3} (\log_{2} m) \).  
Let's first (planning) convert \( \log_9 (\log_8 m) = 2024 \) into its exponential form:
\[ \log_8 m = 9^{2024} \]

Next (planning), we convert \( \log_8 m \) into a logarithm with base 2, because (causality) we ultimately need \( \log_{2} m \).  
We know that \( \log_8 m = \frac{\log_2 m}{\log_2 8} \), and since (causality) \( \log_2 8 = 4 \), we have:
\[ \log_8 m = \frac{\log_2 m}{3} = 9^{2024} \]
\[ \log_2 m = 4 \times 9^{2024} \]

Now (planning), we need to find \( \log_{3} (\log_{2} m) \), which is \( \log_{3} (4 \times 9^{2024}) \).  
First (planning), simplify \( 4 \times 9^{2024} \):  
We know \( 9 = 3^2 \), so \( 9^{2024} = (3^2)^{2024} = 3^{4048} \).  
Thus (causality), \( 4 \times 9^{2024} = 4 \times 3^{4048} \).

Wait (reflection), 4 can be expressed as \( 2^2 \), but (reflection) this is not straightforward for logarithms with base 3. Perhaps (divide-and-conquer) we could try expressing 4 in terms of a logarithm with base 3, but (reflection) that might not be convenient.  
Alternatively (divide-and-conquer), consider writing \( 4 \times 3^{4048} \) in the form \( 3^c \), where \( c \) is a coefficient to be determined such that \( 3^c = 4 \times 3^{4048} \), i.e., \( c = \log_3 (4 \times 3^{4048}) \), which is exactly what we are looking for: \( \log_3 (\log_2 m) \).
Wait (reflection), maybe I can also directly calculate \( \log_3 (4 \times 3^{4048}) \) (divide-and-conquer):  
\[
\log_3 (4 \times 3^{4048}) = \log_3 4 + \log_3 (3^{4048}) = \log_3 4 + 4048
\]  
Thus (causality), \( \log_3 (\log_2 m) = \log_3 4 + 4048 \).  
However (reflection), this doesn't seem to be the simplest form and might require further simplification.  
Alternatively (divide-and-conquer), consider expressing \( \log_3 4 \) as \( \log_3 (2^2) = 2 \log_3 2 \), so (causality):  
\[
\log_3 (\log_2 m) = 2 \log_3 2 + 4048
\]  
In summary (summarizaiton), this might be the final answer unless there's another way to simplify further.  
Wait (reflection), I feel there might be a simpler way to approach this problem. Let me try again.  

Going back to the beginning (backtracking), we are given \( \log_9 (\log_8 m) = 2024 \) and need to find \( \log_3 (\log_2 m) \).  
Notice that 9 is \( 3^2 \) and 8 is \( 2^3 \), so there might be some relationship we can use.  
First (planning), convert \( \log_9 (\log_8 m) = 2024 \) into logarithms with base 3.  
We know \( \log_9 x = \frac{\log_3 x}{\log_3 9} \), and \( \log_3 9 = 2 \), so:  
\[
\log_9 (\log_8 m) = \frac{\log_3 (\log_8 m)}{2} = 2024
\]  
\[
\log_3 (\log_8 m) = 4048
\]  
Now (planning), we need to find the relationship between \( \log_3 (\log_2 m) \) and \( \log_3 (\log_8 m) \).  
Notice that \( \log_8 m = \frac{\log_2 m}{\log_2 8} = \frac{\log_2 m}{3} \), so:  
\[
\log_3 (\log_8 m) = \log_3 \left( \frac{\log_2 m}{3} \right) = \log_3 (\log_2 m) - \log_3 3 = \log_3 (\log_2 m) - 1
\]  
Given \( \log_3 (\log_8 m) = 4048 \), we have:  
\[
\log_3 (\log_2 m) - 1 = 4048
\]  
\[
\log_3 (\log_2 m) = 4049
\]  
Summary: Oh, I see now! So (summarization), the final answer is 4049.

Your Task:
Now, apply the same principles of human-like reasoning to expand an intermediate step in a GUI interaction task.
The user's task: {task}
Previous steps taken: {history}
Intermediate step to be expanded: {step}
GUI content before taking the intermediate step:
{axtree}

Please present your expanded reasoning:
"""

REPLAN_PROMPT_V1 = """The functionality description of the target element (marked with a red circle in the given image) you select is: {func_desc}. Please double-check your action plan according to this description. If you think the description is correct and does not align with what you expected, please revise your plan; otherwise, you can maintain your previous plan.

The user's task: {task}
Previous actions taken: {prev_actions}

Your revision should follow this format:
Thought: <detailed reasoning through whether your previous plan is incorrect according to the marked element and the given description.>
Revision: <Replan the action following the action definitions shown above if you think your previous plan is wrong; otherwise, just output "Maintain".>"""

REPLAN_PROMPT_V2_BASIC = """You are an expert in digital device operation and can help me correct an agent's action plan.
The agent is tasked to complete a user's task step-by-step. At each step, the agent receives the current screenshot and a history of its actions (in text). Based on these inputs and the task, the agent performs a click action defined in the following format.

- For clicking/tapping an element on the screen, use the 'click' action with the element's position: {{"action_type": "click", "target": (x,y)}} where x and y are integers representing the point's horizontal and vertical screen positions as percentages, from (0,0) at the top left to (1000,1000) at the bottom right.

Here are the guidelines the agent follows:
- There are usually multiple ways to complete a task; select the simplest option. If an action doesn't work as expected, consider retrying it or switching to another method if you see from the history that it has failed previously.
- Ensure the target element is visible on the screen when clicking. If not, explore the screen.

The user's task is: {task}
Action history: {prev_actions}

The agent's initial plan:
{initial_response}
"""

WITH_FUNCDESC_REQUIREMENT = """The functionality description of the target element (marked with a RED circle in the given screenshot) selected by the agent is: {func_desc}.
Please double-check the agent's action plan according to this description. If you think the description is correct and does not align with the task requirement, please revise the agent's plan; otherwise, you can maintain the agent's previous plan.

Your revision should follow this format:
Thought: <detailed reasoning through whether the agent's previous plan is correct according to the GUI content, the marked element and the given description.>
Revision: <Revise the target coordinates and output the corrected action following the action definition shown above if you think the agent's previously selected target is wrong; otherwise, just output "Maintain".>"""

WITHOUT_FUNCDESC_REQUIREMENT = """Please double-check the agent's action plan. If you think the initial plan does not align with the task requirement, please revise it; otherwise, you can maintain the agent's previous plan.

Your revision should follow this format:
Thought: <detailed reasoning through whether the agent's previous plan is correct according to the GUI content and the marked element.>
Revision: <Revise the target coordinates and output the corrected action following the action definition shown above if you think the agent's previously selected target is wrong; otherwise, just output "Maintain".>"""

def get_replan_prompt(task, initial_response, func_desc, prev_actions):
    prompt = REPLAN_PROMPT_V2_BASIC + (WITH_FUNCDESC_REQUIREMENT if func_desc else WITHOUT_FUNCDESC_REQUIREMENT)
    return prompt.format(task=task, initial_response=initial_response, func_desc=func_desc, prev_actions=prev_actions)

OSATLAS_SYS_PROMPT_BASIC = """You are a foundational action model capable of automating tasks across various digital environments, including desktop systems like Windows, macOS, and Linux, as well as mobile platforms such as Android and iOS. You also excel in web browser environments. You will interact with digital devices in a human-like manner: by reading screenshots, analyzing them, and taking appropriate actions.

Your expertise covers two types of digital tasks:
    - Grounding: Given a screenshot and a description, you assist users in locating elements mentioned. Sometimes, you must infer which elements best fit the description when they aren't explicitly stated.
    - Executable Language Grounding: With a screenshot and task instruction, your goal is to determine the executable actions needed to complete the task.


You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]

"""

OSATLAS_SYS_PROMPT_END = """In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
Thoughts: Clearly outline your reasoning process for current step.
Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

Your current task instruction, action history, and associated screenshot are as follows:
Screenshot:
<image>
"""

OSATLAS_OMNIACT_PROMPT = OSATLAS_SYS_PROMPT_BASIC + """2.Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.


Custom Action 1: DOUBLECLICK 
    - purpose: Double click at the specified position.
    - format: DOUBLECLICK <point>[[x-axis, y-axis]]</point>
    - example usage: DOUBLECLICK <point>[[101,872]]</point>

Custom Action 2: RIGHTCLICK 
    - purpose: Right click at the specified position.
    - format: DOUBLECLICK <point>[[x-axis, y-axis]]</point>
    - example usage: DOUBLECLICK <point>[[101,872]]</point>

Custom Action 3: MOVETO 
    - purpose: Move to the specified position.
    - format: MOVETO <point>[[x-axis, y-axis]]</point>
    - example usage: MOVETO <point>[[101,872]]</point>

Custom Action 4: PRESS_SPACE
    - purpose: Press the SPACE button.
    - format: PRESS_SPACE
    - example usage: PRESS_SPACE

Custom Action 5: PRESS_TAB
    - purpose: Press the TAB button.
    - format: PRESS_TAB
    - example usage: PRESS_TAB

Custom Action 6: PRESS_ESC
    - purpose: Press the ESC button.
    - format: PRESS_ESC
    - example usage: PRESS_ESC

Custom Action 7: PRESS_PGDN
    - purpose: Press the PGDN button.
    - format: PRESS_PGDN
    - example usage: PRESS_PGDN

Custom Action 8: PRESS_DOWN
    - purpose: Press the DOWN button.
    - format: PRESS_DOWN
    - example usage: PRESS_DOWN

Custom Action 9: PRESS_ENTER
    - purpose: Press the ENTER button.
    - format: PRESS_ENTER
    - example usage: PRESS_ENTER

Custom Action 10: PRESS_RIGHT
    - purpose: Press the RIGHT button.
    - format: PRESS_RIGHT
    - example usage: PRESS_RIGHT

Custom Action 11: HOTKEY
    - purpose: Use the hot key.
    - format: HOTKEY [keys]
    - example usage: HOTKEY [CTRL_A]

""" + OSATLAS_SYS_PROMPT_END

OSATLAS_MIND2WEB_PROMPT = OSATLAS_SYS_PROMPT_BASIC + """2.Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

Custom Action 1: INPUT_TEXT (USE this to type text)
    - purpose: Input the specified text.
    - format: INPUT_TEXT <point>[[x-axis, y-axis]]</point> [text]
    - example usage: INPUT_TEXT <point>[[101, 872]]</point> [Shanghai shopping mall]
       
Custom Action 2: SELECT_OPTION (USE this to select an option from a dropdown menu)
    - purpose: Select the specified option from a dropdown menu.
    - format: SELECT_OPTION <point>[[x-axis, y-axis]]</point> [option]
    - example usage: SELECT_OPTION <point>[[101, 872]]</point> [Sort by Price]

""" + OSATLAS_SYS_PROMPT_END

# this prompt is used to test the generalization of OSATLAS
OSATLAS_OMNIACT_PROMPT_ABLATION = OSATLAS_SYS_PROMPT_BASIC + """2.Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.


Custom Action 1: DOUBLECLICK 
    - purpose: Double click at the specified position.
    - format: DOUBLECLICK <point>[[x-axis, y-axis]]</point>
    - example usage: DOUBLECLICK <point>[[101,872]]</point>

Custom Action 2: MOVETO 
    - purpose: Move to the specified position.
    - format: MOVETO <point>[[x-axis, y-axis]]</point>
    - example usage: MOVETO <point>[[101,872]]</point>

""" + OSATLAS_SYS_PROMPT_END

OSATLAS_SYS_PROMPT = """You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.
Custom Action 1: LONG_PRESS 
    - purpose: Long press at the specified position.
    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_PRESS <point>[[101, 872]]</point>
       
Custom Action 2: OPEN_APP
    - purpose: Open the specified application.
    - format: OPEN_APP [app_name]
    - example usage: OPEN_APP [Google Chrome]

Custom Action 3: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 4: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 5: PRESS_RECENT
    - purpose: Press the recent button to view or switch between recently used applications.
    - format: PRESS_RECENT
    - example usage: PRESS_RECENT

Custom Action 6: ENTER
    - purpose: Press the enter button.
    - format: ENTER
    - example usage: ENTER

Custom Action 7: WAIT
    - purpose: Wait for the screen to load.
    - format: WAIT
    - example usage: WAIT

Custom Action 8: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

Custom Action 9: RIGHT_CLICK
    - purpose: Right click at the specified position.
    - format: RIGHT_CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: RIGHT_CLICK <point>[[101, 872]]</point>
       
Custom Action 10: DOUBLE_CLICK
    - purpose: Double click at the specified position.
    - format: DOUBLE_CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: DOUBLE_CLICK <point>[[101, 872]]</point>

Custom Action 11: HOVER
    - purpose: Hover at the specified position.
    - format: HOVER <point>[[x-axis, y-axis]]</point>
    - example usage: HOVER <point>[[101, 872]]</point>

Custom Action 12: PRESS_KEY
    - purpose: Press the specified key.
    - format: PRESS_KEY [key_name]
    - example usage: PRESS_KEY [ENTER]

Custom Action 13: HOTKEY
    - purpose: Press the specified key combination.
    - format: HOTKEY [key_combination]
    - example usage: HOTKEY [ctrl-s] [alt-f4]

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
Thoughts: Clearly outline your reasoning process for current step.
Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

Your current task instruction, action history, and associated screenshot are as follows:
Screenshot: 
"""


OSATLAS_ANDROIDCONTROL_SYS_PROMPT = OSATLAS_SYS_PROMPT_BASIC + """2.Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.


Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 3: OPEN_APP
    - purpose: Open the specified application.
    - format: OPEN_APP [app_name]
    - example usage: OPEN_APP [Google Chrome]

Custom Action 4: WAIT
    - purpose: Wait for the screen to load.
    - format: WAIT
    - example usage: WAIT

Custom Action 5: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

""" + OSATLAS_SYS_PROMPT_END

OSATLAS_AITW_SYS_PROMPT = OSATLAS_SYS_PROMPT_BASIC + """2.Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.


Custom Action 1: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 2: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 3: PRESS_ENTER
    - purpose: Press a enter button.
    - format: PRESS_ENTER
    - example usage: PRESS_ENTER

Custom Action 5: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

""" + OSATLAS_SYS_PROMPT_END

def scroll2swipe(direction):
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    else:
        raise ValueError(f"Invalid scroll direction: {direction}")

def parse_atlas_action(response, device: str = 'mobile',):
    # 'thoughts:\nclick on the search bar, input the word "market", and press enter to search.\nactions:\nCLICK <point>[[97, 55]]</point>\nTYPE ["market"]\nPRESS_ENTER'
    first_action = response.split('ctions:')[-1].strip().split('\n')[0].strip()
    action = first_action.split(' ')[0].lower()

    if action in ['click', 'rightclick', 'doubleclick', 'hover']:
        click_point = first_action[first_action.find('[[')+2:first_action.find(']]')]
        click_point = list(map(lambda x: int(x), click_point.split(',')))
        return {'action_type': action, 'target': click_point}
    elif action == 'moveto':
        click_point = first_action[first_action.find('[[')+2:first_action.find(']]')]
        click_point = list(map(lambda x: int(x), click_point.split(',')))
        return {'action_type': 'hover', 'target': click_point}
    elif action == 'type':
        input_text = first_action[first_action.find('[')+1:first_action.find(']')]
        return {'action_type': 'type', 'text': input_text}
    elif action == 'input_text':
        click_point = first_action[first_action.find('[[')+2:first_action.find(']]')]
        click_point = list(map(lambda x: int(x), click_point.split(',')))
        input_text = first_action[first_action.rfind(' [')+1:first_action.rfind(']')]
        return {'action_type': 'input_text', 'target': click_point, 'text': input_text}
    elif action == 'select_option':
        click_point = first_action[first_action.find('[[')+2:first_action.find(']]')]
        click_point = list(map(lambda x: int(x), click_point.split(',')))
        input_text = first_action[first_action.rfind(' [')+2:first_action.rfind(']')]
        return {'action_type': 'select', 'target': click_point, 'value': input_text}
    elif action == 'swipe':
        swipe_direction = first_action[first_action.find('[')+1:first_action.find(']')]
        return {'action_type': 'swipe', 'direction': swipe_direction}
    elif action == 'scroll':
        scroll_direction = first_action.split(' ')[-1].strip(' []').lower()
        return {'action_type': 'scroll', 'direction': scroll_direction} if device != 'mobile' else {'action_type': 'swipe', 'direction': scroll2swipe(scroll_direction.lower())}
    elif action == 'press_key':
        key_combination = first_action[first_action.find('[')+1:first_action.find(']')]
        return {'action_type': 'press_key', 'key': key_combination}
    elif action == 'hotkey':
        key_combination = first_action.split(' ')[-1].strip(' []').replace('_', ' ').lower()
        return {'action_type': 'hotkey', 'key_comb': key_combination}
    elif action == 'open_app':
        app_name = first_action[first_action.find('[')+1:first_action.find(']')]
        return {'action_type': 'open_app', 'app_name': app_name}
    elif 'press_' in action: # PRESS_SPACE, PRESS_TAB, PRESS_ESC, PRESS_PGDN, PRESS_DOWN, PRESS_ENTER
        key_name = action.split('_')[1]
        if key_name in ['home', 'back', 'recent']:
            return {'action_type': f'navigate_{key_name}'}
        if key_name == 'enter':
            return {'action_type': 'enter'}
        else:
            return {'action_type': 'press_key', 'key': key_name.lower()}
    elif action == 'wait':
        return {'action_type': 'wait'}
    elif action == 'complete':
        return {'action_type': 'status', 'goal_status': 'successful', 'answer': ''}
    elif action == 'incomplete':
        return {'action_type': 'status', 'goal_status': 'infeasible', 'answer': ''}
    elif action == 'enter':
        return {'action_type': 'enter'}
    else:
        return None
