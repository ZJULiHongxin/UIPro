import re
import json

def extract_thought_components(response):
    response = response.replace("```json", "").replace("```", "").strip()

    # HTML-tag format
    if any(k in response for k in ['<observation>', '<thought>', '<target_functionality>', '<action>', '<summary>']):
        obs = response.split('servation>')[1].split('</')[0].strip() if 'servation>' in response else ''
        thought = response.split('hought>')[1].split('</')[0].strip() if 'hought>' in response else ''
        funcdesc = response.split('ionality>')[1].split('</')[0].strip() if 'ionality>' in response else ''
        if funcdesc == '' and 'This element' in response:
            funcdesc = response[response.find('This element'):].split('</')[0].strip()
        action_pred_raw = response.split('ction>')[1].split('</')[0].strip() if '<action>' in response else ''
        if action_pred_raw == '':
            action_start = response.rfind('{"action_type')
            action_end = response.find('}', action_start)
            action_pred_raw = response[action_start:action_end+1]

        if 'ummary>' in response:
            summary = response.split('ummary>')[1].split('</')[0].strip() if 'ummary>' in response else ''
        else: summary = ''

        return obs, thought, funcdesc, action_pred_raw, summary
    elif response.startswith('{'):
        try:
            dict_action = eval(response)
            # extract obs
            for k in dict_action.keys():
                if 'observ' in k.lower():
                    obs = dict_action[k]; break
            # extract thought
            for k in dict_action.keys():
                if 'thought' in k.lower():
                    thought = dict_action[k]; break
            # extract funcdesc
            for k in dict_action.keys():
                if 'functionality' in k.lower():
                    funcdesc = dict_action[k]; break
            else: funcdesc = ''

            # extract action
            for k in dict_action.keys():
                if 'action' in k.lower():
                    action_pred_raw = dict_action[k]; break
            # extract summary
            for k in dict_action.keys():
                if 'summary' in k.lower():
                    summary = dict_action[k]; break
            
            return obs, thought, funcdesc, str(action_pred_raw), summary
        except:
            pass

    # extract observation
    obs_start = response.find(":")
    thought_start = response.find("Thought", obs_start)
    obs = response[obs_start+1:thought_start].strip(' :,\n"\'*')
    
    # extract thought
    target_funcdesc_start = response.find("Target's", thought_start)
    thought_start = response.find(":", thought_start)
    thought = response[thought_start+1:target_funcdesc_start].strip(' :,\n"\'*')
    
    # extract funcdesc
    funcdesc_start = response.find(":", target_funcdesc_start)
    funcdesc_end = response.find("\n", funcdesc_start)
    funcdesc = response[funcdesc_start+1:funcdesc_end].strip(' :, \n"\'*')
    
    # extract action
    action_start = response.find(":", response.find("Action", funcdesc_end))
    action_dict_start = response.rfind('{"')
    action_dict_end = response.find("}", action_dict_start)
    action_pred_raw = response[action_dict_start:action_dict_end+1]
    assert action_pred_raw.startswith('{"') and action_pred_raw.endswith('}')

    # extract action summary
    summary_start = response.find(":", response.find('Summary', action_dict_end))
    summary = response[summary_start:].strip(' :, \n"\'*}')
    return obs, thought, funcdesc, action_pred_raw, summary

def extract_SEEACT_thought_components(response):
    """<Current Webpage Identification>
    This is the TikTok Creative Center's "Audio Library" section, where users can explore pre-cleared music categorized by various criteria.

    <Previous Action Analysis>
    There are no previous actions, so this is the starting point.

    <Screenshot Details Analysis>
    The webpage contains filters and dropdowns to refine music searches. The filters of interest include:
    - A dropdown for country at the top left, currently set to "United States" (box 16).
    - A filter for "Usage" with an option for "TikTok Series" (box 24).
    - Tabs labeled "All Music" (box 14) and "Playlist" (box 15).
    Other areas like the list of recommended playlists or featured tracks remain untargeted at this moment.

    <Next Action Based on Webpage and Analysis>
    To refine our query to look for music for TikTok series, the logical first step is to select the "TikTok Series" filter under "Usage."

    <Reiteration>
    Target element: Box labeled with the white letter "D" at the bottom left of "TikTok Series."
    Location: Filter option labeled "TikTok Series" under "Usage."
    Action: SELECT.

    <Verification with the Screenshot>
    The target element "TikTok Series" is marked with a red bounding box and the letter "D."

    <Final Answer>
    ELEMENT: D  
    ACTION: CLICK  
    VALUE: None"""
    parts = response.split("<Final Answer>")
    if len(parts) == 2:
        thought, action_pred = response.split("<Final Answer>")
    else:
        action_start = response.find("ELEMENT")
        thought, action_pred = response[:action_start], response[action_start:]
    
    # Extract the three arguments from the action_pred using regex
    action_pred = action_pred.strip()
    action_type = re.search(r'ACTION: (\w+)', action_pred).group(1)
    element = re.search(r'ELEMENT: (\w+)', action_pred).group(1)
    # judge if the element is a number
    if element.isdigit():
        element = int(element)
    else:
        element = None

    value = re.search(r'VALUE: (\w+)', action_pred).group(1)

    if action_type == 'CLICK':
        action_type = 'click'
        action_str = str({'action_type': action_type, 'target': element})
    elif action_type == 'TYPE':
        action_type = 'input_text'
        action_str = str({'action_type': action_type, 'target': element, 'text': value})
    elif action_type == 'SELECT':
        action_type = 'select'
        action_str = str({'action_type': action_type, 'target': element, 'value': value})
    
    return '', thought, '', action_str, ''

thought_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
thinking_pattern = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)
task_pattern = re.compile(r'<task>(.*?)</task>', re.DOTALL)
observation_pattern = re.compile(r'<observation>(.*?)</observation>', re.DOTALL)
progress_pattern = re.compile(r'<progress>(.*?)</progress>', re.DOTALL)
exception_pattern = re.compile(r'<exception>(.*?)</exception>', re.DOTALL)
decision_pattern = re.compile(r'<decision>(.*?)</decision>', re.DOTALL)
intent_pattern = re.compile(r'<intent>(.*?)</intent>', re.DOTALL)
anticipation_pattern = re.compile(r'<prediction>(.*?)</prediction>', re.DOTALL)

def extract_protocol_components(response):
    """'<think>\n<task> The user wants to find the best route from their home to Takai Sushi Restaurant using the Maps app.</task>\n<observation> The current screen shows a search bar at the top with the placeholder text "Choose destination." Below the search bar, there are several suggested locations, including "Home," "Rua Bento Gonçalves, 1607," "Takai Sushi," "Torres," "Rio de Janeiro," and "São Paulo." The keyboard is open, indicating that the user can type a destination.</observation>\n<progress> The user has already selected "Home" as the starting point. Now, they need to enter "Takai Sushi" as the destination to proceed with finding the route.</progress>\n<exception> No exceptions have been detected so far.</exception>\n<decision> Since the user has already selected "Home" as the starting point, the next logical step is to enter "Takai Sushi" as the destination. This will allow the app to calculate the route from the user\'s home to the restaurant.</decision>\n<intent> Type "Takai Sushi" into the search bar.</intent>\n<prediction> After typing "Takai Sushi," the app will suggest the restaurant location, allowing the user to select it as the destination and proceed with finding the route.</prediction>\n</think>\n<action> {"action_type": "type", "text": "Takai Sushi"} </action>'"""

    thought_match = thought_pattern.search(response)
    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        thought_match = thinking_pattern.search(response)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = ""

    task = task_pattern.search(thought).group(1).strip() if task_pattern.search(thought) else ""
    observation = observation_pattern.search(thought).group(1).strip() if observation_pattern.search(thought) else ""
    progress = progress_pattern.search(thought).group(1).strip() if progress_pattern.search(thought) else ""
    exception = exception_pattern.search(thought).group(1).strip() if exception_pattern.search(thought) else ""
    decision = decision_pattern.search(thought).group(1).strip() if decision_pattern.search(thought) else ""
    intent = intent_pattern.search(thought).group(1).strip() if intent_pattern.search(thought) else ""
    prediction = anticipation_pattern.search(thought).group(1).strip() if anticipation_pattern.search(thought) else ""

    action_part = response[response.rfind('<action>')+8:].strip()
    if '{"action' in response:
        action = response[response.find('{"action'):response.rfind('}')+1]
    elif '[[' in action_part:
        xy = action_part.split('[[')[-1].split(']]')[0]
        action = f"""{{"action_type": "click", "target": ({xy})}}"""
    elif 'SCROLL' in action_part:
        direction = action_part.split('[')[1].split(']')[0].lower()
        if direction == 'down': direction = 'up'
        elif direction == 'up': direction = 'down'
        action = f"""{{"action_type": "swipe", "direction": {direction}}}"""
    elif 'OPEN_APP' in action_part:
        app = action_part.split('[')[1].split(']')[0].lower()
        action = f"""{{"action_type": "open_app", "app": {app}}}"""

    return {
        "task": task,
        "observation": observation,
        "progress": progress,
        "exception": exception,
        "decision": decision,
        "intent": intent,
        "prediction": prediction,
        "thought": thought,
        "action": action
    }
 
def extract_all_action_jsons(text):
    """
    Extracts all JSON action dictionaries from a given string.

    Args:
        text: The input string containing action dictionaries.

    Returns:
        A list of dictionaries, where each dictionary represents an action.
    """
    actions = []
    # Regular expression to find JSON objects that look like actions
    # We are looking for {"action": "...", ...}
    # This regex is a bit more robust to capture full JSON objects
    # It looks for "{" followed by "action": and then captures
    # everything until the matching "}"
    # However, a simpler approach given the specific format is to find all
    # lines that start with "{" and end with "}" and try to parse them.

    # A more robust way would be to use a proper JSON parser that can
    # extract multiple JSON objects from a string, but for this specific
    # format where each action is on its own line (or clearly delimited),
    # we can iterate and try to parse.

    for line in text.splitlines():
        line = line.strip()
        if '{"action' in line:
            try:
                action = json.loads(line)
                if isinstance(action, dict) and "action" in action:
                    actions.append(action)
            except json.JSONDecodeError:
                # Not a valid JSON object, or not a complete one
                actions.append(None)
    return actions

def lower_first_letter(s):
    if not s:
        return s
    return s[0].lower() + s[1:]

def extract_thought_components_UITARS(response):
    response = response.replace("```json", "").replace("```", "").strip()

    parts = response.split("Action:")
    thought, action_pred_raw = parts[0].strip(), parts[-1].strip()

    summary = thought.strip().split('\n')[-1].strip()
    funcdesc = "This element is used to " + summary

    return '', thought, funcdesc, action_pred_raw, summary

def extract_thought_components_aguvis(response):
    # "Action: Click on the 'TikTok Series' option to explore music suitable for TikTok series.\n\nassistantall\nThought: To find the romantic reggae music from BCD Studio for TikTok series in Andorra, I need to select the 'TikTok Series' option to narrow down the search to relevant content.\nAction: Click on the 'TikTok Series' option to explore music suitable for TikTok series.\n\nassistantos\npyautogui.click(x=0.097, y=0.7667)"
    response = response.replace("```json", "").replace("```", "").strip()

    if 'Thought:' in response:
        response = response[response.find("Thought:"):].strip()
        parts = response[8:].split("Action:")
        thought, action_raw = parts[0].strip(), parts[-1].strip()
    else:
        thought = ''
        action_raw = response.split("Action:")[-1]

    summary = lower_first_letter(action_raw.split("\n")[0].strip())
    action_pred_raw = action_raw.split("\n")[-1].strip()
    funcdesc = "This element is used to " + summary

    return '', thought, funcdesc, action_pred_raw, summary

def extract_thought_components_llamav(response):
    # **Step 1: Observation**

    # The current screenshot shows a Google search page with a search bar at the top and various icons below it, including Gmail, Photos, YouTube, Google Maps, Chrome, Google Drive, and Google Calendar. The search bar is empty, and the cursor is blinking, indicating that it is ready to receive input.

    # **Step 2: Thought**

    # To find the news in Indonesia, we need to search for relevant keywords on Google. Since the task is to find news in Indonesia, we can start by typing "news" in the search bar and then add "Indonesia" to narrow down the results.

    # **Step 3: Target's Functionality**

    # This element enables users to input text into the search bar to retrieve relevant search results.

    # **Step 4: Action**

    # Input text: {"action_type": "input_text", "text": "news Indonesia"}

    # **Step 5: Summary**

    # I typed "news Indonesia" into the search bar to retrieve relevant search results.

    # extract observation
    obs_start = response.find("Observation")
    thought_start = response.find("Thought")
    obs = response[obs_start:thought_start].strip().split("**")[1].strip()

    # extract thought
    funcdesc_start = response.find("Target's")
    thought = response[thought_start:funcdesc_start].strip().split("**")[1].strip()

    # extract funcdesc
    action_start = response.find(": Action")
    funcdesc = response[funcdesc_start:action_start].strip().split("**")[1].strip()

    # extract action
    action_start = response.find('{"action_type":')
    action_end = response.find('}', action_start)
    action_pred_raw = response[action_start:action_end+1]

    # extract summary
    summary_start = response.find("Summary")
    summary = response[summary_start:].strip().split("**")[1].strip()

    return obs, thought, funcdesc, action_pred_raw, summary
