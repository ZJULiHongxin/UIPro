import cv2, re, random, json, os, math, magic
import unicodedata
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

from typing import Any
from tqdm import tqdm

from utils.data_utils.task_prompt_lib import *

def convert_conv_tags(data: list, new: str, old: str) -> list:
    for x in tqdm(data, total=len(data)):
        if CONV_TAGS[old]['conv_tag'] not in x: continue
        convs = x.pop(CONV_TAGS[old]['conv_tag'])
        
        for turn_i, turn in enumerate(convs):
            convs[turn_i].pop(CONV_TAGS[old]['role_tag'])

            if turn_i % 2 == 0:
                convs[turn_i][CONV_TAGS[new]['role_tag']] = CONV_TAGS[new]['user_tag']
                convs[turn_i][CONV_TAGS[new]['content_tag']] = convs[turn_i].pop(CONV_TAGS[old]['content_tag'])
            else:
                convs[turn_i][CONV_TAGS[new]['role_tag']] = CONV_TAGS[new]['assistant_tag']
                convs[turn_i][CONV_TAGS[new]['content_tag']] = convs[turn_i].pop(CONV_TAGS[old]['content_tag'])
        
        x[CONV_TAGS[new]['conv_tag']] = convs

    return data

def load_json(file: str) -> list:
    if file.endswith('json'):
        return json.load(open(file))
    elif file.endswith('jsonl'):
        data = []
        with open(file) as f:
            for line in f:
                data.append(json.loads(line))
        return data

def write_json(data: list, file: str):
    if file.endswith('json'):
        with open(file, 'w') as f:
            json.dump(data, f, indent=2 if len(data) < 2000 else 0)
    elif file.endswith('jsonl'):
        with open(file, 'w') as f:
            for x in data:
                f.write(json.dumps(x) + '\n')

def add_text(img, text, font_scale=0.6, font_color=(0,0,0)):
    height, width = img.shape[:2]
    # Define the size of the new region to add
    new_region_height = 40  # Adjust as needed

    # Create a new blank image with extra space
    new_image = np.zeros((height + new_region_height, width, 3), dtype=np.uint8)

    # Copy the original image to the top of the new image
    new_image[:height, :] = img

    # Fill the new region with a color (e.g., white)
    new_image[height:, :] = [255, 255, 255]  # White color

    # Now you can add text to the new region
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height + (new_region_height + text_size[1]) // 2
    cv2.putText(new_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
    
    return new_image

def classify_node(node):
    """
    Classify a node as either an icon without text or a pure-text element.
    
    Parameters:
        node (dict): A dictionary representing the node in the AX tree.

    Returns:
        str: 'Icon' if the node is an icon without text, 'Text' if it is a pure-text element,
             'Unknown' if it does not fit either category.
    """
    if 'button' in node.get('desc',''):
        return 'Icon'

    # Check if the node is an image
    if 'image' in node.get('class', '').lower():
        return 'Image'

    # Check if the node has text
    text = node.get('text', None)
    has_text = text is not None and bool(text.strip())
    
    content_description = node.get('content_description', node.get('content-desc', None))
    has_content_description = content_description is not None and bool(content_description.strip())

    # Rule: If it has text, it's a pure-text element
    if has_text and not has_content_description:
        return 'Text'

    # If it doesn't match any of the rules, return 'Icon'
    return 'Icon'

def is_uniform_region(image, roi):
    """
    Check if a region in a grayscale image contains a uniform color.

    Parameters:
        image (numpy.ndarray): The input grayscale image.
        roi (tuple): A tuple (x, y, width, height) defining the region of interest.

    Returns:
        bool: True if the region is of uniform color, False otherwise.
    """
    x1, y1, x2, y2 = roi
    region = image[y1:y2, x1:x2]

    # Check if all pixel values in the region are the same
    if region.ndim == 2:  # Grayscale image
        is_uniform = np.all(region == region[0, 0])
    else:
        # Reshape the region to a 2D array of pixel colors
        reshaped_region = region.reshape(-1, 3)  # Assuming a 3-channel (RGB/BGR) image

        # Check if all pixels are the same color
        is_uniform = np.all(reshaped_region == reshaped_region[0])
    
    return is_uniform

def is_pure_color(image, roi, threshold=5):
    """
    Determine if a region of an image is of pure color, supporting both grayscale and color images.
    
    Parameters:
        image (numpy.ndarray): The image in which to check the color purity.
        roi (tuple): A tuple of (x, y, width, height) specifying the region of interest.
        threshold (int or float): The threshold for the standard deviation to consider the region as pure color.
    
    Returns:
        bool: True if the region is of pure color, False otherwise.
    """
    # Extract the region of interest from the image
    x1, y1, x2, y2 = roi
    region = image[y1:y2, x1:x2]

    # Check if the image is grayscale or color
    if len(region.shape) == 2:
        # Grayscale image (2D array)
        std_dev = np.std(region)
    else:
        # Color image (3D array)
        std_dev = np.std(region, axis=(0, 1))

    # Check if the standard deviation is below the threshold for all color channels (or the single channel in grayscale)
    if isinstance(std_dev, np.ndarray):
        return np.all(std_dev < threshold)
    else:
        return std_dev < threshold

def contains_chinese(text):
    # Regular expression to match Chinese characters
    pattern = re.compile('[\u4e00-\u9fff]')
    return bool(pattern.search(text))
    
def contains_japanese(text):
    return bool(re.search(r'[\u3040-\u30FF\uFF00-\uFFEF\u30A0-\u30FF]', text))

def contains_russian(text):
    return bool(re.search(r'[\u0400-\u04FF]', text))

def contains_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def contains_korean(text):
    return bool(re.search(r'[\uAC00-\uD7AF]', text))

def detect_invalid_lang(text):
    return (contains_japanese(text) or 
            contains_russian(text) or 
            contains_arabic(text) or 
            contains_korean(text))


def is_valid_string(s):
    """
    Check if the input string contains only English letters, Chinese characters, digits, and common symbols.

    Parameters:
        s (str): The input string to check.

    Returns:
        bool: True if the string contains only valid characters, False otherwise.
    """
    pattern = r'^[A-Za-z0-9\u4e00-\u9fff.,!?;:()\[\]{}<>@#$%^&*+\-=_~\s]*$'
    return bool(re.match(pattern, s))


import spacy

# Load the spaCy model

class TextProcessor:
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")

    def extract_main_clause(self, sentence):
        """
        # Example sentence
        sentence = 'Click the "Change Location" button to update the city to New York City.'
        main_clause = extract_main_clause(sentence)
        print(main_clause)
        # Output: Click the "Change Location" button
        """
        # Parse the sentence
        doc = self.nlp(sentence)

        # Initialize variables to store the main clause
        main_clause_tokens = []

        # Iterate through the tokens in the sentence
        for token in doc:
            # Stop when encountering an infinitive marker ("to") or a preposition
            if token.text.lower() == "to" and (token.dep_ == "aux" or token.dep_ == "prep"):
                break
            main_clause_tokens.append(token.text)
        
        # Join the tokens to form the main clause, preserving original spacing
        main_clause = "".join([token.text_with_ws for token in doc[:len(main_clause_tokens)]])
            
        # Join the tokens to form the main clause
        main_clause = " ".join(main_clause_tokens) if len(main_clause_tokens) >= 4 else sentence
        return lower_first_letter(main_clause).replace('" ', '"')


import numpy as np

def bbox_iou_np(target_box, other_boxes):
    # Convert to numpy arrays
    target_box = np.array(target_box)
    other_boxes = np.array(other_boxes)

    # Calculate coordinates of intersection
    inter_x1 = np.maximum(target_box[0], other_boxes[:, 0])
    inter_y1 = np.maximum(target_box[1], other_boxes[:, 1])
    inter_x2 = np.minimum(target_box[2], other_boxes[:, 2])
    inter_y2 = np.minimum(target_box[3], other_boxes[:, 3])

    # Calculate intersection area
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    # Calculate area of the target box and other boxes
    target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
    other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

    # Calculate IoU
    union_area = target_area + other_areas - inter_area
    iou = inter_area / union_area

    return iou

def is_box_overlapping_np(target_box, other_boxes, threshold):
    iou_values = bbox_iou_np(target_box, other_boxes)
    return np.any(iou_values > threshold)


import base64

def decode_img_base64(base64_string):
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    return image_data

"""
LinearLayout text: '' 
        TextView text: 'WebView Browser Tester 69.0.3497.100' resource-id: android:id/title
        LinearLayout text: '' resource-id: org.chromium.webview_shell:id/container
                LinearLayout text: '' 
                        EditText text: 'de' resource-id: org.chromium.webview_shell:id/url_field, clickable: true, focused: true
                        ImageButton text: 'Load URL' clickable: true
                        ImageButton text: 'About WebView' clickable: true
                WebView text: '' clickable: true
"""
"""
An Axtree node owns these attrs: ['unique_id', 'bounds_in_screen', 'class_name', 'content_description', 'hint_text', 'package_name', 'text', 'text_selection_start', 'text_selection_end', 'view_id_resource_name', 'window_id', 'is_checkable', 'is_checked', 'is_clickable', 'is_editable', 'is_enabled', 'is_focusable', 'is_focused', 'is_long_clickable', 'is_password', 'is_scrollable', 'is_selected', 'is_visible_to_user', 'actions', 'child_ids', 'clickable_spans', 'depth', 'labeled_by_id', 'label_for_id', 'drawing_order', 'tooltip_text']
"""

# ANDROID_ACTION_ID References: 
# 1. https://github.com/Scarabei/AndroidArtifacts/blob/master/com.android.support.support-compat.26.0.0/src/android/support/v4/view/accessibility/AccessibilityNodeInfoCompat.java
# 2. https://www.capa.run/static/image/img_media/Android%E6%97%A0%E9%9A%9C%E7%A2%8D%E7%B1%BB_AccessibilityNodeInfo.pdf
ANDROID_ACTION_ID = {
    1: 'is_focusable',
    2: 'is_focusable',
    64: 'is_focusable',
    128: 'is_focusable',
    256: 'is_focusable',
    512: 'is_focusable',
    4: 'is_selected',
    8: 'is_selected',
    16: 'is_clickable',
    262144: 'is_clickable',
    524288: 'is_clickable',
    32: 'is_long_clickable',
    4096: 'is_scrollable',
    8192: 'is_scrollable',
    32768: 'is_editable',
    2097152: 'is_editable',
    131072: 'is_checkable',
    16908342: '',
    16908343: '',# AccessibilityActionShowOnScreen = 16908342: The Resource.Id.AccessibilityActionShowOnScreen field in Android refers to an accessibility action that allows a user to show an element on the screen. This action is particularly useful in scenarios where the UI element is not currently visible to the user, such as when it is off-screen or obscured by other elements.
}

def set_node_attrs(node):
    for action_id in node.actions:
        action_type = ANDROID_ACTION_ID.get(action_id.id, '')
        if action_type == 'is_focusable':
            node.is_focusable = True
        elif action_type == 'is_selected':
            node.is_selected = True
        elif action_type == 'is_clickable':
            node.is_clickable = True
        elif action_type == 'is_long_clickable':
            node.is_long_clickable = True
        elif action_type == 'is_editable':
            node.is_editable = True
        elif action_type == 'is_checkable':
            node.is_editable = True

# In XML, the following characters have special meanings and must be escaped to avoid parsing errors:

#     Ampersand (&) → Use &amp;
#     Less than (<) → Use &lt;
#     Greater than (>) → Use &gt;
#     Double quote (") → Use &quot; when inside an attribute value
#     Single quote (') → Use &apos; when inside an attribute value

def replace_special_chars(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
    return text

def tree_to_string_and_attrs(node, tree_dict, ratio, level=0):
    # NODE TEXT-RELATED -ATTRS: text, content_description, hint_text, tooltip_text
    node_desc = ''
    text_key = ''
    
    node.text = replace_special_chars(node.text)
    node.content_description = replace_special_chars(node.content_description)
    node.hint_text = replace_special_chars(node.hint_text)
    node.tooltip_text = replace_special_chars(node.tooltip_text)
    
    if node.text:
        node_desc = node.text; text_key = 'text'
    elif node.content_description:
        node_desc = node.content_description; text_key = 'content-desc'
    elif node.hint_text:
        node_desc = node.hint_text; text_key = 'hint-text'
    elif node.tooltip_text:
        node_desc = node.tooltip_text; text_key = 'tooltip-text'
    # elif node.view_id_resource_name:
    #     node_desc = node.view_id_resource_name.split('/')[-1]

    node_cls = node.class_name.split('.')[-1]
    is_leaf = len(node.child_ids) == 0
    
    set_node_attrs(node)
    is_interactable = any([node.is_clickable, node.is_checkable, node.is_editable, node.is_long_clickable, node.is_scrollable, node.is_focusable, node.is_password])

    # 保存UI截图时，会降低其分辨率，所以这里一并调整bbox尺度
    x1, y1, x2, y2 = round(node.bounds_in_screen.left * ratio), round(node.bounds_in_screen.top * ratio), round(node.bounds_in_screen.right * ratio), round(node.bounds_in_screen.bottom * ratio)

    node_key_attrs = []
    if node.is_visible_to_user and text_key:
        node_key_attrs.append({'class': node_cls, 'box': [x1, y1, x2, y2], 'node_text': node.text,
                       'node_desc': node_desc, 'resource_id': node.view_id_resource_name.split('/')[-1], 'is_leaf': is_leaf, 'is_interactable': is_interactable, 'text_key': text_key, 'action_ids': [x.id for x in node.actions]})
        result = ' ' * level * 2 + f'<{node.class_name} index="{node.unique_id}" package="{node.package_name}" class="{node.class_name}" text="{node.text}" content-desc="{node.content_description}" hint-text="{node.hint_text}" tooltip-text="{node.tooltip_text}" checkable="{node.is_checkable}" checked="{node.is_checked}" clickable="{node.is_clickable}" enabled="{node.is_enabled}" focusable="{node.is_focusable}" focused="{node.is_focused}" long-clickable="{node.is_long_clickable}" password="{node.is_password}" scrollable="{node.is_scrollable}" selected="{node.is_selected}" bounds="[{x1},{y1}][{x2},{y2}]" displayed="{node.is_visible_to_user}">\n'
    else:
        result = ''

    for child_id in node.child_ids:
        if child_id not in tree_dict: continue
        child_string, child_attrs = tree_to_string_and_attrs(tree_dict[child_id], tree_dict, ratio, level + 1)
        result += child_string
        node_key_attrs.extend(child_attrs)
    
    if node.is_visible_to_user and text_key:
        result += ' ' * level * 2 + f'</{node.class_name}>\n'

    return result, node_key_attrs


def parse_axtrees_proto(axtrees, ratio):
    tree_str_lst, node_key_attrs_lst = [], []
    for axtree in axtrees:
        nodes = axtree.tree.nodes
        
        tree_dict = {node.unique_id: node for node in nodes} # unique_id = 0 不意味着这个结点是根节点，现在取列表第一个元素为根节点

        tree_str, node_key_attrs = tree_to_string_and_attrs(nodes[0], tree_dict, ratio)
        
        tree_str_lst.append(tree_str); node_key_attrs_lst.append(node_key_attrs)
    
    return tree_str_lst, node_key_attrs_lst

def resize_image(img, max_size=1008):
    height, width = img.shape[:2]

    if max(width, height) <= max_size:
        return img, 1.0  # No resizing, return original ratio

    # Calculate the new dimensions while maintaining aspect ratio
    if width > height:
        new_width = max_size
        ratio = max_size / width
        new_height = int(height * ratio)
    else:
        new_height = max_size
        ratio = max_size / height
        new_width = int(width * ratio)

    resized_img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img, ratio

import numpy as np

def calculate_iou_matrix(bounding_boxes):
    """
    Calculate the IoU matrix for all pairs of bounding boxes using numpy.
    bounding_boxes should be a numpy array of shape (n, 4) where each row is [x1, y1, x2, y2].
    """
    # Extract the coordinates
    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    x2 = bounding_boxes[:, 2]
    y2 = bounding_boxes[:, 3]
    
    # Calculate the area of each bounding box
    areas = (x2 - x1) * (y2 - y1)
    
    # Calculate the intersection coordinates
    x1_inter = np.maximum(x1[:, None], x1[None, :])
    y1_inter = np.maximum(y1[:, None], y1[None, :])
    x2_inter = np.minimum(x2[:, None], x2[None, :])
    y2_inter = np.minimum(y2[:, None], y2[None, :])
    
    # Calculate the intersection area
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Calculate the union area
    union_area = areas[:, None] + areas[None, :] - inter_area
    
    # Calculate IoU
    iou_matrix = np.where(union_area > 0, inter_area / union_area, 0)
    
    return iou_matrix

def average_iou(bounding_boxes):
    """
    Compute the average IoU for all pairs of bounding boxes.
    """
    iou_matrix = calculate_iou_matrix(bounding_boxes)
    
    # Take the upper triangle of the matrix (excluding diagonal) to avoid double-counting
    n = len(bounding_boxes)
    iou_sum = np.sum(np.triu(iou_matrix, k=1))
    
    # Compute the number of unique pairs (n choose 2)
    num_pairs = n * (n - 1) / 2
    
    # Return the average IoU
    return iou_sum / num_pairs if num_pairs > 0 else 0.0

# # Example usage:
# bounding_boxes = np.array([[10, 10, 20, 20], [15, 15, 25, 25], [18, 18, 28, 28]])
# average_overlap = average_iou(bounding_boxes)
# print(f"Average IoU: {average_overlap:.4f}")

def find_smallest_box_containing_point(point, boxes):
    """
    Find the smallest box that contains the point from a list of boxes using NumPy for efficiency,
    and return both the box and its index in the original array.
    :param point: A tuple (px, py) representing the point.
    :param boxes: A NumPy array of shape (n, 4) where each row is a box defined as (x1, y1, x2, y2).
    :return: A tuple containing the smallest box and its index, or (None, None) if no such box exists.
    """
    px, py = point
    # Extract coordinates for easier reading
    if isinstance(boxes, list):
        boxes = np.array(boxes)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Check if the point is inside each box using broadcasting
    inside = (x1 <= px) & (px <= x2) & (y1 <= py) & (py <= y2)

    # Calculate the area of each box
    areas = (x2 - x1) * (y2 - y1)

    # Filter boxes that contain the point and have the smallest area
    if np.any(inside):
        # Find the index of the box with the smallest area among those that contain the point
        filtered_areas = areas[inside]
        min_area_index = np.argmin(filtered_areas)
        # Find the index in the original array
        original_index = np.where(inside)[0][min_area_index]
        return (boxes[original_index].tolist(), original_index)
    else:
        return (None, None)


from xml.etree.ElementTree import Element
from lxml import html

def load_html_from_file(filepath):
    """Loads an HTML file using lxml.

    Args:
        filepath: The path to the HTML file.

    Returns:
        An lxml.html.HtmlElement object representing the parsed HTML tree, or None if an error occurs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:  # Explicitly handle encoding
            html_content = f.read()
            tree = html.fromstring(html_content)
            return tree
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except lxml.etree.ParserError as e: #Catching parsing errors
        print(f"Error parsing HTML: {e}")
        return None
    except Exception as e: #Catching other exceptions
        print(f"An unexpected error occurred: {e}")
        return None

XML_BOX_PATTERN = re.compile(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]')
def find_all_elem_texts_boxes(element: Element):
    """
    Recursively find all interactable elements in the XML tree.
    """
    elem_texts_boxes = []

    box = None
    is_leaf = False

    if (element.get('clickable') == 'true' or
        element.get('focusable') == 'true' or
        element.get('long-clickable') == 'true' or
        element.get('password') == 'true' or
        element.get('checkable') == 'true'):
        is_interactable = True
    else: is_interactable = False

    box_str = element.get('bounds', None)
    if box_str is not None:
        coords = XML_BOX_PATTERN.search(box_str)
        if coords:
            x1, y1, x2, y2 = map(int, coords.groups())
            box = [x1, y1, x2, y2]

    if len(element) == 0:
        is_leaf = True
    else:
        # Recursively check children
        for child in element:
            elem_texts_boxes.extend(find_all_elem_texts_boxes(child))
    
    elem_cls = element.tag if getattr(element, 'tag', None) not in ['node', None] else element.attrib['class']

    elem_texts_boxes.append({'tag':elem_cls.split('.')[-1], 'text':element.attrib.get('text', None), 'box': box, 'package': element.attrib.get('package', None), 'content-desc': element.attrib.get('content-desc', None), 'resource_id': element.attrib.get('resource_id', element.attrib.get('resource-id', None)), 'is_leaf': is_leaf, 'is_interactable': is_interactable})
    return elem_texts_boxes



NUMBERING_PATTERN = re.compile(r'\[\d+\]')

IN_VIEWPORT_RATIO_THRESHOLD = 0.6
INVALID_NODE_ROLES = ["generic", "img", "list", "strong", "paragraph", "banner", "navigation", "Section", "LabelText", "Legend", "listitem", "alert", "superscript", "LineBreak", "Canvas"]
IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "disabled",
    "describedby",
    "roledescription"
)

def prune_accessibility_tree_wo_bound(
    accessibility_tree,
) -> tuple[str, dict[str, Any]]:
    """Parse the accessibility tree into a string text"""
    
    def remove_node_in_graph(node) -> None:
        # update the node information in the accessibility tree
        nodeid = node["nodeId"]
        parent_nodeid = node["parentId"]
        
        # If the node's parent is not in the accessibility tree, remove it. This case happen when the AXtree only contains the nodes inside the viewport.
        if parent_nodeid not in accessibility_tree:
            accessibility_tree[nodeid]["parentId"] = "[REMOVED]"
            return

        children_nodeids = node["childIds"]
        # update the children of the parent node
        assert (
            accessibility_tree[parent_nodeid].get("parentId", "Root")
            is not None
        )
        # remove the nodeid from parent's childIds
        try:
            index = accessibility_tree[parent_nodeid]["childIds"].index(
                nodeid
            )
            accessibility_tree[parent_nodeid]["childIds"].pop(index)
        except:
            index = len(accessibility_tree[parent_nodeid]["childIds"])
        
        # Insert children_nodeids in the same location
        for child_nodeid in children_nodeids:
            accessibility_tree[parent_nodeid]["childIds"].insert(
                index, child_nodeid
            )
            index += 1
        # update children node's parent
        for child_nodeid in children_nodeids:
            if child_nodeid not in accessibility_tree: continue
            accessibility_tree[child_nodeid][
                "parentId"
            ] = parent_nodeid
        # mark as removed
        accessibility_tree[nodeid]["parentId"] = "[REMOVED]"
    
    for obs_node_id, node in accessibility_tree.items():
        valid_node = True
        try:
            role = node["role"]["value"]
            name = node["name"]["value"]

            node_str = f"[{obs_node_id}] {role} {repr(name)}"
            properties = []
            
            if role == 'textbox':
                for x in node["name"]['sources']:
                    if x['type'] == 'placeholder' and 'value' in x.keys():
                        properties.append(f"placeholder: [{x['value']['value']}]")
            
            for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    properties.append(
                        f'{property["name"]}: {property["value"]["value"]}'
                    )
                except KeyError:
                    pass

            if properties:
                node_str += " " + " ".join(properties)

            # check valid
            if not node_str.strip():
                valid_node = False

            # empty generic node
            if not name.strip():
                if not properties:
                    if role in INVALID_NODE_ROLES:
                        valid_node = False
                elif role in ["listitem"]:
                    valid_node = False

            if not valid_node:
                remove_node_in_graph(node)
                continue
        except Exception as e:
            valid_node = False
            remove_node_in_graph(node)
    
    for nodeId in list(accessibility_tree.keys()):
        if accessibility_tree[nodeId].get("parentId", "-1") == "[REMOVED]":
            del accessibility_tree[nodeId]

    return accessibility_tree

BROKEN_WEB_TAGS = [
    '400 Bad Request',
    '403 Forbidden',
    '404 Not Found',
    '405 Method Not Allowed',
    '406 Not Acceptable',
    '408 Request Timeout',
    '409 Conflict',
    '410 Gone',
    '411 Length Required',
    '412 Precondition Failed',
    '413 Payload Too Large',
    '414 URI Too Long',
    '415 Unsupported Media Type',
    '416 Range Not Satisfiable',
    '417 Expectation Failed',
    '502 Bad Gateway',
    '503 Service Unavailable',
    '504 Gateway Timeout',
    '505 HTTP Version Not Supported',
    '506 Variant Also Negotiates',
    '507 Insufficient Storage',
    '508 Loop Detected',
    '510 Not Extended',
    '511 Network Authentication Required'
]

def contain_network_errors(text):
    for error in BROKEN_WEB_TAGS:
        if error in text:
            return True
    return False

import spacy
from http import HTTPStatus
from typing import List

class VerbExtactor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def find_first_verb(self, text):
        # Process the text
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ in ["VERB", 'AUX', "MD"] and token.tag_ not in ["VBD", "VBN"] and token.dep_ not in ["amod", "acomp"]:  # Check if the token is a verb, auxiliary verb such as "is" or "are," or modal verb. The verb should not be used as an adjective.
                # Find the start position of the verb in the paragraph
                start_index = token.idx
                return token.text, start_index
        return None, -1  # Return a message and -1 if no verb is found

def box2center(box, W, H, scale):
    center_x, center_y = (box[0]+box[2]) / 2, (box[1]+box[3]) / 2
    normalized_center = [max(0, min(scale-1, round(center_x / W * scale))), max(0, min(scale-1, round(center_y / H * scale)))]
    return normalized_center

def generate_neg_point(gt_box, W, H, scale, num_points=10):
    neg_points = []
    for _ in range(num_points):
        while True:
            x = random.randint(0, W)
            y = random.randint(0, H)
            if not (gt_box[0] <= x <= gt_box[2] and gt_box[1] <= y <= gt_box[3]):
                neg_points.append([max(0, min(scale-1, round(x / W * scale))), max(0, min(scale-1, round(y / H * scale)))])
                break
        
    return neg_points

def is_point_in_box(point, box):
    is_in = False
    if isinstance(box, str):
        box = XML_BOX_PATTERN.search(box)

    if box is not None:
        box = box.groups()
        if len(box) == 4:
            box = list(map(int, box))
            is_in = box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
    return is_in

def remove_abnormal_unicode(text):
    # Filter out characters that are not considered printable
    return ''.join(
        c for c in text
        if unicodedata.category(c)[0] != 'C'  # 'C' stands for control characters
    )

DEFAULT_NODE_TAG = 'android.widget.TextView'
CORRECT_NESTED_TAG_CLASS_PATTERN = r'(<[^/>]*?)\$([^>]*?>)|(<\/[^>]*?)\$([^>]*?>)'
def preprocess_xml_content(xml_content):
    if '< index=' in xml_content:
        xml_content = xml_content.replace('< index=', f'<{DEFAULT_NODE_TAG} index=').replace('</>', f'</{DEFAULT_NODE_TAG}>')
    
    if '$' in xml_content:
        xml_content = re.sub(CORRECT_NESTED_TAG_CLASS_PATTERN, lambda m: f"{m.group(1) or m.group(3)}_{m.group(2) or m.group(4)}", xml_content)

    xml_content = remove_abnormal_unicode(xml_content)

    return xml_content

def parse_xml_to_tree(xml_content):
    # Parse the XML content
    root = ET.fromstring(preprocess_xml_content(xml_content))
    return root

# In XML, the following characters have special meanings and must be escaped to avoid parsing errors:

#     Ampersand (&) → Use &amp;
#     Less than (<) → Use &lt;
#     Greater than (>) → Use &gt;
#     Double quote (") → Use &quot; when inside an attribute value
#     Single quote (') → Use &apos; when inside an attribute value
def decode_special_chars(text):
    return text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&apos;', "'")


def simplify_tree(node, target_box=None, target_point=[-1,-1]):
    """
    Recursively simplify the tree:
    1. Remove nodes with only one child if they do not add any significant info.
    2. Remove unnecessary attributes.
    """
    # Simplify the child nodes first
    children = list(node)
    for child in children:
        simplify_tree(child, target_box, target_point)
    
    # Remove unnecessary attributes from the node
    useful_attributes = ['text', 'resource-id', 'clickable', 'bounds', 'content-desc', 'hint-text', 'tooltip-text']
    if node.tag == 'node' and node.attrib.get('class', ''):
        node.tag = node.attrib['class']
    if node.attrib.get('checkable', 'false') == 'true': useful_attributes.append('checked')
    if node.attrib.get('focusable', 'false') == 'true': useful_attributes.append('focused')
    if node.attrib.get('password', 'false') == 'true': useful_attributes.append('password')

    node.attrib = {key: node.attrib[key] for key in useful_attributes if key in node.attrib and node.attrib[key]}
    #'resource-id' not in node.attrib and 
    # Merge the node with its only child if appropriate
    node_bound = node.attrib.get('bounds','')
    if len(children) == 1 and node_bound != target_box and not is_point_in_box(target_point, node_bound) and not node.attrib.get('text', '') and not node.attrib.get('content-desc', '') and all(node.attrib.get(k, 'false') == 'false' for k in ['clickable', 'checked', 'focused', 'password']):
        child = children[0]
        # If the current node and the child node are of the same type and no significant info is lost, merge them
        node.tag = child.tag
        node.attrib.update(child.attrib)
        node[:] = child[:]  # Adopt the children of the child node
        return

    # # If the node has more than one child, ensure it's kept
    # if len(children) > 1:
    #     return

XML_BOX_PATTERN = re.compile(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]')

def tree_to_text(node, level=0, all_boxes=None, skip_statusbar=False):
    """
    Convert the simplified tree to a text format.
    """
    # Recursively add children
    if skip_statusbar and '_statusbar' in node.tag or len(node) == 0 and not node.attrib.get('text', '') and not node.attrib.get('content-desc', '') and all(node.attrib.get(k, 'false') == 'false' for k in ['clickable', 'checked', 'focused', 'password']):
        subtree_str = ''
    else:
        indent = '\t' * level
        
        node_text = decode_special_chars(node.attrib.get('text',''))
        if len(node_text) == 0:
            node_text = decode_special_chars(node.attrib.get('content-desc',''))

        node_info = "{} text: '{}' ".format(node.tag.split('.')[-1], node_text.replace('\n', ' '))
        
        # Add additional descriptions
        hint_text = node.attrib.get('hint-text', '').strip()
        if hint_text and hint_text != node_text:
            node_info += f"hint-text: '{hint_text}' "
        
        tooltip_text = node.attrib.get('tooltip-text', '').strip()
        if tooltip_text and tooltip_text != node_text:
            node_info += f"tooltip-text: '{tooltip_text}' "

        # Add attributes info
        node_info += ', '.join(f"{k}: {v}" for k,v in node.attrib.items() if k not in ['text', 'bounds', 'content-desc', 'hint-text', 'tooltip-text'] and v != 'false')
        
        # box
        coords = XML_BOX_PATTERN.search(node.attrib.get('bounds',''))
        if coords:
            x1, y1, x2, y2 = map(int, coords.groups())
            box = [x1, y1, x2, y2]
        
            node_info += f', Box: [{x1},{y1},{x2},{y2}]'
        else: box = None

        if all_boxes is not None:
            all_boxes.append(box)
        
        child_texts = []
        for child in node:
            child_texts.append(tree_to_text(child, level + 1, all_boxes=all_boxes, skip_statusbar=skip_statusbar))

        subtree_str = f"{indent}{node_info}\n" + ''.join(child_texts)
    
    return subtree_str

def process_xml(xml_file, target_box=None, target_point=[-1,-1], resume=True, skip_statusbar=True, skip_lang=False):
    proc_file = xml_file.replace('.xml','_axtree.json')
    if xml_file.endswith(".xml") or xml_file.endswith("_xml.txt"):
        if resume and os.path.exists(proc_file):
            with open(proc_file) as f:
                axtree_info = json.load(f)
            
            return axtree_info['axtree'].split('\n'), axtree_info['all_boxes']
        else:
            with open(xml_file) as f:
                xml = f.read()
    else: xml = xml_file
    root = parse_xml_to_tree(xml)
    
    # Step 2: Simplify the tree
    if isinstance(target_box, list):
        target_box = f"[{target_box[0]},{target_box[1]}][{target_box[2]},{target_box[3]}]"
    simplify_tree(root, target_box=target_box, target_point=target_point)
    
    # Step 3: Convert the tree to text format
    all_boxes = []
    ax_tree_text = tree_to_text(root, all_boxes=all_boxes, skip_statusbar=skip_statusbar).strip()
    
    with open(proc_file, "w") as f:
        json.dump({'axtree': ax_tree_text, 'all_boxes': all_boxes}, f, indent=2)

    tree_lines = ax_tree_text.split('\n')
    
    # Skip non English/Chinese samples
    if skip_lang:
        if detect_invalid_lang(xml):
            all_boxes = []

    return tree_lines, all_boxes

def extract_label_prob(logprobs, label_start='summary', label='yes'):
    i = len(logprobs) - 1
    while i >= 0:
        tok = logprobs[i].token.lower().strip()
        if len(tok) >= len(label_start) and label_start in tok:
            break
        i -= 1
    else:
        i = 0
    
    i += 1

    prob = 0.0
    while i < len(logprobs):
        tok = logprobs[i].token.lower().strip()

        # Hardcoded string length limit for binary label Yes/No
        if len(tok) <= 3 and label in tok:
            prob = math.exp(logprobs[i].logprob)
            break
        i += 1

    return prob

def random_substring(s, mode='sub-string'):
    if not s:
        return ""
    
    words = s.split(' ')
    start = random.randint(0, len(words) - 1)
    end = random.randint(start + 1, len(words))
        
    if mode == 'sub-string':
        rand_str = ' '.join(words[start:end])
    else:
        rand_str = ' '.join(random.sample(words, end - start))
    return rand_str

RAW_SWIPE_PATTERN = r"(Scroll|Swipe)\s+(up|down|left|right)"
def revise_swipe_action(text, swipe_action_str):
    return re.sub(RAW_SWIPE_PATTERN, swipe_action_str, text, flags=re.IGNORECASE)

from utils.data_utils.task_prompt_lib import *

def generate_negative_action_plans(gt_act_type, W, H, scale, gt_center=[-1,-1], boxes=None, direction='', text='', goal_status='', drag_start=None, drag_end=None):
    # NOTE: all coordinates are original (unnormalized)

    neg_action_plans = []
    neg_actions_other_types = []

    # Pick negative boxes
    # NOTE: the GT click target may not fall in any boxes.
    neg_boxes = [b for b in boxes if not (b[0] <= gt_center[0] <= b[2] and b[1] <= gt_center[1] <= b[3])]

    # If the GT action is click, generate negatives using other elem boxes
    if len(neg_boxes) >= 1:
        if gt_act_type == 'click':
            neg_cands = []
            selected_idxs = random.sample(list(range(len(neg_boxes))), min(len(neg_boxes), 9))
            for idx in selected_idxs:
                box = neg_boxes[idx]
                normalized_center = box2center(box, W, H, scale)
                neg_click = CLICK_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1])
                neg_cands.append([INCORRECT_CLICK_TARGET.format(action='click'), neg_click])
            
            random.shuffle(neg_cands)
            neg_action_plans.append(neg_cands)
        else:
            box = random.choice(neg_boxes)
            normalized_center = box2center(box, W, H, scale)
            neg_actions_other_types.append([INCORRECT_ACTION,
                                            CLICK_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1])])
    
    if gt_act_type == 'swipe':
        neg_cands = []
        for neg_direction in ['up', 'down', 'left', 'right']:
            if direction == neg_direction: continue
            
            if gt_center != [-1,-1]:
                normalized_center = [max(0, min(scale-1, round(gt_center[0] / W * scale))), max(0, min(scale-1, round(gt_center[1] / H * scale)))]

                neg_swipe = SWIPE_TEMPLATE.format(start_x=normalized_center[0], start_y=normalized_center[1], direction=neg_direction, distance="medium")
            else:
                _, start, end = format_swiping_dual_points(neg_direction, scale=scale, scroll2swipe=False)
                neg_swipe = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=neg_direction, distance="medium")

            neg_cands.append([INCORRECT_SWIPE_DIRECTION + f'. Should swipe {direction} instead of {neg_direction}', neg_swipe])
        
        random.shuffle(neg_cands)
        neg_action_plans.append(neg_cands)
    else:
        direction, start, end = format_swiping_dual_points(random.choice(['up', 'down', 'left', 'right']), scale=scale, scroll2swipe=False)
        neg_swipe = SWIPE_TEMPLATE.format(start_x=start[0], start_y=start[1], direction=direction, distance="medium")
        neg_actions_other_types.append([INCORRECT_ACTION, neg_swipe])

    if gt_act_type == 'drag':
        neg_cands = [] # TODO
        rand_drag_end = [random.randint(0, scale-1), random.randint(0, scale-1)]
        neg_cands.append([INCORRECT_DRAG_DIRECTION, DRAG_TEMPLATE.format(
            start_x=max(0, min(scale-1, round(drag_start[0]/1000*scale))), start_y=max(0, min(scale-1, round(drag_start[1]/1000*scale))),
            end_x=rand_drag_end[0],
            end_y=rand_drag_end[1])])
        
        rand_drag_start = [random.randint(0, scale-1), random.randint(0, scale-1)]
        neg_cands.append([INCORRECT_DRAG_DIRECTION, DRAG_TEMPLATE.format(
            start_x=rand_drag_start[0],
            start_y=rand_drag_start[1],
            end_x=max(0, min(scale-1, round(drag_end[0]/1000*scale))),
            end_y=max(0, min(scale-1, round(drag_end[1]/1000*scale))))])

        rand_drag_start = [random.randint(0, scale-1), random.randint(0, scale-1)]
        rand_drag_end = [random.randint(0, scale-1), random.randint(0, scale-1)]
        neg_cands.append([INCORRECT_DRAG_DIRECTION, DRAG_TEMPLATE.format(
            start_x=rand_drag_start[0],
            start_y=rand_drag_start[1],
            end_x=rand_drag_end[0],
            end_y=rand_drag_end[1])])
        random.shuffle(neg_cands)
        neg_action_plans.append(neg_cands)

    if gt_act_type == 'input_text':
        neg_cands = []
        used = [text]
        for _ in range(9):
            rand_text = random_substring(text)
            if rand_text in used: continue
            used.append(rand_text)
            neg_type = INPUT_TEMPLATE.format(text=rand_text)
            neg_cands.append([INCORRECT_INPUT_TEXT + f'. Should be "{text}" instead of "{rand_text}"', neg_type])
        random.shuffle(neg_cands)
        neg_action_plans.append(neg_cands)

    app = random.choice([
        "Facebook", "Instagram", "WhatsApp", "TikTok", "Snapchat",
        "YouTube", "Twitter", "Spotify", "Netflix", "Zoom",
        "Google Maps", "Amazon", "Gmail", "Pinterest", "Reddit",
        "LinkedIn", "Telegram", "Discord", "Uber", "Airbnb"
    ])

    neg_open_app = OPEN_APP_TEMPLATE.format(app_name=app)
    incorr_reason = INCORRECT_OPEN_APP + f'. Should be {text} instead of {app}.' if gt_act_type == 'open_app' else INCORRECT_ACTION
    neg_actions_other_types.append([incorr_reason, neg_open_app])

    # Generate negatives only for input_text actions as input_text is not a very confusing negative candidate for other action types
    if gt_act_type != 'navigate_back':
        neg_back = NAVIGATE_BACK_TEMPLATE
        incorr_reason = INCORRECT_NAVIGATION_ACTION + '. Should navigate home instead of navigate back.' if 'home' in gt_act_type else INCORRECT_ACTION
        neg_actions_other_types.append([incorr_reason, neg_back])
    
    if gt_act_type != 'navigate_home':
        neg_home = NAVIGATE_HOME_TEMPLATE
        incorr_reason = INCORRECT_NAVIGATION_ACTION + '. Should navigate back instead of navigate home.' if 'back' in gt_act_type else INCORRECT_ACTION
        neg_actions_other_types.append([incorr_reason, neg_home])
    
    if gt_act_type != 'enter': neg_actions_other_types.append([INCORRECT_ACTION, ENTER_TEMPLATE])

    if gt_act_type == 'status':
        if goal_status == 'successful': neg_status = 'infeasible'
        elif goal_status == 'infeasible': neg_status = 'successful'
        incorr_reason = INCORRECT_STATUS
    else:
        neg_status = random.choice(['successful', 'infeasible'])
        incorr_reason = INCORRECT_ACTION
        
    neg_status = STATUS_TEMPLATE.format(goal_status=neg_status, answer='')
    neg_actions_other_types.append([incorr_reason, neg_status])

    random.shuffle(neg_actions_other_types)
    neg_action_plans.append(neg_actions_other_types)
    
    return neg_action_plans


def generate_negative_action_plans_for_web(gt_act_type, W, H, scale, gt_box, boxes=None, text=''):
    neg_action_plans = []
    neg_actions_other_types = []
    
    # Pick negative boxes
    gt_center = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2

    neg_boxes = [b for b in boxes if not (b[0] <= gt_center[0] <= b[2] and b[1] <= gt_center[1] <= b[3])]

    def generate_neg_actions(neg_boxes, W, H, scale, action_template, text='', value='', incorr_reason=INCORRECT_ACTION, num_samples=5):
        neg_cands = []

        for neg_box in random.sample(neg_boxes,min(len(neg_boxes),num_samples)):
            neg_center = max(0, min(scale-1, round((neg_box[0] + neg_box[2]) / 2 / W * scale))), max(0, min(scale-1, round((neg_box[1] + neg_box[3]) / 2 / H * scale)))
            neg_action = action_template.format(target_x=neg_center[0], target_y=neg_center[1], text=text, value=value)
            neg_cands.append([incorr_reason, neg_action])
        
        random.shuffle(neg_cands)
        return neg_cands

    # If the GT action is click, generate negatives using other elem boxes
    for act_type, template in zip(['hover', 'click'], [HOVER_TEMPLATE, CLICK_TEMPLATE]):
        if gt_act_type == act_type:
            if len(neg_boxes):
                neg_cands = generate_neg_actions(neg_boxes, W, H, scale, template, incorr_reason=INCORRECT_CLICK_TARGET.format(action=gt_act_type))
                neg_action_plans.append(neg_cands)
        else:
            reason = INCORRECT_TOUCH_MODE + f". Should {gt_act_type} instead of {act_type}" if gt_act_type in ['hover', 'click'] else INCORRECT_ACTION
            neg_cands = generate_neg_actions([gt_box], W, H, scale, template, incorr_reason=reason, num_samples=1)
            neg_actions_other_types.append(neg_cands[0])

            if len(neg_boxes):
                neg_cands = generate_neg_actions(neg_boxes, W, H, scale, template, incorr_reason=INCORRECT_ACTION, num_samples=1)
                neg_actions_other_types.append(neg_cands[0])

    # Generate negatives only for input_text actions as input_text is not a very confusing negative candidate for other action types
    if gt_act_type == 'input_text':
        if len(neg_boxes):
            neg_cands = generate_neg_actions(neg_boxes, W, H, scale, INPUT_TARGET_TEMPLATE, text=text, incorr_reason=INCORRECT_CLICK_TARGET.format(action=gt_act_type), num_samples=5)
            
            neg_action_plans.append(neg_cands)

            neg_actions_other_types.extend(generate_neg_actions(neg_boxes + [gt_box], W, H, scale, HOVER_TEMPLATE, incorr_reason=INCORRECT_ACTION, num_samples=2))

            neg_actions_other_types.extend(generate_neg_actions(neg_boxes + [gt_box], W, H, scale, CLICK_TEMPLATE, incorr_reason=INCORRECT_ACTION, num_samples=2))

        neg_select = generate_neg_actions(neg_boxes + [gt_box], W, H, scale, SELECT_TEMPLATE, value=text,  incorr_reason=INCORRECT_ACTION, num_samples=2)
        neg_actions_other_types.extend(neg_select)
            

    if gt_act_type == 'select':
        if len(neg_boxes):
            neg_cands = generate_neg_actions(neg_boxes, W, H, scale, SELECT_TEMPLATE, text=text, incorr_reason=INCORRECT_CLICK_TARGET.format(action=gt_act_type), num_samples=5)
        
            neg_action_plans.append(neg_cands)

            neg_input = generate_neg_actions(neg_boxes, W, H, scale, INPUT_TARGET_TEMPLATE, text=text, incorr_reason=INCORRECT_ACTION, num_samples=2)

            neg_actions_other_types.extend(neg_input)

        neg_actions_other_types.extend(generate_neg_actions(neg_boxes + [gt_box], W, H, scale, HOVER_TEMPLATE, incorr_reason=INCORRECT_ACTION, num_samples=2))
        neg_actions_other_types.extend(generate_neg_actions(neg_boxes + [gt_box], W, H, scale, CLICK_TEMPLATE,  incorr_reason=INCORRECT_ACTION, num_samples=2))
    
    if gt_act_type != 'enter':
        neg_actions_other_types.append([INCORRECT_ACTION, ENTER_TEMPLATE])

    if gt_act_type != 'scroll':
        neg_actions_other_types.append([INCORRECT_ACTION, SIMPLE_SCROLL_TEMPLATE.format(direction=random.choice(['up','down']))])

    if gt_act_type != 'status':
        neg_status = STATUS_TEMPLATE.format(goal_status='successful', answer='')
        incorr_reason = EAYLY_STOPPING
        neg_actions_other_types.append([incorr_reason, neg_status])

    # else:
    #     normalized_center = generate_neg_point(gt_box,  W, H, scale, num_points=1)[0]
    #     neg_actions_other_types.append(SELECT_TEMPLATE.format(target_x=normalized_center[0], target_y=normalized_center[1], value=text))
    random.shuffle(neg_actions_other_types)
    neg_action_plans.append(neg_actions_other_types)
    
    return neg_action_plans

# point (str) -> point
BRACKET_COORD_PATTERN = re.compile(r'\[(.*?)\]')
GENERAL_COORD_PATTERN = re.compile(r'-?\d+\.?\d*')

# bbox (qwen str) -> bbox
SEECLICK_BOX_PATTERN = re.compile(r"\((\d+,\d+)\),\((\d+,\d+)\)")
def extract_bbox(pred):
    # Regular expression to find the content inside <box> and </box>
    matches = SEECLICK_BOX_PATTERN.findall(pred)
    # Convert the tuples of strings into tuples of integers
    
    try:
        points = []
        
        for point in matches[-1]:
            x, y = point.split(',')
            points.extend([int(x), int(y)])
    except:
        points = None

    return points

    click_point = None
def pred_2_point(pred, keep_box=True, scale=1000):
    click_point = None
    if isinstance(pred, str):
        if '[[' in pred: # For CogAgent
            coords_start = pred.find('[[')
            if coords_start != -1:
                coords_end = pred.find(']]')
                if coords_end != -1:
                    coords_str = pred[coords_start+2:coords_end].replace('[','').replace(']','')
                    try:
                        # The bounding box coordinates in the CogAgent's output use the format [[x1, y1, x2, y2]], with the origin at the top left corner, the x-axis to the right, and the y-axis downward. (x1, y1) and (x2, y2) are the top-left and bottom-right corners, respectively, with values as relative coordinates multiplied by 1000 (prefixed with zeros to three digits).
                        click_point = [x / scale for x in map(float, coords_str.split(','))]
                    except:
                        raise ValueError("Cannot extract click point from {}".format(pred))
        elif '[' in pred:
            matches = [(match.group(), (match.start(), match.end())) for match in BRACKET_COORD_PATTERN.finditer(pred)]

            if matches:
                # We take the last one
                last_valid_match_id = len(matches) - 1
                while last_valid_match_id >=0:
                    click_point_str, start_end = matches[last_valid_match_id]
                    try:
                        click_point = list(map(float, click_point_str[1:-1].split(',')))
                        break
                    except: pass
                    last_valid_match_id -= 1
                else:
                    raise ValueError("Cannot extract click point from {}".format(pred))

                # If there are two coordinates enclosed with brackets and they are different and their appearances in the response are not far away, they may be represent the top-left and bottom-right corners, respectively.
                if len(click_point) == 2 and last_valid_match_id > 0 and (start_end[0] - matches[last_valid_match_id-1][1][1]) < 30:
                    try:
                        another_point = list(map(float, matches[last_valid_match_id-1][0][1:-1].split(', ')))
                        if len(another_point) == 2:
                            click_point = [(another_point[0] + click_point[0]) / 2, (another_point[1] + click_point[1]) / 2]
                    except: pass
        elif pred.startswith('<box>'): # '<box>598 102 673 406</box>.'
            coords = re.findall(r'\d+', pred)

            # Convert to integers
            click_point = [int(num) for num in coords]
    else:
        click_point = pred

    if click_point is None: # For SeeClick
        if '<box>' in pred: # For QWen-VL-Chat
            click_point = extract_bbox(pred)
        else:
            floats = GENERAL_COORD_PATTERN.findall(pred)
            
            if floats:
                click_point = []
                for num in floats:
                    try:
                        num = float(num)
                        click_point.append(num)
                    except: pass
        
    assert click_point is not None, "Cannot extract click point from {}".format(pred)
    assert len(click_point) in [2,4], "Invalid click point {} found in {}".format(click_point, pred)
    
    if not keep_box and len(click_point) == 4:
        click_point = [(click_point[0]+click_point[2])/2, (click_point[1]+click_point[3])/2]

    # In case where the coordinates are normalized in the range [0, 1000)
    if any(x > 1 for x in click_point):
        click_point = [x / scale for x in click_point]

    return click_point

def remove_redundant_spaces(text):
  """
  Removes all redundant spaces from a given string using regular expressions.

  Args:
    text: The input string.

  Returns:
    A string with all redundant spaces removed.
    
    # Example usage:
    text = "   This   is   a   string   with   redundant   spaces.   "
    result = remove_redundant_spaces(text)
    print(result)  # Output: "This is a string with redundant spaces."
  """
  return re.sub(r"\s+", " ", text).strip()

def keep_unique_actions(history: list[str]) -> list[str]:
    cleaned_history = []
    last_action = None

    retained_idxs = []
    for i, action in enumerate(history):
        if action != last_action and len(action.strip()) > 0:
            cleaned_history.append(action)
            last_action = action
            retained_idxs.append(i)

    return retained_idxs, cleaned_history

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_ui_tars_messages(initial_instruc, past_obs, past_action_strs, past_actions, max_history=5, conv_mode: bool = True):
    # messages = [
    #     {"role": "system", "content": initial_instruc},
    # ]
    messages = []
    for i, (obs, action_str, action) in enumerate(zip(past_obs, past_action_strs + [None], past_actions + [None])):
        if i < len(past_obs) - max_history:
            messages.append({"role": "user", "content": [{
                    "type": "text",
                    "text": f"Step {i} observation"
                },
                {
                    "type": "text",
                    "text": "** GUI content placeholder **"
                }]})
        else:
            messages.append({"role": "user", "content": [{
                "type": "text",
                "text": f"Step {i} observation"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(obs)}"
                }
            }]})

        if action is not None:
            messages.append({"role": "assistant", "content": f"{' Thought: ...' if 'Thought:' not in action_str else action_str}\nAction: {action}"})
    
    # insert the initial prompt
    messages[0]['content'].insert(0, {
                    "type": "text",
                    "text": initial_instruc
                })
    
    if not conv_mode:
        new_messages = [
            {'role': 'user', 'content': [{'type': 'text', 'text': messages[0]['content'][0]['text']}]},
        ]
        for i, turn in enumerate(messages):
            if turn['role'] == 'user':
                new_messages[0]['content'].append(turn['content'][-1])
            else:
                new_messages[0]['content'].append({'type': 'text', 'text': f'Step {(i+1)//2}: ' + turn['content']})
        messages = new_messages
    # add the current observation
    # messages.append({"role": "user", "content": [{
    #     "type": "text",
    #     "text": f"Step {len(past_obs)} observation"
    # },
    # {
    #     "type": "image_url",
    #     "image_url": {
    #         "url": f"data:image/jpeg;base64,{encode_image(past_obs[-1])}"
    #     }
    # }]})

    return messages

def JSON2UITARS_action(action_str, platform='mind2web'):
    action = eval(action_str)
    action_type = action['action_type']
    if action_type == 'click': # click(start_box='<|box_start|>(x1,y1)<|box_end|>') # For clicking an element on the screen
        target = action['target']
        uitars_action = f"click(start_box='<|box_start|>({target[0]},{target[1]})<|box_end|>')"
    elif action_type == 'input_text':
        target = action['target']
        text = action.get('text', action.get('value'))
        uitars_action = f"type(start_box='<|box_start|>({target[0]},{target[1]})<|box_end|>', content='{text}')"
    elif action_type == 'select':
        target = action['target']
        value = action['value']
        uitars_action = f"select(start_box='<|box_start|>({target[0]},{target[1]})<|box_end|>', option='{value}')"

    return uitars_action

def parse_UITARS_action(action_str, platform='mind2web'):
    if 'click' in action_str:
        target = eval(action_str[action_str.find("='(")+2:action_str.find(')')+1])
        action = CLICK_TEMPLATE.format(target_x=target[0], target_y=target[1])
    elif 'type' in action_str:
        if platform == 'mind2web':
            target = eval(action_str[action_str.find("='(")+2:action_str.find(')')+1])
            if 'content=' in action_str:
                text = action_str[action_str.find('content=')+9:action_str.rfind("'")].strip()
            else:
                text = action_str[action_str.find('text=')+6:].strip()
            action = INPUT_TARGET_TEMPLATE.format(target_x=target[0], target_y=target[1], text=text)
        else:
            text = action_str[action_str.find('=')+1:].strip()
            action = INPUT_TEMPLATE.format(text=text)
    elif 'select' in action_str:
        # select(start_box='<|box_start|>(x1,y1)<|box_end|>', option='')
        target = eval(action_str[action_str.find("='(")+2:action_str.find(")'")+1])
        value = action_str[action_str.find("option='")+8:action_str.rfind("'")].strip()
        action = SELECT_TEMPLATE.format(target_x=target[0], target_y=target[1], value=value)

    return action

def lower_first_letter(s):
    if not s:
        return s
    return s[0].lower() + s[1:]

def parse_AGUVIS_action(action_str, platform='mind2web'):
    if 'click' in action_str:
        coords = re.findall(r'x=([\d.]+),\s*y=([\d.]+)', action_str)
        target = [float(coords[0][0]), float(coords[0][1])]
        action = CLICK_TEMPLATE.format(target_x=target[0], target_y=target[1])
    elif 'write' in action_str:
        # example: "pyautogui.write(message='BCD Studio')"
        text = action_str[action_str.find('=')+2:-2].strip()
        action = INPUT_TARGET_TEMPLATE.format(target_x=-1, target_y=-1, text=text)
    elif 'select' in action_str:
        # select(start_box='<|box_start|>(x1,y1)<|box_end|>', option='')
        coords = re.findall(r'x=([\d.]+),\s*y=([\d.]+)', action_str)
        target = [float(coords[0][0]), float(coords[0][1])]
        value = action_str[action_str.find("value='")+7:action_str.rfind("')")].strip()
        action = SELECT_TEMPLATE.format(target_x=target[0], target_y=target[1], value=value)
    elif 'scroll' in action_str:
        # 'pyautogui.scroll(page=-0.26)'
        direction = float(action_str[action_str.find("page=")+5:-1].strip())
        action = SIMPLE_SCROLL_TEMPLATE.format(direction='up' if direction > 0 else 'down')
    return action

def add_screenshot_label(screenshot: np.ndarray, label: str):
  """Add a text label to the right bottom of the screenshot.

  Args:
    screenshot: The screenshot as a numpy ndarray.
    label: The text label to add, just a single word.
  """
  height, width, _ = screenshot.shape
  screenshot[height - 30 : height, width - 150 : width, :] = (255, 255, 255)
  cv2.putText(
      screenshot,
      label,
      (width - 120, height - 5),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 0, 0),
      thickness=2,
  )

PATTERN = re.compile(r'\d+')
def extract_integers(text: str):
    return list(map(int, PATTERN.findall(text)))

def scroll2swipe(direction):
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'


def restore_unified_actions(action: dict, only_scroll: bool = False) -> dict:
    act_type = action['action_type'].lower()
    if act_type in ['tap']:
        action['action_type'] = 'click'
        if 'element' in action:
            action['target'] = action['element']
    elif act_type in ['scroll']:
        if 'relative_down' in action: # GUIAct-Web
            direction = 'down'
            if action['relative_down'] != 0:
                direction = 'down' if action['relative_down'] > 0 else 'up'
            else:
                direction = 'right' if action['relative_right'] > 0 else 'left'
            action['direction'] = direction
        elif 'touch' in action:
            x_shift, y_shift = abs(action['touch'][0] - action['lift'][0]), abs(action['touch'][1] - action['lift'][1])
            if x_shift > y_shift:
                direction = 'right' if action['touch'][0] > action['lift'][0] else 'left'
            else:
                direction = 'down' if action['touch'][1] > action['lift'][1] else 'up'
            action['direction'] = direction
    elif act_type in ['swipe']:
        if only_scroll:
            if 'from' in action:
                action['action_type'] = 'scroll'
                x_shift, y_shift = abs(action['from'][0] - action['to'][0]), abs(action['from'][1] - action['to'][1])
                if x_shift > y_shift:
                    action['direction'] = 'right' if action['from'][0] < action['to'][0] else 'left'
                else:
                    action['direction'] = 'down' if action['from'][1] < action['to'][1] else 'up'
        else:
            x_shift, y_shift = abs(action['from'][0] - action['to'][0]), abs(action['from'][1] - action['to'][1])
            if x_shift > y_shift:
                action['direction'] = 'right' if action['from'][0] < action['to'][0] else 'left'
            else:
                action['direction'] = 'down' if action['from'][1] < action['to'][1] else 'up'
    elif act_type in ['input', 'type']:
        action['action_type'] = 'input_text'
        if 'content' in action:
            action['text'] = action['content']
    elif act_type in ['select_text']:
        action['action_type'] = 'drag'
    elif act_type in ['copy']:
        action['action_type'] = 'hotkey'
        action['key_comb'] = 'ctrl+c'
    elif act_type in ['press_enter']:
        action['action_type'] = 'press_key'
        action['key'] = 'Enter'
    elif act_type in 'press_back':
        action['action_type'] = 'navigate_back'
    elif act_type in 'press_home':
        action['action_type'] = 'navigate_home'
    elif act_type in 'press_recent':
        action['action_type'] = 'navigate_recent'
    return action


def qwen2vl_to_nornal_action(action_json, use_assert: bool = False):
    """Convert JSONAction to QwenAction"""
    if isinstance(action_json, str):
        action_json = json.loads(action_json)

    action_type = action_json['action']
    if action_type == 'click':
        target = action_json.get('target', action_json.get('coordinate'))
        if target is None:
            if 'x' in action_json:
                target = [round(action_json['x']), round(action_json['y'])]

        act = CLICK_TEMPLATE.format(target_x=target[0], target_y=target[1]) # '{{"action": "click", "coordinate": [{target_x}, {target_y}]}}'
    elif action_type == 'long_press':
        target = action_json.get('target', action_json.get('coordinate'))
        if target is None:
            if 'x' in action_json:
                target = [round(action_json['x']), round(action_json['y'])]
        act = LONG_PRESS_TEMPLATE.format(target_x=target[0], target_y=target[1], duration=action_json.get('duration', 1))
    elif action_type in ['swipe', 'scroll']:
        start_x, start_y = action_json['coordinate']
        end_x, end_y = action_json['coordinate2']
        direction = get_swipe_direction(start=[start_x, start_y], end=[end_x, end_y], is_swipe=True)

        act = SWIPE_TEMPLATE.format(start_x=start_x, start_y=start_y, direction=direction, distance='medium')
    elif action_type == 'type':
        txt = action_json['text']
        if txt.count('"') % 2 == 1:
            txt = txt.replace('"', '')
        act = INPUT_TEMPLATE.format(text=txt)
    elif action_type == 'system_button':
        button = action_json['button'].lower()
        if 'back' in button:
            act = NAVIGATE_BACK_TEMPLATE # '{{"action": "system_button", "button": "{button}"}}'
        elif 'home' in button:
            act = NAVIGATE_HOME_TEMPLATE
        elif 'enter' in button:
            act = ENTER_TEMPLATE
        elif 'menu' in button:
            act = NAVIGATE_RECENT_TEMPLATE
    elif action_type in ['navigate_back', 'back']:
        act = NAVIGATE_BACK_TEMPLATE
    elif action_type in ['navigate_home', 'home']:
        act = NAVIGATE_HOME_TEMPLATE
    elif action_type == 'press_key':
        act = PRESSKEY_TEMPLATE.format(key=action_json['key'])
    elif action_type in ['press_enter', 'enter']:
        act = ENTER_TEMPLATE
    elif action_type == 'open_app':
        act = OPEN_APP_TEMPLATE.format(app_name=action_json.get('app_name', action_json.get('text')))
    elif action_type == 'terminate':
        if any(k in action_json['status'] for k in ['unsuccessful', 'failure', 'failed', 'infeasible', 'incomplete']):
            action_json['status'] = 'failed'
        elif any(k in action_json['status'] for k in ['successful', 'success', 'feasible', 'complete']):
            action_json['status'] = 'successful'
        else:
            if use_assert: 
                assert False, f"Invalid goal status: {action_json['status']}"
            else:
                return None, None
        act = STATUS_TEMPLATE.format(goal_status='success' if action_json['status'] == 'successful' else 'failure', answer='') # {"action": "terminate", "status": <"success" or "failure">}
    elif action_type == 'answer':
        ans = action_json.get('answer', action_json.get('text'))
        act = ANSWER_TEMPLATE.format(text=ans)
    elif action_type == 'wait':
        act = WAIT_TEMPLATE
    else:
        raise ValueError(f"Unsupported action type: {action_type}")
    return act

def to_qwen_action(action_json, W, H, type_w_coords: bool = True, scale: int = -1, use_assert: bool = True):
    """Convert JSONAction to QwenAction"""
    action_type = action_json.get('action_type', 'action')
    if action_type == 'click':
        target = action_json.get('target', action_json.get('coordinate'))
        if target is None:
            if 'x' in action_json:
                target = [round(action_json['x']), round(action_json['y'])]

        act = CLICK_TEMPLATE_QWEN.format(target_x=target[0], target_y=target[1]) # '{{"action": "click", "coordinate": [{target_x}, {target_y}]}}'
    elif action_type == 'long_press':
        target = action_json.get('target', action_json.get('coordinate'))
        if target is None:
            if 'x' in action_json:
                target = [round(action_json['x']), round(action_json['y'])]
        act = LONG_PRESS_TEMPLATE_QWEN.format(target_x=target[0], target_y=target[1], duration=action_json.get('duration', 1))
    elif action_type in ['swipe', 'scroll']:
        if 'coordinate2' in action_json:
            start_x, start_y = action_json['coordinate']
            end_x, end_y = action_json['coordinate2']
        else:
            direction = action_json['direction']
            
            if action_type == 'scroll':
                direction = scroll2swipe(direction)

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
        txt = action_json['text']
        if txt.count('"') % 2 == 1:
            txt = txt.replace('"', '')
        act = INPUT_TEMPLATE_QWEN_NO_COORD.format(text=txt)
    elif action_type == 'navigate_back':
        act = SYSTEM_BUTTON_TEMPLATE_QWEN.format(button='BACK') # '{{"action": "system_button", "button": "{button}"}}'
    elif action_type == 'navigate_home':
        act = SYSTEM_BUTTON_TEMPLATE_QWEN.format(button='HOME')
    elif action_type == 'press_key':
        act = SYSTEM_BUTTON_TEMPLATE_QWEN.format(button=action_json['key'])
    elif action_type in ['press_enter', 'enter']:
        act = SYSTEM_BUTTON_TEMPLATE_QWEN.format(button='ENTER')
    elif action_type == 'open_app':
        act = OPEN_APP_TEMPLATE_QWEN.format(app_name=action_json.get('app_name', action_json.get('text')))
    elif action_type == 'status':
        if any(k in action_json['goal_status'] for k in ['unsuccessful', 'failed', 'infeasible', 'incomplete']):
            action_json['goal_status'] = 'failed'
        elif any(k in action_json['goal_status'] for k in ['successful', 'success', 'feasible', 'complete']):
            action_json['goal_status'] = 'successful'
        else:
            if use_assert: 
                assert False, f"Invalid goal status: {action_json['goal_status']}"
            else:
                return None, None
        act = TERMINATE_TEMPLATE_QWEN.format(status='success' if action_json['goal_status'] == 'successful' else 'failure') # {"action": "terminate", "status": <"success" or "failure">}
    elif action_type == 'answer':
        ans = action_json.get('answer', action_json.get('text'))
        act = ANSWER_TEMPLATE_QWEN.format(answer=ans)
    elif action_type == 'wait':
        act = WAIT_TEMPLATE_QWEN.format(time=2)
    else:
        raise ValueError(f"Unsupported action type: {action_type}")
    return act, json.loads(act)


def get_swipe_direction(start, end, is_swipe: bool = True):
    vertical_shift, horizontal_shift = end[1] - start[1], end[0] - start[0]

    # judged the scrolling direction
    if abs(vertical_shift) > abs(horizontal_shift):
        direction = 'down' if vertical_shift > 0 else 'up'
        distance = discretize_dist(abs(vertical_shift))
    else:
        direction = 'right' if horizontal_shift > 0 else 'left'
        distance = discretize_dist(abs(horizontal_shift))
    
    if not is_swipe:
        if direction == 'down': direction = 'up'
        elif direction == 'up': direction = 'down'
        elif direction == 'right': direction = 'left'
        elif direction == 'left': direction = 'right'
    
    return direction, distance


def extract_actions_from_response(response):
    """Extract actions from the response string."""
    actions = []
    action_pattern = r'<action>\n(.*?)\n</action>'
    matches = re.findall(action_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            action = json.loads(match)
            actions.append(action)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse action: {match}")
    
    return actions

def visualize_action(image, action, action_idx):
    """Draw the action on the image with a numbered marker."""
    draw = ImageDraw.Draw(image)
    
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    
    # Action type color mapping
    color_map = {
        "click": (255, 0, 0),       # Red
        "long_press": (0, 255, 0),  # Green
        "type": (0, 0, 255),        # Blue
        "swipe": (255, 165, 0)      # Orange
    }
    
    # Get action type
    action_type = ""
    if "action" in action:
        action_type = action["action"]
    
    # Default color (red) if action type not recognized
    color = color_map.get(action_type, (255, 0, 0))
    
    # Handle different action types
    if action_type == "click" and "coordinate" in action:
        # Click action
        x, y = action["coordinate"][0], action["coordinate"][1]
        
        # Draw a circle at the click point
        radius = 20
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], outline=color, width=3)
        
        # Draw the action number
        draw.text((x-10, y-radius-25), f"{action_idx+1}", fill=color, font=font)
        
        # Label the action type
        draw.text((x-20, y+radius+5), "Click", fill=color, font=font)
        
    elif action_type == "long_press" and "coordinate" in action:
        # Long press action
        x, y = action["coordinate"][0], action["coordinate"][1]
        
        # Draw a double circle for long press
        radius1 = 20
        radius2 = 28
        draw.ellipse([(x-radius1, y-radius1), (x+radius1, y+radius1)], outline=color, width=3)
        draw.ellipse([(x-radius2, y-radius2), (x+radius2, y+radius2)], outline=color, width=2)
        
        # Draw the action number
        draw.text((x-10, y-radius2-25), f"{action_idx+1}", fill=color, font=font)
        
        # Label the action type
        draw.text((x-40, y+radius2+5), "Long Press", fill=color, font=font)
        
    elif action_type == "type" and "coordinate" in action:
        # Type action
        x, y = action["coordinate"][0], action["coordinate"][1]
        text = action.get("text", "")
        
        # Draw a rectangle at the type point
        radius = 20
        draw.rectangle([(x-radius, y-radius), (x+radius, y+radius)], outline=color, width=3)
        
        # Draw the action number
        draw.text((x-10, y-radius-25), f"{action_idx+1}", fill=color, font=font)
        
        # Label the action type and text
        draw.text((x-15, y+radius+5), "Type", fill=color, font=font)
        
        # Draw the text that will be typed (truncate if too long)
        if text:
            truncated_text = text if len(text) < 20 else text[:17] + "..."
            draw.text((x-30, y+radius+30), f'"{truncated_text}"', fill=color, font=small_font)
        
    elif action_type == "swipe" and "coordinate" in action and "coordinate2" in action:
        # Swipe action
        x1, y1 = action["coordinate"][0], action["coordinate"][1]
        x2, y2 = action["coordinate2"][0], action["coordinate2"][1]
        
        # Draw a line with an arrow for the swipe
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
        
        # Draw arrow at the end
        arrow_size = 15
        angle = math.atan2(y2 - y1, x2 - x1)
        x_arrow1 = x2 - arrow_size * math.cos(angle - math.pi/6)
        y_arrow1 = y2 - arrow_size * math.sin(angle - math.pi/6)
        x_arrow2 = x2 - arrow_size * math.cos(angle + math.pi/6)
        y_arrow2 = y2 - arrow_size * math.sin(angle + math.pi/6)
        
        draw.line([(x2, y2), (x_arrow1, y_arrow1)], fill=color, width=3)
        draw.line([(x2, y2), (x_arrow2, y_arrow2)], fill=color, width=3)
        
        # Draw circles at start and end points
        small_radius = 10
        draw.ellipse([(x1-small_radius, y1-small_radius), (x1+small_radius, y1+small_radius)], outline=color, width=2)
        draw.ellipse([(x2-small_radius, y2-small_radius), (x2+small_radius, y2+small_radius)], outline=color, width=2, fill=color)
        
        # Draw the action number at the midpoint
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        offset_x = 15 * math.sin(angle)  # Offset perpendicular to the swipe direction
        offset_y = -15 * math.cos(angle)
        draw.text((mid_x + offset_x - 10, mid_y + offset_y - 10), f"{action_idx+1}", fill=color, font=font)
        
        # Label the action type
        draw.text((mid_x + offset_x - 20, mid_y + offset_y + 15), "Swipe", fill=color, font=font)
    
    else:
        # Handle legacy or other action formats by looking for x/y coordinates
        x, y = None, None
        if 'x' in action and 'y' in action:
            x, y = float(action['x']), float(action['y'])
        elif 'touch' in action and isinstance(action['touch'], dict):
            if 'x' in action['touch'] and 'y' in action['touch']:
                x, y = float(action['touch']['x']), float(action['touch']['y'])
        
        if x is not None and y is not None:
            # Draw a dotted circle for unknown action type
            radius = 20
            # Create a dotted circle effect
            for i in range(0, 360, 20):  # Draw 18 short arcs to create a dotted effect
                start_angle = i
                end_angle = (i + 10) % 360
                draw.arc([(x-radius, y-radius), (x+radius, y+radius)], 
                         start=start_angle, end=end_angle, fill=(128, 128, 128), width=3)
            
            # Draw the action number
            draw.text((x-10, y-radius-25), f"{action_idx+1}", fill=(128, 128, 128), font=font)
            
            # Label as unknown action type
            if "action" in action:
                draw.text((x-30, y+radius+5), f"{action['action']}", fill=(128, 128, 128), font=font)
            else:
                draw.text((x-30, y+radius+5), "Unknown", fill=(128, 128, 128), font=font)
    
    return image

def get_image_dimensions(image_path):
    img_info = magic.from_file(image_path)
    
    if 'precision' in img_info:
        W, H = list(map(int, re.search('precision 8, (\d+)x(\d+)', img_info).groups(1)))
    else:
        W, H = list(map(int, re.search('(\d+) x (\d+)', img_info).groups(1)))
    
    return W, H