import torch
from PIL import Image
from typing import Dict, Tuple, List
import io
import base64

# from ultralytics import YOLO
import os
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import json
import requests
# utility function
import os
from openai import AzureOpenAI

import json
import sys
import os
import cv2
import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt
import easyocr
from paddleocr import PaddleOCR
reader = easyocr.Reader(['en'])
paddle_ocr = PaddleOCR(
    lang='en',  # other lang also available
    use_angle_cls=False,
    use_gpu=False,  # using cuda will conflict with pytorch in the same process
    show_log=False,
    max_batch_size=1024,
    use_dilation=True,  # improves accuracy
    det_db_score_mode='slow',  # improves accuracy
    rec_batch_num=1024)
import time
import base64

import os
import ast
import torch
from typing import Tuple, List
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T

from typing import List, Optional, Union, Tuple

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette


class BoxAnnotator:
    """
    A class for drawing bounding boxes on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the bounding box,
            can be a single color or a color palette
        thickness (int): The thickness of the bounding box lines, default is 2
        text_color (Color): The color of the text on the bounding box, default is white
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box,
            default is 1
        text_padding (int): The padding around the text on the bounding box,
            default is 5

    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 3, # 1 for seeclick 2 for mind2web and 3 for demo
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5, # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
        text_thickness: int = 2, #1, # 2 for demo
        text_padding: int = 10,
        avoid_overlap: bool = True,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.avoid_overlap: bool = avoid_overlap

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the
                bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels
                corresponding to each detection. If `labels` are not provided,
                corresponding `class_id` will be used as label.
            skip_label (bool): Is set to `True`, skips bounding box label annotation.
        Returns:
            np.ndarray: The image with the bounding boxes drawn on it

        Example:
            ```python
            import supervision as sv

            classes = ['person', ...]
            image = ...
            detections = sv.Detections(...)

            box_annotator = sv.BoxAnnotator()
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ in detections
            ]
            annotated_frame = box_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels
            )
            ```
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            if skip_label:
                continue

            text = (
                f"{class_id}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            if not self.avoid_overlap:
                text_x = x1 + self.text_padding
                text_y = y1 - self.text_padding

                text_background_x1 = x1
                text_background_y1 = y1 - 2 * self.text_padding - text_height

                text_background_x2 = x1 + 2 * self.text_padding + text_width
                text_background_y2 = y1
                # text_x = x1 - self.text_padding - text_width
                # text_y = y1 + self.text_padding + text_height
                # text_background_x1 = x1 - 2 * self.text_padding - text_width
                # text_background_y1 = y1
                # text_background_x2 = x1
                # text_background_y2 = y1 + 2 * self.text_padding + text_height
            else:
                text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2 = get_optimal_label_pos(self.text_padding, text_width, text_height, x1, y1, x2, y2, detections, image_size)

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            # import pdb; pdb.set_trace()
            box_color = color.as_rgb()
            luminance = 0.299 * box_color[0] + 0.587 * box_color[1] + 0.114 * box_color[2]
            text_color = (0,0,0) if luminance > 160 else (255,255,255)
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                # color=self.text_color.as_rgb(),
                color=text_color,
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene
    

def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

def IoU(box1, box2, return_max=True):
    intersection = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - intersection
    if box_area(box1) > 0 and box_area(box2) > 0:
        ratio1 = intersection / box_area(box1)
        ratio2 = intersection / box_area(box2)
    else:
        ratio1, ratio2 = 0, 0
    if return_max:
        return max(intersection / union, ratio1, ratio2)
    else:
        return intersection / union


def get_optimal_label_pos(text_padding, text_width, text_height, x1, y1, x2, y2, detections, image_size):
    """ check overlap of text and background detection box, and get_optimal_label_pos, 
        pos: str, position of the text, must be one of 'top left', 'top right', 'outer left', 'outer right' TODO: if all are overlapping, return the last one, i.e. outer right
        Threshold: default to 0.3
    """

    def get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size):
        is_overlap = False
        for i in range(len(detections)):
            detection = detections.xyxy[i].astype(int)
            if IoU([text_background_x1, text_background_y1, text_background_x2, text_background_y2], detection) > 0.3:
                is_overlap = True
                break
        # check if the text is out of the image
        if text_background_x1 < 0 or text_background_x2 > image_size[0] or text_background_y1 < 0 or text_background_y2 > image_size[1]:
            is_overlap = True
        return is_overlap
    
    # if pos == 'top left':
    text_x = x1 + text_padding
    text_y = y1 - text_padding

    text_background_x1 = x1
    text_background_y1 = y1 - 2 * text_padding - text_height

    text_background_x2 = x1 + 2 * text_padding + text_width
    text_background_y2 = y1
    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2
    
    # elif pos == 'outer left':
    text_x = x1 - text_padding - text_width
    text_y = y1 + text_padding + text_height

    text_background_x1 = x1 - 2 * text_padding - text_width
    text_background_y1 = y1

    text_background_x2 = x1
    text_background_y2 = y1 + 2 * text_padding + text_height
    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2
    

    # elif pos == 'outer right':
    text_x = x2 + text_padding
    text_y = y1 + text_padding + text_height

    text_background_x1 = x2
    text_background_y1 = y1

    text_background_x2 = x2 + 2 * text_padding + text_width
    text_background_y2 = y1 + 2 * text_padding + text_height

    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    # elif pos == 'top right':
    text_x = x2 - text_padding - text_width
    text_y = y1 - text_padding

    text_background_x1 = x2 - 2 * text_padding - text_width
    text_background_y1 = y1 - 2 * text_padding - text_height

    text_background_x2 = x2
    text_background_y2 = y1

    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2


def get_caption_model_processor(model_name, device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if "blip2" in model_name.lower():
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, device_map=None, torch_dtype=torch.float32
        ) 
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, device_map=None, torch_dtype=torch.float16
        ).to(device)
    elif 'florence' in model_name.lower():
        from transformers import AutoProcessor, AutoModelForCausalLM 
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    else:
        print("[omniparser.py] No model specified. Omniparser will be used to draw SoM only.")
        model = processor = None

    return {'model': model, 'processor': processor}


def get_yolo_model(model_path):
    from ultralytics import YOLO
    # Load the model.
    model = YOLO(model_path)
    return model


@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=None):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    batch_size = 10  # Number of samples per batch
    generated_texts = []
    device = model.device

    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i:i+batch_size]
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
        if 'florence' in model.config.name_or_path:
            generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=1024,num_beams=3, do_sample=False)
        else:
            generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1) # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    return generated_texts



def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}] 
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs ={'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(input['input_ids'])
            inputs['attention_mask'].append(input['attention_mask'])
            inputs['pixel_values'].append(input['pixel_values'])
            inputs['image_sizes'].append(input['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
            inputs['attention_mask'][i] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = { 
            "max_new_tokens": 25, 
            "temperature": 0.01, 
            "do_sample": False, 
        } 
        generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # # remove input tokens 
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts

def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                if not any(IoU(box1, box3) > iou_threshold for k, box3 in enumerate(ocr_bbox)):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def annotate_som(image_source: np.ndarray, boxes: torch.Tensor, text_scale: float, text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape

    detections = sv.Detections(xyxy=boxes)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    return annotated_frame

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def predict(model, image, caption, box_threshold, text_threshold):
    """ Use huggingface model to replace the original model
    """
    model, processor = model['model'], model['processor']
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold, # 0.4,
        text_threshold=text_threshold, # 0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


def predict_yolo(model, image_path, box_threshold, imgsz):
    """ Use huggingface model to replace the original model
    """
    # model = model['model']
    
    result = model.predict(
    source=image_path,
    conf=box_threshold,
    imgsz=imgsz
    # iou=0.5, # default 0.7
    )
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases


def get_som_labeled_img(img_path, model=None, BOX_TRESHOLD = 0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, caption_model_processor=None, ocr_text=None, use_local_semantics=True, iou_threshold=0.9,prompt=None,imgsz=640):
    """ ocr_bbox: list of xyxy format bbox
    """
    TEXT_PROMPT = "clickable buttons on the screen"
    # BOX_TRESHOLD = 0.02 # 0.05/0.02 for web and 0.1 for mobile
    TEXT_TRESHOLD = 0.01 # 0.9 # 0.01
    
    if isinstance(img_path, str): 
        image_source = Image.open(img_path).convert("RGB")
        w, h = image_source.size
    elif isinstance(img_path, np.ndarray):
        image_source = img_path
        h, w = image_source.shape[:2]
    # import pdb; pdb.set_trace()
    if False: # TODO
        xyxy, logits, phrases = predict(model=model, image=image_source, caption=TEXT_PROMPT, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD)
    else:
        xyxy, logits, phrases = predict_yolo(model=model, image_path=img_path, box_threshold=BOX_TRESHOLD, imgsz=imgsz)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # annotate the image with labels
    h, w, _ = image_source.shape
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox=ocr_bbox.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox = None
    filtered_boxes = remove_overlap(boxes=xyxy, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox)
    
    # get parsed icon local semantics
    if use_local_semantics:
        caption_model = caption_model_processor['model']
        if 'phi3_v' in caption_model.config.model_type: 
            parsed_content_icon = get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor)
        else:
            parsed_content_icon = get_parsed_content_icon(filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=prompt)
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        parsed_content_merged = ocr_text + parsed_content_icon_ls
    elif ocr_text is not None:
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]
    
    # draw boxes
    h, w, _ = image_source.shape
    xyxy = box_convert(boxes=filtered_boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy()

    if output_coord_in_ratio:
        # h, w, _ = image_source.shape
        xyxy[:,0] /= w; xyxy[:,2] /= w; xyxy[:,1] /= h; xyxy[:,3] /= h
    
    # if draw_bbox_config:
        # annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
    # else:
    #     annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
    
    # pil_img = Image.fromarray(annotated_frame)
    # buffered = io.BytesIO()
    # pil_img.save(buffered, format="PNG")
    # encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    # if output_coord_in_ratio:
    #     # h, w, _ = image_source.shape
    #     label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
    #     assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return xyxy, parsed_content_merged


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h
    


def check_ocr_box(image_path, display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None, use_paddleocr=False):
    if use_paddleocr:
        result = paddle_ocr.ocr(image_path, cls=False)[0]
        coord = [item[0] for item in result]
        text = [item[1][0] for item in result]
    else:  # EasyOCR
        if easyocr_args is None:
            easyocr_args = {}
        result = reader.readtext(image_path, **easyocr_args)
        # print('goal filtering pred:', result[-5:])
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    # read the image using cv2
    if display_img:
        opencv_img = cv2.imread(image_path)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            # print(x, y, a, b)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
        
        # Display the image
        plt.imshow(opencv_img)
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
        # print('bounding box!!!', bb)
    return (text, bb), goal_filtering



class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        
        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        
        if config['caption_model_path'] is not None:
            self.caption_model_processor = get_caption_model_processor(config['caption_model_path'], device=config['device'])
            self.caption_model_processor['model'].to(torch.float32)
        else:
            self.caption_model_processor = None

    def parse(self, image_path: str):
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9})
        text, ocr_bbox = ocr_bbox_rslt

        draw_bbox_config = self.config['draw_bbox_config']
        BOX_TRESHOLD = self.config['BOX_TRESHOLD']
        label_coordinates, parsed_content_list = get_som_labeled_img(image_path, self.som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=False, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=False)
        
        # image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        # formating output
        return_list = [{'box': box.tolist(),
                        'text': parsed_content_list[i].split(': ')[1], 'type':'text'} for i, box in enumerate(label_coordinates) if i < len(parsed_content_list)]
        return_list.extend(
            [{'box': box.tolist(),
                        'text': 'None', 'type':'icon'} for i, box in enumerate(label_coordinates) if i >= len(parsed_content_list)]
              )

        return return_list
    
