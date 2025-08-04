from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
import torch

import os
from typing import List
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM
)
from qwen_vl_utils import process_vision_info

from PIL import Image
import re


def prepare_inputs(image_path, text, processor):
    img = Image.open(image_path).convert("RGB")
    size = img.size

    input_text = ("What to do to execute the command? " + text.strip()).lower()

    encoding = processor(
        images=img,
        text=input_text,
        return_tensors="pt",
        do_resize=True,
    )
    encoding["image_size"] = size

    return encoding


def postprocess(text: str, image_size: tuple[int]):
    """Function that decodes model's generation into action json.

    Args:
        text: single generated sample
        image_size: corresponding image size
    """
    pattern = r"</s><s>(<[^>]+>|[^<\s]+)\s*([^<]*?)(<loc_\d+>.*)"
    point_pattern = r"<loc_(\d+)><loc_(\d+)>"


    try:
        location = re.findall(point_pattern, text)[0]
        if len(location) > 0:
            point = [int(loc) for loc in location]

    except Exception:
        point = (0, 0)

    return point

class TinyClick:
    def __init__(self, device, model_name: str = 'Samsung/TinyClick'):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(device)# load_in_4bit=True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

    def get_model_response(self, prompt: str, image: str, max_new_tokens: int = 1024):
        if "<image>" in prompt:
            prompt = prompt.replace("<image>", "")
        
        img = Image.open(image.replace("file://","")).convert("RGB")
        img_size = img.size

        inputs = self.processor(
            images=img,
            text=prompt,
            return_tensors="pt",
            do_resize=True,
        ).to(self.model.device, dtype=self.model.dtype)

        outputs = self.model.generate(
                    **inputs,
                    do_sample= False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )

        text_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        text_output = postprocess(text_output, img_size)
        return text_output