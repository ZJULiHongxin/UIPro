import time
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

from transformers import AutoTokenizer
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from utils.openai_utils.osatlas4b_conv import get_conv_template

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class OSATLAS4B:
    def __init__(self, device, model_name: str = 'OS-Copilot/OS-Atlas-Base-7B'):
        self.model_name = model_name
        
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
                ).eval().cuda()
     
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def test_speed(self, prompt: str, image: str = None, max_new_tokens: int = 512, sys_prompt: str = ''):
        prompt = prompt.replace("<image>", "").strip()

        # set the max number of tiles in `max_num`
        pixel_values = load_image(image, max_num=6).to(torch.bfloat16).to(dtype=self.model.dtype, device=self.model.device)
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
        
        start = time.time()
        with torch.inference_mode():
            text_outputs, history = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
        duration = time.time() - start
        
        output_ids = self.tokenizer.encode(text_outputs)

        return duration, output_ids # assume bs = 1

        
    def get_model_response(self, prompt: str, image: str = None, temperature: float = 0.0, max_new_tokens: int = 512, sys_prompt: str = ''):
        question = sys_prompt + '\n' + prompt

        pixel_values = load_image(image.replace("file://", ""), max_num=6).to(torch.bfloat16).cuda()

        generation_config = dict(max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True if temperature > 0.0 else False)

        #response, history = self.model.chat(self.tokenizer, pixel_values, sys_prompt + '\n' + prompt, generation_config, history=None, return_history=False)
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
                
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = get_conv_template(self.model.template)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)

        history = []
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()

        return response