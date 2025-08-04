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

import transformers
if transformers.__version__ >= '4.45.0':
    from transformers import Qwen2VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

class UITARS:
    def __init__(self, device, model_name: str = 'bytedance-research/UI-TARS-2B-SFT'):
        self.model_name = model_name
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2",
            )
     
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1008*1008)

    def test_speed(self, prompt: str, image: str = None, max_new_tokens: int = 512, sys_prompt: str = ''):
        prompt = f"{prompt.replace('<image>','').strip()}"
        conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]

        if sys_prompt:
            conversation[0]['content'].insert(0, {"type": "text", "text": sys_prompt})
        texts = [self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)]

        image_inputs, video_inputs = process_vision_info(conversation)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device) # input_ids: B x L | attention_mask: B x L | pixel_values: (B*X) x Y | image_grid_thw: B x num_grids
        
        start = time.time()
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0, do_sample=False, use_cache=False)
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

        
    def get_model_response(self, prompt: str, image: str = None, temperature: float = 0.0, max_new_tokens: int = 512, sys_prompt: str = ''):
        prompt = f"{prompt.replace('<image>','').strip()}"
        conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]

        if sys_prompt:
            conversation[0]['content'].insert(0, {"type": "text", "text": sys_prompt})
        texts = [self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)]

        image_inputs, video_inputs = process_vision_info(conversation)

        inputs = self.processor(
            text=texts,
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