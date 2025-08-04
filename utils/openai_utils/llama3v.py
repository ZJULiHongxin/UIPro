import os
from typing import List
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from peft import PeftModel
from transformers import MllamaForConditionalGeneration, AutoProcessor
class Llama3V:
    def __init__(self, device, model_name: str = ''):
        self.model_name = model_name
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_model_response(self, prompt: str, image: str = None, temperature: float = 0.0, max_new_tokens: int = 1024):
        if isinstance(image, str):
            image = Image.open(image.replace("file://",""))
            
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            do_sample=True if temperature > 0 else False,
            temperature=temperature
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = self.processor.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        return output