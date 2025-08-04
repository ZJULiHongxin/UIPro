import os
from typing import List
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from peft import PeftModel
from transformers import pipeline

class Llama3:
    def __init__(self, device, model_name: str = ''):
        self.model_name = model_name
        
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def get_model_response(self, prompt: str, image: str = None, temperature: float = 0.0, max_new_tokens: int = 1024):
        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": prompt},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False if temperature == 0.0 else True,
            temperature=temperature
        )

        return outputs[0]["generated_text"][-1]['content']