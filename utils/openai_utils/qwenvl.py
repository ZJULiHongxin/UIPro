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

class QwenVL:
    def __init__(self, device, model_name: str = 'Qwen/Qwen2-VL-7B-Instruct'):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).eval() # load_in_4bit=True
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        
        self.model.to(device)

    def get_model_response(self, prompt: str, image: str, max_new_tokens: int = 1024):

        if "<image>" in prompt:
            prompt = prompt.replace("<image>", "")
                
        query = [
            {"image": image.replace("file://","")},
            {"text": prompt}
        ]

        questions = self.tokenizer.from_list_format(query)
        
        text_output, history = self.model.chat(self.tokenizer, 
                                                query=questions, 
                                                history=None,
                                                chat_format="chatml",
                                                eos_token_id=151643,
                                                pad_token_id=151643,
                                                top_k=0,
                                                do_sample=False,
                                                max_new_tokens=max_new_tokens)
                



        return text_output