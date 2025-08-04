import torch
from uipro import constants
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates, SeparatorStyle
from PIL import Image

class FuncPred_UIPro:
    def __init__(self, device, model_name: str = '/home/jingran_su/slime_ckpt/1030_SliME-Gemma-1p1-2B_5MBoxQAs+SimpUI5p3M+UIPro18p6M/checkpoint-32656'):
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_name, None, model_name, use_flash_attn=True)
        self.model.to(device)
        self.model.generation_config.eos_token_id = 107 # '<end_of_turn>'
        self.gen_kwargs = {"temperature": 0.0, "top_p": None, "num_beams": 1}
        self.prompt_template = "What happens when you click point {} on the screen?"

    def caption(self, image: Image.Image, points: list, max_new_tokens: int = 128):
        funcpred = []
        for point in points:
            prompt = self.prompt_template.format(f"({point[0]},{point[1]})")
            conv = conv_templates['gemma'].copy()
            conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
            conv.append_message(conv.roles[1], None)
            prompt_formatted = conv.get_prompt()
            
            img = [image.convert('RGB')]
            img_tensor = process_images(img, self.image_processor, self.model.config).to(dtype=self.model.dtype, device=self.model.device) # B x num_patches x 3 x 336 x 336
            
            self.gen_kwargs["image_sizes"] = [img[0].size]

            input_ids = tokenizer_image_token(prompt_formatted, self.tokenizer, constants.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=self.model.device)
            
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=img_tensor,
                    image_sizes=self.gen_kwargs["image_sizes"],
                    do_sample=True if self.gen_kwargs["temperature"] > 0 else False,
                    temperature=self.gen_kwargs["temperature"],
                    top_p=self.gen_kwargs["top_p"],
                    num_beams=self.gen_kwargs["num_beams"],
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
                response = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

            funcpred.append(response)
        return funcpred

from typing import List
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration
)

def detect_repeated_phrases(text, min_repeats=3):
    words = text.split()
    n = len(words)
    
    # Check for repeated patterns of various lengths
    for length in range(1, n // 2 + 1):
        for start in range(n - length):
            phrase = tuple(words[start:start + length])
            count = 0
            for i in range(start, n - length + 1, length):
                if tuple(words[i:i + length]) == phrase:
                    count += 1
                else:
                    break
            if count >= min_repeats:
                return True, ' '.join(phrase)
    
    return False, None

class QWenVL_Captioner:
    def __init__(self, device, model_name: str = 'Qwen/Qwen2-VL-2B-Instruct'):
        self.model_name = model_name
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1008*1008)

    def caption(self, images: List[Image.Image], points: List[str] = None, max_new_tokens: int = 128):
        to_proc = list(range(len(images)))
        captions = ['' for _ in range(len(images))]

        while True:
            conversations = []
            for i in range(len(images)):
                if i in to_proc:
                    target = f"({points[i][0]},{points[i][1]})" if isinstance(points[i], list) else points[i]
                    conversations.append([
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": "<image>\nDescribe the functionality description of the element at {target} in detail.".format(target=target)}
                            ]
                        }
                    ])

            texts = [self.processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
            inputs = self.processor(
                text=texts,
                images=[images[i] for i in range(len(images)) if i in to_proc],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device) # input_ids: B x L | attention_mask: B x L | pixel_values: (B*X) x Y | image_grid_thw: B x num_grids
                    
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=1.0)

            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            new_to_proc = []
            for i, cap in zip(to_proc, captions):
                has_repeats, repeated_phrase = detect_repeated_phrases(cap)
                
                if has_repeats:
                    captions[i] = cap[:cap.find(repeated_phrase)].strip()
                else: captions[i] = cap.strip()
                
                # Qwen2-vl keeps outputting repeated patterns even adjusting the temperature
                # if has_repeats: new_to_proc.append(i)
                # else: captions[i] = cap

            to_proc = new_to_proc
            if len(to_proc) == 0: break

        return captions