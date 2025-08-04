import time
import torch

from llava_uground.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava_uground.conversation import conv_templates
from llava_uground.mm_utils import tokenizer_image_token, process_images, pre_resize_by_width
from llava_uground.model.builder import load_pretrained_model
from llava_uground.utils import disable_torch_init

DEFAULT_IMAGE_TOKEN = '<image>'

class UGround_LLAVA:
    def __init__(self, device, model_name: str = 'Qwen/Qwen2-VL-7B-Instruct'):
        self.model_name = model_name
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_name, None, model_name, device='cuda')


    def test_speed(self, prompt: str, image: str = None, max_new_tokens: int = 512):
        if DEFAULT_IMAGE_TOKEN not in prompt:
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}".strip()

        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Apply chat template
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                
        resized_image, pre_resize_scale = pre_resize_by_width(image)
        image_tensor, image_new_size = process_images([resized_image], self.image_processor, self.model.config)

        start = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.to(device=self.model.device, dtype=self.model.dtype),
                image_sizes=[image_new_size],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                use_cache=False
            )
        duration = time.time() - start

        return duration, output_ids # assume bs = 1

        
    def get_model_response(self, prompt: str, image: str = None, temperature: float = 0.0, max_new_tokens: int = 512, sys_prompt: str = ''):
        pass