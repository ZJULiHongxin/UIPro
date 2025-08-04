import time
import torch
from functools import partial

from utils.openai_utils.ferretui_utils.inference import inference_and_run
from utils.openai_utils.ferretui_utils.conversation import conv_templates, SeparatorStyle
from utils.openai_utils.ferretui_utils.mm_utils import tokenizer_image_token, process_images
from utils.openai_utils.ferretui_utils.builder import load_pretrained_model


class FERRETUI:
    def __init__(self, device, model_name: str = 'jadechoghari/Ferret-UI-Gemma2b'):
        self.model_name = model_name

        if 'gemma' in model_name.lower():
            model_type = 'ferret_gemma'
        elif 'llama' or 'vicuna' in model_name.lower():
            model_type = 'ferret_llama'

        self.conv_mode = 'ferret_gemma_instruct' if 'gemma' in model_type else 'ferret_llama_3'
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_name, None, model_type)


    def test_speed(self, prompt: str, image: str = None, max_new_tokens: int = 512, sys_prompt: str = ''):
        if "<image>" in prompt:
            prompt = prompt.split('<image>')[1]
        prompt = "<image>\n" + prompt
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()

        image_size = image.size

        if self.model.config.image_aspect_ratio == "square_nocrop":
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt', do_resize=True, 
                                                do_center_crop=False, size=[336, 336])['pixel_values'][0]
        elif self.model.config.image_aspect_ratio == "anyres":
            image_process_func = partial(self.image_processor.preprocess, return_tensors='pt', do_resize=True, do_center_crop=False, size=[336, 336])
            image_tensor = process_images([image], self.image_processor, self.model.config, image_process_func=image_process_func)[0]
        else:
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        image_tensor = image_tensor.to(device=self.model.device, dtype=self.model.dtype)
             
        start = time.time()
        with torch.inference_mode():
            self.model.orig_forward = self.model.forward
            self.model.forward = partial(
                self.model.orig_forward,
                region_masks=None
            )
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                region_masks=None,
                image_sizes=[image_size],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                use_cache=False)
            self.model.forward = self.model.orig_forward
        duration = time.time() - start

        return duration, output_ids # assume bs = 1

        
    def get_model_response(self, prompt: str, image: str = None, temperature: float = 0.0, max_new_tokens: int = 512, sys_prompt: str = ''):
        pass