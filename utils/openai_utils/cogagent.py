import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image

class CogAgent:
    """
    Cogagent Chat Model from Hugging Face # from https://huggingface.co/THUDM/cogagent-chat-hf
    
    Example usage:
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m lmms_eval \
        --model cogagent_chat_hf 
        --model_args pretrained=THUDM/cogagent-chat-hf,device_map='' 
        --tasks motif_bbox_test 
        --batch_size 1 
        --log_samples 
        --log_samples_suffix cogagent-chat-hf_motif_bbox_test 
        --output_path ./logs/
    
    NOTE: transformers == 4.43.2 is requried
    """

    def __init__(self, device, model_name: str = 'Qwen/Qwen2-VL-7B-Instruct'):

        self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map=device).eval()
        self.tokenizer =  LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    
    def get_model_response(self, prompt: str, image: str, max_new_tokens: int = 1024):
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]
        
        def collate_fn(features, tokenizer) -> dict:
            images = [feature.pop('images') for feature in features]
            tokenizer.padding_side = 'left'
            padded_features = tokenizer.pad(features)
            inputs = {**padded_features, 'images': images}
            return inputs
        

        def recur_move_to(item, tgt, criterion_func):
            if criterion_func(item):
                device_copy = item.to(tgt)
                return device_copy
            elif isinstance(item, list):
                return [recur_move_to(v, tgt, criterion_func) for v in item]
            elif isinstance(item, tuple):
                return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
            elif isinstance(item, dict):
                return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
            else:
                return item

        samples = []
        sample = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[Image.open(image.replace("file://",""))])
        samples.append(sample)
        inputs = collate_fn(samples, self.tokenizer)
        # batch infernece: https://github.com/THUDM/CogVLM/issues/143
        inputs = recur_move_to(inputs, self.model.device, lambda x: isinstance(x, torch.Tensor))
        inputs = recur_move_to(inputs, self.model.dtype, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

        cont = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

        output = cont[0, inputs['input_ids'].shape[1]:]
        text_output = self.tokenizer.decode(output).split("</s>")[0].strip()

        return text_output