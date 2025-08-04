import base64, openai, random
from PIL import Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_np(np_img):
    return base64.b64encode(Image.fromarray(np_img).tobytes()).decode('utf-8')

class OpenAIModel:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 1.0, max_tokens: int = 1024):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.p_tokens = self.cmpl_tokens = 0

    def get_completion(self, prompt: str, logprobs: int = 20, temperature: float=0.0, max_new_tokens: int = 256):
        
        logprob_args = {"logprobs": logprobs} if logprobs != -1 else {}

        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            timeout=30,
            **logprob_args
        )
        
        # response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            usage = response.usage
            self.p_tokens += usage.prompt_tokens
            self.cmpl_tokens += usage.completion_tokens
            status = True
            resp = response.choices[0].text
            
            logprob_info = []
            if logprobs != -1:
                for token, text_offset, token_logprob, top_logprobs in zip(response.choices[0].logprobs.tokens, response.choices[0].logprobs.text_offset, response.choices[0].logprobs.token_logprobs, response.choices[0].logprobs.top_logprobs):
                    logprob_info.append({
                        'token': token, # str
                        'text_offset': text_offset, # int
                        'token_logprob': token_logprob, # float
                        'top_logprobs': top_logprobs # dict[<candidate token>: <logprob>]
                        })
        else:
            status, resp, logprob_info = False, response.error["message"], []
        
        return status, resp, logprob_info

    def get_model_response(self, prompt: str, images: list = None, use_img_url: bool = True, logprob: bool = False, use_system: bool = True, sys_prompt: str = '', temperature: float | None = None, num_gen: int = 1, timeout: int = 60):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        if images is not None:
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(img)}" if use_img_url else f"data:image/jpeg;base64,{encode_image_np(img)}"
                    }
                })
        
        logprob_args = {"logprobs": True, "top_logprobs": 4} if logprob else {}

        messages = [{"role": "system", "content": sys_prompt if sys_prompt else "You are a helpful assistant."}] if use_system else []
        messages.append({
                    "role": "user",
                    "content": content
                })

        self.client.api_key = random.choice(
            ["ak-68d2efa11e2ab28ccac3e3e6c825b6a1",
                "ak-61d3efgh29i8jkl75mno46pqrs32tuv7k0"]
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
            timeout=timeout,
            n=num_gen,
            **logprob_args
        )

        # response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            usage = response.usage
            self.p_tokens += usage.prompt_tokens
            self.cmpl_tokens += usage.completion_tokens
            status = True
            
            resp_list, logprobs_list, reasoning_content_list = [], [], []

            for i in range(len(response.choices)):
                reasoning_content = getattr(response.choices[i].message, 'reasoning_content', '')
                resp = response.choices[i].message.content

                if reasoning_content:
                    resp = f"<think>\n{reasoning_content}\n</think>\n{resp}"

                logprobs = response.choices[i].logprobs.content if logprob and response.choices[i].logprobs is not None else None
                
                resp_list.append(resp)
                logprobs_list.append(logprobs)
                reasoning_content_list.append(reasoning_content)
        else:
            status, resp, logprobs = False, response.error["message"], None
        
        if num_gen == 1:
            return status, resp_list[0], logprobs_list[0]
        else:
            return status, resp_list, logprobs_list

    def get_model_response_with_prepared_messages(self, messages: list, logprob: bool = False, temperature: float | None = None, max_new_tokens: int = None):
        logprob_args = {"top_logprobs": 4} if logprob else {}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_new_tokens if max_new_tokens is not None else self.max_tokens,
            timeout=300,
            logprobs=logprob,
            **logprob_args
        )
        
        # response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            usage = response.usage
            self.p_tokens += usage.prompt_tokens
            self.cmpl_tokens += usage.completion_tokens
            status = True
            resp = response.choices[0].message.content

            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
            if reasoning_content:
                resp = f"<think>\n{reasoning_content}\n</think>\n{resp}"

            logprobs = response.choices[0].logprobs.content if logprob and response.choices[0].logprobs is not None else None
        else:
            status, resp, logprobs = False, response.error["message"], None
        
        return status, resp, logprobs
    
    def get_model_response_history(self, prompt: str, images: list = None, history: list = None, use_img_url: bool = True, logprob: bool = False, use_system: bool = True):
        messages = [{"role": "system", "content": "You are a helpful assistant."}] if use_system else []

        # 1st turn
        first_turn_content = [
            {
                "type": "text",
                "text": history[0]
            }
        ]

        if images is not None:
            for img in images:
                first_turn_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(img)}" if use_img_url else f"data:image/jpeg;base64,{encode_image_np(img)}"
                    }
                })

        first_turn = {
                    "role": "user",
                    "content": first_turn_content
                }

        messages.append(first_turn)

        # subsequent history
        for turn_idx, turn in enumerate(history[1:]):
            messages.append(
                {
                    'role': 'assistant' if turn_idx % 2 == 0 else 'user',
                    'content': turn
                }
            )

        # add the latest prompt
        messages.append({
                    "role": "user",
                    "content": prompt
                })

        logprob_args = {"top_logprobs": 4} if logprob else {}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=300,
            logprobs=logprob,
            **logprob_args
        )
        
        # response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            usage = response.usage
            self.p_tokens += usage.prompt_tokens
            self.cmpl_tokens += usage.completion_tokens
            status = True
            resp = response.choices[0].message.content
            logprobs = response.choices[0].logprobs.content if logprob and response.choices[0].logprobs is not None else None
        else:
            status, resp, logprobs = False, response.error["message"], None
        
        return status, resp, logprobs