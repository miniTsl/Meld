import torch
from PIL import Image
import time
from base import BaseExecutor

## qwen2-VL
class Qwen2VLExecutor(BaseExecutor):
    def __init__(self, model_name, model_quant=None, device="cuda"):
        super().__init__(model_name, model_quant, device)

    def load_model(self):
        from transformers import Qwen2VLForConditionalGeneration
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     self.model_name,
        #     torch_dtype="auto",
        #     device_map="auto"
        # )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.model.eval()

    def load_processor(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    def pre_process(self, messages):
        from qwen_vl_utils import process_vision_info

        start_time = time.perf_counter_ns()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Excepted output:
        # '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
        image_inputs, video_inputs = process_vision_info(messages)  # open image or video file
        model_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self.device)
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9
        
        # width, height = image_inputs[0].size
        # self.pixels_processed += width * height
        
        # # get the input token num of inputs["input_ids"], which is a torch.tensor
        # # TODO: is that right? Seems the input_ids is much longer than user-request ...
        # self.tokens_processed += inputs["input_ids"].numel()
        
        return model_inputs
    
    def generate_output(self, inputs:dict):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": inputs["image"],
                    },
                    {
                        "type": "text",
                        "text": inputs["prompt"]
                    },
                ],
            }
        ]
        input_data = self.pre_process(messages)
        
        start_time = time.perf_counter_ns()
        generated_ids = self.model.generate(
            **input_data,
            max_new_tokens=inputs["max_output_size"]
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_data.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]   # output_text is a list of strings
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9
        
        # self.tokens_processed += generated_ids_trimmed[0].numel()
        
        return output_text


## Florence-2
class FlorenceExecutor(BaseExecutor):
    def __init__(self, model_name, model_quant=None, device="cuda"):
        super().__init__(model_name, model_quant, device)
    
    def load_model(self):
        from transformers import AutoModelForCausalLM
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code = True,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.model.eval()

    def load_processor(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

    def pre_process(self, messages):
        start_time = time.perf_counter_ns()
        model_inputs = self.processor(text=messages["text"], images=messages["image"], return_tensors="pt")
        # dtype is very important, must be specified here
        model_inputs = model_inputs.to(self.device, self.torch_dtype)
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return model_inputs
    
    def generate_output(self, inputs:dict):
        # check if inputs has "text" key:
        if inputs["text"] != "":
            prompt = inputs["text"] + inputs["prompt"]
        else:
            prompt = inputs["prompt"]
            
        # open image file according to the image file path
        image = Image.open(inputs["image"])

        messages = {
            "text": prompt,
            "image": image
        }
        input_data = self.pre_process(messages)
        
        start_time = time.perf_counter_ns()
        generated_ids = self.model.generate(
            input_ids=input_data["input_ids"],
            pixel_values=input_data["pixel_values"],
            max_new_tokens=inputs["max_output_size"],
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        output_text = self.processor.post_process_generation(generated_text, task=inputs["prompt"], image_size=(image.width, image.height))
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9
        
        return output_text


## llava
class LLaVAExecutor(BaseExecutor):
    def __init__(self, model_name, model_quant=None, device="cuda"):
        super().__init__(model_name, model_quant, device)

    def load_model(self):
        if "1.5" in self.model_name:
            from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True) if self.model_quant == "4bit" else None
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                quantization_config=quantization_config # You shouldn't move a model that is dispatched using accelerate hooks.
            ).to(self.device)
        else:
            from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True) if self.model_quant == "4bit" else None
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                quantization_config=quantization_config # You shouldn't move a model that is dispatched using accelerate hooks.
            ).to(self.device)
        self.model.eval()

    def load_processor(self):
        if "1.5" in self.model_name:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        else:
            from transformers import LlavaNextProcessor
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)

    def pre_process(self, messages):
        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image")
        start_time = time.perf_counter_ns()
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        # get image data from messages, but now we support only one image
        content = messages[0]["content"]
        for c in content:
            if c["type"] == "image":
                image = c["image"]
        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self.device, self.torch_dtype)
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return model_inputs

    def generate_output(self, inputs:dict):

        # open image file according to the image file path
        image = Image.open(inputs["image"])

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": inputs["prompt"]
                    },
                ],
            }
        ]
        input_data = self.pre_process(messages)

        start_time = time.perf_counter_ns()
        generated_ids = self.model.generate(
            **input_data,
            max_new_tokens=inputs["max_output_size"],
            do_sample=False
        )
        output_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return output_text


## Phi-3.5-vision-instruct todo


## InternVL todo


## Emu3-Chat todo