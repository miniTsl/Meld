import torch
from PIL import Image
import time

# TODO: now Executors only support single media input(with/without dependency with previous Executors) besides prompt, should be modified to support multiple pictures
# TODO: check dtype format of different models
# TODO: check device map, default now is "auto", which means using cuda if available
# TODO: check process details, especially the time of inputs loading
# TODO: check generate details

class BaseExecutor:
    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = self.load_model()
        self.processor = self.load_processor()
        
        self.latency = 0.0
        self.memory = 0.0
        self.tokens_processed = 0
        self.pixels_processed = 0
    
    def load_model(self):
        pass
    
    def load_processor(self):
        pass
    
    def pre_process(self, messages):
        pass
    
    def generate_output(self, inputs:dict):
        pass
    
    def get_memory(self):
        self.memory = self.model.get_memory_footprint()/1014/1024/1024
        return self.memory
    
    def get_latency(self):
        return self.latency
    
    def get_tokens_processed(self):
        return self.tokens_processed
    
    def get_pixels_processed(self):
        return self.pixels_processed
    
    # TODO: add other metrics!!!


### image & text to text models

## qwen2-VL
class Qwen2VLExecutor(BaseExecutor):
    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name, device)
    
    def load_model(self):
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
        model.eval()
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     self.model_name,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )
        return model
    
    def load_processor(self):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self.model_name)
        # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        return processor
    
    def pre_process(self, messages):
        from qwen_vl_utils import process_vision_info
        start_time = time.perf_counter_ns()
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9
        
        # width, height = image_inputs[0].size
        # self.pixels_processed += width * height
        
        # # get the input token num of inputs["input_ids"], which is a torch.tensor
        # # TODO: is that right? Seems the input_ids is much longer than user-request ...
        # self.tokens_processed += inputs["input_ids"].numel()
        
        return inputs
    
    def generate_output(self, inputs:dict):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": inputs["image"],
                    },
                    {"type": "text", "text": inputs["prompt"]},
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
        )
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9
        
        # self.tokens_processed += generated_ids_trimmed[0].numel()
        
        return output_text


## Florence-2
class FlorenceExecutor(BaseExecutor):
    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name, device)
    
    def load_model(self):
        from transformers import AutoModelForCausalLM 
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code = True,
            torch_dtype="auto",
        ).to(self.device)
        model.eval()

        return model
    
    def load_processor(self):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        return processor
    
    def pre_process(self, messages):
        start_time = time.perf_counter_ns()
        inputs = self.processor(text=messages["prompt"], images=messages["image"], return_tensors="pt")
        # dtype is very important
        inputs = inputs.to(self.device, self.torch_dtype)
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return inputs
    
    def generate_output(self, inputs:dict):
        # check if inputs has "text_input" key:
        if inputs["text"] != "":
            prompt = inputs["text"] + inputs["prompt"]
        else:
            prompt = inputs["prompt"]
            
        # open image file according to the image file path
        image = Image.open(inputs["image"])
        print(image.size)

        messages = {
            "prompt": prompt,
            "image": image
        }
        input_data = self.pre_process(messages)
        
        start_time = time.perf_counter_ns()
        generated_ids = self. model.generate(
            input_ids=input_data["input_ids"],
            pixel_values=input_data["pixel_values"],
            max_new_tokens=inputs["max_output_size"],
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=inputs["prompt"], image_size=(image.width, image.height))
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9
        
        return parsed_answer
    



### text to text models

## qwen2.5
class Qwen2Executor(BaseExecutor):
    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name, device)
    
    def load_model(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        model.eval()
        return model
    
    def load_processor(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
    
    def pre_process(self, messages):
        start_time = time.perf_counter_ns()
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.processor([text], return_tensors="pt").to(self.device)
        
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9
    
        return model_inputs
    
    def generate_output(self, inputs:dict):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": inputs["prompt"]}
        ]
        input_data = self.pre_process(messages)
       
        start_time = time.perf_counter_ns()
        generated_ids = self.model.generate(
            **input_data,
            max_new_tokens=inputs["max_output_size"]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_data.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9
        
        return response
    


### text to image models

## stable-diffusion
class StableDiffusionExecutor(BaseExecutor):
    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name, device)
    
    def load_model(self):
        if self.model_name == "stable-diffusion-v1-5/stable-diffusion-v1-5":
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(self.model_name, torch_dtype=torch.float16)

        elif self.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        # The pipeline is set in evaluation mode (`model.eval()`) by default.
        pipe.to(self.device)
        return pipe
    
    def generate_output(self, inputs:dict):
        from compel import Compel, ReturnedEmbeddingsType
        if self.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
            compel = Compel(tokenizer=[self.model.tokenizer, self.model.tokenizer_2],
                            text_encoder=[self.model.text_encoder, self.model.text_encoder_2],
                            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                            requires_pooled=[False, True])
            conditioning, pooled = compel(inputs["prompt"])
        elif self.model_name == "stable-diffusion-v1-5/stable-diffusion-v1-5":
            compel = Compel(tokenizer=self.model.tokenizer, text_encoder=self.model.text_encoder)
            conditioning = compel(inputs["prompt"])

        start_time = time.perf_counter_ns()
        max_output_size = inputs["max_output_size"]
        if self.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
            images = self.model(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                num_images_per_prompt=max_output_size[0],
                height=max_output_size[1],
                width=max_output_size[2],
            ).images[0]
        elif self.model_name == "stable-diffusion-v1-5/stable-diffusion-v1-5":
            images = self.model(
                prompt_embeds=conditioning,
                num_images_per_prompt = max_output_size[0],
                height = max_output_size[1],
                width = max_output_size[2]
            ).images[0] # TODO: now we only support single image generation
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        # save image
        images.save(inputs["text"])

        # return the image path
        return inputs["text"]
    
    def get_memory(self):
        for name, module in self.model.components.items():
            if isinstance(module, torch.nn.Module):
                # check whether the model has "get_memory_footprint" method
                if hasattr(module, "get_memory_footprint"):
                    self.memory += module.get_memory_footprint()/1014/1024/1024
                else:
                    mem = sum([param.nelement() * param.element_size() for param in module.parameters()])
                    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in module.buffers()])
                    mem = mem + mem_bufs
                    self.memory += mem/1024/1024/1024
        return self.memory

