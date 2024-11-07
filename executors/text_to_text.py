import torch
from PIL import Image
import time
from .base import BaseExecutor


# class TextToTextExecutor(BaseExecutor):
#     def __init__(self, model_name, model_quant=None, device="cuda"):
#         super().__init__(model_name, model_quant, device)

#     def load_model(self):
#         from transformers import AutoModelForCausalLM
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             torch_dtype="auto",
#             device_map="auto"
#         )
#         self.model.eval()

#     def load_processor(self):
#         from transformers import AutoTokenizer
#         self.processor = AutoTokenizer.from_pretrained(self.model_name)

#     def pre_process(self, messages):
#         start_time = time.perf_counter_ns()
#         model_inputs = self.processor(messages["prompt"], return_tensors="pt").to(self.device)
#         end_time = time.perf_counter_ns()
#         self.latency += (end_time - start_time) / 1e9

#         return model_inputs

#     def generate_output(self, inputs: dict):
#         messages = inputs
#         input_data = self.pre_process(messages)

#         start_time = time.perf_counter_ns()
#         generated_ids = self.model.generate(
#             **input_data,
#             max_new_tokens=inputs["max_output_size"]
#         )
#         response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
#         end_time = time.perf_counter_ns()
#         self.latency += (end_time - start_time) / 1e9

#         return response


def merge_text(inputs:dict):
    if "text" in inputs and inputs["text"] != None:
        with open(inputs["text"], "r") as f:
            text = f.read()
        inputs["prompt"] = inputs["prompt"] + " The text material is: " + text
    return inputs

## qwen2.5
class Qwen2Executor(BaseExecutor):
    def __init__(self, model_name, model_quant=None, device="cuda"):
        super().__init__(model_name, model_quant, device)

    def load_model(self):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto", # torch.bfloat16
            device_map="auto"
        )
        self.model.eval()

    def load_processor(self):
        from transformers import AutoTokenizer
        self.processor = AutoTokenizer.from_pretrained(self.model_name)

    def pre_process(self, messages):
        start_time = time.perf_counter_ns()
        # # apply_chat_template():
        # # Converts a list of dictionaries with "role" and "content" keys to a list of token ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to determine the format and control tokens to use when converting.
        # text = self.processor.apply_chat_template(
        #     messages,
        #     tokenize=False, # Whether to tokenize the output. If False, the output will be a string.
        #     add_generation_prompt=True  # if this is set, a prompt with the token(s) that indicate the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model. Note that this argument will be passed to the chat template, and so it must be supported in the template for this argument to have any effect.
        # )
        # model_inputs = self.processor([text], return_tensors="pt").to(self.device)
        
        # the above code are equivalent to the following code
        model_inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return model_inputs

    def generate_output(self, inputs: dict, generate_limit):
        inputs = merge_text(inputs)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": inputs["prompt"]
            }
        ]
        input_data = self.pre_process(messages)

        start_time = time.perf_counter_ns()
        generated_ids = self.model.generate(
            **input_data,
            max_new_tokens=generate_limit[0]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_data.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return response


## gemma2
class Gemma2Executor(BaseExecutor):
    def __init__(self, model_name, model_quant=None, device="cuda"):
        super().__init__(model_name, model_quant, device)

    def load_model(self):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        if self.model_quant == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.model_quant == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        else:
            quantization_config = None
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            quantization_config=quantization_config
        )
        self.model.eval()

    def load_processor(self):
        from transformers import AutoTokenizer
        self.processor = AutoTokenizer.from_pretrained(self.model_name)

    def pre_process(self, messages):
        start_time = time.perf_counter_ns()
        model_inputs = self.processor.apply_chat_template(messages, add_generation_prompt = True, return_tensors="pt", return_dict=True).to(self.device)  # the format is for instruction-tuning models but should work for all
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return model_inputs

    def generate_output(self, inputs: dict, generate_limit):
        inputs = merge_text(inputs)
        messages = [
            {
                "role": "user",
                "content": inputs["prompt"]
            },
        ]   # Gemma does not support System role
        input_data = self.pre_process(messages)

        start_time = time.perf_counter_ns()
        generated_ids = self.model.generate(
            **input_data,
            max_new_tokens=generate_limit[0]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_data.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return response


class LLaMAExecutor(BaseExecutor):
    def __init__(self, model_name, model_quant=None, device="cuda"):
        super().__init__(model_name, model_quant, device)

    def load_model(self):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if self.model_quant == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.model_quant == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                    bnb_4bit_compute_dtype=torch.bfloat16,  # This sets the computational type which might be different than the input type. For example, inputs might be fp32, but computation can be set to bf16 for speedups.
                                                    bnb_4bit_use_double_quant=True, # This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.
                                                    bnb_4bit_quant_type= "nf4") # This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by fp4 or nf4.
        else:
            quantization_config = None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            quantization_config=quantization_config
        )
        self.model.eval()

    def load_processor(self):
        from transformers import AutoTokenizer
        self.processor = AutoTokenizer.from_pretrained(self.model_name)

    def pre_process(self, messages):
        start_time = time.perf_counter_ns()
        model_inputs = self.processor.apply_chat_template(messages, add_generation_prompt = True, return_tensors="pt", return_dict=True).to(self.device)
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return model_inputs

    def generate_output(self, inputs: dict, generate_limit):
        inputs = merge_text(inputs)
        messages = inputs
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": messages["prompt"]
            },
        ]
        input_data = self.pre_process(messages)

        start_time = time.perf_counter_ns()
        generated_ids = self.model.generate(
            **input_data,
            max_new_tokens=generate_limit[0]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_data.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return response


# ## mistral
# class MistralExecutor(BaseExecutor):
#     def __init__(self, model_name, model_quant=None, device="cuda"):
#         super().__init__(model_name, model_quant, device)

#     def load_model(self):
#         from transformers import AutoModelForCausalLM
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             torch_dtype="auto",
#             device_map="auto"
#         )
#         self.model.eval()

#     def load_processor(self):
#         from transformers import AutoTokenizer
#         self.processor = AutoTokenizer.from_pretrained(self.model_name)

#     def pre_process(self, messages):
#         start_time = time.perf_counter_ns()
#         model_inputs = self.processor(messages["prompt"], return_tensors="pt").to(self.device)
#         end_time = time.perf_counter_ns()
#         self.latency += (end_time - start_time) / 1e9

#         return model_inputs

#     def generate_output(self, inputs: dict):
#         messages = inputs
#         input_data = self.pre_process(messages)

#         start_time = time.perf_counter_ns()
#         generated_ids = self.model.generate(
#             **input_data,
#             max_new_tokens=inputs["max_output_size"]
#         )
#         response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
#         end_time = time.perf_counter_ns()
#         self.latency += (end_time - start_time) / 1e9

#         return response