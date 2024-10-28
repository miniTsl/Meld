import torch
from PIL import Image
import time
from base import BaseExecutor


## qwen2.5
class Qwen2Executor(BaseExecutor):
    def __init__(self, model_name, model_quant=None, device="cuda"):
        super().__init__(model_name, model_quant, device)

    def load_model(self):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model.eval()

    def load_processor(self):
        from transformers import AutoTokenizer
        self.processor = AutoTokenizer.from_pretrained(self.model_name)

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

    def generate_output(self, inputs: dict):
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
            max_new_tokens=inputs["max_output_size"]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_data.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return response


## gemma2
class Gemm2Executor(BaseExecutor):
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
        model_inputs = self.processor.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(self.device)  # the format is for instruction-tuning models but should work for all
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return model_inputs

    def generate_output(self, inputs: dict):
        messages = [
            {
                "role": "user",
                "content": inputs["prompt"]
            },
        ]
        input_data = self.pre_process(messages)

        start_time = time.perf_counter_ns()
        generated_ids = self.model.generate(
            **input_data,
            max_new_tokens=inputs["max_output_size"]
        )
        response = self.processor.decode(generated_ids[0])
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
                                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_quant_type= "nf4")
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
        model_inputs = self.processor(messages["prompt"], return_tensors="pt").to("cuda")
        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        return model_inputs

    def generate_output(self, inputs: dict):
        messages = inputs
        input_data = self.pre_process(messages)

        start_time = time.perf_counter_ns()
        generated_ids = self.model.generate(
            **input_data,
            max_new_tokens=inputs["max_output_size"]
        )
        response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
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