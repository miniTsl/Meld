import torch
from PIL import Image
import time

# TODO: now Executors only support single media input(with/without dependency with previous Executors) besides prompt, should be modified to support multiple media files
# TODO: now Executors only support up to single GPU, should be modified to support multiple GPUs?
# TODO: check dtype format of different models
# TODO: check device map, default now is "auto", which means using cuda if available
# TODO: check process details, especially the time of inputs loading
# TODO: check generate details

class BaseExecutor:
    def __init__(self, model_name:str, model_quant:str=None, device:str="cuda"):
        self.model_name = model_name
        self.model_quant = model_quant  # optional quantization
        self.model = None   # model or pipeline(for diffusers)
        self.processor = None   # processors or tokenizers
        self.torch_dtype = None
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.load_model()
        self.load_processor()

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

    def generate_output(self, inputs: dict):
        pass

    def get_memory(self):
        # for pretrained models from transformers
        self.memory = self.model.get_memory_footprint() / 1014 / 1024 / 1024
        return self.memory

    def get_latency(self):
        return self.latency

    def get_tokens_processed(self):
        return self.tokens_processed

    def get_pixels_processed(self):
        return self.pixels_processed

    # TODO: add other metrics!

