import torch
from PIL import Image
import time
from .base import BaseExecutor

# TODO: now we only support single image generation

## stable-diffusion
class StableDiffusionExecutor(BaseExecutor):
    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name, device)

    def load_model(self):
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_name, torch_dtype=self.torch_dtype, use_safetensors=True, variant="fp16")
        # keep the variant="fp16" casue models supported now all have this variant
        # The pipeline is set in evaluation mode (`model.eval()`) by default.
        pipe.to(self.device)
        
        # # use torch.compile for pytorch>2.0
        # torch._inductor.config.conv_1x1_as_mm = True
        # torch._inductor.config.coordinate_descent_tuning = True
        # torch._inductor.config.epilogue_fusion = False
        # torch._inductor.config.coordinate_descent_check_all_directions = True
        # pipe.unet.to(memory_format=torch.channels_last)
        # pipe.vae.to(memory_format=torch.channels_last)
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        self.model = pipe

    def generate_output(self, inputs: dict):
        from compel import Compel, ReturnedEmbeddingsType
        start_time = time.perf_counter_ns()
        if self.model_name == "stable-diffusion-v1-5/stable-diffusion-v1-5":
            compel = Compel(tokenizer=self.model.tokenizer, text_encoder=self.model.text_encoder)
            conditioning = compel(inputs["prompt"])
            images = self.model(
                prompt_embeds=conditioning,
                num_inference_steps=inputs["num_inference_steps"],
                height=512,
                width=512,
            ).images[0]  
        elif self.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
            # XL models need both prompt_embeds and pooled_prompt_embeds
            compel = Compel(tokenizer=[self.model.tokenizer, self.model.tokenizer_2],
                            text_encoder=[self.model.text_encoder, self.model.text_encoder_2],
                            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                            requires_pooled=[False, True])
            conditioning, pooled = compel(inputs["prompt"])
            images = self.model(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                num_inference_steps=inputs["num_inference_steps"],
                height=1024,
                width=1024,
            ).images[0]
        

        end_time = time.perf_counter_ns()
        self.latency += (end_time - start_time) / 1e9

        images.save(inputs["text"])
        print(images.size)

        return inputs["text"]

    def get_memory(self):
        for name, module in self.model.components.items():
            if isinstance(module, torch.nn.Module):
                # check whether the model has "get_memory_footprint" method
                if hasattr(module, "get_memory_footprint"):
                    self.memory += module.get_memory_footprint() / 1014 / 1024 / 1024
                else:
                    mem = sum([param.nelement() * param.element_size() for param in module.parameters()])
                    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in module.buffers()])
                    mem = mem + mem_bufs
                    self.memory += mem / 1024 / 1024 / 1024
        return self.memory

