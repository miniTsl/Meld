import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


from text_to_image import StableDiffusionExecutor
# pipe_name = "stabilityai/stable-diffusion-xl-base-1.0"    # 1024, 1024
pipe_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"   # 512, 512
tmp = StableDiffusionExecutor(pipe_name)
prompt = "A man++ walking along side by a river, movie style, high quality"
inputs = {
    "prompt": prompt,
    "text": "test.png",
    "num_inference_steps": 40,
}
tmp.generate_output(inputs)
print(tmp.get_latency())
print(tmp.get_memory())
print(tmp.model.components["vae"].dtype)