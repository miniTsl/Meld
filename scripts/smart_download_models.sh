#!/bin/bash

# 模型列表
models=(
# TEXT_IMAGE_TO_TEXT
"Qwen/Qwen2-VL-2B-Instruct"
"Qwen/Qwen2-VL-7B-Instruct"
"Qwen/Qwen2-VL-7B-Instruct-AWQ"
"Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
"microsoft/Florence-2-large"
"microsoft/Florence-2-base"
"llava-hf/llava-1.5-7b-hf"
"llava-hf/llava-1.5-13b-hf"
"llava-hf/llama3-llava-next-8b-hf"
"llava-hf/llava-v1.6-mistral-7b-hf"
# "microsoft/Phi-3.5-vision-instruct"  # to add
# "meta-llama/Llama-3.2-11B-Vision-Instruct" # to add
# "OpenGVLab/InternVL2-1B"   # to add
# "OpenGVLab/InternVL2-2B"   # to add
# "OpenGVLab/InternVL2-4B"   # to add
# "OpenGVLab/InternVL2-8B"   # to add
# "OpenGVLab/InternVL2-8B-AWQ"  # to add
# "OpenGVLab/InternVL2-2B-AWQ"  # to add
# "BAAI/Emu3-Chat"  # to add

# TEXT_TO_TEXT
"google/gemma-2-2b-it"
"google/gemma-2-9b-it"
"Qwen/Qwen2.5-0.5B-Instruct"
"Qwen/Qwen2.5-1.5B-Instruct"
"Qwen/Qwen2.5-3B-Instruct"
"Qwen/Qwen2.5-7B-Instruct"
"meta-llama/Llama-3.2-1B-Instruct"
"meta-llama/Llama-3.2-3B-Instruct"
# "meta-llama/Llama-3.1-8B-Instruct"
# "mistralai/Mistral-7B-Instruct-v0.3"
# "mistralai/Ministral-8B-Instruct-2410" # to add
# # "openai-community/gpt2"  # better download GPT-2 serie from manually todo
# # "facebook/opt-125m" # better download OPT serie from manually todo
# # T5 todo


# TEXT_TO_IMAGE
"stable-diffusion-v1-5/stable-diffusion-v1-5"
"stabilityai/stable-diffusion-xl-base-1.0"
# "black-forest-labs/FLUX.1-dev" # to add
# "stabilityai/stable-diffusion-2-1"   # to add
# "stabilityai/stable-diffusion-2" # to add
# "stabilityai/stable-diffusion-2-base"
# "stabilityai/stable-diffusion-3-medium-diffusers"
# "BAAI/Emu3-Gen" # todo

# # IMAGE_TEXT_TO_IMAGE 
# "timbrooks/instruct-pix2pix"   # to add
# "stabilityai/stable-diffusion-xl-refiner-1.0" # to add
# "stabilityai/stable-diffusion-2-inpainting"   # to add
)

# 遍历每个模型并下载
for model in "${models[@]}"
do
   ./hf_guohong.sh $model  --exclude "*.bin" "*.ckpt" "*.pth" # --token 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 
done