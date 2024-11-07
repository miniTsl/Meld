from format import *
from problems import *
from examples import *
from utils import *
import json
from itertools import product
import copy

# model_candidates = {
#     "image_text_to_text": {
#         "Qwen/Qwen2-VL-2B-Instruct": ["original"],
#         "Qwen/Qwen2-VL-7B-Instruct": ["original"],
#         # "Qwen/Qwen2-VL-7B-Instruct-AWQ": ["original"],
#         # "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4": ["original"],
#         # "microsoft/Florence-2-large": ["original"],
#         # "microsoft/Florence-2-base": ["original"],
#         "llava-hf/llava-1.5-7b-hf": ["original", "4bit"],
#         "llava-hf/llava-1.5-13b-hf": ["original", "4bit"],
#         "llava-hf/llama3-llava-next-8b-hf": ["original", "4bit"],
#         "llava-hf/llava-v1.6-mistral-7b-hf": ["original", "4bit"],
#     },
#     "text_to_text": {
#         "google/gemma-2-2b-it": ["original", "4bit", "8bit"],
#         # "google/gemma-2-9b-it": ["original", "4bit", "8bit"],
#         "Qwen/Qwen2.5-0.5B-Instruct": ["original"],
#         "Qwen/Qwen2.5-1.5B-Instruct": ["original"],
#         "Qwen/Qwen2.5-3B-Instruct": ["original"],
#         "Qwen/Qwen2.5-7B-Instruct": ["original"],
#         "meta-llama/Llama-3.2-1B-Instruct": ["original", "4bit", "8bit"],
#         "meta-llama/Llama-3.2-3B-Instruct": ["original", "4bit", "8bit"],
#     },
#     "text_to_image": {
#         "stable-diffusion-v1-5/stable-diffusion-v1-5": ["original"],
#         "stabilityai/stable-diffusion-xl-base-1.0": ["original"],
#     }
# }

model_candidates = {
    "image_text_to_text": {
        "Qwen/Qwen2-VL-2B-Instruct": ["original"],
        "Qwen/Qwen2-VL-7B-Instruct": ["original"],
        "llava-hf/llava-1.5-7b-hf": ["original", "4bit"],
        "llava-hf/llava-1.5-13b-hf": ["original", "4bit"],
        "llava-hf/llama3-llava-next-8b-hf": ["original", "4bit"],
        "llava-hf/llava-v1.6-mistral-7b-hf": ["original", "4bit"],
    },
    "text_to_text": {
        "google/gemma-2-2b-it": ["original", "4bit", "8bit"],
        "google/gemma-2-9b-it": ["original", "4bit", "8bit"],
        "Qwen/Qwen2.5-0.5B-Instruct": ["original"],
        # "Qwen/Qwen2.5-1.5B-Instruct": ["original"],
        "Qwen/Qwen2.5-3B-Instruct": ["original"],
        # "Qwen/Qwen2.5-7B-Instruct": ["original"],
        "meta-llama/Llama-3.2-1B-Instruct": ["original", "4bit", "8bit"],
        "meta-llama/Llama-3.2-3B-Instruct": ["original", "4bit", "8bit"],
    },
    "text_to_image": {
        "stable-diffusion-v1-5/stable-diffusion-v1-5": ["original"],
        "stabilityai/stable-diffusion-xl-base-1.0": ["original"],
    }
}

user_request = problem23
user_inputs = {
    "text": "/home/sunyi/Meld/assets/text/problem_23.txt",
}

demonstrations = []
examples = ["program_1_v1.json", "program_2_v1.json", "program_6_v1.json", "program_7_v1.json"]
for example in examples:
    with open(f"examples/{example}", "r") as f:
        demonstrations.append(json.load(f))

prompt = Prompt.format(user_request=user_request, user_inputs=user_inputs, model_candidates=model_candidates, demonstrations=demonstrations)
# save the extended prompt into a file
with open("prompt.txt", "w") as f:
    f.write(prompt)
    
model_name = "gpt-4o-2024-08-06"
temperature = 0.9

# program_list = query(prompt, model_name=model_name, json_format=ProgramList, temperature=temperature)
# print(f"Programs are generated.")

# program_list_data = program_list.model_dump_json(indent=4)  # return A JSON string representation of the model.
# with open("program_list.json", "w") as f:
#     f.write(program_list_data)

with open("program_list.json", "r") as f:
    program_list_data = json.load(f)
    program_list_data = program_list_data["programs"]
    # travse the text_to_text models and max new tokens
    for index, program in enumerate(program_list_data):
        count = 1
        model_names = list(model_candidates["text_to_text"].keys())
        generate_limits = [1000]
        task_count = len(program["tasks"])
        for model_limit_combination in product(product(model_names, generate_limits), repeat=task_count):
            variant = copy.deepcopy(program)
            
            for i, (model, limit) in enumerate(model_limit_combination):
                variant["tasks"][i]["model"] = model
                variant["tasks"][i]["generate_limit"] = [limit]

            variant_data = json.dumps(variant, indent=4)
            print(f"Saving program {index + 1}_v{count}")
            with open(f"/home/sunyi/Meld/programs/problem23/program_{index + 1}_v{count}.json", "w") as f:
                f.write(variant_data)
            count += 1