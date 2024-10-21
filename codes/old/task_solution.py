from ..questions import *
from ..models import *
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt



__TEXT_IMAGE_TO_TEXT_MODEL__ = [
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct-AWQ",
    "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large-ft"
]

__TEXT_TO_TEXT_MODEL__ = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct"
]

__TEXT_TO_IMAGE__MODEL__ = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0"
]

__IMAGE_TEXT_TO_IMAGE__MODEL__ = [
    "timbrooks/instruct-pix2pix"
]

### question1
print("#"*50)
print("Question is: ", question2)
print("#"*20)
    

for i in range(5):
    for j in range(2):
        user_request = question2
        image_file_path = "../../assets/lion_walking.jpg"
        # task_0
        task_0_type = "text_image_to_text"
        task_0_id = 0
        task_0_dependency = -1
        task_0_model = __TEXT_IMAGE_TO_TEXT_MODEL__[i]
        task_0_prompt = "What kind of animals are in the picture?" if i < 3 else "<DETAILED_CAPTION>"
        task_0_image = image_file_path
        task_0_interpreter = Qwen2VLInterpreter(task_0_model) if i < 3 else FlorenceInterpreter(task_0_model)
        task_0_outputs = task_0_interpreter.generate_output(task_0_prompt, task_0_image)
        task_0_latency = task_0_interpreter.get_latency()
        task_0_memory = task_0_interpreter.get_memory()
        del task_0_interpreter
        TASK_0 = TASK_FORMAT.format(task_type=task_0_type, task_id=task_0_id, task_dependency=task_0_dependency, model=task_0_model, prompt=task_0_prompt, image_file_path=task_0_image, video_file_path="null", audio_file_path="null", text_file_path="null", others="null", outputs="TASK_0_OUTPUTS")
        print("Task_0:\n", TASK_0)

        # task_1
        task_1_type = "text_to_image"
        task_1_id = 1
        task_1_dependency = 0
        task_1_model = __TEXT_TO_IMAGE__MODEL__[j]
        task_1_prompt = "The result of animal kind recognition from previous task is as follows. " \
                        + str(task_0_outputs) \
                        + " Generate a picture with the kinds of animals but in a modern art style."
        task_1_interpreter = StableDiffusionInterpreter(task_1_model)
        task_1_outputs = task_1_interpreter.generate_output(prompt=task_1_prompt)
        task_1_latency = task_1_interpreter.get_latency()
        task_1_memory = task_1_interpreter.get_memory()
        del task_1_interpreter
        new_image_path = "../results/task_1_outputs_" + str(i) + "_" + str(j) + ".jpg"
        task_1_outputs.save(new_image_path)
        TASK_1 = TASK_FORMAT.format(task_type=task_1_type, task_id=task_1_id, task_dependency=task_1_dependency, model=task_1_model, prompt=task_1_prompt, image_file_path="null", video_file_path="null", audio_file_path="null", text_file_path="null", others="null", outputs=new_image_path)
        print("Task_1:\n", TASK_1)

        # Program_0, must use "task_0": task_0, "task_1": task_1 format to fill the task_list.
        task_list = "task_0: " + TASK_0 + ",\n" + "task_1: " + TASK_1
        PROGRAM = PROGRAM_FORMAT.format(user_request=user_request, image_file_path=image_file_path, video_file_path="null", audio_file_path="null", task_list=task_list)

        # save program and results into a txt file
        with open("../results/program.txt", "a") as f:
            f.write(PROGRAM)
            f.write("#"*20 + "\n")
            f.write("Task_0 Outputs:\n")
            f.write(str(task_0_outputs) + "\n")
            f.write("#"*20 + "\n")
            f.write("Task_0 latency: " + str(task_0_latency) + " s\n")
            f.write("Task_0 memory: " + str(task_0_memory) + " GB\n")
            f.write("#"*20 + "\n")
            f.write("Task_1 latency: " + str(task_1_latency) + " s\n")
            f.write("Task_1 memory: " + str(task_1_memory) + " GB\n")