import json
from questions import *

TEXT_IMAGE_TO_TEXT_MODEL = {
    "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
    # "Qwen2-VL-7B-Instruct-AWQ": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
    # "Qwen2-VL-7B-Instruct-GPTQ-Int4": "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
    "Florence-2-base-ft": "microsoft/Florence-2-base-ft",
    "Florence-2-large-ft": "microsoft/Florence-2-large-ft"
}

TEXT_TO_TEXT_MODEL  = {
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct"
}

TEXT_TO_IMAGE_MODEL = {
    "stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0"
}

def generate_task(task_id, task_type, task_dependency, model, inputs:dict):
    task = {
        "task_type": task_type,
        "task_id": task_id,
        "task_dependency": task_dependency,
        "model": model,
        "inputs": {
            "prompt": inputs["prompt"],
            "image": inputs["image"],
            "video": inputs["video"],
            "audio": inputs["audio"],
            "text": inputs["text"],
            "others": inputs["others"],
            "max_output_size": inputs["max_output_size"]
        },
        "outputs": f"TASK_{task_id}_OUTPUTS"
    }
    return task


def generate_program(user_request:str, user_inputs:dict, tasks:list, output_path:str):
    program = {
        "User Request": user_request,
        "Inputs": {
            "images": user_inputs["images"],
            "videos": user_inputs["videos"],
            "audios": user_inputs["audios"],
            "texts": user_inputs["texts"],
            "others": user_inputs["others"]
        },
        "Tasks": {}
    }

    for task in tasks:
        program["Tasks"][f"task_{task['task_id']}"] = task

    with open(output_path, 'w') as json_file:
        json.dump(program, json_file, indent=4)

    print(f"Program is generated and saved to {output_path}")


if __name__ == "__main__":
    token_range = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    pixel_range = [256, 512, 1024]

    ## for question1
    user_request = question1
    user_inputs = {
        "images": "../assets/cat_dog.jpg",
        "videos": "",
        "audios": "",
        "texts": "",
        "others": ""
    }
    # 2 tasks version
    for key_0, model_0 in TEXT_IMAGE_TO_TEXT_MODEL.items():
        for key_1, model_1 in TEXT_TO_IMAGE_MODEL.items():
            for max_token in token_range:
                for max_pixel in pixel_range:
                    tasks = []

                    task_id = 0
                    task_type = "text_image_to_text"
                    task_dependency = -1
                    model = model_0
                    inputs = {
                        "prompt": "<DETAILED_CAPTION>" if "Florence" in model_0 else "What kind of animals are in the picture?",
                        "image": user_inputs["images"],
                        "video": user_inputs["videos"],
                        "audio": user_inputs["audios"],
                        "text": user_inputs["texts"],
                        "others": user_inputs["others"],
                        "max_output_size": max_token
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    task_id = 1
                    task_type = "text_to_image"
                    task_dependency = 0
                    model = model_1
                    inputs = {
                        "prompt": "Animals recognition from previous task is: +{TASK_0_OUTPUTS}+ Generate a picture with the kinds of animals but in a Monet style.",
                        "image": "",
                        "video": "",
                        "audio": "",
                        "text": "../eval_1020/question_1" + f"_{key_0}_{max_token}_{key_1}_{max_pixel}.jpg",
                        "others": "",
                        "max_output_size": [1, max_pixel, max_pixel]
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    output_path = f"../programs/program_1" + f"_{key_0}_{max_token}_{key_1}_{max_pixel}.json"
                    generate_program(user_request, user_inputs, tasks, output_path)


    ## for question2
    user_request = question2
    user_inputs = {
        "images": "../assets/food.jpg",
        "videos": "",
        "audios": "",
        "texts": "",
        "others": ""
    }
    # 1 task version
    for key_0, model_0 in TEXT_IMAGE_TO_TEXT_MODEL.items():
        if "Florence" in model_0:
            continue
        for max_token in token_range:
            tasks = []

            task_id = 0
            task_type = "text_image_to_text"
            task_dependency = -1
            model = model_0
            inputs = {
                "prompt": user_request,
                "image": user_inputs["images"],
                "video": user_inputs["videos"],
                "audio": user_inputs["audios"],
                "text": user_inputs["texts"],
                "others": user_inputs["others"],
                "max_output_size": max_token
            }
            task = generate_task(task_id, task_type, task_dependency, model, inputs)
            tasks.append(task)

            output_path = f"../programs/program_2" + f"_{key_0}_{max_token}.json"
            generate_program(user_request, user_inputs, tasks, output_path)

    # 2 tasks version
    for key_0, model_0 in TEXT_IMAGE_TO_TEXT_MODEL.items():
        for key_1, model_1 in TEXT_TO_TEXT_MODEL.items():
            for max_token_0 in token_range:
                for max_token_1 in token_range:
                    tasks = []

                    task_id = 0
                    task_type = "text_image_to_text"
                    task_dependency = -1
                    model = model_0
                    inputs = {
                        "prompt": "<DETAILED_CAPTION>" if "Florence" in model_0 else "In this picture, What kind of food is on the table?",
                        "image": user_inputs["images"],
                        "video": user_inputs["videos"],
                        "audio": user_inputs["audios"],
                        "text": user_inputs["texts"],
                        "others": user_inputs["others"],
                        "max_output_size": max_token_0
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    task_id = 1
                    task_type = "text_to_text"
                    task_dependency = 0
                    model = model_1
                    inputs = {
                        "prompt": "The result of food recognition from previous task is as follows. +{TASK_0_OUTPUTS}+ Generate a dinner invitation to Jane to invite her to my home for dinner. You can describe what food is already set for her.",
                        "image": "",
                        "video": "",
                        "audio": "",
                        "text": "",
                        "others": "",
                        "max_output_size": max_token_1
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    output_path = f"../programs/program_2" + f"_{key_0}_{max_token_0}_{key_1}_{max_token_1}.json"
                    generate_program(user_request, user_inputs, tasks, output_path)

    ## for question3
    user_request = question3
    user_inputs = {
        "images": "../assets/famous_people.jpg",
        "videos": "",
        "audios": "",
        "texts": "",
        "others": ""
    }
    # 1 task version
    for key_0, model_0 in TEXT_IMAGE_TO_TEXT_MODEL.items():
        for max_token in token_range:
            tasks = []

            task_id = 0
            task_type = "text_image_to_text"
            task_dependency = -1
            model = model_0
            inputs = {
                "prompt": user_request if "Florence" not in model_0 else "<MORE_DETAILED_CAPTION>",
                "image": user_inputs["images"],
                "video": user_inputs["videos"],
                "audio": user_inputs["audios"],
                "text": user_inputs["texts"],
                "others": user_inputs["others"],
                "max_output_size": max_token
            }
            task = generate_task(task_id, task_type, task_dependency, model, inputs)
            tasks.append(task)

            output_path = f"../programs/program_3" + f"_{key_0}_{max_token}.json"
            generate_program(user_request, user_inputs, tasks, output_path)

    # 2 tasks version
    for key_0, model_0 in TEXT_IMAGE_TO_TEXT_MODEL.items():
        for key_1, model_1 in TEXT_TO_TEXT_MODEL.items():
            for max_token_0 in token_range:
                for max_token_1 in token_range:
                    tasks = []

                    task_id = 0
                    task_type = "text_image_to_text"
                    task_dependency = -1
                    model = model_0
                    inputs = {
                        "prompt": "<MORE_DETAILED_CAPTION>" if "Florence" in model_0 else "Who are the famous people in the picture?",
                        "image": user_inputs["images"],
                        "video": user_inputs["videos"],
                        "audio": user_inputs["audios"],
                        "text": user_inputs["texts"],
                        "others": user_inputs["others"],
                        "max_output_size": max_token_0
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    task_id = 1
                    task_type = "text_to_text"
                    task_dependency = 0
                    model = model_1
                    inputs = {
                        "prompt": "The result of famous people recognition from previous task is as follows. +{TASK_0_OUTPUTS}+ Tell me something interesting or important about them.",
                        "image": "",
                        "video": "",
                        "audio": "",
                        "text": "",
                        "others": "",
                        "max_output_size": max_token_1
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    output_path = f"../programs/program_3" + f"_{key_0}_{max_token_0}_{key_1}_{max_token_1}.json"
                    generate_program(user_request, user_inputs, tasks, output_path)

    ## for question4
    user_request = question4
    user_inputs = {
        "images": "../assets/math_1.jpg",
        "videos": "",
        "audios": "",
        "texts": "",
        "others": ""
    }
    # 1 task version
    for key_0, model_0 in TEXT_IMAGE_TO_TEXT_MODEL.items():
        for max_token in token_range:
            tasks = []

            task_id = 0
            task_type = "text_image_to_text"
            task_dependency = -1
            model = model_0
            inputs = {
                "prompt": user_request if "Florence" not in model_0 else "<MORE_DETAILED_CAPTION>",
                "image": user_inputs["images"],
                "video": user_inputs["videos"],
                "audio": user_inputs["audios"],
                "text": user_inputs["texts"],
                "others": user_inputs["others"],
                "max_output_size": max_token
            }
            task = generate_task(task_id, task_type, task_dependency, model, inputs)
            tasks.append(task)

            output_path = f"../programs/program_4" + f"_{key_0}_{max_token}.json"
            generate_program(user_request, user_inputs, tasks, output_path)

    # 2 tasks version
    for key_0, model_0 in TEXT_IMAGE_TO_TEXT_MODEL.items():
        for key_1, model_1 in TEXT_TO_TEXT_MODEL.items():
            for max_token_0 in token_range:
                for max_token_1 in token_range:
                    tasks = []

                    task_id = 0
                    task_type = "text_image_to_text"
                    task_dependency = -1
                    model = model_0
                    inputs = {
                        "prompt": "<MORE_DETAILED_CAPTION>" if "Florence" in model_0 else "For the question in the input picture, please use markdown to express it.",
                        "image": user_inputs["images"],
                        "video": user_inputs["videos"],
                        "audio": user_inputs["audios"],
                        "text": user_inputs["texts"],
                        "others": user_inputs["others"],
                        "max_output_size": max_token_0
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    task_id = 1
                    task_type = "text_to_text"
                    task_dependency = 0
                    model = model_1
                    inputs = {
                        "prompt": "The result of math problem recognition from previous task is as follows. +{TASK_0_OUTPUTS}+ Give me an answer to the question.",
                        "image": "",
                        "video": "",
                        "audio": "",
                        "text": "",
                        "others": "",
                        "max_output_size": max_token_1
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    output_path = f"../programs/program_4" + f"_{key_0}_{max_token_0}_{key_1}_{max_token_1}.json"
                    generate_program(user_request, user_inputs, tasks, output_path)

    ## for question5
    user_request = question5
    user_inputs = {
        "images": "../assets/places.jpg",
        "videos": "",
        "audios": "",
        "texts": "",
        "others": ""
    }

    # 1 task version
    for key_0, model_0 in TEXT_IMAGE_TO_TEXT_MODEL.items():
        if "Florence" in model_0:
            continue
        for max_token in token_range:
            tasks = []

            task_id = 0
            task_type = "text_image_to_text"
            task_dependency = -1
            model = model_0
            inputs = {
                "prompt": user_request,
                "image": user_inputs["images"],
                "video": user_inputs["videos"],
                "audio": user_inputs["audios"],
                "text": user_inputs["texts"],
                "others": user_inputs["others"],
                "max_output_size": max_token
            }
            task = generate_task(task_id, task_type, task_dependency, model, inputs)
            tasks.append(task)

            output_path = f"../programs/program_5" + f"_{key_0}_{max_token}.json"
            generate_program(user_request, user_inputs, tasks, output_path)

    # 2 tasks version
    for key_0, model_0 in TEXT_IMAGE_TO_TEXT_MODEL.items():
        for key_1, model_1 in TEXT_TO_TEXT_MODEL.items():
            for max_token_0 in token_range:
                for max_token_1 in token_range:
                    tasks = []

                    task_id = 0
                    task_type = "text_image_to_text"
                    task_dependency = -1
                    model = model_0
                    inputs = {
                        "prompt": "Where is this place in the picture?" if "Florence" not in model_0 else "<MORE_DETAILED_CAPTION>",
                        "image": user_inputs["images"],
                        "video": user_inputs["videos"],
                        "audio": user_inputs["audios"],
                        "text": user_inputs["texts"],
                        "others": user_inputs["others"],
                        "max_output_size": max_token_0
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    task_id = 1
                    task_type = "text_to_text"
                    task_dependency = 0
                    model = model_1
                    inputs = {
                        "prompt": "The result of place recognition from previous task is as follows. +{TASK_0_OUTPUTS}+ Generate some traveling advice if I go there sometime.",
                        "image": "",
                        "video": "",
                        "audio": "",
                        "text": "",
                        "others": "",
                        "max_output_size": max_token_1
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    output_path = f"../programs/program_5" + f"_{key_0}_{max_token_0}_{key_1}_{max_token_1}.json"
                    generate_program(user_request, user_inputs, tasks, output_path)

    ## for question6
    user_request = question6
    user_inputs = {
        "images": "",
        "videos": "",
        "audios": "",
        "texts": "",
        "others": ""
    }
    # 1 task version
    for key_0, model_0 in TEXT_TO_IMAGE_MODEL.items():
        for max_pixel in pixel_range:
            tasks = []

            task_id = 0
            task_type = "text_to_image"
            task_dependency = -1
            model = model_0
            inputs = {
                "prompt": "a hero is taking an adventure in the forest and encountering a dragon. Generate a picture based on this.",
                "image": "",
                "video": user_inputs["videos"],
                "audio": user_inputs["audios"],
                "text": "../eval_1020/question_6" + f"_{key_0}_{max_pixel}.jpg",
                "others": user_inputs["others"],
                "max_output_size": [1, max_pixel, max_pixel]
            }
            task = generate_task(task_id, task_type, task_dependency, model, inputs)
            tasks.append(task)

            output_path = f"../programs/program_6" + f"_{key_0}_{max_pixel}.json"
            generate_program(user_request, user_inputs, tasks, output_path)

    # 2 tasks version
    for key_0, model_0 in TEXT_TO_TEXT_MODEL.items():
        for key_1, model_1 in TEXT_TO_IMAGE_MODEL.items():
            for max_token_0 in token_range:
                for max_pixel in pixel_range:
                    tasks = []

                    task_id = 0
                    task_type = "text_to_text"
                    task_dependency = -1
                    model = model_0
                    inputs = {
                        "prompt": "continue: a hero is taking an adventure in the forest and encountering a dragon. then ...",
                        "image": "",
                        "video": "",
                        "audio": "",
                        "text": "",
                        "others": "",
                        "max_output_size": max_token_0
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    task_id = 1
                    task_type = "text_to_image"
                    task_dependency = 0
                    model = model_1
                    inputs = {
                        "prompt": "Story is as follows. +{TASK_0_OUTPUTS}+ Generate a picture based on this.",
                        "image": "",
                        "video": "",
                        "audio": "",
                        "text": "../eval_1020/question_6" + f"_{key_0}_{max_token_0}_{key_1}_{max_pixel}.jpg",
                        "others": "",
                        "max_output_size": [1, max_pixel, max_pixel]
                    }
                    task = generate_task(task_id, task_type, task_dependency, model, inputs)
                    tasks.append(task)

                    output_path = f"../programs/program_6" + f"_{key_0}_{max_token_0}_{key_1}_{max_pixel}.json"
                    generate_program(user_request, user_inputs, tasks, output_path)


