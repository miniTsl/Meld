import json
import os

TASK_FORMAT = """
{{
    "task_type": {task_type},
    "task_id": {task_id},
    "task_dependency": {task_dependency},
    "model": {model},
    "inputs": {{
        "prompt": {prompt},
        "image": {image_file_path},
        "video": {video_file_path},
        "audio": {audio_file_path},
        "text": {text_file_path},
        "others": {others}
    }}
    "outputs": {outputs}
}}
"""

PROGRAM_FORMAT = """
{{
    "User request": {user_request},
    "Inputs": {{
        "images": {image_file_path},
        "videos": {video_file_path},
        "audio": {audio_file_path}
    }}
    "Tasks": {{
        {task_list}
    }}
}}
"""

# TODO: how to describe the models, which will be chosen and combined in the program
MODEL_DESCRIPTION_FORMAT = """
{{
    "model_name": {model_name},
    "description": {description}
}}
"""


Prompt = """
System prompt: You are a PROGRAM generator to solve some complex real-world problems. 
For each problem, user shall give a user-request and some inputs(maybe some images, videos, etc). 
The PROGRAM consists of several TASKs, each TASK follows the json format below. 
You should think step by step and generate a PROGRAM according to the following json format, and there are several DEMONSTRATIONs after that, which you could refer to.

TASK FORMAT:
```json
{{
    "task_type": {task_type},
    "task_id": {task_id},
    "task_dependency": {task_dependency},
    "model": {model},
    "inputs": {
        "prompt": {prompt},
        "image": {image_file_path},
        "video": {video_file_path},
        "audio": {audio_file_path},
        "text": {text_file_path},
        "others": {others}
    }
    "outputs": {outputs}
}}
```

PROGRAM FORMAT:
```json
{{
    "User request": {user_request},
    "Inputs": {{
        "images": {image_file_path},
        "videos": {video_file_path},
        "audio": {audio_file_path}
    }}
    "Tasks": {{
        "task_0": the first task follows the TASK FORMAT,
        "task_1": the second task follows the TASK FORMAT,
        ...
    }}
}}
```

DEMONSTRATIONs:

Program_0:
```json
    "User request": "Tell me what animals are in the picture and generate another picture with these kinds of animals but in a modern art style."
    Inputs: {
        "images": "../assets/lion_walking.jpg"
        }
    "Tasks": {
        "task_0": {
            "task_type": "text_image_to_text",
            "task_id": 0,
            "task_dependency": -1,
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "inputs": {
                "prompt": "What kind of animals are in the picture?",
                "image": Inputs["images"]
            }
            "outputs": TASK_0_OUTPUTS
        },
        "task_1": {
            "task_type": "text_to_image",
            "task_id": 1,
            "task_dependency": 0,
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "inputs": {
                "prompt": "The result of animal kind recognition from previous task is as follows."
                            + TASK_0_OUTPUTS 
                            + "Generate a picture with the kinds of animals but in a modern art style."
            }
            "outputs": TASK_1_OUTPUTS
        }
    }
```

Program_1:
...

"""