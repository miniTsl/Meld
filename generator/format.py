import json
import os

Task_Format = """
{{
    "task_type": {task_type},
    "task_id": {task_id},
    "task_dependency": {task_dependency},
    "model_name": {model_name},
    "model_quant": {model_quant},
    "prompt": {prompt},
    "media_inputs": {{
        "image": {image_file_path},
        "video": {video_file_path},
        "audio": {audio_file_path},
        "text": {text_file_path},
        "others": {others}
    }},
    "generate_limit": {generate_limit},
    "outputs": {outputs}
}}
"""

Program_Format = """
{{
    "user_request": {user_request},
    "user_inputs": {{
        "images": {image_file_path},
        "videos": {video_file_path},
        "audio": {audio_file_path},
        "text": {text_file_path},
        "others": {others}
    }}
    "tasks": {{
        {task_list}
    }}
}}
"""

# TODO: how to describe the models, which will be chosen and combined in the program
Model_Description_Format = """
{{
    "model_name": {model_name},
    "description": {description}
}}
"""


Prompt = """
System Prompt: 
You are a Program generator for solving given complex real-world problems. The Program you generated should be composed of a list of Tasks.
For the given problem, the User shall give a user_request and some user_inputs(maybe paths to some images, videos, audios etc).
You should analyse the problem and disassemble it step by step to use combinations of Tasks to solve it. The prompt and needed task_inputs of each Task should be generated through your analysis.
The formats and descriptions of Task, Program and corresponding MediaInputs are given afterwords.

MediaInputs Format:
```json
{{
    "image": image_file_path (optional),
    "video": video_file_path (optional),
    "audio": audio_file_path (optional),
    "text": text_file_path (optional),
    "others": others (optional)
}}
```


Task Format:
```json
{{
    "type": task type,
    "id": task id inside the program,
    "dependency": ids of dependent tasks inside the program, -1 if no dependency,
    "model": model name,
    "quant": quantization precision of the model, such as "4bit" and "8bit", default is "original",
    "prompt": generated prompt for this task,
    "inputs": media files needed for this task in MediaInputs Format (optional),
    "generate_limit": "max_new_tokens" OR "num_inference_steps". 
    "outputs": outputs
}}
```

Instructions for Task generation:
1. "type", "model" and "quant": for each task you should fisrt decide the type of the task. Then you MUST choose one model and quant_option of THAT chosen type from a dict of candidate models. For example, if the task is to generate a text from an image, the type should be image_text_to_text, and the model should be only be chosen from the part of image_text_to_text models. The model_quant can be chosen from options of the model, such as "original", "4bit" and "8bit". The model dict is {model_candidates}.

2. "id" and "dependency": for each task you should generate a unique proper integer "id", and an integer "dependency" (for now we only consider single dependency), which is the "id" that this task depends on. The "dependency" is -1 if this task does not depend on any data from previous tasks.

3. "prompt": should be generated considering both the aim of this task amid the whole program and the data dependencies from previous tasks inside the program. For example, if the task is to just generate a text from an image, the prompt should be like "Describe the image in the picture.". If the task is to generate an audio from text description derived from the image given, then the prompt should be like "Generate an audio from the text. The text is generated by previous image understanding task with the output:{{Task_X_Outputs}}". The previous_task_outputs inside the prompt MUST be in {{Task_x_Outputs}} format, where x is the task_id of the previous task.

4. "inputs": for each task you should generate proper media inputs, considering the data dependencies from previous task. The "image", "video", "audio" and "text" are paths to the files (given by the User or from previous task). The "text" is some additional input text besides "prompt". The "others" are the other inputs needed by the model. If the task does not need any media input, you can leave this item empty.
    
5. The "generate_limit" is an integer for the max_new_tokens OR num_inference_steps, which are the maximum number of tokens generated by a text model or the number of inference steps for an image generaton model respectively. Feel free to randomly choose the value from ranges: max_new_tokens: range(100, 2000, 100), num_inference_steps: range(5, 100, 5).
    
6. "outputs": should be the name of generated data from the model, which MUST be in the form Task_x_Outputs, where x is the task_id of this task. 


Program Format:
```json
{{
    "user_request": request from the User
    "user_inputs": media files provided by the User in MediaInputs Format,
    "tasks": {{
        "task_0": the first task in Task Format,
        "task_1": the second task in Task Format,
        ...
        "task_n": the last task in Task Format
    }}
}}
```


Here are several Demonstrations which you could refer to, but MUSTN'T follow the values and model selection inside strictly. You MUST freely choose model and generate_limit, and you MUST feel free to generate prompts for tasks as variously as possible.
{demonstrations}


For the following user_request and user_inputs, please generate a Program list composed of 10 programs. Each program should be independent and can slove the problem alone. Which means you should try to repeat solving this problem for 10 times. Want to see the variety of the generated programs.
user_request: {user_request}
user_inputs: {user_inputs} 

However, there are also something you MUST notice while generating the Program:
1. The same problem may have different ways to complete. Sometimes you can use one model to do two things if you think that's feasible and reasonable. And soemtimes you may be able to change the order of tasks to make the program more various. 
2. Although you can freely choose each model's generate limit, you should also notice the complexity of different tasks. For example, summarizing a news may need at least 500 tokens, so you souldn't set the generate limit too small. Another example is that the task of generating a longer story may need more generation token limit than the task of summarizing a news. So you should choose the generate limit properly.
3. Each task may have data dependencies from previous tasks and may have extra media inputs besides prompt, and they will be dealt with somehow in later stages. For example, as for text dependency, the {{Task_x_Outputs}} in the prompt will be directly replaced with the output content from Task x. As for media inputs, I(not you) shall append something like "The text/image/video/audio material is: " and then the file content at the end of prompt. So for each task's prompt generation, you should be aware of the potential of augmentation and replacement. And you MUST consider the organization of words in the prompt so that the model can tell easily which part inside the prompt is task requirements and which parts are dependency or expended materials. Different parts MUST be declared clearly.
4. Basically, the execution of different tasks is totally independent (after data dependency is solved). So when generating prompts for each task, your primacy is to let the model know what it should do, what it should use and what it should output. There is no need to mention other tasks in the prompt. But you can add some explanation if needed to clarify the task requirements more clearly.

If you find it impossible to solve this problem, just return an empty list.

"""


from pydantic import BaseModel
from typing import Optional
# TODO: now we just support single file (as a path string) for each media type
class MediaInputs(BaseModel):
    image: Optional[str]
    video: Optional[str]
    audio: Optional[str]
    text: Optional[str]
    others: Optional[str]

class Task(BaseModel):
    type: str
    id: int
    dependency: int
    model: str
    quant: str
    prompt: str
    inputs: Optional[MediaInputs]
    generate_limit: list[int]
    outputs: str
    
class Program(BaseModel):
    user_request: str
    user_inputs: Optional[MediaInputs]
    tasks: list[Task]

class ProgramList(BaseModel):
    programs: list[Program]