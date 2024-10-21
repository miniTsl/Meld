from time import sleep

from task import *
import json
import os
import torch

__TASKS__ = {
    "text_image_to_text": TextImageToTextTask,
    "text_to_image": TextToImageTask,
    "text_to_text": TextToTextTask
}

class Interpreter:
    def __init__(self, program:str):
        self.program = json.load(open(program, "r"))
        self.user_request = self.program["User Request"]
        self.inputs = self.program["Inputs"]
        self.tasks_dict = self.program["Tasks"]
        self.tasks = self.load_tasks()
        self.tasks_output_pool = {}
        self.tasks_latency = {} # s
        self.tasks_memory = {}  # GB

    def load_tasks(self):
        task_objects = []
        for _, task_config in self.tasks_dict.items():
            task_type = task_config["task_type"]
            task = __TASKS__[task_type](task_config)
            task_objects.append(task)
        return task_objects

    def check_dependency(self, task):
        # check dependency
        # TODO: now we only support up to single dependency
        task_dependency = task.task_dependency
        if task_dependency == -1:
            pass
        else:
            dependent_data = self.tasks_output_pool["task_" + str(task_dependency)]
            task.set_dependent_data(dependent_data)

    def execute(self):
        for task in self.tasks:
            self.check_dependency(task)
            task.execute()
            self.tasks_output_pool["task_" + str(task.task_id)] = task.outputs
            self.tasks_latency["task_" + str(task.task_id)] = task.executor.get_latency()
            self.tasks_memory["task_" + str(task.task_id)] = task.executor.get_memory()
            task.release_executor() # release GPU memory

    def __str__(self):
        return f"User Request: {self.user_request}\nInputs: {self.inputs}\nTasks: {self.tasks}"

    def __repr__(self):
        return f"User Request: {self.user_request}\nInputs: {self.inputs}\nTasks: {self.tasks}"



if __name__ == "__main__":
    # run one program
    for _ in range(10):
        PROGRAM_FOLDER = "../programs/"
        interpreter = Interpreter(PROGRAM_FOLDER + "program_2_Qwen2-VL-2B-Instruct_500_Qwen2.5-1.5B-Instruct_500.json")
        interpreter.execute()
        print(interpreter.tasks_output_pool)
        print(interpreter.tasks_latency)
        print(interpreter.tasks_memory)

    del interpreter
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # run multiple programs
    PROGRAM_FOLDER = "../programs/"
    for filename in os.listdir(PROGRAM_FOLDER):
        if filename.endswith(".json"):
            print(f"Start evaluating {filename}")
            file_path = os.path.join(PROGRAM_FOLDER, filename)
            interpreter = Interpreter(file_path)
            interpreter.execute()

            # read the json file
            with open(file_path, "r") as f:
                program = json.load(f)
            # first add new item "eval" into each task in the program
            # write the latency, memory and output into each task in the program and save the new program into a new json
            for task_id, task_config in program["Tasks"].items():
                task_config["eval"] = {}
                task_config["eval"]["latency"] = interpreter.tasks_latency[task_id]
                task_config["eval"]["memory"] = interpreter.tasks_memory[task_id]
                task_config["eval"]["output_content"] = interpreter.tasks_output_pool[task_id]

            save_path = os.path.join("../eval_1020/", filename)
            with open(save_path, "w") as f:
                json.dump(program, f, indent=4)

            print(f"Finish evaluating {filename}")
            del interpreter
            torch.cuda.empty_cache()
            torch.cuda.synchronize()