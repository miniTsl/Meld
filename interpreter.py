from time import sleep

from task import *
import json
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

__TASKS__ = {
    "image_text_to_text": ImageTextToTextTask,
    "text_to_image": TextToImageTask,
    "text_to_text": TextToTextTask
}

class Interpreter:
    def __init__(self, program:str):
        self.program = json.load(open(program, "r"))
        self.user_request = self.program["user_request"]
        self.user_inputs = self.program["user_inputs"]
        self.tasks = self.program["tasks"]
        self.tasks_output_pool = {}
        self.tasks_latency = {} # s
        self.tasks_model_memory = {}  # GB
        self.tasks_gpu_memory = {}  # GB

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
        for task_config in self.tasks:
            index = "task_" + str(task_config["id"])
            task_type = task_config["type"]
            task = __TASKS__[task_type](task_config)
            
            # clear GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            try:    
                self.check_dependency(task)
                task.execute()
            
                max_memory = torch.cuda.max_memory_allocated() / 1024**3 # GB
                self.tasks_output_pool[index] = task.outputs
                self.tasks_latency[index] = task.executor.get_latency()
                self.tasks_model_memory[index] = task.executor.get_memory()
                self.tasks_gpu_memory[index] = max_memory
            except Exception as e:
                print(f"Error in task {index}: {e}")
                self.tasks_output_pool[index] = "Error: " + str(e)
                self.tasks_latency[index] = 0
                self.tasks_model_memory[index] = 0
                self.tasks_gpu_memory[index] = 0
                continue
            
            del task
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def save_evaluated_program(self, save_dir:str):
        for task_config in self.program["tasks"]:
            task_id = "task_" + str(task_config["id"])
            task_config["eval"] = {}
            task_config["eval"]["latency"] = self.tasks_latency[task_id]
            task_config["eval"]["model memory"] = self.tasks_model_memory[task_id]
            task_config["eval"]["gpu memory"] = self.tasks_gpu_memory[task_id]
            task_config["eval"]["output_content"] = self.tasks_output_pool[task_id]
            
        with open(save_dir, "w") as f:
            json.dump(self.program, f, indent=4)



if __name__ == "__main__":
    # run one program to warm up
    for _ in range(10):
        PROGRAM_FOLDER = "/home/sunyi/Meld/programs/problem23/"
        interpreter = Interpreter(PROGRAM_FOLDER + "program_1_v1.json")
        interpreter.execute()
        print(interpreter.tasks_output_pool)
        print(interpreter.tasks_latency)
        print(interpreter.tasks_model_memory)
        print(interpreter.tasks_gpu_memory)
        del interpreter
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # run multiple programs
    PROGRAM_FOLDER = "/home/sunyi/Meld/programs/problem23/"
    for filename in os.listdir(PROGRAM_FOLDER):
        if filename.endswith(".json"):
            print(f"Start evaluating {filename}")
            file_path = os.path.join(PROGRAM_FOLDER, filename)
            interpreter = Interpreter(file_path)
            interpreter.execute()
            save_dir = PROGRAM_FOLDER + "/eval/" + filename
            interpreter.save_evaluated_program(save_dir)
            print(f"Finish evaluating {filename}")
            del interpreter
            torch.cuda.empty_cache()
            torch.cuda.synchronize()