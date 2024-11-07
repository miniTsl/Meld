from executors import *
import torch

class BaseTask:
    def __init__(self, task_config:dict):
        self.task_type = task_config["type"]
        self.task_id = task_config["id"]
        self.task_dependency = task_config["dependency"]
        self.model_name = task_config["model"]
        self.model_quant = task_config["quant"]
        self.prompt = task_config["prompt"]
        self.task_inputs = task_config["inputs"]
        self.generate_limit = task_config["generate_limit"]
        self.outputs_name = task_config["outputs"]
        self.outputs = None
        self.executor = None
        self.load_executor()
        
    def load_executor(self):
        pass

    def set_dependent_data(self, dependent_data):
        # TODO: now we only support up to single dependency
        mark = "{" + "Task_" + str(self.task_dependency) + "_Outputs" + "}"
        if mark in self.prompt:
            self.prompt = self.prompt.replace(mark, "\n" + str(dependent_data) + "\n")
    
    def set_media_inputs(self):
        inputs = {"prompt": self.prompt}
        return inputs
                
    def execute(self):
        inputs = self.set_media_inputs()
        self.outputs = self.executor.generate_output(inputs, self.generate_limit)

class ImageTextToTextTask(BaseTask):
    def __init__(self, task_config):
        super().__init__(task_config)
        
    def load_executor(self):
        self.executor = IMAGE_TEXT_TO_TEXT_EXECUTORS[self.model_name](self.model_name, self.model_quant)

    def set_dependent_data(self, dependent_data):
        # prompt and image can be dependent with previous tasks
        mark = "{" + "Task_" + str(self.task_dependency) + "_Outputs" + "}"
        if mark in self.prompt:
            self.prompt = self.prompt.replace(mark, "\n" + str(dependent_data) + "\n")
        if mark in self.task_inputs["image"]:
            self.task_inputs["image"] = dependent_data
    

class TextToTextTask(BaseTask):
    def __init__(self, task_config):
        super().__init__(task_config)
        
    def load_executor(self):
        self.executor = TEXT_TO_TEXT_EXECUTORS[self.model_name](self.model_name, self.model_quant)
    
    def set_media_inputs(self):
        if self.task_inputs != None:
            text_to_text_inputs = {
                "prompt": self.prompt,
                "text": self.task_inputs["text"]
            }
        else:
            text_to_text_inputs = {"prompt": self.prompt}
        return text_to_text_inputs
    
class TextToImageTask(BaseTask):
    def __init__(self, task_config):
        super().__init__(task_config)
        
    def load_executor(self):
        self.executor = TEXT_TO_IMAGE_EXECUTORS[self.model_name](self.model_name, self.model_quant)