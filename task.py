from executors import *
import torch

TEXT_IMAGE_TO_TEXT_EXECUTOR = {
    "Qwen/Qwen2-VL-2B-Instruct": Qwen2VLExecutor,
    "Qwen/Qwen2-VL-7B-Instruct-AWQ": Qwen2VLExecutor,
    "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4": Qwen2VLExecutor,
    "microsoft/Florence-2-base-ft": FlorenceExecutor,
    "microsoft/Florence-2-large-ft": FlorenceExecutor
}

TEXT_TO_TEXT_EXECUTOR  = {
    "Qwen/Qwen2.5-0.5B-Instruct": Qwen2Executor,
    "Qwen/Qwen2.5-1.5B-Instruct": Qwen2Executor,
    "Qwen/Qwen2.5-3B-Instruct": Qwen2Executor,
    "Qwen/Qwen2.5-7B-Instruct": Qwen2Executor
}

TEXT_TO_IMAGE_EXECUTOR = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": StableDiffusionExecutor,
    "stabilityai/stable-diffusion-xl-base-1.0": StableDiffusionExecutor
}

class BaseTask:
    def __init__(self, task_config:dict):
        self.task_type = task_config["task_type"]
        self.task_id = task_config["task_id"]
        self.task_dependency = task_config["task_dependency"]
        self.model = task_config["model"]
        self.inputs = task_config["inputs"]
        self.outputs_name = task_config["outputs"]
        self.outputs = None
        self.executor = None

    def load_executor(self):
        pass

    def execute(self):
        self.load_executor()
        self.outputs = self.executor.generate_output(self.inputs)

    def __str__(self):
        return f"Task Type: {self.task_type}\nTask ID: {self.task_id}\nTask Dependency: {self.task_dependency}\nModel: {self.model}\nPrompt: {self.prompt}\nImage File Path: {self.image_file_path}\nVideo File Path: {self.video_file_path}\nAudio File Path: {self.audio_file_path}\nText File Path: {self.text_file_path}\nOthers: {self.others}\nOutputs: {self.outputs}"

    def __repr__(self):
        return f"Task Type: {self.task_type}\nTask ID: {self.task_id}\nTask Dependency: {self.task_dependency}\nModel: {self.model}\nPrompt: {self.prompt}\nImage File Path: {self.image_file_path}\nVideo File Path: {self.video_file_path}\nAudio File Path: {self.audio_file_path}\nText File Path: {self.text_file_path}\nOthers: {self.others}\nOutputs: {self.outputs}"

    def set_dependent_data(self, dependent_data):
        # TODO: now we only support up to single dependency
        mark = "{" + "TASK_" + str(self.task_dependency) + "_OUTPUTS" + "}"
        if mark in self.inputs["prompt"]:
            self.inputs["prompt"] = self.inputs["prompt"].replace(mark, "\n" + str(dependent_data) + "\n")

    def release_executor(self):
        self.executor = None

class TextImageToTextTask(BaseTask):
    def load_executor(self):
        self.executor = TEXT_IMAGE_TO_TEXT_EXECUTOR[self.model](self.model)

    def set_dependent_data(self, dependent_data):
        # prompt and image can be dependent with previous tasks
        mark = "{" + "TASK_" + str(self.task_dependency) + "_OUTPUTS" + "}"
        if mark in self.inputs["prompt"]:
            self.inputs["prompt"] = self.inputs["prompt"].replace(mark, "\n" + str(dependent_data) + "\n")
        if mark in self.inputs["image"]:
            self.inputs["image"] = dependent_data


class TextToTextTask(BaseTask):
    def load_executor(self):
        self.executor = TEXT_TO_TEXT_EXECUTOR[self.model](self.model)


class TextToImageTask(BaseTask):
    def load_executor(self):
        self.executor = TEXT_TO_IMAGE_EXECUTOR[self.model](self.model)
