from .base import BaseExecutor
from .image_text_to_text import *
from .text_to_image import *
from .text_to_text import *
from .image_text_to_image import *

IMAGE_TEXT_TO_TEXT_EXECUTORS = {
    "Qwen/Qwen2-VL-2B-Instruct": Qwen2VLExecutor,
    "Qwen/Qwen2-VL-7B-Instruct": Qwen2VLExecutor,
    "Qwen/Qwen2-VL-7B-Instruct-AWQ": Qwen2VLExecutor,
    "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4": Qwen2VLExecutor,
    "microsoft/Florence-2-base-ft": FlorenceExecutor,
    "microsoft/Florence-2-large-ft": FlorenceExecutor
}

TEXT_TO_TEXT_EXECUTORS = {
    "Qwen/Qwen2.5-0.5B-Instruct": Qwen2Executor,
    "Qwen/Qwen2.5-1.5B-Instruct": Qwen2Executor,
    "Qwen/Qwen2.5-3B-Instruct": Qwen2Executor,
    "Qwen/Qwen2.5-7B-Instruct": Qwen2Executor

}

TEXT_TO_IMAGE_EXECUTORS = {

    "stable-diffusion-v1-5/stable-diffusion-v1-5": StableDiffusionExecutor,
    "stabilityai/stable-diffusion-xl-base-1.0": StableDiffusionExecutor
}

IMAGE_TEXT_TO_IMAGE_EXECUTORS = {

}