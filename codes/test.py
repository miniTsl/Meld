# TODO: for now we just consider single image input, we will consider multiple images later.

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




image = "../assets/car.jpg"

    
# # test qwen2VL
# from models import Qwen2VLExecutor
# tmp = Qwen2VLExecutor("Qwen/Qwen2-VL-2B-Instruct")
# prompt = "detect and locate objects in this picture, give me the labels and bboxes"
# image = "../assets/lion_walking.jpg"
# answer = tmp.generate_output(prompt, image)
# print(answer)
# print(tmp.get_latency())
# print(tmp.get_memory())
# print(tmp.get_pixels_processed())
# print(tmp.get_tokens_processed())
# test qwen2VL token limit
from models import Qwen2VLExecutor
tmp = Qwen2VLExecutor("Qwen/Qwen2-VL-2B-Instruct")
prompt = "What is the animal in the picture and what is it doing?"
image = "../assets/lion_walking.jpg"
inputs = {
    "prompt": prompt,
    "image": image
}
answer = tmp.generate_output(inputs)
print(answer)
print(tmp.get_latency())
print(tmp.get_memory())
print(tmp.get_pixels_processed())
print(tmp.get_tokens_processed())


# # test florence-2
# from interpreters import FlorenceInterpreter
# model_name = "microsoft/Florence-2-base-ft"
# tmp = FlorenceInterpreter(model_name)
# import matplotlib.pyplot as plt  
# import matplotlib.patches as patches  
# def plot_bbox(image, data):
#    # Create a figure and axes  
#     fig, ax = plt.subplots()  
      
#     # Display the image(file str)
#     img = plt.imread(image)
#     ax.imshow(img)
      
#     # Plot each bounding box  
#     for bbox, label in zip(data['bboxes'], data['labels']):  
#         # Unpack the bounding box coordinates  
#         x1, y1, x2, y2 = bbox  
#         # Create a Rectangle patch  
#         rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
#         # Add the rectangle to the Axes  
#         ax.add_patch(rect)  
#         # Annotate the label  
#         plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
#     # Remove the axis ticks and labels  
#     ax.axis('off')  
      
#     # Show the plot  
#     plt.savefig('florence_test.jpg')

# # Caption
# task_prompt = '<CAPTION>'
# answer = tmp.generate_output(task_prompt, image)
# print(answer)
# task_prompt = '<DETAILED_CAPTION>'
# answer = tmp.generate_output(task_prompt, image)
# print(answer)
# task_prompt = '<MORE_DETAILED_CAPTION>'
# answer = tmp.generate_output(task_prompt, image)
# print(answer)

# # Object detection
# task_prompt = '<OD>'
# answer = tmp.generate_output(task_prompt, image)
# print(answer)
# plot_bbox(image, answer['<OD>'])

# task_prompt = "<DENSE_REGION_CAPTION>"
# answer = tmp.generate_output(task_prompt, image)
# print(answer)
# plot_bbox(image, answer['<DENSE_REGION_CAPTION>'])

# # ocr
# image = "../assets/cuda.jpg"
# task_prompt = '<OCR>'
# answer = tmp.generate_output(task_prompt, image)
# print(answer)
# task_prompt = '<OCR_WITH_REGION>'
# answer = tmp.generate_output(task_prompt, image)
# print(answer)

# # More Detailed Caption + Phrase Grounding 
# task_prompt = '<MORE_DETAILED_CAPTION>'
# results = tmp.generate_output(task_prompt, image)
# text_input = results[task_prompt]
# task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
# results = tmp.generate_output(task_prompt, image, text_input)
# results['<MORE_DETAILED_CAPTION>'] = text_input
# print(results)
# plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])

# # test qwen2.5
# from interpreters import Qwen2Interpreter
# model_name = "Qwen/Qwen2.5-3B-Instruct"
# tmp = Qwen2Interpreter(model_name)
# prompt = "Tell me something about Lebron James"
# results = tmp.generate_output(prompt)
# print(results)
# print("Inference latency is： ", tmp.get_latency(), "s")
# print("Model memory footprint is： ", tmp.get_memory(), "GB")

# # test stable diffusion
# from interpreters import StableDiffusionInterpreter
# model_name = "sd-legacy/stable-diffusion-v1-5"
# tmp = StableDiffusionInterpreter(model_name)
# prompt = "Generate a picture of future life in morden city. The technology is advanced and people are living a happy life."
# image = tmp.generate_output(prompt)
# # image is a PIL.Image.Image object, save this picture
# image.save("generated_image.jpg")
# print("Inference latency is： ", tmp.get_latency(), "s")
# print("Model memory footprint is： ", tmp.get_memory(), "GB")