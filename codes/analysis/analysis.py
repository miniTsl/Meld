import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ## get mean, std and maybe distribution of latency and memory for each question
# eval_dir = "../../eval_1020/"
# prefix = "program_"
# for index in range(1, 7):
#     count = 0
#     latency = []
#     memory = []
#     # find all programs for this question
#     for file in os.listdir(eval_dir):
#         total_latency = 0
#         total_memory = 0
#         if file.startswith(prefix + str(index)):
#             count += 1
#             with open(eval_dir + file, "r") as f:
#                 program = json.load(f)
#             for task_id, task_config in program["Tasks"].items():
#                 total_latency += task_config["eval"]["latency"]
#                 total_memory += task_config["eval"]["memory"]
#             latency.append(total_latency)
#             memory.append(total_memory)
#     print(f"Program_{index}:")
#     print(f"Total programs: {count}")
#     # round to 0.001, print mean, std, std/mean(in %), max, min
#     print(f"Latency(s): mean: {np.mean(latency):.3f}, std: {np.std(latency):.3f}, std/mean: {np.std(latency)/np.mean(latency)*100:.3f}%, max: {np.max(latency):.3f}, min: {np.min(latency):.3f}")
#     print(f"Memory(GB): mean: {np.mean(memory):.3f}, std: {np.std(memory):.3f}, std/mean: {np.std(memory)/np.mean(memory)*100:.3f}%, max: {np.max(memory):.3f}, min: {np.min(memory):.3f}")
#     print()
#
#     # plot and save cdf of latency and memory
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#     ax[0].hist(latency, bins=10, cumulative=True, density=True, color='skyblue', edgecolor='black')
#     ax[0].set_title(f"Latency CDF for Program_{index}")
#     ax[0].set_xlabel("Latency(s)")
#     ax[0].set_ylabel("CDF")
#     ax[1].hist(memory, bins=10, cumulative=True, density=True, color='skyblue', edgecolor='black')
#     ax[1].set_title(f"Memory CDF for Program_{index}")
#     ax[1].set_xlabel("Memory(GB)")
#     ax[1].set_ylabel("CDF")
#     plt.savefig(f"program_{index}_cdf.png")


TEXT_IMAGE_TO_TEXT_MODEL = [
    "Qwen2-VL-2B-Instruct",
    # "Qwen2-VL-7B-Instruct-AWQ",
    # "Qwen2-VL-7B-Instruct-GPTQ-Int4",
    "Florence-2-base-ft",
    "Florence-2-large-ft"
]

TEXT_TO_TEXT_MODEL  = [
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    # "Qwen2.5-3B-Instruct",
    # "Qwen2.5-7B-Instruct"
]

TEXT_TO_IMAGE_MODEL = [
    "stable-diffusion-v1-5",
    "stable-diffusion-xl-base-1.0"
]



eval_dir = "../../eval_1020/"

# # for question_1, programs contain 2 kinds of models: text-image-to-text and text-to-image, so there are 3*2=6 combinations
# # for each combination, there are variants with different max_new_tokens and max_new_pixels, and we want to plot the latency and memory for each combination in heatmap
# prefix = "program_1_"
# # get all programs for question_1
# programs = []
# for file in os.listdir(eval_dir):
#     if file.startswith(prefix):
#         with open(eval_dir + file, "r") as f:
#             program = json.load(f)
#         programs.append(program)
# # plot a 3*2 subplots and each subplot is a heatmap of latency or memory
# fig, ax = plt.subplots(3, 2, figsize=(12, 18))
# position = 0
# for text_image_to_text in TEXT_IMAGE_TO_TEXT_MODEL:
#     for text_to_image in TEXT_TO_IMAGE_MODEL:
#         latency = []
#         memory = []
#         max_new_tokens = []
#         max_new_pixels = []
#         for program in programs:
#             if  text_image_to_text in program["Tasks"]["task_0"]["model"] and text_to_image in program["Tasks"]["task_1"]["model"]:
#                 max_new_tokens.append(program["Tasks"]["task_0"]["inputs"]["max_output_size"])
#                 max_new_pixels.append(program["Tasks"]["task_1"]["inputs"]["max_output_size"][1])
#                 latency.append(program["Tasks"]["task_0"]["eval"]["latency"] + program["Tasks"]["task_1"]["eval"]["latency"])
#                 memory.append(program["Tasks"]["task_0"]["eval"]["memory"] + program["Tasks"]["task_1"]["eval"]["memory"])
#         # reorganize latency and memory to 2D array according to max_new_tokens and max_new_pixels
#         # first find all unique max_new_tokens and max_new_pixels
#         unique_max_new_tokens = list(set(max_new_tokens))
#         unique_max_new_pixels = list(set(max_new_pixels))
#         unique_max_new_tokens.sort()
#         unique_max_new_pixels.sort()
#         # create 2D array
#         latency_2d = np.zeros((len(unique_max_new_tokens), len(unique_max_new_pixels)))
#         memory_2d = np.zeros((len(unique_max_new_tokens), len(unique_max_new_pixels)))
#         for i in range(len(max_new_tokens)):
#             row = unique_max_new_tokens.index(max_new_tokens[i])
#             col = unique_max_new_pixels.index(max_new_pixels[i])
#             latency_2d[row][col] = latency[i]
#             memory_2d[row][col] = memory[i]
#         # plot heatmap
#         sns.heatmap(latency_2d, ax=ax[position//2][position%2], annot=True, fmt=".3f", cmap="YlGnBu")
#         ax[position//2][position%2].set_title(f"{text_image_to_text} and {text_to_image}")
#         ax[position//2][position%2].set_xticklabels(unique_max_new_pixels)
#         ax[position//2][position%2].set_yticklabels(unique_max_new_tokens)
#         ax[position//2][position%2].set_xlabel("max_new_pixels")
#         ax[position//2][position%2].set_ylabel("max_new_tokens")
#         position += 1
# plt.savefig("question_1_latency_heatmap.png")


# # for question_2 and question_5, programs may contain only 1 kind of model: text-image-to-text, and there is only 1 choice: "Qwen2-VL-2B-Instruct"
# # programs may also contain 2 kinds of models: text-image-to-text and text-to-text, so there are 3*2=6 combinations
# for question in [2, 5]:
#     prefix = "program_" + str(question) + "_"
#     programs = []
#     for file in os.listdir(eval_dir):
#         if file.startswith(prefix):
#             with open(eval_dir + file, "r") as f:
#                 program = json.load(f)
#             programs.append(program)
#
#     # programs contain 1 kind model, only plot the latency over max_new_tokens
#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#     latency = []
#     memory = []
#     max_new_tokens = []
#     for program in programs:
#         if TEXT_IMAGE_TO_TEXT_MODEL[0] in program["Tasks"]["task_0"]["model"] and len(program["Tasks"]) == 1:
#             max_new_tokens.append(program["Tasks"]["task_0"]["inputs"]["max_output_size"])
#             latency.append(program["Tasks"]["task_0"]["eval"]["latency"])
#             memory.append(program["Tasks"]["task_0"]["eval"]["memory"])
#     # plot latency over max_new_tokens and save
#     sns.scatterplot(x=max_new_tokens, y=latency, ax=ax)
#     ax.set_title(f"{TEXT_IMAGE_TO_TEXT_MODEL[0]}")
#     ax.set_xlabel("max_new_tokens")
#     ax.set_ylabel("latency(s)")
#     plt.savefig("question_" + str(question) + "_1_kind_latency.png")
#
#     # programs contain 2 kinds of models, plot the latency over max_new_tokens1 and max_new_tokens2
#     fig, ax = plt.subplots(3, 2, figsize=(12, 18))
#     position = 0
#     for text_image_to_text in TEXT_IMAGE_TO_TEXT_MODEL:
#         for text_to_text in TEXT_TO_TEXT_MODEL:
#             latency = []
#             memory = []
#             max_new_tokens1 = []
#             max_new_tokens2 = []
#             for program in programs:
#                 if len(program["Tasks"]) == 2 and text_image_to_text in program["Tasks"]["task_0"]["model"] and text_to_text in program["Tasks"]["task_1"]["model"]:
#                     max_new_tokens1.append(program["Tasks"]["task_0"]["inputs"]["max_output_size"])
#                     max_new_tokens2.append(program["Tasks"]["task_1"]["inputs"]["max_output_size"])
#                     latency.append(program["Tasks"]["task_0"]["eval"]["latency"] + program["Tasks"]["task_1"]["eval"]["latency"])
#                     memory.append(program["Tasks"]["task_0"]["eval"]["memory"] + program["Tasks"]["task_1"]["eval"]["memory"])
#             # reorganize latency and memory to 2D array according to max_new_tokens1 and max_new_tokens2
#             # first find all unique max_new_tokens1 and max_new_tokens2
#             unique_max_new_tokens1 = list(set(max_new_tokens1))
#             unique_max_new_tokens2 = list(set(max_new_tokens2))
#             unique_max_new_tokens1.sort()
#             unique_max_new_tokens2.sort()
#             # create 2D array
#             latency_2d = np.zeros((len(unique_max_new_tokens1), len(unique_max_new_tokens2)))
#             memory_2d = np.zeros((len(unique_max_new_tokens1), len(unique_max_new_tokens2)))
#             for i in range(len(max_new_tokens1)):
#                 row = unique_max_new_tokens1.index(max_new_tokens1[i])
#                 col = unique_max_new_tokens2.index(max_new_tokens2[i])
#                 latency_2d[row][col] = round(latency[i], 1)
#                 memory_2d[row][col] = round(memory[i], 1)
#             # plot heatmap
#             sns.heatmap(latency_2d, ax=ax[position//2][position%2], annot=True, fmt=".1f", cmap="YlGnBu")
#             ax[position//2][position%2].set_title(f"{text_image_to_text} and {text_to_text}")
#             ax[position//2][position%2].set_xticklabels(unique_max_new_tokens2)
#             ax[position//2][position%2].set_yticklabels(unique_max_new_tokens1)
#             ax[position//2][position%2].set_xlabel("max_new_tokens2")
#             ax[position//2][position%2].set_ylabel("max_new_tokens1")
#             position += 1
#     plt.savefig("question_" + str(question) + "_2_kinds_latency_heatmap.png")

# # for question_3 and question_4, programs may contain only 1 kind of model: text-image-to-text, and there are 3 choices: "Qwen2-VL-2B-Instruct", "Florence-2-base-ft", "Florence-2-large-ft"
# # programs may also contain 2 kinds of models: text-image-to-text and text-to-text, so there are 3*2=6 combinations
# for question in [3, 4]:
#     prefix = "program_" + str(question) + "_"
#     programs = []
#     for file in os.listdir(eval_dir):
#         if file.startswith(prefix):
#             with open(eval_dir + file, "r") as f:
#                 program = json.load(f)
#             programs.append(program)
#
#     # programs contain 1 kind model, only plot the latency over max_new_tokens
#     fig, ax = plt.subplots(1, 3, figsize=(18, 6))
#     position = 0
#     for text_image_to_text in TEXT_IMAGE_TO_TEXT_MODEL:
#         latency = []
#         memory = []
#         max_new_tokens = []
#         for program in programs:
#             if text_image_to_text in program["Tasks"]["task_0"]["model"] and len(program["Tasks"]) == 1:
#                 max_new_tokens.append(program["Tasks"]["task_0"]["inputs"]["max_output_size"])
#                 latency.append(program["Tasks"]["task_0"]["eval"]["latency"])
#                 memory.append(program["Tasks"]["task_0"]["eval"]["memory"])
#         # plot latency over max_new_tokens and save
#         sns.scatterplot(x=max_new_tokens, y=latency, ax=ax[position])
#         ax[position].set_title(f"{text_image_to_text}")
#         ax[position].set_xlabel("max_new_tokens")
#         ax[position].set_ylabel("latency(s)")
#         position += 1
#     plt.savefig("question_" + str(question) + "_1_kind_latency.png")
#
#     # programs contain 2 kinds of models, plot the latency over max_new_tokens1 and max_new_tokens2
#     fig, ax = plt.subplots(3, 2, figsize=(12, 18))
#     position = 0
#     for text_image_to_text in TEXT_IMAGE_TO_TEXT_MODEL:
#         for text_to_text in TEXT_TO_TEXT_MODEL:
#             latency = []
#             memory = []
#             max_new_tokens1 = []
#             max_new_tokens2 = []
#             for program in programs:
#                 if len(program["Tasks"]) == 2 and text_image_to_text in program["Tasks"]["task_0"]["model"] and text_to_text in program["Tasks"]["task_1"]["model"]:
#                     max_new_tokens1.append(program["Tasks"]["task_0"]["inputs"]["max_output_size"])
#                     max_new_tokens2.append(program["Tasks"]["task_1"]["inputs"]["max_output_size"])
#                     latency.append(program["Tasks"]["task_0"]["eval"]["latency"] + program["Tasks"]["task_1"]["eval"]["latency"])
#                     memory.append(program["Tasks"]["task_0"]["eval"]["memory"] + program["Tasks"]["task_1"]["eval"]["memory"])
#             # reorganize latency and memory to 2D array according to max_new_tokens1 and max_new_tokens2
#             # first find all unique max_new_tokens1 and max_new_tokens2
#             unique_max_new_tokens1 = list(set(max_new_tokens1))
#             unique_max_new_tokens2 = list(set(max_new_tokens2))
#             unique_max_new_tokens1.sort()
#             unique_max_new_tokens2.sort()
#             # create 2D array
#             latency_2d = np.zeros((len(unique_max_new_tokens1), len(unique_max_new_tokens2)))
#             memory_2d = np.zeros((len(unique_max_new_tokens1), len(unique_max_new_tokens2)))
#             for i in range(len(max_new_tokens1)):
#                 row = unique_max_new_tokens1.index(max_new_tokens1[i])
#                 col = unique_max_new_tokens2.index(max_new_tokens2[i])
#                 latency_2d[row][col] = round(latency[i], 1)
#                 memory_2d[row][col] = round(memory[i], 1)
#             # plot heatmap
#             sns.heatmap(latency_2d, ax=ax[position//2][position%2], annot=True, fmt=".1f", cmap="YlGnBu")
#             ax[position//2][position%2].set_title(f"{text_image_to_text} and {text_to_text}")
#             ax[position//2][position%2].set_xticklabels(unique_max_new_tokens2)
#             ax[position//2][position%2].set_yticklabels(unique_max_new_tokens1)
#             ax[position//2][position%2].set_xlabel("max_new_tokens2")
#             ax[position//2][position%2].set_ylabel("max_new_tokens1")
#             position += 1
#     plt.savefig("question_" + str(question) + "_2_kinds_latency_heatmap.png")

# for question_6, programs may contain only 1 kind of model: text-to-image, and there are 2 choices: "stable-diffusion-v1-5", "stable-diffusion-xl-base-1.0"
# programs may also contain 2 kinds of models: text-to-text and text-to-image, so there are 2*2=4 combinations
for question in [6]:
    prefix = "program_" + str(question) + "_"
    programs = []
    for file in os.listdir(eval_dir):
        if file.startswith(prefix):
            with open(eval_dir + file, "r") as f:
                program = json.load(f)
            programs.append(program)

    # programs contain 1 kind model, only plot the latency over max_new_pixels
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    position = 0
    for text_to_image in TEXT_TO_IMAGE_MODEL:
        latency = []
        memory = []
        max_new_pixels = []
        for program in programs:
            if text_to_image in program["Tasks"]["task_0"]["model"] and len(program["Tasks"]) == 1:
                max_new_pixels.append(program["Tasks"]["task_0"]["inputs"]["max_output_size"][1])
                latency.append(program["Tasks"]["task_0"]["eval"]["latency"])
                memory.append(program["Tasks"]["task_0"]["eval"]["memory"])
        # plot latency over max_new_tokens and save
        sns.scatterplot(x=max_new_pixels, y=latency, ax=ax[position])
        ax[position].set_title(f"{text_to_image}")
        ax[position].set_xlabel("max_new_pixels")
        ax[position].set_ylabel("latency(s)")
        position += 1
    plt.savefig("question_" + str(question) + "_1_kind_latency.png")

    # programs contain 2 kinds of models, plot the latency over max_new_tokens1 and max_new_pixels
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    position = 0
    for text_to_text in TEXT_TO_TEXT_MODEL:
        for text_to_image in TEXT_TO_IMAGE_MODEL:
            latency = []
            memory = []
            max_new_tokens = []
            max_new_pixels = []
            for program in programs:
                if len(program["Tasks"]) == 2 and text_to_text in program["Tasks"]["task_0"]["model"] and text_to_image in program["Tasks"]["task_1"]["model"]:
                    max_new_tokens.append(program["Tasks"]["task_0"]["inputs"]["max_output_size"])
                    max_new_pixels.append(program["Tasks"]["task_1"]["inputs"]["max_output_size"][1])
                    latency.append(program["Tasks"]["task_0"]["eval"]["latency"] + program["Tasks"]["task_1"]["eval"]["latency"])
                    memory.append(program["Tasks"]["task_0"]["eval"]["memory"] + program["Tasks"]["task_1"]["eval"]["memory"])
            # reorganize latency and memory to 2D array according to max_new_tokens and max_new_pixels
            # first find all unique max_new_tokens and max_new_pixels
            unique_max_new_tokens = list(set(max_new_tokens))
            unique_max_new_pixels = list(set(max_new_pixels))
            unique_max_new_tokens.sort()
            unique_max_new_pixels.sort()
            # create 2D array
            latency_2d = np.zeros((len(unique_max_new_tokens), len(unique_max_new_pixels)))
            memory_2d = np.zeros((len(unique_max_new_tokens), len(unique_max_new_pixels)))
            for i in range(len(max_new_tokens)):
                row = unique_max_new_tokens.index(max_new_tokens[i])
                col = unique_max_new_pixels.index(max_new_pixels[i])
                latency_2d[row][col] = round(latency[i], 1)
                memory_2d[row][col] = round(memory[i], 1)
            # plot heatmap
            sns.heatmap(latency_2d, ax=ax[position//2][position%2], annot=True, fmt=".1f", cmap="YlGnBu")
            ax[position//2][position%2].set_title(f"{text_to_text} and {text_to_image}")
            ax[position//2][position%2].set_xticklabels(unique_max_new_pixels)
            ax[position//2][position%2].set_yticklabels(unique_max_new_tokens)
            ax[position//2][position%2].set_xlabel("max_new_pixels")
            ax[position//2][position%2].set_ylabel("max_new_tokens")
            position += 1
    plt.savefig("question_" + str(question) + "_2_kinds_latency_heatmap.png")