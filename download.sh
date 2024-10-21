#!/bin/bash

# 模型列表
models=(
"timbrooks/instruct-pix2pix"
)

# 遍历每个模型并下载
for model in "${models[@]}"
do
   huggingface-cli download $model
done