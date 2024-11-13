#!/bin/bash

python src/export_model.py \
    --model_name_or_path /home/scb123/huggingface_weight/Qwen2.5-7b-Instruct \
    --adapter_name_or_path /home/scb123/PyProject/LLaMA-Factory/train_weight_test \
    --template qwen \
    --finetuning_type lora \
    --export_dir /home/scb123/PyProject/LLaMA-Factory/export_model \
    --export_size 2 \
    --export_device auto \
    --export_legacy_format false \


    
