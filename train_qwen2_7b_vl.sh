deepspeed --master_port=25678 --num_gpus=4 src/train.py \
    --model_name_or_path /home/scb123/HuggingfaceWeight/Qwen2-VL-2B-Instruct \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset mllm_video_demo \
    --template qwen2_vl \
    --cutoff_len 1024 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir ./train_weight_vl \
    --logging_steps 1 \
    --save_steps 100 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4  \
    --learning_rate 1e-5\
    --warmup_ratio 0.05 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --fp16 true \
    --ddp_timeout 180000000 \
    --eval_steps 100 \
    --val_size 0.5 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy steps \
    --load_best_model_at_end true \
    --save_total_limit 1 \
    # --run_name test \
    # --save_only_model true \
    
