deepspeed --master_port=25678 --num_gpus=2 src/train.py \
    --model_name_or_path /home/scb123/HuggingfaceWeight/Qwen2.5-7B-Instruct \
    --stage pt \
    --do_train \
    --finetuning_type lora \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset jingyong \
    --template qwen \
    --cutoff_len 2048 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir ./train_weight_jingyong_lora \
    --logging_steps 1 \
    --save_steps 0.2 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8  \
    --learning_rate 1e-4\
    --warmup_ratio 0.1 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --fp16 true \
    --ddp_timeout 180000000 \
    --eval_steps 0.2 \
    --val_size 0.05 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy steps \
    # --load_best_model_at_end true \
    # --save_total_limit 1 \
    # --run_name test \
    # --save_only_model true \
    
