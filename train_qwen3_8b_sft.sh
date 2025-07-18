
deepspeed --master_port=25678 --num_gpus=4 src/train.py \
    --model_name_or_path /home/scb123/HuggingfaceWeight/Qwen3-8B \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset synsql_8000 \
    --template qwen3 \
    --cutoff_len 4096 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir /home/scb123/PyProject/Train_Weights/qwen3-8b \
    --logging_steps 1 \
    --save_steps 0.6 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4  \
    --learning_rate 1e-5\
    --warmup_steps 50 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --fp16 true \
    --ddp_timeout 180000000 \
    --eval_steps 50 \
    --val_size 0.01 \
    --per_device_eval_batch_size 2 \
    --run_name qwen3_8b \
    --save_only_model true \
    # --load_best_model_at_end true \
    # --save_total_limit 1 \
    # --run_name test \
    
  
    
