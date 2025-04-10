
deepspeed --master_port=25678 --num_gpus=4 src/train.py \
    --model_name_or_path /home/scb123/HuggingfaceWeight/Qwen2.5-7B-Instruct \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset r1_o1_sft_110k \
    --template qwen \
    --cutoff_len 4096 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir ./qwen2.5_r1_o1_sft \
    --logging_steps 1 \
    --save_steps 0.2 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4  \
    --learning_rate 1e-5\
    --warmup_steps 50 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --fp16 true \
    --ddp_timeout 180000000 \
    --eval_steps 100 \
    --val_size 0.01 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy steps \
    --run_name r1-o1-110k-sft-03291041 \
    --save_only_model true \
    # --load_best_model_at_end true \
    # --save_total_limit 1 \
    # --run_name test \
    # --save_only_model true \
    
