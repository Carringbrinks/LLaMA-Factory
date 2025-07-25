deepspeed --master_port=25678 --num_gpus=4 src/train.py \
    --model_name_or_path /home/scb123/HuggingfaceWeight/Qwen2.5-1.5B \
    --stage pt \
    --do_train \
    --finetuning_type full \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset 1_14_pre \
    --cutoff_len 1024 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir /home/scb123/PyProject/Train_Weights/qwen2.5-1.5b-pt_1-14 \
    --logging_steps 1 \
    --save_steps 0.6 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1  \
    --learning_rate 1e-5\
    --warmup_ratio 0.1 \
    --num_train_epochs 10 \
    --lr_scheduler_type cosine \
    --fp16 true \
    --ddp_timeout 180000000 \
    --eval_steps 50 \
    --val_size 0.05 \
    --per_device_eval_batch_size 1 \
    --save_only_model true \
    --run_name 1_14_pt
    # --load_best_model_at_end true \
    # --save_total_limit 1 \
    # --run_name test \
    # --evaluation_strategy steps \
    # --template qwen \
    
