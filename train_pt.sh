deepspeed --master_port=25678 --num_gpus=2 src/train.py \
    --model_name_or_path /home/cbsu/huggingface/Qwen2.5-1.5B \
    --stage pt \
    --do_train \
    --finetuning_type full \
    --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
    --dataset book_pre \
    --cutoff_len 4096 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir /home/cbsu/Pyproject/LLaMA-Factory/Train_Weights \
    --logging_steps 1 \
    --save_steps 0.6 \
    --plot_loss true \
    --overwrite_output_dir yes \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2  \
    --learning_rate 1e-5\
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --fp16 true \
    --ddp_timeout 180000000 \
    --eval_steps 50 \
    --eval_strategy steps \
    --val_size 0.05 \
    --per_device_eval_batch_size 1 \
    --save_only_model true \
    --report_to swanlab \
    --run_name book_pt_test \
    --disable_shuffling \
    --packing false
    # --load_best_model_at_end true \
    # --save_total_limit 1 \
    # --run_name test \
    # --evaluation_strategy steps \
    # --template qwen \
    
