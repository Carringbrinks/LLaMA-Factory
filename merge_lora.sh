# python src/export_model.py \
#     --model_name_or_path /home/scb123/HuggingfaceWeight/Qwen2-VL-2B-Instruct \
#     --adapter_name_or_path /home/scb123/PyProject/LLaMA-Factory/train_weight_vl \
#     --template qwen2_vl \
#     --finetuning_type lora \
#     --export_dir /home/scb123/PyProject/LLaMA-Factory/export_model_qwen2_vl \
#     --export_size 2 \
#     --export_device auto \
#     --export_legacy_format false \


python src/export_model.py \
    --model_name_or_path /home/scb123/PyProject/LLaMA-Factory/Hongkong/export_model_9000 \
    --template deepseek3 \
    --trust_remote_code true \
    --export_dir /home/scb123/PyProject/LLaMA-Factory/Hongkong/export_model_9000_gptq_int4_fp16 \
    --export_quantization_bit 4 \
    --export_quantization_dataset data/c4_demo.json \
    --export_size 5 \
    --export_device cpu \
    --export_legacy_format false \