#!/bin/bash
# default MODEL_PATH or use the one from the environment
MODEL_PATH="/home/oberon/projects/Magma/Magma-8B"
# default OUTPUT_DIR
OUTPUT_DIR="/home/oberon/projects/Magma/FT_checkpoints"

torchrun --nproc_per_node=1 train.py \
    --deepspeed ./trainer/deepspeed/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --version magma_instruct \
    --data_path "/home/oberon/projects/Magma/data_configs/mind2web_som_finetune.yaml" \
    --vision_tower convnext_xxlarge \
    --img_size 512 \
    --max_num_crops 4 \
    --img_anyres_strategy crop \
    --vision_backbone "convnextxxlarge" \
    --is_multimodal True \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_vision_tokenizer 'none' \
    --mm_vision_select_layer -2 \
    --mm_use_image_start_end False \
    --mm_use_som_tom True \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 100 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --flash_attn_2_enabled True \
    --local_run True \
    --show_trace False \
    --run_name finetune_anyres \
    --remove_static_trace_pts True