#!/bin/bash
# Training script for BLT Adapter v2 with Qwen2.5 1.5B - ALL global transformer layers frozen
# Ablation study: Same hyperparameters as original except lm_weight=0.0 (all layers frozen)

cd /data/home/zhangsj/AST_decoding

python blt_focused_training_v2.py \
    --model_path /data/home/zhangsj/AST_decoding \
    --trial_name qwen2.5_1.5b_all_frozen \
    --parquet /data/home/zhangsj/Data/more_big_code_language/python/python_ast_parsed.parquet \
    --epochs 10 \
    --batch_size 4 \
    --lr 3e-05 \
    --max_length 328 \
    --dtype auto \
    --min_span_len 1 \
    --max_span_len 64 \
    --max_nodes_per_sample 64 \
    --lm_weight 0.0 \
    --warmup_lm_weight 0.0 \
    --node_recon_weight 1.0 \
    --boundary_weight 0.5 \
    --latent_mse_weight 0.2 \
    --warmup_steps 500 \
    --warmup_node_weight 0.5 \
    --warmup_boundary_weight 0.3 \
    --warmup_mse_weight 0.2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler cosine \
    --lr_warmup_steps 500 \
    --min_lr_ratio 0.1 \
    --local_num_layers 2 \
    --val_split 0.0 \
    --humaneval_parquet /data/home/zhangsj/Data/HumanEval/humaneval_ast_parsed.parquet \
    --eval_every_n_epochs 1

