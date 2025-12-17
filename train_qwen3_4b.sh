#!/bin/bash
# Training script for BLT Adapter v2 with Qwen3 4B

cd /data/home/zhangsj/AST_decoding

python blt_focused_training_v2.py \
    --model_path /data/home/zhangsj/qwen3_4b \
    --trial_name focused_qwen3_4b_v2 \
    --parquet /data/home/zhangsj/Data/more_big_code_language/python/python_ast_parsed.parquet \
    --epochs 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr 3e-5 \
    --max_length 328 \
    --dtype auto \
    --local_num_layers 2 \
    --min_span_len 1 \
    --max_span_len 64 \
    --max_nodes_per_sample 64 \
    --lm_weight 0.05 \
    --warmup_lm_weight 0.0 \
    --node_recon_weight 1.0 \
    --boundary_weight 0.5 \
    --latent_mse_weight 0.2 \
    --warmup_steps 500 \
    --warmup_node_weight 0.5 \
    --warmup_boundary_weight 0.3 \
    --warmup_mse_weight 0.2 \
    --lr_scheduler cosine \
    --lr_warmup_steps 500 \
    --min_lr_ratio 0.1 \
    --humaneval_parquet /data/home/zhangsj/Data/HumanEval/humaneval_ast_parsed.parquet \
    --eval_every_n_epochs 1


