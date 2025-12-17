#!/bin/bash
# Run EvalPlus evaluation on BLT adapter model
# Checkpoint: /data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/focused_sep_embedding_global_kv_residual_LM_NTP/epoch_5
# Model trained with blt_focused_training.py - uses BLT adapter features (local decoder, learned boundary patching)

cd /data/home/zhangsj/AST_decoding

python run_evalplus_blt.py \
    --checkpoint /data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/focused_sep_embedding_global_kv_residual_LM_NTP/epoch_5 \
    --model_path /data/home/zhangsj/AST_decoding \
    --device auto \
    --dtype auto \
    --dataset humaneval \
    --n_samples 1 \
    --max_new_tokens 512 \
    --patcher learned \
    --boundary_threshold 0.65 \
    --min_steps_between_patches 4 \
    --max_patch_len 128 \
    --temperature 0.0 \
    --top_p 1.0 \
    --repetition_penalty 1.0 \
    --overwrite


