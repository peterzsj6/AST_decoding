#!/bin/bash
# Run EvalPlus evaluation on baseline model (global transformer only)
# Checkpoint: /data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/freezon_baseline/epoch_10

cd /data/home/zhangsj/AST_decoding

python run_evalplus_blt.py \
    --checkpoint /data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/freezon_baseline/epoch_10 \
    --model_path /data/home/zhangsj/AST_decoding \
    --device auto \
    --dtype auto \
    --dataset humaneval \
    --n_samples 1 \
    --max_new_tokens 512 \
    --disable_local_decoder \
    --temperature 0.0 \
    --top_p 1.0 \
    --repetition_penalty 1.0 \
    --overwrite

