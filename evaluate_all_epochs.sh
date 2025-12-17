#!/bin/bash
# Evaluate all epochs in focused_sep_embedding_global_kv_residual_LM_NTP
# and compile pass@1 rates into a JSONL file

cd /data/home/zhangsj/AST_decoding

CHECKPOINT_DIR="/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/focused_sep_embedding_global_kv_residual_LM_NTP"
MODEL_PATH="/data/home/zhangsj/AST_decoding"
OUTPUT_JSONL="${CHECKPOINT_DIR}/epoch_results.jsonl"

echo "=========================================="
echo "Evaluating all epochs"
echo "=========================================="
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Model path: ${MODEL_PATH}"
echo "Output JSONL: ${OUTPUT_JSONL}"
echo "=========================================="
echo ""

python evaluate_all_epochs.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --gpu 7 \
    --dataset humaneval \
    --patcher learned \
    --boundary_threshold 0.65 \
    --min_steps_between_patches 4 \
    --max_patch_len 128 \
    --temperature 0.0 \
    --top_p 1.0 \
    --repetition_penalty 1.0 \
    --max_new_tokens 512

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_JSONL}"
echo "=========================================="

