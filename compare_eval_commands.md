# Comparison: BLT Adapter vs Baseline Evaluation Commands

## BLT Adapter Model (focused_sep_embedding_global_kv_residual_LM_NTP/epoch_5)

**Checkpoint:** `/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/focused_sep_embedding_global_kv_residual_LM_NTP/epoch_5`

**Training:** Trained with `blt_focused_training.py` - includes BLT adapter components (local decoder, boundary head, etc.)

**CLI Command:**
```bash
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
```

**Key Features:**
- ✅ **Uses BLT adapter features**: Local decoder refinement enabled
- ✅ **Learned boundary patching**: `--patcher learned` uses trained boundary head
- ✅ **Boundary threshold**: 0.65 (trained boundary head confidence threshold)
- ✅ **Local decoder**: Refines spans using learned span latents
- ✅ **Span refinement**: Patches detected spans with local decoder output

---

## Baseline Model (freezon_baseline/epoch_10)

**Checkpoint:** `/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/freezon_baseline/epoch_10`

**Training:** Baseline model (global transformer only, all layers frozen)

**CLI Command:**
```bash
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
```

**Key Features:**
- ❌ **No BLT adapter features**: `--disable_local_decoder` disables all adapter components
- ❌ **No patching**: Automatically sets `patcher="none"` when `--disable_local_decoder` is used
- ❌ **No boundary detection**: Boundary head is not used
- ❌ **No local decoder**: Only uses global transformer for generation
- ✅ **Pure baseline**: Equivalent to evaluating the base model directly

---

## Key Differences Summary

| Feature | BLT Adapter Model | Baseline Model |
|---------|------------------|----------------|
| **Local Decoder** | ✅ Enabled (refines spans) | ❌ Disabled |
| **Boundary Patching** | ✅ Learned (trained boundary head) | ❌ None |
| **Patcher** | `learned` | `none` (auto-set) |
| **Boundary Threshold** | 0.65 | N/A |
| **Min Steps Between Patches** | 4 | N/A |
| **Max Patch Length** | 128 | N/A |
| **Model Components Used** | Global transformer + Local decoder + Boundary head + Span encoder | Global transformer only |

---

## What Each Model Does During Generation

### BLT Adapter Model:
1. Generates tokens using global transformer
2. Uses trained boundary head to detect span boundaries
3. When boundary detected (confidence ≥ 0.65):
   - Extracts span latent from global hidden state
   - Uses local decoder to refine/regenerate the span
   - Replaces original span with refined version
4. Continues generation with refined code

### Baseline Model:
1. Generates tokens using global transformer only
2. No boundary detection
3. No span refinement
4. Pure autoregressive generation (standard LM behavior)

---

## Expected Performance Difference

- **BLT Adapter**: Should show improved code quality due to span-aware refinement
- **Baseline**: Standard transformer performance (no span-aware improvements)


