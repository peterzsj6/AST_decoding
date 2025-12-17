#!/usr/bin/env python3
"""Test if different checkpoints produce different inference results"""

import os
import sys

# Set CUDA_VISIBLE_DEVICES before importing
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Make project root importable
PROJECT_ROOT = "/data/home/zhangsj"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from AST_decoding.blt_inference import load_adapter_and_tokenizer, incremental_generate, select_device, select_dtype

def test_inference(checkpoint_path, model_path, test_prompt="def hello_world():\n    \"\"\""):
    """Test inference with a checkpoint"""
    print(f"\n{'='*80}")
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    device = select_device("auto")
    dtype = select_dtype(device, "auto")
    
    # Load checkpoint
    adapter, tokenizer = load_adapter_and_tokenizer(
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        device=device,
        dtype=dtype,
        peft_adapter=None,
    )
    
    # Get boundary_head weight sum for verification
    if hasattr(adapter, 'boundary_head') and adapter.boundary_head is not None:
        boundary_params = list(adapter.boundary_head.parameters())
        if boundary_params:
            boundary_sum = sum(p.sum().item() for p in boundary_params)
            print(f"Loaded boundary_head sum: {boundary_sum:.6f}")
    
    # Run inference with same prompt
    print(f"Running inference with prompt: {repr(test_prompt)}")
    output = incremental_generate(
        model=adapter,
        tokenizer=tokenizer,
        prompt_text=test_prompt,
        max_new_tokens=50,
        patcher="learned",
        boundary_threshold=0.65,
        min_steps_between_patches=4,
        max_patch_len=128,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        disable_patching_in_docstring=True,
        use_local_decoder=True,
    )
    
    print(f"\nGenerated output (first 200 chars):")
    print(output[:200])
    print(f"\nFull output length: {len(output)}")
    
    return output, boundary_sum if 'boundary_sum' in locals() else None

if __name__ == "__main__":
    model_path = "/data/home/zhangsj/AST_decoding"
    test_prompt = "def hello_world():\n    \"\"\""
    
    checkpoints = [
        "/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/focused_sep_embedding_global_kv_residual_LM_NTP/epoch_5",
        "/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/qwen2.5_1.5b_all_frozen/epoch_5",
    ]
    
    results = {}
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            output, boundary_sum = test_inference(ckpt, model_path, test_prompt)
            results[ckpt] = {
                'output': output,
                'boundary_sum': boundary_sum,
                'output_hash': hash(output[:500])  # Hash first 500 chars
            }
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}\n")
    
    if len(results) >= 2:
        ckpt_names = list(results.keys())
        print(f"Checkpoint 1: {os.path.basename(ckpt_names[0])}")
        print(f"  boundary_head sum: {results[ckpt_names[0]]['boundary_sum']:.6f}")
        print(f"  output hash: {results[ckpt_names[0]]['output_hash']}")
        print(f"\nCheckpoint 2: {os.path.basename(ckpt_names[1])}")
        print(f"  boundary_head sum: {results[ckpt_names[1]]['boundary_sum']:.6f}")
        print(f"  output hash: {results[ckpt_names[1]]['output_hash']}")
        
        if results[ckpt_names[0]]['output_hash'] == results[ckpt_names[1]]['output_hash']:
            print(f"\n⚠ WARNING: Outputs are IDENTICAL despite different boundary_head weights!")
            print(f"This suggests the model isn't using boundary_head during inference.")
        else:
            print(f"\n✓ Outputs are different - model is working correctly")

