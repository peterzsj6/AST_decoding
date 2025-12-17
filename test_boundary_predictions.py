#!/usr/bin/env python3
"""Test boundary_head predictions from different checkpoints"""

import os
import sys
import torch

# Set CUDA_VISIBLE_DEVICES before importing
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Make project root importable
PROJECT_ROOT = "/data/home/zhangsj"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from AST_decoding.blt_inference import load_adapter_and_tokenizer, select_device, select_dtype

def test_boundary_predictions(checkpoint_path, model_path, test_text="def hello_world():\n    \"\"\""):
    """Test boundary_head predictions"""
    print(f"\n{'='*80}")
    print(f"Testing: {os.path.basename(checkpoint_path)}")
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
    
    # Tokenize test text
    enc = tokenizer(test_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    
    # Get hidden states
    with torch.no_grad():
        outputs = adapter(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        last_hidden = hidden_states[:, -1, :]  # Last token hidden state
    
    # Get boundary_head predictions
    if hasattr(adapter, 'boundary_head') and adapter.boundary_head is not None:
        with torch.no_grad():
            boundary_logits = adapter.boundary_head(last_hidden)
            probs = torch.softmax(boundary_logits, dim=-1)
            boundary_confidence = float(probs[0, 1].item())
            boundary_logit_0 = float(boundary_logits[0, 0].item())
            boundary_logit_1 = float(boundary_logits[0, 1].item())
        
        print(f"Boundary head predictions:")
        print(f"  Logit[0] (no boundary): {boundary_logit_0:.4f}")
        print(f"  Logit[1] (boundary): {boundary_logit_1:.4f}")
        print(f"  Confidence (prob of boundary): {boundary_confidence:.4f}")
        
        # Check boundary_head weights
        boundary_params = list(adapter.boundary_head.parameters())
        if boundary_params:
            boundary_sum = sum(p.sum().item() for p in boundary_params)
            print(f"  Boundary head weight sum: {boundary_sum:.6f}")
        
        return {
            'boundary_confidence': boundary_confidence,
            'boundary_logit_0': boundary_logit_0,
            'boundary_logit_1': boundary_logit_1,
            'boundary_weight_sum': boundary_sum if 'boundary_sum' in locals() else None
        }
    else:
        print("ERROR: boundary_head not found!")
        return None

if __name__ == "__main__":
    model_path = "/data/home/zhangsj/AST_decoding"
    test_text = "def hello_world():\n    \"\"\""
    
    checkpoints = [
        "/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/focused_sep_embedding_global_kv_residual_LM_NTP/epoch_5",
        "/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/qwen2.5_1.5b_all_frozen/epoch_5",
    ]
    
    results = {}
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            result = test_boundary_predictions(ckpt, model_path, test_text)
            if result:
                results[ckpt] = result
    
    # Compare
    if len(results) >= 2:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}\n")
        ckpt_names = list(results.keys())
        print(f"Checkpoint 1 ({os.path.basename(ckpt_names[0])}):")
        print(f"  boundary_confidence: {results[ckpt_names[0]]['boundary_confidence']:.6f}")
        print(f"  boundary_weight_sum: {results[ckpt_names[0]]['boundary_weight_sum']:.6f}")
        print(f"\nCheckpoint 2 ({os.path.basename(ckpt_names[1])}):")
        print(f"  boundary_confidence: {results[ckpt_names[1]]['boundary_confidence']:.6f}")
        print(f"  boundary_weight_sum: {results[ckpt_names[1]]['boundary_weight_sum']:.6f}")
        
        conf_diff = abs(results[ckpt_names[0]]['boundary_confidence'] - results[ckpt_names[1]]['boundary_confidence'])
        print(f"\nConfidence difference: {conf_diff:.6f}")
        if conf_diff < 0.001:
            print("⚠ WARNING: Boundary predictions are nearly identical despite different weights!")
            print("This suggests the hidden states are the same, or boundary_head isn't being used correctly.")

