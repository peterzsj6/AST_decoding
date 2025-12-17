#!/usr/bin/env python3
"""Test script to verify checkpoint loading is working correctly"""

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

def test_checkpoint_loading(checkpoint_path, model_path):
    """Test loading a checkpoint and verify weights are correct"""
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
    
    # Get weight checksums
    checksums = {}
    if hasattr(adapter, 'boundary_head') and adapter.boundary_head is not None:
        boundary_params = list(adapter.boundary_head.parameters())
        if boundary_params:
            boundary_sum = sum(p.sum().item() for p in boundary_params)
            checksums['boundary_head'] = boundary_sum
    
    if hasattr(adapter, 'latent_from_global') and adapter.latent_from_global is not None:
        latent_params = list(adapter.latent_from_global.parameters())
        if latent_params:
            latent_sum = sum(p.sum().item() for p in latent_params)
            checksums['latent_from_global'] = latent_sum
    
    # Also load checkpoint file directly to compare
    checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_file):
        checkpoint_sd = torch.load(checkpoint_file, map_location="cpu")
        
        # Get checksums from checkpoint file
        file_checksums = {}
        if 'boundary_head.weight' in checkpoint_sd and 'boundary_head.bias' in checkpoint_sd:
            boundary_file_sum = checkpoint_sd['boundary_head.weight'].sum().item() + checkpoint_sd['boundary_head.bias'].sum().item()
            file_checksums['boundary_head'] = boundary_file_sum
        
        latent_keys = [k for k in checkpoint_sd.keys() if 'latent_from_global' in k]
        if latent_keys:
            latent_file_sum = sum(checkpoint_sd[k].sum().item() for k in latent_keys)
            file_checksums['latent_from_global'] = latent_file_sum
        
        print(f"\nComparison:")
        print(f"  Loaded model boundary_head: {checksums.get('boundary_head', 'N/A')}")
        print(f"  Checkpoint file boundary_head: {file_checksums.get('boundary_head', 'N/A')}")
        if 'boundary_head' in checksums and 'boundary_head' in file_checksums:
            diff = abs(checksums['boundary_head'] - file_checksums['boundary_head'])
            if diff < 0.01:
                print(f"  ✓ Match! (diff={diff:.6f})")
            else:
                print(f"  ✗ MISMATCH! (diff={diff:.6f})")
        
        print(f"  Loaded model latent_from_global: {checksums.get('latent_from_global', 'N/A')}")
        print(f"  Checkpoint file latent_from_global: {file_checksums.get('latent_from_global', 'N/A')}")
        if 'latent_from_global' in checksums and 'latent_from_global' in file_checksums:
            diff = abs(checksums['latent_from_global'] - file_checksums['latent_from_global'])
            if diff < 0.01:
                print(f"  ✓ Match! (diff={diff:.6f})")
            else:
                print(f"  ✗ MISMATCH! (diff={diff:.6f})")
    
    return checksums

if __name__ == "__main__":
    model_path = "/data/home/zhangsj/AST_decoding"
    base_dir = "/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/focused_sep_embedding_global_kv_residual_LM_NTP"
    
    # Test a few epochs
    epochs_to_test = [1, 5, 10]
    all_checksums = {}
    
    for epoch in epochs_to_test:
        checkpoint_path = os.path.join(base_dir, f"epoch_{epoch}")
        if os.path.exists(checkpoint_path):
            checksums = test_checkpoint_loading(checkpoint_path, model_path)
            all_checksums[epoch] = checksums
    
    # Compare checksums between epochs
    print(f"\n{'='*80}")
    print("Summary: Comparing checksums across epochs")
    print(f"{'='*80}\n")
    
    if len(all_checksums) > 1:
        epochs = sorted(all_checksums.keys())
        for i in range(len(epochs) - 1):
            epoch1, epoch2 = epochs[i], epochs[i+1]
            print(f"Epoch {epoch1} vs Epoch {epoch2}:")
            for key in ['boundary_head', 'latent_from_global']:
                if key in all_checksums[epoch1] and key in all_checksums[epoch2]:
                    val1 = all_checksums[epoch1][key]
                    val2 = all_checksums[epoch2][key]
                    diff = abs(val1 - val2)
                    print(f"  {key}: {val1:.6f} vs {val2:.6f} (diff={diff:.6f})")
                    if diff < 1e-3:
                        print(f"    ⚠ WARNING: Values are nearly identical!")

