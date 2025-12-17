"""
Test script for BLTAdapterModel v2 with Qwen3 4B
"""
import torch
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from blt_adapter_model_v2 import create_blt_adapter_model, BLTAdapterModel

def test_qwen3_4b():
    print("=" * 60)
    print("Testing BLTAdapterModel v2 with Qwen3 4B")
    print("=" * 60)
    
    model_path = "/data/home/zhangsj/qwen3_4b"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        return False
    
    print(f"\n[1/5] Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False
    
    print(f"\n[2/5] Creating BLTAdapterModel with base model from {model_path}...")
    try:
        adapter = create_blt_adapter_model(
            model_path=model_path,
            local_num_layers=2,
            local_dropout=0.1,
            max_node_length=64,
            num_node_types=128
        )
        print(f"✓ Adapter model created successfully")
        print(f"  - Hidden size: {adapter.hidden_size}")
        print(f"  - Vocab size: {adapter.vocab_size}")
        print(f"  - Base model type: {type(adapter.base_model).__name__}")
        print(f"  - Has base_model.model.layers: {hasattr(adapter.base_model, 'model') and hasattr(adapter.base_model.model, 'layers')}")
        if hasattr(adapter.base_model, 'model') and hasattr(adapter.base_model.model, 'layers'):
            print(f"  - Number of layers: {len(adapter.base_model.model.layers)}")
    except Exception as e:
        print(f"✗ Failed to create adapter: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[3/5] Testing device/dtype properties...")
    try:
        device = adapter.device
        dtype = adapter.dtype
        print(f"✓ Device: {device}, Dtype: {dtype}")
    except Exception as e:
        print(f"✗ Failed to access device/dtype: {e}")
        return False
    
    print(f"\n[4/5] Testing forward pass with dummy inputs...")
    try:
        # Create dummy input
        test_text = "def hello_world():\n    print('Hello, World!')"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Keep on CPU for testing (GPU may not have enough memory)
        # Forward pass on CPU
        adapter.eval()
        with torch.no_grad():
            outputs = adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_metadata=None,
                labels=None
            )
        
        print(f"✓ Forward pass successful")
        print(f"  - Output keys: {list(outputs.keys())}")
        if hasattr(outputs, 'logits'):
            print(f"  - Logits shape: {outputs.logits.shape}")
        if hasattr(outputs, 'loss'):
            print(f"  - Loss: {outputs.loss}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[5/5] Testing save/load...")
    try:
        test_checkpoint_dir = "/tmp/test_blt_v2_checkpoint"
        os.makedirs(test_checkpoint_dir, exist_ok=True)
        
        print(f"  Saving to {test_checkpoint_dir}...")
        adapter.save_pretrained(test_checkpoint_dir)
        tokenizer.save_pretrained(test_checkpoint_dir)
        print(f"  ✓ Saved successfully")
        
        print(f"  Loading from {test_checkpoint_dir}...")
        loaded_adapter = BLTAdapterModel.from_pretrained(test_checkpoint_dir)
        print(f"  ✓ Loaded successfully")
        print(f"  - Base model type: {type(loaded_adapter.base_model).__name__}")
        print(f"  - Hidden size: {loaded_adapter.hidden_size}")
        
        # Test forward pass with loaded model (on CPU)
        loaded_adapter.eval()
        with torch.no_grad():
            loaded_outputs = loaded_adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_metadata=None,
                labels=None
            )
        print(f"  ✓ Forward pass with loaded model successful")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_checkpoint_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_qwen3_4b()
    sys.exit(0 if success else 1)

