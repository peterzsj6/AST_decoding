#!/usr/bin/env python3
"""
Simple PEFT LoRA Training Runner

This script runs PEFT LoRA training for the span-aware model.
Uses the official PEFT library for better stability and compatibility.
No complex configuration needed - just run it.
"""

import os
# Set CUDA_VISIBLE_DEVICES before importing torch to ensure GPU 1 is used
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from lora_span_aware_model import (
    create_lora_span_aware_model,
    setup_lora_training,
    train_lora_model
)

def main():
    print("🚀 STARTING PEFT LORA SPAN-AWARE TRAINING")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. PEFT LoRA training requires GPU.")
        return
    
    print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    config = {
        'embedding_approach': 'mean_pooling',  # Options: 'mean_pooling', 'mean_pooling_only', 'span_type_only', 'multi_component', 'placeholder'
        'trail_name': 'peft_lora_v2_11_6_mean_pooling',
        'parquet_files': [
            '/data/home/zhangsj/Data/more_big_code_language/python/python_ast_parsed.parquet'
        ],
        'lora_config': {
            'rank': 16,
            'alpha': 32,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        },
        'training': {
            'learning_rate': 5e-4,
            'batch_size': 16,
            'max_length': 1024,
            'num_epochs': 3,  # Reduced for faster training
            'save_steps': 5000,
            'log_steps': 50,
            'span_type_loss_weight': 0.2  # Weight (λ) for combining span type loss with cross-entropy loss
        }
    }
    
    print(f"\n📋 CONFIGURATION (Using PEFT):")
    print(f"   Embedding: {config['embedding_approach']}")
    print(f"   LoRA rank: {config['lora_config']['rank']}")
    print(f"   LoRA alpha: {config['lora_config']['alpha']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Epochs: {config['training']['num_epochs']}")
    print(f"   Span type loss weight (λ): {config['training']['span_type_loss_weight']}")
    print(f"   Parquet files: {len(config['parquet_files'])} file(s)")
    for pf in config['parquet_files']:
        print(f"     - {pf}")
    
    try:
        # Step 1: Create PEFT LoRA model
        print(f"\n🔧 Creating PEFT LoRA span-aware model...")
        lora_model = create_lora_span_aware_model(
            model_path="/data/home/zhangsj/qwen_coder_1.5b",
            embedding_approach=config['embedding_approach'],
            lora_config=config['lora_config']
        )
        
        # Move to GPU
        lora_model = lora_model.cuda()
        print(f"✅ PEFT LoRA model moved to GPU")
        
        # Step 2: Setup training
        print(f"\n📊 Setting up PEFT LoRA training...")
        training_setup = setup_lora_training(
            model=lora_model,
            learning_rate=config['training']['learning_rate'],
            batch_size=config['training']['batch_size'],
            max_length=config['training']['max_length'],
            num_epochs=config['training']['num_epochs'],
            trail_name=config['trail_name'],
            span_type_loss_weight=config['training']['span_type_loss_weight'],
            parquet_files=config['parquet_files']
        )
        
        if training_setup is None:
            print("❌ Failed to setup training. Check if dataset exists.")
            return
        
        # Step 3: Start training
        print(f"\n🏋️ Starting PEFT LoRA training...")
        trained_model = train_lora_model(
            training_setup,
            num_epochs=config['training']['num_epochs'],
            save_steps=config['training']['save_steps'],
            log_steps=config['training']['log_steps']
        )
        
        print(f"\n🎉 PEFT LORA TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Check TensorBoard logs: {training_setup['log_dir']}")
        print(f"   Best model saved to: /data/home/zhangsj/qwen_coder_1.5b/best_peft_lora_span_aware_{config['trail_name']}")
        
        # Memory cleanup
        del lora_model
        del trained_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error during PEFT LoRA training: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        torch.cuda.empty_cache()

def quick_test():
    """Quick test to verify PEFT LoRA model creation works."""
    print("🧪 QUICK PEFT LORA TEST")
    print("="*40)
    
    try:
        # Test model creation
        print("Creating PEFT LoRA model...")
        lora_model = create_lora_span_aware_model(
            model_path="/data/home/zhangsj/qwen_coder_1.5b",
            embedding_approach="mean_pooling",
            lora_config={'rank': 8, 'alpha': 16, 'dropout': 0.1, 'target_modules': ['q_proj', 'v_proj']}
        )
        
        print("✅ PEFT LoRA model created successfully!")
        
        # Test parameter counting
        stats = lora_model.print_trainable_parameters()
        
        # Test forward pass
        print("\nTesting forward pass...")
        test_input = torch.randint(0, 1000, (1, 10)).cuda()
        lora_model = lora_model.cuda()
        
        with torch.no_grad():
            output = lora_model(test_input)
            print(f"✅ Forward pass successful: output shape available")
        
        # Cleanup
        del lora_model
        torch.cuda.empty_cache()
        
        print("✅ Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        main()
