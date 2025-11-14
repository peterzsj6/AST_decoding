#!/usr/bin/env python3
"""
LoRA Span-Aware Qwen2 Model with PEFT

This script creates a LoRA version of the span-aware model that can train
all transformer layers efficiently using PEFT's Low-Rank Adaptation (LoRA).

Key features:
- PEFT LoRA adapters on all transformer layers (attention + MLP)
- Span-aware embedding layer (trainable)
- Memory efficient training
- Compatible with existing span metadata
- Uses official PEFT library for better stability and compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import datetime
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Import components from the original load_model.py
from load_model import (
    ASTSpanDataset,
    create_dataloader,
    load_language_datasets,
    MeanPooledSpanEmbeddingLayer,
    OnlyMeanPooledSpanEmbeddingLayer,
    SpanTypeOnlyEmbeddingLayer,
    SpanAwareEmbeddingLayer,
    PlaceholderCustomEmbeddingLayer,
    process_code_for_spans,
    tokenizer,
    config
)


class PEFTSpanAwareQwen2Model(nn.Module):
    """
    Span-aware Qwen2 model with PEFT LoRA adapters on all transformer layers.
    
    Features:
    - PEFT LoRA adapters on all attention and MLP layers
    - Span-aware embedding layer (fully trainable)
    - Memory-efficient training
    - Compatible with existing training pipeline
    """
    
    def __init__(self, base_model, embedding_type="mean_pooling"):
        super().__init__()
        
        self.base_model = base_model
        self.embedding_type = embedding_type
        self.config = base_model.config
        
        # PEFT wraps the model, so we need to access through base_model attribute
        # PeftModel.base_model -> LoraModel -> model -> Qwen2ForCausalLM -> model -> Qwen2Model
        if hasattr(base_model, 'base_model'):
            # After PEFT wrapping: PeftModel -> base_model (LoraModel) -> model (Qwen2ForCausalLM) -> model (Qwen2Model)
            actual_model = base_model.base_model.model.model
        else:
            # Before PEFT wrapping: Qwen2ForCausalLM -> model (Qwen2Model)
            actual_model = base_model.model
        
        # Store original embedding for potential restoration
        self.original_embed_tokens = actual_model.embed_tokens
        
        # Replace embedding layer with span-aware version
        if embedding_type == "multi_component":
            actual_model.embed_tokens = SpanAwareEmbeddingLayer(base_model.config)
            print("Using Multi-Component AST span-aware embedding layer with PEFT LoRA")
        elif embedding_type == "mean_pooling":
            actual_model.embed_tokens = MeanPooledSpanEmbeddingLayer(base_model.config)
            print("Using Mean Pooling AST span-aware embedding layer with PEFT LoRA")
        elif embedding_type == "mean_pooling_only":
            actual_model.embed_tokens = OnlyMeanPooledSpanEmbeddingLayer(base_model.config)
            print("Using Mean Pooling Only AST span-aware embedding layer with PEFT LoRA")
        elif embedding_type == "span_type_only":
            actual_model.embed_tokens = SpanTypeOnlyEmbeddingLayer(base_model.config)
            print("Using Span Type Only AST span-aware embedding layer with PEFT LoRA")
        else:
            actual_model.embed_tokens = PlaceholderCustomEmbeddingLayer(base_model.config)
            print("Using placeholder embedding layer with PEFT LoRA")
    
    def forward(self, input_ids, attention_mask=None, span_metadata=None, **kwargs):
        """Enhanced forward pass that handles span metadata."""
        # Get the actual embedding layer (handle PEFT wrapping)
        if hasattr(self.base_model, 'base_model'):
            embed_tokens = self.base_model.base_model.model.model.embed_tokens
        else:
            embed_tokens = self.base_model.model.embed_tokens
        
        if self.embedding_type == "span_type_only":
            # Special handling for span type prediction
            embedding_output = embed_tokens(input_ids, span_metadata)
            
            # Extract embeddings and span type logits
            inputs_embeds = embedding_output['embeddings']
            span_type_logits = embedding_output['span_type_logits']
            
            # Forward pass through transformer with embeddings
            outputs = self.base_model(
                input_ids=None,
                attention_mask=attention_mask, 
                inputs_embeds=inputs_embeds,
                **kwargs
            )
            
            # Add span type predictions to outputs
            outputs.span_type_logits = span_type_logits
            
            # Compute span type loss if span_metadata contains target span types
            if span_metadata is not None and 'span_types' in span_metadata:
                span_type_loss = embed_tokens.compute_span_type_loss(
                    span_type_logits, 
                    span_metadata['span_types'], 
                    attention_mask
                )
                outputs.span_type_loss = span_type_loss
            
            return outputs
            
        elif hasattr(embed_tokens, 'forward') and 'span_metadata' in embed_tokens.forward.__code__.co_varnames:
            # Other span-aware embedding layers
            embedding_output = embed_tokens(input_ids, span_metadata)
            
            # Handle both tensor and dict returns
            if isinstance(embedding_output, dict):
                inputs_embeds = embedding_output['embeddings']
            else:
                inputs_embeds = embedding_output
                
            # Forward through base model
            outputs = self.base_model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
            
            # If auxiliary span type logits are provided by the embedding layer, propagate and compute loss
            if isinstance(embedding_output, dict) and 'span_type_logits' in embedding_output:
                outputs.span_type_logits = embedding_output['span_type_logits']
                
                if span_metadata is not None and 'span_types' in span_metadata and hasattr(embed_tokens, 'compute_span_type_loss'):
                    span_type_loss = embed_tokens.compute_span_type_loss(
                        embedding_output['span_type_logits'],
                        span_metadata['span_types'],
                        attention_mask
                    )
                    outputs.span_type_loss = span_type_loss
            
            # Propagate gating information for thorough logging if available
            if isinstance(embedding_output, dict):
                if 'gate' in embedding_output:
                    outputs.gate = embedding_output['gate']
                if 'span_mask' in embedding_output:
                    outputs.span_mask = embedding_output['span_mask']
            
            return outputs
        else:
            # Regular embedding layer
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
    
    def get_embedding_parameters(self):
        """Get embedding layer parameters."""
        if hasattr(self.base_model, 'base_model'):
            return list(self.base_model.base_model.model.model.embed_tokens.parameters())
        else:
            return list(self.base_model.model.embed_tokens.parameters())
    
    def get_trainable_parameters(self):
        """Get all trainable parameters (LoRA + embeddings)."""
        trainable_params = []
        
        # Add all parameters that require gradients
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        return trainable_params
    
    def print_trainable_parameters(self):
        """Print statistics about trainable parameters."""
        trainable_params = []
        lora_params = []
        embedding_params = []
        
        # Count LoRA parameters
        if isinstance(self.base_model, PeftModel):
            for name, param in self.base_model.named_parameters():
                if param.requires_grad:
                    if 'lora' in name.lower():
                        lora_params.append(param)
                    elif 'embed_tokens' in name:
                        embedding_params.append(param)
                    trainable_params.append(param)
        else:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if 'embed_tokens' in name:
                        embedding_params.append(param)
                    trainable_params.append(param)
        
        # Get total parameters
        total_params = list(self.parameters())
        
        lora_count = sum(p.numel() for p in lora_params)
        embedding_count = sum(p.numel() for p in embedding_params)
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in total_params)
        
        print(f"\n📊 PARAMETER STATISTICS:")
        print(f"   LoRA parameters: {lora_count:,}")
        print(f"   Embedding parameters: {embedding_count:,}")
        print(f"   Total trainable: {trainable_count:,}")
        print(f"   Total model: {total_count:,}")
        print(f"   Trainable ratio: {trainable_count/total_count*100:.2f}%")
        
        return {
            'lora_params': lora_count,
            'embedding_params': embedding_count,
            'trainable_params': trainable_count,
            'total_params': total_count,
            'trainable_ratio': trainable_count/total_count*100
        }
    
    def save_pretrained(self, save_directory):
        """Save the model (PEFT adapter + custom embeddings)."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save PEFT model (LoRA adapters)
        if isinstance(self.base_model, PeftModel):
            self.base_model.save_pretrained(save_directory)
        
        # Get embedding layer (handle PEFT wrapping)
        if hasattr(self.base_model, 'base_model'):
            embed_tokens = self.base_model.base_model.model.model.embed_tokens
        else:
            embed_tokens = self.base_model.model.embed_tokens
        
        # Save custom embedding layer separately
        embedding_path = os.path.join(save_directory, "custom_embeddings.pt")
        torch.save({
            'embedding_type': self.embedding_type,
            'embedding_state_dict': embed_tokens.state_dict()
        }, embedding_path)
        
        print(f"✅ Saved PEFT model and custom embeddings to {save_directory}")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
    
    @property
    def device(self):
        """Get device of the model."""
        return next(self.parameters()).device


def create_lora_span_aware_model(
    model_path="/data/home/zhangsj/qwen_coder_1.5b",
    embedding_approach="mean_pooling",
    lora_config=None
):
    """
    Create a LoRA span-aware model using PEFT from the original Qwen2 model.
    
    Args:
        model_path: Path to the original Qwen2 model
        embedding_approach: Type of span-aware embedding to use
        lora_config: LoRA configuration dict
        
    Returns:
        PEFT LoRA span-aware model with copied weights
    """
    print("🚀 CREATING PEFT LORA SPAN-AWARE MODEL")
    print("="*60)
    
    # Default LoRA config
    if lora_config is None:
        lora_config = {
            'rank': 16,
            'alpha': 32,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        }
    
    print(f"Model path: {model_path}")
    print(f"Embedding approach: {embedding_approach}")
    print(f"LoRA config: {lora_config}")
    
    # Load original model
    print("\n📂 Loading original model...")
    original_model = Qwen2ForCausalLM.from_pretrained(model_path)
    model_config = original_model.config
    
    # Store original embedding weights
    original_embedding = original_model.model.embed_tokens.weight.clone()
    
    # Create PEFT LoRA config
    print("\n🔧 Creating PEFT LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config['rank'],
        lora_alpha=lora_config['alpha'],
        lora_dropout=lora_config['dropout'],
        target_modules=lora_config['target_modules'],
        bias="none",
        modules_to_save=None  # We'll handle embeddings separately
    )
    
    # Apply PEFT LoRA to the model
    print("\n🎯 Applying PEFT LoRA adapters...")
    peft_model = get_peft_model(original_model, peft_config)
    peft_model.print_trainable_parameters()
    
    # Wrap in our custom span-aware wrapper
    print("\n🎨 Adding span-aware embedding layer...")
    span_aware_model = PEFTSpanAwareQwen2Model(peft_model, embedding_type=embedding_approach)
    
    # Copy original embedding weights to the new embedding layer
    print("\n📋 Copying embedding weights...")
    
    # Get the actual embedding layer (handle PEFT wrapping)
    if hasattr(span_aware_model.base_model, 'base_model'):
        embed_tokens = span_aware_model.base_model.base_model.model.model.embed_tokens
    else:
        embed_tokens = span_aware_model.base_model.model.embed_tokens
    
    if embedding_approach in ["multi_component", "mean_pooling", "mean_pooling_only", "span_type_only"]:
        # Copy to token embeddings component of span-aware layer
        with torch.no_grad():
            embed_tokens.token_embeddings.weight.copy_(original_embedding)
            print(f"   ✅ Copied token embeddings to {embedding_approach} layer: {original_embedding.shape}")
            print(f"   🎲 {embedding_approach} embedding components initialized randomly (including token adapter)")
    else:
        # Copy to placeholder layer
        with torch.no_grad():
            embed_tokens.embed_tokens.weight.copy_(original_embedding)
            print(f"   ✅ Copied embedding weights to placeholder layer: {original_embedding.shape}")
    
    print(f"\n✅ PEFT LoRA model creation completed")
    
    # Print parameter statistics
    span_aware_model.print_trainable_parameters()
    
    return span_aware_model


def setup_lora_training(
    model,
    learning_rate=5e-4,  # Higher LR for LoRA
    batch_size=4,
    max_length=1024,
    num_epochs=5,
    parquet_files=None,
    log_dir=None,
    trail_name="lora_span_aware",
    span_type_loss_weight=0.2  # Weight for span type loss when combining with cross-entropy
):
    """
    Set up LoRA training with span-aware embeddings.
    
    Args:
        model: PEFT LoRA span-aware model
        learning_rate: Learning rate (typically higher for LoRA)
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_epochs: Number of training epochs
        log_dir: Directory for TensorBoard logs
        trail_name: Name for this training run
        span_type_loss_weight: Weight (λ) for span type loss when combining with cross-entropy loss (default: 0.2)
        
    Returns:
        Training setup dictionary
    """
    print("\n" + "="*60)
    print("🚀 SETTING UP PEFT LORA SPAN-AWARE TRAINING")
    print("="*60)
    
    # Load dataset(s)
    default_parquet = "/data/home/zhangsj/Data/more_big_code_language/python/python_ast_parsed.parquet"
    if parquet_files is None:
        dataset_paths = [default_parquet]
    else:
        dataset_paths = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)

    try:
        print(f"📊 Loading training data from {len(dataset_paths)} parquet file(s)...")
        for p in dataset_paths:
            print(f"   - {p}")
        language_tag = "multi" if len(dataset_paths) > 1 else "python"
        train_dataloader = create_dataloader(
            dataset_paths,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            language=language_tag,
            shuffle=True
        )
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        return None
    
    print(f"✅ Training dataloader created:")
    print(f"   Dataset size: {len(train_dataloader.dataset):,} samples")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches: {len(train_dataloader):,}")
    print(f"   Max sequence length: {max_length}")
    
    # Get trainable parameters
    print(f"\n🔧 Setting up optimizer for PEFT LoRA training...")
    trainable_params = model.get_trainable_parameters()
    
    param_stats = model.print_trainable_parameters()
    
    # Create optimizer (higher learning rate for LoRA)
    from torch.optim import AdamW
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataloader))
    
    print(f"   Optimizer: AdamW (lr={learning_rate}, weight_decay=0.01)")
    print(f"   Scheduler: CosineAnnealingLR")
    print(f"   Trainable parameters: {param_stats['trainable_params']:,} ({param_stats['trainable_ratio']:.2f}%)")
    
    # Set up TensorBoard logging
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"/data/home/zhangsj/qwen_coder_1.5b/tensorboard_logs/peft_lora_{model.embedding_type}_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"\n📊 TensorBoard logging setup:")
    print(f"   Log directory: {log_dir}")
    print(f"   View logs with: tensorboard --logdir {log_dir}")
    
    # Log configuration
    writer.add_text("config/embedding_approach", model.embedding_type)
    writer.add_text("config/learning_rate", str(learning_rate))
    writer.add_text("config/batch_size", str(batch_size))
    writer.add_text("config/max_length", str(max_length))
    writer.add_text("config/num_epochs", str(num_epochs))
    writer.add_text("config/trainable_params", str(param_stats['trainable_params']))
    writer.add_text("config/trainable_ratio", f"{param_stats['trainable_ratio']:.2f}%")
    writer.add_text("config/peft_type", "LoRA")
    writer.add_text("config/span_type_loss_weight", str(span_type_loss_weight))
    
    print(f"\n📊 Loss configuration:")
    print(f"   Span type loss weight (λ): {span_type_loss_weight}")
    if model.embedding_type == "span_type_only":
        print(f"   Total loss = cross_entropy_loss + {span_type_loss_weight} * span_type_loss")
    else:
        print(f"   Total loss = cross_entropy_loss (span_type_loss_weight not used for {model.embedding_type})")
    
    return {
        'model': model,
        'train_dataloader': train_dataloader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'tokenizer': tokenizer,
        'writer': writer,
        'log_dir': log_dir,
        'trail_name': trail_name,
        'param_stats': param_stats,
        'span_type_loss_weight': span_type_loss_weight
    }


def train_lora_model(training_setup, num_epochs=5, save_steps=500, log_steps=50):
    """
    Train the PEFT LoRA span-aware model.
    
    Args:
        training_setup: Dict returned from setup_lora_training()
        num_epochs: Number of training epochs
        save_steps: Save checkpoint every N steps
        log_steps: Log progress every N steps
    """
    print("\n" + "="*60)
    print("🏋️ STARTING PEFT LORA SPAN-AWARE TRAINING")
    print("="*60)
    
    model = training_setup['model']
    train_dataloader = training_setup['train_dataloader']
    optimizer = training_setup['optimizer']
    scheduler = training_setup['scheduler']
    tokenizer = training_setup['tokenizer']
    writer = training_setup['writer']
    trail_name = training_setup['trail_name']
    span_type_loss_weight = training_setup.get('span_type_loss_weight', 0.2)
    
    model.train()
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    total_steps = len(train_dataloader) * num_epochs
    step = 0
    
    print(f"Training configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Save every: {save_steps} steps")
    print(f"   Log every: {log_steps} steps")
    print(f"   Using PEFT LoRA adapters")
    print(f"   Span type loss weight (λ): {span_type_loss_weight}")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_span_type_loss = 0
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"PEFT LoRA Epoch {epoch+1}")):
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # Move span metadata to device
            span_metadata = {}
            for key, value in batch['span_metadata'].items():
                if isinstance(value, torch.Tensor):
                    span_metadata[key] = value.to(model.device)
                else:
                    span_metadata[key] = value
            
            # Forward pass
            if model.embedding_type in ["multi_component", "mean_pooling", "mean_pooling_only", "span_type_only"]:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    span_metadata=span_metadata,
                    labels=input_ids  # Causal LM training
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
            
            # Get base cross-entropy loss
            loss = outputs.loss
            
            # Combine with span type loss if available
            if hasattr(outputs, 'span_type_loss') and outputs.span_type_loss is not None:
                total_loss = loss + span_type_loss_weight * outputs.span_type_loss
            else:
                total_loss = loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += total_loss.item()
            epoch_ce_loss += loss.item()
            
            # Track span type loss
            current_span_type_loss = None
            if hasattr(outputs, 'span_type_loss') and outputs.span_type_loss is not None:
                current_span_type_loss = outputs.span_type_loss.item()
                epoch_span_type_loss += current_span_type_loss
            
            epoch_steps += 1
            step += 1
            
            # TensorBoard logging
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('train/total_loss', total_loss.item(), step)
            writer.add_scalar('train/cross_entropy_loss', loss.item(), step)
            
            # Always log span type loss (log 0 if not available for consistency)
            if current_span_type_loss is not None:
                writer.add_scalar('train/span_type_loss', current_span_type_loss, step)
                writer.add_scalar('train/span_type_loss_weighted', (span_type_loss_weight * current_span_type_loss), step)
            else:
                writer.add_scalar('train/span_type_loss', 0.0, step)
                writer.add_scalar('train/span_type_loss_weighted', 0.0, step)
            
            writer.add_scalar('train/learning_rate', current_lr, step)
            
            # Log embedding weights based on embedding type
            if step % (log_steps * 2) == 0:
                # Get embedding layer (handle PEFT wrapping)
                if hasattr(model.base_model, 'base_model'):
                    embed_tokens = model.base_model.base_model.model.model.embed_tokens
                else:
                    embed_tokens = model.base_model.model.embed_tokens
                
                if model.embedding_type == "mean_pooling":
                    # Thorough gating statistics
                    gate_tensor = None
                    span_mask_tensor = None
                    if hasattr(outputs, 'gate'):
                        gate_tensor = outputs.gate.detach()
                    elif hasattr(embed_tokens, 'last_gate'):
                        gate_tensor = embed_tokens.last_gate
                    if hasattr(outputs, 'span_mask'):
                        span_mask_tensor = outputs.span_mask.detach()
                    elif hasattr(embed_tokens, 'last_span_mask'):
                        span_mask_tensor = embed_tokens.last_span_mask
                    
                    if gate_tensor is not None:
                        # Reduce over hidden dimension to get per-token gate strength
                        gate_token = gate_tensor.mean(dim=-1)  # [B, L]
                        attn_mask_bool = attention_mask.bool()
                        if span_mask_tensor is not None:
                            span_mask_bool = (span_mask_tensor > 0.5)
                        else:
                            # Fallback: consider all tokens as span tokens to avoid division by zero
                            span_mask_bool = torch.ones_like(attn_mask_bool, dtype=torch.bool)
                        
                        # Valid positions
                        valid_all = attn_mask_bool
                        valid_span = attn_mask_bool & span_mask_bool
                        
                        # Flatten
                        gate_all = gate_token[valid_all]
                        writer.add_scalar('gate/mean_all', gate_all.mean().item(), step)
                        writer.add_scalar('gate/std_all', gate_all.float().std(unbiased=False).item(), step)
                        writer.add_histogram('gate/values_all', gate_all, step)
                        
                        # Span-only stats
                        if valid_span.any():
                            gate_span = gate_token[valid_span]
                            writer.add_scalar('gate/coverage', valid_span.float().mean().item(), step)
                            writer.add_scalar('gate/mean_span', gate_span.mean().item(), step)
                            writer.add_scalar('gate/std_span', gate_span.float().std(unbiased=False).item(), step)
                            writer.add_scalar('gate/min_span', gate_span.min().item(), step)
                            writer.add_scalar('gate/max_span', gate_span.max().item(), step)
                            
                            # Percentiles
                            try:
                                qs = torch.tensor([0.1, 0.5, 0.9], device=gate_span.device, dtype=gate_span.dtype)
                                p = torch.quantile(gate_span, qs)
                                writer.add_scalar('gate/p10_span', p[0].item(), step)
                                writer.add_scalar('gate/p50_span', p[1].item(), step)
                                writer.add_scalar('gate/p90_span', p[2].item(), step)
                            except Exception:
                                pass
                            
                            # Extremes fractions
                            writer.add_scalar('gate/fraction_span_le_0.1', (gate_span <= 0.1).float().mean().item(), step)
                            writer.add_scalar('gate/fraction_span_ge_0.9', (gate_span >= 0.9).float().mean().item(), step)
                            
                            writer.add_histogram('gate/values_span', gate_span, step)
                            
                            # Per-example mean over span tokens
                            # Avoid divide by zero
                            mask_ex = (span_mask_bool & attn_mask_bool).float()
                            num_per_ex = mask_ex.sum(dim=1).clamp(min=1.0)
                            sum_per_ex = (gate_token * mask_ex).sum(dim=1)
                            mean_per_ex = sum_per_ex / num_per_ex
                            writer.add_histogram('gate/mean_per_example_span', mean_per_ex, step)
                            
                            # Boundary-wise gating if boundaries available
                            if span_metadata and 'boundaries' in span_metadata:
                                boundaries = span_metadata['boundaries']  # [B, L]
                                bmask = attn_mask_bool & span_mask_bool
                                for btype, bname in [(0, 'middle'), (1, 'start'), (2, 'end'), (3, 'single')]:
                                    sel = (boundaries == btype) & bmask
                                    if sel.any():
                                        writer.add_scalar(f'gate/by_boundary/{bname}', gate_token[sel].mean().item(), step)
                                    else:
                                        writer.add_scalar(f'gate/by_boundary/{bname}', 0.0, step)
                        else:
                            # No span positions in this batch (rare)
                            writer.add_scalar('gate/coverage', 0.0, step)
                            writer.add_scalar('gate/mean_span', 0.0, step)
                            writer.add_scalar('gate/std_span', 0.0, step)
                            writer.add_scalar('gate/min_span', 0.0, step)
                            writer.add_scalar('gate/max_span', 0.0, step)
                            writer.add_scalar('gate/p10_span', 0.0, step)
                            writer.add_scalar('gate/p50_span', 0.0, step)
                            writer.add_scalar('gate/p90_span', 0.0, step)
                            writer.add_scalar('gate/fraction_span_le_0.1', 0.0, step)
                            writer.add_scalar('gate/fraction_span_ge_0.9', 0.0, step)
                    
                    # Log adapter statistics (new with token adapter)
                    if hasattr(embed_tokens, 'token_adapter'):
                        adapter_weight_norm = torch.norm(embed_tokens.token_adapter[0].weight)
                        adapter_output_norm = torch.norm(embed_tokens.token_adapter[2].weight)
                        writer.add_scalar('embeddings/adapter_weight_norm', adapter_weight_norm.item(), step)
                        writer.add_scalar('embeddings/adapter_output_norm', adapter_output_norm.item(), step)
                        
                elif model.embedding_type == "mean_pooling_only":
                    # Log layer norm statistics for mean_pooling_only
                    layer_norm_weight = torch.norm(embed_tokens.layer_norm.weight)
                    writer.add_scalar('embeddings/layer_norm_weight_norm', layer_norm_weight.item(), step)
                elif model.embedding_type == "span_type_only":
                    # Log span type prediction head and layer norm statistics
                    layer_norm_weight = torch.norm(embed_tokens.layer_norm.weight)
                    writer.add_scalar('embeddings/layer_norm_weight_norm', layer_norm_weight.item(), step)
                    
                    # Log span type head weights
                    for i, layer in enumerate(embed_tokens.span_type_head):
                        if hasattr(layer, 'weight'):
                            head_weight_norm = torch.norm(layer.weight)
                            writer.add_scalar(f'embeddings/span_type_head_layer_{i}_weight_norm', head_weight_norm.item(), step)
            
            # Console logging
            if step % log_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                loss_str = f"Total: {total_loss.item():.4f} (CE: {loss.item():.4f}"
                if current_span_type_loss is not None:
                    loss_str += f", Span: {current_span_type_loss:.4f})"
                else:
                    loss_str += ")"
                print(f"   Step {step:,}/{total_steps:,} | {loss_str} | Avg: {avg_loss:.4f} | LR: {current_lr:.2e}")
            
            # Save checkpoint
            if step % save_steps == 0:
                checkpoint_path = f"/data/home/zhangsj/qwen_coder_1.5b/checkpoints_peft_lora_{trail_name}/checkpoint_step_{step}"
                os.makedirs(checkpoint_path, exist_ok=True)
                
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                # Save training state
                checkpoint_state = {
                    'step': step,
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'total_loss': total_loss.item(),
                    'cross_entropy_loss': loss.item(),
                    'embedding_approach': model.embedding_type,
                    'best_loss': best_loss,
                    'span_type_loss_weight': span_type_loss_weight,
                    'span_type_loss': current_span_type_loss if current_span_type_loss is not None else 0.0
                }
                torch.save(checkpoint_state, f"{checkpoint_path}/training_state.pt")
                
                print(f"   💾 PEFT LoRA checkpoint saved: {checkpoint_path}")
                writer.add_text('checkpoints/saved', f"Step {step}: {checkpoint_path}", step)
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        avg_epoch_ce_loss = epoch_ce_loss / epoch_steps if epoch_steps > 0 else 0.0
        avg_epoch_span_type_loss = epoch_span_type_loss / epoch_steps if epoch_steps > 0 else 0.0
        
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
        writer.add_scalar('train/epoch_cross_entropy_loss', avg_epoch_ce_loss, epoch)
        writer.add_scalar('train/epoch_span_type_loss', avg_epoch_span_type_loss, epoch)
        
        epoch_summary = f"   Epoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f} (CE: {avg_epoch_ce_loss:.4f}"
        if avg_epoch_span_type_loss > 0:
            epoch_summary += f", Span: {avg_epoch_span_type_loss:.4f})"
        else:
            epoch_summary += ")"
        print(epoch_summary)
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = f"/data/home/zhangsj/qwen_coder_1.5b/best_peft_lora_span_aware_{trail_name}"
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"   🏆 New best PEFT LoRA model saved: {best_model_path} (loss: {best_loss:.4f})")
    
    # Save final model
    final_path = f"/data/home/zhangsj/qwen_coder_1.5b/finetuned_peft_lora_span_aware_{trail_name}"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    writer.add_text('training/status', 'COMPLETED', step)
    # Always log final span_type_loss (0 if not available)
    final_span_type_loss = 0.0
    try:
        writer.add_scalar('training/final_total_loss', total_loss.item(), step)
        writer.add_scalar('training/final_cross_entropy_loss', loss.item(), step)
        if 'outputs' in locals() and hasattr(outputs, 'span_type_loss') and outputs.span_type_loss is not None:
            final_span_type_loss = outputs.span_type_loss.item()
        writer.add_scalar('training/final_span_type_loss', final_span_type_loss, step)
    except:
        pass
    writer.add_scalar('training/best_loss', best_loss, step)
    
    # Log final embedding weights
    # Get embedding layer (handle PEFT wrapping)
    if hasattr(model.base_model, 'base_model'):
        embed_tokens = model.base_model.base_model.model.model.embed_tokens
    else:
        embed_tokens = model.base_model.model.embed_tokens
    
    if model.embedding_type == "mean_pooling":
        # Final gating summary (if available)
        final_gate = None
        if hasattr(embed_tokens, 'last_gate'):
            final_gate = embed_tokens.last_gate
        if final_gate is not None:
            final_gate_token = final_gate.mean(dim=-1)  # [B, L]
            writer.add_scalar('gate/final_mean_all', final_gate_token.mean().item(), step)
            print(f"   📊 Final gating mean (all tokens): {final_gate_token.mean().item():.3f}")
        # Log final adapter weights if present
        if hasattr(embed_tokens, 'token_adapter'):
            final_adapter_norm = torch.norm(embed_tokens.token_adapter[0].weight)
            writer.add_scalar('embeddings/final_adapter_norm', final_adapter_norm.item(), step)
            print(f"   📊 Final adapter weight norm: {final_adapter_norm.item():.3f}")
            
    elif model.embedding_type == "mean_pooling_only":
        final_layer_norm = torch.norm(embed_tokens.layer_norm.weight)
        writer.add_scalar('embeddings/final_layer_norm_weight_norm', final_layer_norm.item(), step)
        print(f"   📊 Final layer norm weight norm: {final_layer_norm.item():.3f}")
    elif model.embedding_type == "span_type_only":
        final_layer_norm = torch.norm(embed_tokens.layer_norm.weight)
        writer.add_scalar('embeddings/final_layer_norm_weight_norm', final_layer_norm.item(), step)
        
        # Log final span type head weights
        for i, layer in enumerate(embed_tokens.span_type_head):
            if hasattr(layer, 'weight'):
                head_weight_norm = torch.norm(layer.weight)
                writer.add_scalar(f'embeddings/final_span_type_head_layer_{i}_weight_norm', head_weight_norm.item(), step)
        
        print(f"   📊 Final span type prediction model - Layer norm: {final_layer_norm.item():.3f}")
    
    writer.close()
    
    print(f"\n✅ PEFT LORA TRAINING COMPLETED!")
    print(f"   Final model: {final_path}")
    print(f"   Best model: /data/home/zhangsj/qwen_coder_1.5b/best_peft_lora_span_aware_{trail_name}")
    print(f"   Total steps: {step:,}")
    # Note: loss and total_loss from last iteration
    try:
        print(f"   Final total loss: {total_loss.item():.4f}")
        print(f"   Final cross-entropy loss: {loss.item():.4f}")
        print(f"   Final span type loss: {final_span_type_loss:.4f}")
    except:
        pass
    print(f"   Best loss: {best_loss:.4f}")
    
    return model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🎯 PEFT LORA SPAN-AWARE QWEN2 MODEL")
    print("="*60)
    
    # Configuration
    embedding_approach = "span_type_only"  # or "multi_component", "mean_pooling", "mean_pooling_only", "placeholder"
    trail_name = "peft_lora_v1_span_type_only"
    
    lora_config = {
        'rank': 16,        # LoRA rank (lower = fewer parameters)
        'alpha': 32,       # LoRA scaling factor
        'dropout': 0.1,    # LoRA dropout
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    }
    
    print(f"Embedding approach: {embedding_approach}")
    print(f"Trail name: {trail_name}")
    print(f"LoRA configuration: {lora_config}")
    
    # Create PEFT LoRA model
    lora_model = create_lora_span_aware_model(
        model_path="/data/home/zhangsj/qwen_coder_1.5b",
        embedding_approach=embedding_approach,
        lora_config=lora_config
    )
    
    # Move to GPU
    lora_model = lora_model.cuda()
    
    # Setup training
    training_setup = setup_lora_training(
        model=lora_model,
        learning_rate=5e-4,  # Higher LR for LoRA
        batch_size=4,        # Can use larger batch due to memory efficiency
        max_length=1024,
        num_epochs=5,
        trail_name=trail_name
    )
    
    if training_setup is not None:
        # Start training
        trained_model = train_lora_model(
            training_setup,
            num_epochs=5,
            save_steps=5000,
            log_steps=50
        )
        print("🎉 PEFT LoRA training completed successfully!")
    else:
        print("❌ Failed to set up PEFT LoRA training. Check data availability.")
