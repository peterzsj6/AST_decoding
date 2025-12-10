"""
BLT Generation Training Script

Key difference from focused_training.py:
- Train local decoder to generate NEXT span (not reconstruct current span)
- Global hidden at position t → Generate span at position t+1
- This enables true autoregressive generation at inference time

Based on Meta's Byte Latent Transformer approach.
"""

import os
if 'LOCAL_RANK' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU 1 has most free memory (~56GB)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import math
import argparse
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from blt_adapter_model import (
    BLTAdapterModel,
    create_blt_adapter_model,
    SPAN_TYPE_TO_ID,
)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class GenerationTrainingDataset(Dataset):
    """
    Dataset for BLT-style generation training.
    
    For each sample, we create (context, next_span) pairs:
    - context: tokens up to span boundary
    - next_span: tokens of the next span (target for local decoder)
    """
    def __init__(
        self,
        parquet_file_path: str,
        tokenizer,
        max_length: int = 512,
        max_span_len: int = 64,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_span_len = max_span_len
        self.span_type_to_id = SPAN_TYPE_TO_ID
        
        self.df = pd.read_parquet(parquet_file_path)
        
        # Filter valid rows
        content_filter = (self.df['content'].notna()) & (self.df['content'].str.strip() != '')
        ast_span_filter = (self.df['AST_span'].notna()) & (self.df['AST_span'].str.len() > 2)
        self.df = self.df[content_filter & ast_span_filter].reset_index(drop=True)
        
        print(f"[Dataset] Loaded {len(self.df)} samples for generation training")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        content = row['content']
        
        # Tokenize
        enc = self.tokenizer(
            content,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        
        # Parse AST spans
        try:
            ast_spans = json.loads(row['AST_span']) if row['AST_span'] else []
        except:
            ast_spans = []
        
        # Build span pairs for generation training
        span_pairs = self._build_generation_pairs(input_ids, ast_spans)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'span_pairs': span_pairs,  # List of (context_end_pos, target_span_tokens)
        }

    def _build_generation_pairs(self, input_ids: torch.Tensor, ast_spans: List[Dict]) -> List[Dict]:
        """
        Build (context_position, next_span_tokens) pairs.
        
        For each span boundary, we want:
        - context_end_pos: position in sequence where context ends
        - target_tokens: the tokens of the NEXT span to generate
        """
        seq_len = int(input_ids.shape[0])
        pairs = []
        
        # Sort spans by start position
        valid_spans = []
        for sp in ast_spans:
            if not isinstance(sp, dict):
                continue
            token_indices = sp.get('token_indices', [])
            if not token_indices:
                continue
            indices = [i for i in token_indices if 0 <= i < seq_len]
            if indices:
                valid_spans.append({
                    'start': min(indices),
                    'end': max(indices),
                    'indices': indices,
                    'type': sp.get('type', 'unknown')
                })
        
        valid_spans.sort(key=lambda x: x['start'])
        
        # Create pairs: (position before span, span tokens)
        for i, span in enumerate(valid_spans):
            if span['start'] == 0:
                continue  # Skip first span (no context before it)
            
            # Context ends at the position before this span starts
            context_end_pos = span['start'] - 1
            
            # Target is the tokens of this span
            target_tokens = input_ids[span['indices']].tolist()
            
            # Limit span length
            if len(target_tokens) > self.max_span_len:
                target_tokens = target_tokens[:self.max_span_len]
            
            if len(target_tokens) > 0:
                pairs.append({
                    'context_end_pos': context_end_pos,
                    'target_tokens': target_tokens,
                    'span_type': self.span_type_to_id.get(span['type'], 0)
                })
        
        return pairs


def collate_generation_batch(batch):
    """Collate function for generation training."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    span_pairs = [item['span_pairs'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'span_pairs': span_pairs,
    }


def compute_generation_loss(
    model: BLTAdapterModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    span_pairs: List[List[Dict]],
    device: torch.device,
    dtype: torch.dtype,
    max_targets_per_sample: int,
    include_boundary_loss: bool,
    boundary_loss_weight: float,
    include_global_ce_loss: bool = False,
    global_ce_weight: float = 0.5,
    train_global: bool = False,
) -> Tuple[Optional[torch.Tensor], Dict]:
    """
    Compute BLT-style generation loss with optional auxiliary losses.
    
    Losses:
    - gen_loss: Cross-entropy for next-span prediction (main objective)
    - boundary_loss: Keep boundary detection sharp (optional)
    - global_ce_loss: Train global hidden states to encode "what comes next" (optional)
    
    For each span pair:
    1. Get global hidden at context_end_pos
    2. Use local decoder to generate target tokens
    3. Compute cross-entropy loss
    """
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    
    # Forward through global transformer
    # If training global, compute with gradients; otherwise no_grad
    if train_global and include_global_ce_loss:
        # With gradients for global transformer
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # For CE loss computation
            output_hidden_states=True,
        )
        global_ce_loss_value = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
    else:
        # No gradients for global transformer (frozen)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        global_ce_loss_value = None
    
    # Get last hidden states [B, L, H]
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        last_hidden = outputs.hidden_states[-1]
        if not train_global:
            last_hidden = last_hidden.detach()
        del outputs
        torch.cuda.empty_cache()
    else:
        del outputs
        return None, {}
    
    # Collect generation targets
    all_losses = []
    total_tokens = 0
    
    for b in range(batch_size):
        pairs = span_pairs[b]
        if not pairs:
            continue
        
        # Limit pairs per sample for memory
        pairs = pairs[:max_targets_per_sample]
        
        for pair in pairs:
            context_pos = pair['context_end_pos']
            target_tokens = pair['target_tokens']
            
            if context_pos >= last_hidden.shape[1]:
                continue
            
            # Get global hidden at context end position
            global_hidden = last_hidden[b, context_pos, :]  # [H]
            
            # Predict latent for local decoder (or use global hidden directly)
            if hasattr(model, 'latent_from_global'):
                pred_latent = model.latent_from_global(global_hidden.unsqueeze(0))  # [1, H]
            else:
                pred_latent = global_hidden.unsqueeze(0)
            
            # Combine with global hidden if model has latent_combine
            if hasattr(model, 'latent_combine'):
                combined = model.latent_combine(
                    torch.cat([pred_latent, global_hidden.unsqueeze(0)], dim=-1)
                )
            else:
                combined = pred_latent
            
            # Project for local decoder conditioning
            cond = model.latent_proj(combined).unsqueeze(1)  # [1, 1, H]
            
            # Teacher-forced local decoding
            target_tensor = torch.tensor(target_tokens, device=device, dtype=torch.long)
            target_len = len(target_tokens)
            
            # Input: BOS + target[:-1]
            bos_id = getattr(model, 'node_bos_id', model.config.bos_token_id or 0)
            if target_len == 1:
                decoder_input = torch.tensor([[bos_id]], device=device, dtype=torch.long)
            else:
                decoder_input = torch.cat([
                    torch.tensor([[bos_id]], device=device, dtype=torch.long),
                    target_tensor[:-1].unsqueeze(0)
                ], dim=1)
            
            # Local decoder forward
            x = model.local_token_embed(decoder_input)  # [1, T, H]
            h = model.local_transformer(x, cond)  # [1, T, H]
            
            # Apply residual connection if available
            if hasattr(model, 'global_residual_gate') and hasattr(model, 'global_residual_scale'):
                global_expanded = global_hidden.unsqueeze(0).unsqueeze(0).expand_as(h)
                gate_input = torch.cat([h, global_expanded], dim=-1)
                gate = model.global_residual_gate(gate_input)
                h = (1 - gate) * h + gate * model.global_residual_scale * global_expanded
            
            # Output logits
            logits = model.local_out_proj(h)  # [1, T, V]
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_tensor.view(-1),
                reduction='sum'
            )
            
            all_losses.append(loss)
            total_tokens += target_len
    
    if not all_losses:
        return None, {'num_spans': 0, 'num_tokens': 0}
    
    # Average generation loss over tokens
    gen_loss = sum(all_losses) / max(total_tokens, 1)
    total_loss = gen_loss
    
    stats = {
        'num_spans': len(all_losses),
        'num_tokens': total_tokens,
        'avg_span_len': total_tokens / len(all_losses) if all_losses else 0,
        'gen_loss': float(gen_loss.item()),
    }
    
    # Optional: Add boundary loss to maintain boundary detection capability
    if include_boundary_loss and hasattr(model, 'boundary_head'):
        # Collect boundary positions from span_pairs
        boundary_losses = []
        for b in range(batch_size):
            pairs = span_pairs[b]
            if not pairs:
                continue
            for pair in pairs[:max_targets_per_sample]:
                context_pos = pair['context_end_pos']
                if context_pos >= last_hidden.shape[1] - 1:
                    continue
                # Position after context_end is the span start (boundary)
                boundary_pos = context_pos + 1
                if boundary_pos < last_hidden.shape[1]:
                    # Boundary head predicts: is this a span boundary?
                    hidden_at_boundary = last_hidden[b, boundary_pos, :]
                    boundary_logits = model.boundary_head(hidden_at_boundary.unsqueeze(0))
                    # Target: 1 = is boundary (span start)
                    target = torch.tensor([1], device=device, dtype=torch.long)
                    boundary_losses.append(F.cross_entropy(boundary_logits, target))
        
        if boundary_losses:
            boundary_loss = sum(boundary_losses) / len(boundary_losses)
            total_loss = total_loss + boundary_loss_weight * boundary_loss
            stats['boundary_loss'] = float(boundary_loss.item())
    
    # Add global CE loss if available (trains global hidden states)
    if include_global_ce_loss and global_ce_loss_value is not None:
        total_loss = total_loss + global_ce_weight * global_ce_loss_value
        stats['global_ce_loss'] = float(global_ce_loss_value.item())
    
    return total_loss, stats


def train_generation():
    """
    Train local decoder for generation (BLT-style).
    """
    parser = argparse.ArgumentParser(description="BLT Generation Training")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding")
    parser.add_argument("--parquet", type=str, default="/data/home/zhangsj/Data/more_big_code_language/python/python_ast_parsed.parquet")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load from existing checkpoint (e.g., focused_training epoch_6)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--trial_name", type=str, default="generation_training")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)  # Reduced for memory
    parser.add_argument("--lr", type=float, default=1e-5)  # Lower LR for fine-tuning
    parser.add_argument("--max_length", type=int, default=256)  # Reduced for memory
    parser.add_argument("--max_span_len", type=int, default=48)  # Reduced for memory
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # Increased to compensate
    parser.add_argument("--max_targets_per_sample", type=int, default=32)  # Reduced for memory
    
    # Model params
    parser.add_argument("--local_num_layers", type=int, default=2)
    parser.add_argument("--num_node_types", type=int, default=113)
    
    # LR scheduler
    parser.add_argument("--lr_warmup_steps", type=int, default=200)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    
    # What to train
    parser.add_argument("--train_global", action="store_true", 
                        help="Also train global transformer (expensive but better)")
    parser.add_argument("--train_latent_from_global", action="store_true", default=True,
                        help="Train latent_from_global projection")
    
    # Auxiliary losses
    parser.add_argument("--include_boundary_loss", action="store_true", default=True,
                        help="Include boundary loss to maintain boundary detection")
    parser.add_argument("--boundary_loss_weight", type=float, default=0.3,
                        help="Weight for boundary loss")
    parser.add_argument("--include_global_ce_loss", action="store_true", default=False,
                        help="Include global CE loss to train hidden states (requires --train_global)")
    parser.add_argument("--global_ce_weight", type=float, default=0.5,
                        help="Weight for global CE loss")
    
    args = parser.parse_args()
    
    # Validation: global_ce_loss requires train_global
    if args.include_global_ce_loss and not args.train_global:
        print("[WARNING] --include_global_ce_loss requires --train_global to have effect!")
        print("[WARNING] Enabling --train_global automatically...")
        args.train_global = True
    
    # Setup directories
    if not args.output_dir:
        args.output_dir = f"/data/home/zhangsj/AST_decoding/checkpoints/blt_generation/{args.trial_name}"
    if not args.log_dir:
        args.log_dir = f"/data/home/zhangsj/AST_decoding/tensorboard_logs/{args.trial_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Dataset
    dataset = GenerationTrainingDataset(
        args.parquet,
        tokenizer,
        max_length=args.max_length,
        max_span_len=args.max_span_len,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_generation_batch,
        drop_last=True,
    )
    
    # Load model
    if args.checkpoint and os.path.isdir(args.checkpoint):
        print(f"[setup] Loading from checkpoint: {args.checkpoint}")
        model = create_blt_adapter_model(
            args.model_path,
            local_num_layers=args.local_num_layers,
            max_node_length=args.max_span_len,
            num_node_types=args.num_node_types,
        )
        bin_path = os.path.join(args.checkpoint, "pytorch_model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"[setup] Loaded weights from {bin_path}")
    else:
        model = create_blt_adapter_model(
            args.model_path,
            local_num_layers=args.local_num_layers,
            max_node_length=args.max_span_len,
            num_node_types=args.num_node_types,
        )
    
    model = model.to(device=device, dtype=dtype)
    
    # Note: We don't enable gradient checkpointing for global model since it's frozen
    # (would cause "None of the inputs have requires_grad=True" warning)
    # Memory is saved by using torch.no_grad() in compute_generation_loss instead
    
    # Clear cache after loading
    torch.cuda.empty_cache()
    
    # Setup trainable parameters
    trainable_params = []
    
    # Always train local decoder
    for name in ['latent_proj', 'local_transformer', 'latent_combine', 
                 'global_residual_gate', 'global_residual_scale']:
        if hasattr(model, name):
            module = getattr(model, name)
            if isinstance(module, nn.Parameter):
                module.requires_grad = True
                trainable_params.append(module)
            else:
                for p in module.parameters():
                    p.requires_grad = True
                    trainable_params.append(p)
    
    # Optionally train latent_from_global
    if args.train_latent_from_global and hasattr(model, 'latent_from_global'):
        for p in model.latent_from_global.parameters():
            p.requires_grad = True
            trainable_params.append(p)
    
    # Optionally train global transformer (expensive!)
    if args.train_global:
        print("[setup] WARNING: Training global transformer - this is expensive!")
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for p in model.model.layers.parameters():
                p.requires_grad = True
                trainable_params.append(p)
    else:
        # Freeze global
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for p in model.model.layers.parameters():
                p.requires_grad = False
    
    # Freeze embeddings (large, don't need to train)
    if hasattr(model, 'local_token_embed'):
        for p in model.local_token_embed.parameters():
            p.requires_grad = False
    if hasattr(model, 'local_out_proj'):
        for p in model.local_out_proj.parameters():
            p.requires_grad = False
    
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[setup] Trainable: {trainable_count:,} / {total_count:,} parameters")
    
    # Optimizer and scheduler
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = args.epochs * steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    
    # Logging
    writer = SummaryWriter(args.log_dir)
    writer.add_text("config", str(vars(args)))
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            span_pairs = batch['span_pairs']
            
            # Compute generation loss (with optional auxiliary losses)
            loss, stats = compute_generation_loss(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_pairs=span_pairs,
                device=device,
                dtype=dtype,
                max_targets_per_sample=args.max_targets_per_sample,
                include_boundary_loss=args.include_boundary_loss,
                boundary_loss_weight=args.boundary_loss_weight,
                include_global_ce_loss=args.include_global_ce_loss,
                global_ce_weight=args.global_ce_weight,
                train_global=args.train_global,
            )
            
            if loss is None:
                continue
            
            epoch_losses.append(float(loss.item()))
            
            # Backward
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Clear cache periodically to prevent fragmentation
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Optimizer step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                # Logging
                if global_step % 50 == 0:
                    avg_loss = np.mean(epoch_losses[-100:]) if epoch_losses else 0
                    lr = optimizer.param_groups[0]['lr']
                    
                    writer.add_scalar("loss/total", avg_loss, global_step)
                    writer.add_scalar("lr", lr, global_step)
                    writer.add_scalar("stats/num_spans", stats['num_spans'], global_step)
                    
                    # Log individual loss components
                    if 'gen_loss' in stats:
                        writer.add_scalar("loss/generation", stats['gen_loss'], global_step)
                    if 'boundary_loss' in stats:
                        writer.add_scalar("loss/boundary", stats['boundary_loss'], global_step)
                    if 'global_ce_loss' in stats:
                        writer.add_scalar("loss/global_ce", stats['global_ce_loss'], global_step)
                    
                    # Print progress
                    msg = f"epoch {epoch+1} step {global_step} lr {lr:.2e} | total {avg_loss:.4f}"
                    if 'gen_loss' in stats:
                        msg += f" | gen {stats['gen_loss']:.4f}"
                    if 'boundary_loss' in stats:
                        msg += f" | bnd {stats['boundary_loss']:.4f}"
                    if 'global_ce_loss' in stats:
                        msg += f" | lm {stats['global_ce_loss']:.4f}"
                    msg += f" | spans {stats['num_spans']} | avg_len {stats['avg_span_len']:.1f}"
                    print(msg)
        
        # Epoch summary
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
        print(f"\n[Epoch {epoch+1}] Avg generation loss: {avg_epoch_loss:.4f}\n")
        writer.add_scalar("epoch/loss", avg_epoch_loss, epoch)
        
        # Save checkpoint
        save_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        model.config.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved to {save_dir}")
    
    writer.close()
    print(f"\nGeneration training complete!")


if __name__ == "__main__":
    train_generation()

