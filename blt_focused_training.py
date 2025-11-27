"""
Focused BLT Adapter Training Script

Goal: Train span-by-span code generation with minimal losses:
  - node_recon: Primary loss for generating correct code spans
  - boundary: For inference-time span boundary detection
  - latent_mse: For predicting span latent from global hidden state

No LM CE, KL, or InfoNCE losses - just the essentials for span decoding.
"""

from typing import Dict, List, Optional, Tuple
import os
if 'LOCAL_RANK' not in os.environ:
    # Use GPU 0 by default (change if needed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import datetime
import math
import argparse

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

# Import the model components
from blt_adapter_model import (
    BLTAdapterModel, 
    create_blt_adapter_model,
    SPAN_TYPE_TO_ID,
    ID_TO_SPAN_TYPE,
    TEXTUAL_SPAN_TYPES,
)


class FocusedPythonASTSpanDataset(Dataset):
    """
    Dataset that filters spans for better training signal:
    - Optionally filter out very short spans
    - Optionally filter out certain span types (e.g., pure punctuation)
    """
    def __init__(
        self, 
        parquet_file_path: str, 
        tokenizer, 
        max_length: int = 512,
        min_span_len: int = 1,
        max_span_len: int = 64,
        filter_trivial_types: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len
        self.filter_trivial_types = filter_trivial_types
        
        # Span type vocabulary
        self.span_type_to_id = SPAN_TYPE_TO_ID
        self.id_to_span_type = ID_TO_SPAN_TYPE
        self.textual_span_types = TEXTUAL_SPAN_TYPES
        self.num_node_types = len(self.span_type_to_id)
        
        # Types to filter if filter_trivial_types is True
        self.trivial_types = {'punctuation', 'operator', '=', 'in', 'is', 'is not', 'not in'}
        
        if not os.path.exists(parquet_file_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_file_path}")
        
        self.df = pd.read_parquet(parquet_file_path)
        
        # Filter rows
        content_filter = (self.df['content'].notna()) & (self.df['content'].str.strip() != '')
        if 'error' in self.df.columns:
            self.df = self.df[content_filter & (~self.df['error'].notna())]
        else:
            self.df = self.df[content_filter]
        ast_span_filter = (self.df['AST_span'].notna()) & (self.df['AST_span'].str.len() > 2)
        self.df = self.df[ast_span_filter]
        
        print(f"[Dataset] Loaded {len(self.df)} samples")
        print(f"[Dataset] min_span_len={min_span_len}, max_span_len={max_span_len}, filter_trivial={filter_trivial_types}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        content = row['content']
        
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
        except Exception:
            ast_spans = []
        
        span_meta = self._build_span_meta(input_ids, ast_spans)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'span_metadata': span_meta,
            'original_content': content
        }

    def _build_span_meta(self, input_ids: torch.Tensor, ast_spans: List[Dict]) -> Dict:
        seq_len = int(input_ids.shape[0])
        span_types = np.zeros(seq_len, dtype=np.int64)
        positions = np.zeros(seq_len, dtype=np.int64)
        boundaries = np.zeros(seq_len, dtype=np.int64)
        processed = []
        
        for sp in ast_spans:
            if not isinstance(sp, dict):
                continue
            
            token_indices = sp.get('token_indices', [])
            if not token_indices:
                continue
            
            token_indices = np.array(token_indices, dtype=np.int64)
            valid = token_indices[(token_indices >= 0) & (token_indices < seq_len)]
            if valid.size == 0:
                continue
            
            span_len = len(valid)
            
            # Filter by span length
            if span_len < self.min_span_len or span_len > self.max_span_len:
                continue
            
            span_type_str = str(sp.get('type', 'unknown'))
            
            # Optionally filter trivial types
            if self.filter_trivial_types and span_type_str in self.trivial_types:
                continue
            
            span_type_id = int(self.span_type_to_id.get(span_type_str, self.span_type_to_id['unknown']))
            
            # Textual spans => split into single-token spans
            if span_type_str in self.textual_span_types:
                for t in valid.tolist():
                    span_types[t] = span_type_id
                    positions[t] = 0
                    boundaries[t] = 3  # single
                    processed.append({'token_indices': np.array([t], dtype=np.int64), 'span_type_id': span_type_id})
            else:
                for pos, t in enumerate(valid):
                    span_types[t] = span_type_id
                    positions[t] = min(pos, 31)
                    if valid.size == 1:
                        boundaries[t] = 3
                    elif pos == 0:
                        boundaries[t] = 1
                    elif pos == valid.size - 1:
                        boundaries[t] = 2
                    else:
                        boundaries[t] = 0
                processed.append({'token_indices': valid, 'span_type_id': span_type_id})
        
        return {
            'span_types': torch.tensor(span_types, dtype=torch.long),
            'positions': torch.tensor(positions, dtype=torch.long),
            'boundaries': torch.tensor(boundaries, dtype=torch.long),
            'raw_spans': processed
        }


def collate_fn(batch):
    """Custom collate for variable-length raw_spans."""
    input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
    span_types = torch.stack([item['span_metadata']['span_types'] for item in batch], dim=0)
    positions = torch.stack([item['span_metadata']['positions'] for item in batch], dim=0)
    boundaries = torch.stack([item['span_metadata']['boundaries'] for item in batch], dim=0)
    raw_spans = [item['span_metadata']['raw_spans'] for item in batch]
    
    span_metadata = {
        'span_types': span_types,
        'positions': positions,
        'boundaries': boundaries,
        'raw_spans': raw_spans
    }
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'span_metadata': span_metadata
    }


def train_focused():
    """
    Focused training loop with only essential losses for span decoding.
    """
    parser = argparse.ArgumentParser(description="Focused BLT Adapter Training")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding")
    parser.add_argument("--parquet", type=str, default="/data/home/zhangsj/Data/more_big_code_language/python/python_ast_parsed.parquet")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=328)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--trial_name", type=str, default="focused_global_concate_residual")
    
    # Span filtering
    parser.add_argument("--min_span_len", type=int, default=1, help="Minimum span length in tokens")
    parser.add_argument("--max_span_len", type=int, default=64, help="Maximum span length in tokens")
    parser.add_argument("--filter_trivial_types", action="store_true", help="Filter out punctuation/operator spans")
    parser.add_argument("--max_nodes_per_sample", type=int, default=16, help="Max nodes per sample for memory")
    
    # Loss weights (only the 3 essential losses)
    parser.add_argument("--node_recon_weight", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=0.5)
    parser.add_argument("--latent_mse_weight", type=float, default=0.3)
    
    # Warmup
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--warmup_node_weight", type=float, default=0.5)
    parser.add_argument("--warmup_boundary_weight", type=float, default=0.3)
    parser.add_argument("--warmup_mse_weight", type=float, default=0.2)
    
    # Gradient accumulation for effective larger batch
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Checkpoint loading
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint directory")
    
    args = parser.parse_args()
    
    trial_name = args.trial_name
    if not args.output_dir:
        args.output_dir = f"/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/{trial_name}"
    if not args.log_dir:
        args.log_dir = f"/data/home/zhangsj/AST_decoding/tensorboard_logs/{trial_name}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Build dataset
    dataset = FocusedPythonASTSpanDataset(
        args.parquet,
        tokenizer,
        max_length=args.max_length,
        min_span_len=args.min_span_len,
        max_span_len=args.max_span_len,
        filter_trivial_types=args.filter_trivial_types,
    )
    
    # Create model
    if args.resume_from and os.path.isdir(args.resume_from):
        print(f"[setup] Resuming from checkpoint: {args.resume_from}")
        adapter = BLTAdapterModel.from_pretrained(args.resume_from)
    else:
        adapter = create_blt_adapter_model(
            args.model_path,
            local_num_layers=2,
            max_node_length=args.max_span_len,
            num_node_types=dataset.num_node_types
        )
    
    # Device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and args.dtype in ("auto", "bf16") else (
                torch.float16 if args.dtype in ("auto", "fp16") else torch.float32
            )
        except Exception:
            dtype = torch.float16 if args.dtype in ("auto", "fp16") else torch.float32
        adapter = adapter.to(device=device, dtype=dtype)
    else:
        dtype = torch.float32
        adapter = adapter.to(device=device, dtype=dtype)
    
    # Memory optimizations
    try:
        adapter.config.use_cache = False
    except Exception:
        pass
    try:
        adapter.gradient_checkpointing_enable()
    except Exception:
        pass
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Disable LM loss, KL, InfoNCE by setting weights to 0
    adapter.lm_loss_weight = 0.0
    adapter.kl_weight = 0.0
    adapter.infonce_weight = 0.0
    adapter.node_recon_loss_weight = args.node_recon_weight
    adapter.boundary_loss_weight = args.boundary_weight
    adapter.latent_mse_weight = args.latent_mse_weight
    adapter.max_nodes_per_sample = args.max_nodes_per_sample
    
    # Freeze global transformer, train local decoder + boundary + latent_from_global
    trainable_params = []
    
    # Freeze global transformer layers
    if hasattr(adapter, 'model') and hasattr(adapter.model, 'layers'):
        for p in adapter.model.layers.parameters():
            p.requires_grad = False
    
    # Freeze base embeddings
    if hasattr(adapter.model, 'embed_tokens'):
        et = adapter.model.embed_tokens
        if hasattr(et, 'token_embeddings'):
            for p in et.token_embeddings.parameters():
                p.requires_grad = False
        # Train adapter + layer_norm in encoder
        for mod_name in ['token_adapter', 'layer_norm']:
            if hasattr(et, mod_name):
                for p in getattr(et, mod_name).parameters():
                    p.requires_grad = True
                    trainable_params.append(p)
    
    # Train local decoder components (including new latent_combine and residual modules)
    for name in ['latent_proj', 'local_transformer', 'boundary_head', 'latent_from_global', 
                 'latent_combine', 'global_residual_gate']:
        if hasattr(adapter, name):
            for p in getattr(adapter, name).parameters():
                p.requires_grad = True
                trainable_params.append(p)
    
    # Also train the residual scale parameter
    if hasattr(adapter, 'global_residual_scale'):
        adapter.global_residual_scale.requires_grad = True
        trainable_params.append(adapter.global_residual_scale)
    
    # Freeze tied large matrices
    if hasattr(adapter, 'local_token_embed'):
        for p in adapter.local_token_embed.parameters():
            p.requires_grad = False
    if hasattr(adapter, 'local_out_proj'):
        for p in adapter.local_out_proj.parameters():
            p.requires_grad = False
    if hasattr(adapter, 'lm_head'):
        for p in adapter.lm_head.parameters():
            p.requires_grad = False
    
    # Train probe heads (for monitoring, not in loss)
    for name in ['node_type_probe_encoder', 'node_type_probe_decoder']:
        if hasattr(adapter, name) and getattr(adapter, name) is not None:
            for p in getattr(adapter, name).parameters():
                p.requires_grad = True
                trainable_params.append(p)
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    # Optimizer
    if len(trainable_params) == 0:
        trainable_params = [p for p in adapter.parameters() if p.requires_grad]
    opt = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # Logging
    writer = SummaryWriter(args.log_dir)
    
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_count = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    frozen_count = total_params - trainable_count
    
    print(f"[setup] total_params={total_params:,} trainable={trainable_count:,} frozen={frozen_count:,}")
    print(f"[setup] batch_size={args.batch_size}, lr={args.lr}, dtype={dtype}")
    print(f"[setup] Losses: node_recon={args.node_recon_weight}, boundary={args.boundary_weight}, latent_mse={args.latent_mse_weight}")
    print(f"[setup] LM_CE=0, KL=0, InfoNCE=0 (disabled)")
    
    writer.add_text("setup/config", str(vars(args)))
    writer.add_text("setup/trainable_params", str(trainable_count))
    
    # Training loop
    adapter.train()
    global_step = 0
    steps_per_epoch = len(dataloader)
    total_steps = args.epochs * steps_per_epoch
    
    for epoch in range(args.epochs):
        epoch_losses = {'total': [], 'node_recon': [], 'boundary': [], 'latent_mse': []}
        
        for batch_idx, batch in enumerate(dataloader):
            # Warmup schedule
            if global_step < args.warmup_steps:
                warmup_ratio = global_step / args.warmup_steps
                adapter.node_recon_loss_weight = args.warmup_node_weight + warmup_ratio * (args.node_recon_weight - args.warmup_node_weight)
                adapter.boundary_loss_weight = args.warmup_boundary_weight + warmup_ratio * (args.boundary_weight - args.warmup_boundary_weight)
                adapter.latent_mse_weight = args.warmup_mse_weight + warmup_ratio * (args.latent_mse_weight - args.warmup_mse_weight)
            else:
                adapter.node_recon_loss_weight = args.node_recon_weight
                adapter.boundary_loss_weight = args.boundary_weight
                adapter.latent_mse_weight = args.latent_mse_weight
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            span_metadata = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch['span_metadata'].items()}
            
            outputs = adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_metadata=span_metadata,
                labels=input_ids
            )
            
            # Compose loss manually from components (LM CE is disabled via weight=0)
            loss = torch.zeros((), device=device, dtype=dtype)
            
            if hasattr(outputs, 'node_recon_loss') and outputs.node_recon_loss is not None:
                loss = loss + adapter.node_recon_loss_weight * outputs.node_recon_loss
                epoch_losses['node_recon'].append(float(outputs.node_recon_loss.item()))
            
            if hasattr(outputs, 'boundary_loss') and outputs.boundary_loss is not None:
                loss = loss + adapter.boundary_loss_weight * outputs.boundary_loss
                epoch_losses['boundary'].append(float(outputs.boundary_loss.item()))
            
            if hasattr(outputs, 'latent_mse') and outputs.latent_mse is not None:
                loss = loss + adapter.latent_mse_weight * outputs.latent_mse
                epoch_losses['latent_mse'].append(float(outputs.latent_mse.item()))
            
            epoch_losses['total'].append(float(loss.item()))
            
            # Backward
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Separate backward for probes (monitoring only)
            probe_total = None
            if hasattr(outputs, 'type_probe_encoder_loss') and outputs.type_probe_encoder_loss is not None:
                probe_total = outputs.type_probe_encoder_loss
            if hasattr(outputs, 'type_probe_decoder_loss') and outputs.type_probe_decoder_loss is not None:
                probe_total = outputs.type_probe_decoder_loss if probe_total is None else probe_total + outputs.type_probe_decoder_loss
            if probe_total is not None:
                try:
                    probe_total.backward()
                except Exception:
                    pass
            
            # Optimizer step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % 50 == 0:
                    writer.add_scalar("loss/total", float(loss.item() * args.gradient_accumulation_steps), global_step)
                    if hasattr(outputs, 'node_recon_loss') and outputs.node_recon_loss is not None:
                        writer.add_scalar("loss/node_recon", float(outputs.node_recon_loss.item()), global_step)
                    if hasattr(outputs, 'boundary_loss') and outputs.boundary_loss is not None:
                        writer.add_scalar("loss/boundary", float(outputs.boundary_loss.item()), global_step)
                    if hasattr(outputs, 'latent_mse') and outputs.latent_mse is not None:
                        writer.add_scalar("loss/latent_mse", float(outputs.latent_mse.item()), global_step)
                    if hasattr(outputs, 'type_probe_encoder_loss') and outputs.type_probe_encoder_loss is not None:
                        writer.add_scalar("loss/type_probe_encoder", float(outputs.type_probe_encoder_loss.item()), global_step)
                    if hasattr(outputs, 'type_probe_decoder_loss') and outputs.type_probe_decoder_loss is not None:
                        writer.add_scalar("loss/type_probe_decoder", float(outputs.type_probe_decoder_loss.item()), global_step)
                    if hasattr(outputs, 'type_probe_encoder_acc'):
                        writer.add_scalar("acc/type_probe_encoder", float(outputs.type_probe_encoder_acc.item()), global_step)
                    if hasattr(outputs, 'type_probe_decoder_acc'):
                        writer.add_scalar("acc/type_probe_decoder", float(outputs.type_probe_decoder_acc.item()), global_step)
                    
                    # GPU memory
                    if torch.cuda.is_available():
                        writer.add_scalar("mem/alloc_GB", torch.cuda.memory_allocated() / (1024**3), global_step)
                    
                    # Print progress
                    msg = f"epoch {epoch+1} step {global_step} | total {float(loss.item() * args.gradient_accumulation_steps):.4f}"
                    if hasattr(outputs, 'node_recon_loss') and outputs.node_recon_loss is not None:
                        msg += f" | node_recon {float(outputs.node_recon_loss.item()):.4f}"
                    if hasattr(outputs, 'boundary_loss') and outputs.boundary_loss is not None:
                        msg += f" | boundary {float(outputs.boundary_loss.item()):.4f}"
                    if hasattr(outputs, 'latent_mse') and outputs.latent_mse is not None:
                        msg += f" | latent_mse {float(outputs.latent_mse.item()):.4f}"
                    if hasattr(outputs, 'type_probe_encoder_loss'):
                        msg += f" | probe_enc_ce {float(outputs.type_probe_encoder_loss.item()):.4f}"
                    if hasattr(outputs, 'type_probe_encoder_acc'):
                        msg += f" | probe_enc_acc {float(outputs.type_probe_encoder_acc.item()):.3f}"
                    if hasattr(outputs, 'type_probe_decoder_loss'):
                        msg += f" | probe_dec_ce {float(outputs.type_probe_decoder_loss.item()):.4f}"
                    if hasattr(outputs, 'type_probe_decoder_acc'):
                        msg += f" | probe_dec_acc {float(outputs.type_probe_decoder_acc.item()):.3f}"
                    print(msg)
        
        # Epoch summary
        avg_total = np.mean(epoch_losses['total']) if epoch_losses['total'] else 0
        avg_node = np.mean(epoch_losses['node_recon']) if epoch_losses['node_recon'] else 0
        avg_bnd = np.mean(epoch_losses['boundary']) if epoch_losses['boundary'] else 0
        avg_mse = np.mean(epoch_losses['latent_mse']) if epoch_losses['latent_mse'] else 0
        
        print(f"\n[Epoch {epoch+1}] Avg losses: total={avg_total:.4f}, node_recon={avg_node:.4f}, boundary={avg_bnd:.4f}, latent_mse={avg_mse:.4f}\n")
        
        writer.add_scalar("epoch/total_loss", avg_total, epoch)
        writer.add_scalar("epoch/node_recon_loss", avg_node, epoch)
        writer.add_scalar("epoch/boundary_loss", avg_bnd, epoch)
        writer.add_scalar("epoch/latent_mse_loss", avg_mse, epoch)
        
        # Save checkpoint
        save_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        try:
            # Try standard save first
            adapter.save_pretrained(save_dir, safe_serialization=False)
        except RuntimeError as e:
            if "shared tensors" in str(e):
                # Workaround for tied weights issue
                print(f"[save] Using torch.save fallback due to tied weights")
                torch.save(adapter.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                adapter.config.save_pretrained(save_dir)
            else:
                raise
        tokenizer.save_pretrained(save_dir)
        print(f"Saved checkpoint to {save_dir}")
    
    writer.add_text("training/status", "COMPLETED", global_step)
    writer.close()
    print(f"\nTraining complete! Final checkpoint: {args.output_dir}")


if __name__ == "__main__":
    train_focused()

