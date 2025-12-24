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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Reduce CUDA allocator fragmentation; MUST be set before first CUDA context init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Avoid HuggingFace tokenizer's "forked after parallelism" spam when DataLoader uses multiprocessing.
# This must be set before any `transformers`/`tokenizers` usage.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import datetime
import math
import argparse
import re

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

# Prefer SDPA/Flash-SDP when available (v1 did this; helps both speed and memory).
try:
    if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
        # These are safe no-ops on older PyTorch builds.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            # Keep math SDPA enabled as a fallback.
            torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Cosine learning rate scheduler with linear warmup.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

# Import the model components
from blt_adapter_model_v2 import (
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
                    # Keep token_indices as Python lists so DataLoader workers can pickle efficiently.
                    processed.append({'token_indices': [int(t)], 'span_type_id': span_type_id})
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
                processed.append({'token_indices': valid.tolist(), 'span_type_id': span_type_id})
        
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


class HumanEvalPromptSolutionDataset(Dataset):
    """
    HumanEval JSONL (prompt + canonical_solution) for inference-aligned LM validation.

    We compute LM CE on *completion tokens only* by masking prompt tokens to -100.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"HumanEval JSONL not found: {jsonl_path}")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.rows: List[Tuple[str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                prompt = str(obj.get("prompt", "") or "")
                sol = str(obj.get("canonical_solution", "") or "")
                if prompt.strip() and sol.strip():
                    self.rows.append((prompt, sol))
        print(f"[Dataset] Loaded {len(self.rows)} HumanEval prompt+solution pairs from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, sol = self.rows[idx]
        full_text = f"{prompt}{sol}"
        enc_prompt = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc_full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc_full["input_ids"].squeeze(0)
        attention_mask = enc_full["attention_mask"].squeeze(0)
        prompt_len = int(enc_prompt["attention_mask"].sum().item())
        labels = input_ids.clone()
        if prompt_len > 0:
            labels[:prompt_len] = -100
        labels = labels.masked_fill(attention_mask == 0, -100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_lm_completion_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_focused():
    """
    Focused training loop with only essential losses for span decoding.
    """
    parser = argparse.ArgumentParser(description="Focused BLT Adapter Training")
    # Default to your DeepSeek distilled model
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/deepseek_qwen1.5_distill")
    parser.add_argument("--parquet", type=str, default="/data/home/zhangsj/Data/more_big_code_language/python/deepseek_qwen1.5_python_ast.Parquet")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=328)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--trial_name", type=str, default="focused_deepseek_distill_blt_v2")
    # Attention backend: pass into HF model loading if supported (Transformers 4.36+).
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "eager"],
                        help="Attention implementation for HF models. 'sdpa' enables PyTorch SDPA backend.")
    
    # Span filtering
    parser.add_argument("--min_span_len", type=int, default=1, help="Minimum span length in tokens")
    parser.add_argument("--max_span_len", type=int, default=64, help="Maximum span length in tokens")
    parser.add_argument("--filter_trivial_types", action="store_true", help="Filter out punctuation/operator spans")
    parser.add_argument("--max_nodes_per_sample", type=int, default=64, help="Max nodes per sample for memory")
    
    # Loss weights (only the 3 essential losses)
    parser.add_argument("--lm_weight", type=float, default=0.05, help="Global LM CE loss weight")
    parser.add_argument("--warmup_lm_weight", type=float, default=0.0, help="LM CE weight during warmup")
    parser.add_argument("--node_recon_weight", type=float, default=0.4)
    parser.add_argument("--boundary_weight", type=float, default=0.2)
    # Default off: user requested LM + node_recon + boundary only
    parser.add_argument("--latent_mse_weight", type=float, default=0.0)
    parser.add_argument("--boundary_include_singles", action="store_true", default=False,
                        help="If set, treat single-token spans as positives for boundary training")
    parser.add_argument("--boundary_class_weight", type=float, default=None,
                        help="Weight for boundary positive class (class=1). If set, uses weights [1.0, w_pos].")
    parser.add_argument("--boundary_focal_gamma", type=float, default=0.0,
                        help="Focal loss gamma for boundary head (0.0=disabled, 2.0=standard).")
    
    # Warmup
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--warmup_node_weight", type=float, default=0.5)
    parser.add_argument("--warmup_boundary_weight", type=float, default=0.3)
    parser.add_argument("--warmup_mse_weight", type=float, default=0.0)
    
    # Gradient accumulation for effective larger batch
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    
    # LR Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Learning rate scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="LR warmup steps (separate from loss warmup)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR as ratio of initial LR")
    
    # Local decoder configuration
    parser.add_argument("--local_num_layers", type=int, default=2, help="Number of local decoder layers")
    # Chunking for local decoder projection to vocab (memory/speed tradeoff)
    parser.add_argument("--node_ce_chunk_tokens", type=int, default=1024,
                        help="Chunk size (#token positions) when projecting local-decoder hidden states to vocab for CE. Larger=faster, more VRAM.")
    parser.add_argument("--node_argmax_chunk_tokens", type=int, default=1024,
                        help="Chunk size (#token positions) when computing argmax over vocab for self-conditioning. Larger=faster, more VRAM.")
    parser.add_argument("--node_logits_mode", type=str, default="auto", choices=["auto", "full", "chunked"],
                        help="How to compute local-decoder vocab projection for CE/argmax. 'full' is fastest but can OOM on span-heavy batches. 'auto' uses full below a token threshold.")
    parser.add_argument("--node_full_logits_max_tokens", type=int, default=4096,
                        help="When --node_logits_mode=auto, use full logits if #valid positions <= this threshold, else chunked.")
    parser.add_argument("--node_compute_teacher_ce", action="store_true", default=False,
                        help="If set, compute teacher_ce (monitoring-only). Disabling can improve speed.")
    parser.add_argument("--node_teacher_ce_max_tokens", type=int, default=1024,
                        help="Cap teacher_ce computation to at most this many valid positions (monitoring-only).")

    # Local decoder free-run / self-conditioning (reduce teacher-forcing exposure bias).
    # "teacher" = current behavior (BOS + gold[:-1]).
    # "self_condition" = run one teacher-forced pass to get token predictions, then run a second pass where
    # decoder inputs are BOS + predicted_tokens[:-1]. This approximates free-run generation but stays efficient.
    # NOTE: implemented inside `blt_adapter_model_v2.py` (local node recon path).
    parser.add_argument(
        "--local_decoder_train_mode",
        type=str,
        default="self_condition",
        choices=["teacher", "self_condition"],
        help="Training mode for local decoder inputs. 'self_condition' reduces exposure bias by conditioning on the model's own predicted tokens.",
    )
    parser.add_argument("--local_decoder_self_condition_p_start", type=float, default=0.0,
                        help="Start probability of using self-conditioning in local decoder training.")
    parser.add_argument("--local_decoder_self_condition_p_end", type=float, default=1.0,
                        help="End probability of using self-conditioning in local decoder training.")
    parser.add_argument("--local_decoder_self_condition_schedule_frac", type=float, default=0.5,
                        help="Fraction of total optimizer steps over which self-conditioning probability ramps from start to end.")
    
    # Validation (use pre-parsed parquet file from preprocess_validation_data.py)
    parser.add_argument("--val_split", type=float, default=0.0, help="Split this fraction from train for validation (default: 0, use HumanEval only)")
    parser.add_argument("--humaneval_parquet", type=str,
                        default="/data/home/zhangsj/Data/HumanEval/humaneval_ast_parsed.parquet",
                        help="Path to pre-parsed HumanEval parquet for validation")
    parser.add_argument(
        "--humaneval_jsonl",
        type=str,
        default="/data/home/zhangsj/Data/HumanEval/human-eval-v2-20210705.jsonl",
        help="Optional: HumanEval JSONL for completion-only LM validation (prompt+canonical_solution).",
    )
    parser.add_argument("--eval_every_n_epochs", type=int, default=1, help="Run validation every N epochs")
    
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
    # Many code LMs don't define a pad token; but this script uses padding='max_length'.
    if getattr(tokenizer, "pad_token", None) is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"[setup] tokenizer.pad_token was None; set pad_token=eos_token (id={tokenizer.eos_token_id})")
        except Exception:
            pass
    
    # Build dataset
    dataset = FocusedPythonASTSpanDataset(
        args.parquet,
        tokenizer,
        max_length=args.max_length,
        min_span_len=args.min_span_len,
        max_span_len=args.max_span_len,
        filter_trivial_types=args.filter_trivial_types,
    )
    
    # Handle validation dataset
    val_dataset = None
    lm_val_dataset = None
    
    # Option 1: Use pre-parsed HumanEval parquet (default)
    if args.humaneval_parquet and os.path.exists(args.humaneval_parquet):
        print(f"[Dataset] Loading HumanEval validation from: {args.humaneval_parquet}")
        val_dataset = FocusedPythonASTSpanDataset(
            args.humaneval_parquet,
            tokenizer,
            max_length=args.max_length,
            min_span_len=args.min_span_len,
            max_span_len=args.max_span_len,
            filter_trivial_types=args.filter_trivial_types,
        )
    # Option 2: Split from training data (fallback)
    elif args.val_split > 0:
        from torch.utils.data import random_split
        total_len = len(dataset)
        val_len = int(total_len * args.val_split)
        train_len = total_len - val_len
        dataset, val_dataset = random_split(dataset, [train_len, val_len])
        print(f"[Dataset] Train/val split: {train_len} train, {val_len} val")

    # Optional LM completion-only validation on HumanEval JSONL (inference-aligned loss).
    if args.humaneval_jsonl and os.path.exists(args.humaneval_jsonl):
        try:
            lm_val_dataset = HumanEvalPromptSolutionDataset(
                args.humaneval_jsonl,
                tokenizer,
                max_length=args.max_length,
            )
        except Exception as e:
            print(f"[Dataset] Warning: could not build HumanEval JSONL LM val dataset: {e}")
            lm_val_dataset = None

    # Prepare boundary class weight tensor (optional)
    boundary_class_weight_tensor = None
    if args.boundary_class_weight is not None:
        boundary_class_weight_tensor = torch.tensor([1.0, float(args.boundary_class_weight)], dtype=torch.float32)
        print(f"[setup] Using boundary class weights: [non-boundary=1.0, boundary={args.boundary_class_weight:.2f}]")
    
    # Create model
    if args.resume_from and os.path.isdir(args.resume_from):
        print(f"[setup] Resuming from checkpoint: {args.resume_from}")
        adapter = BLTAdapterModel.from_pretrained(args.resume_from, attn_implementation=args.attn_implementation)
    else:
        adapter = create_blt_adapter_model(
            args.model_path,
            local_num_layers=args.local_num_layers,
            max_node_length=args.max_span_len,
            num_node_types=dataset.num_node_types if hasattr(dataset, 'num_node_types') else len(SPAN_TYPE_TO_ID),
            attn_implementation=args.attn_implementation,
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
    # (allocator env var is set at process start; keep this line as a no-op for backward compatibility)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Enable LM CE (NTP) for global transformer if requested; disable KL/InfoNCE
    adapter.lm_loss_weight = float(args.lm_weight)
    # Ensure LM CE flows gradients outside the model call if used directly from outputs.lm_ce
    try:
        adapter.expose_lm_ce_grad = True
    except Exception:
        pass
    adapter.kl_weight = 0.0
    adapter.infonce_weight = 0.0
    adapter.node_recon_loss_weight = args.node_recon_weight
    adapter.boundary_loss_weight = args.boundary_weight
    adapter.latent_mse_weight = args.latent_mse_weight
    adapter.max_nodes_per_sample = args.max_nodes_per_sample
    # Boundary loss options (implemented in model v2 via attributes)
    try:
        adapter.boundary_focal_gamma = float(args.boundary_focal_gamma)
    except Exception:
        pass
    if boundary_class_weight_tensor is not None:
        try:
            adapter.register_buffer("boundary_class_weight", boundary_class_weight_tensor.to(adapter.device))
        except Exception:
            # safe fallback: attach as attribute
            adapter.boundary_class_weight = boundary_class_weight_tensor
    # Boundary behavior: by default exclude singles; enable with flag
    try:
        adapter.boundary_include_singles = bool(args.boundary_include_singles)
    except Exception:
        pass
    # Disable textual span clamping on boundary targets for this training run
    if hasattr(adapter, "textual_span_type_ids"):
        adapter.textual_span_type_ids = torch.tensor([], dtype=torch.long)
    
    # Freeze global transformer, train local decoder + boundary + latent_from_global
    trainable_params = []
    
    # Freeze global transformer layers
    if hasattr(adapter, 'base_model') and hasattr(adapter.base_model, 'model') and hasattr(adapter.base_model.model, 'layers'):
        for p in adapter.base_model.model.layers.parameters():
            p.requires_grad = False
    
    # Freeze base embeddings
    if hasattr(adapter.base_model, 'model') and hasattr(adapter.base_model.model, 'embed_tokens'):
        et = adapter.base_model.model.embed_tokens
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
    
    # If LM CE is enabled, unfreeze last global transformer layer and lm_head for NTP
    if float(adapter.lm_loss_weight) > 0.0:
        try:
            if hasattr(adapter, 'base_model') and hasattr(adapter.base_model, 'model') and hasattr(adapter.base_model.model, 'layers') and len(adapter.base_model.model.layers) > 0:
                for p in adapter.base_model.model.layers[-1].parameters():
                    p.requires_grad = True
                    trainable_params.append(p)
        except Exception:
            pass
        if hasattr(adapter.base_model, 'lm_head'):
            try:
                for p in adapter.base_model.lm_head.parameters():
                    p.requires_grad = True
                    trainable_params.append(p)
            except Exception:
                pass
    
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
    # Freeze lm_head only if LM CE is disabled (lm_weight == 0.0)
    if hasattr(adapter.base_model, 'lm_head') and float(adapter.lm_loss_weight) == 0.0:
        for p in adapter.base_model.lm_head.parameters():
            p.requires_grad = False
    
    # Train probe heads (for monitoring, not in loss)
    for name in ['node_type_probe_encoder', 'node_type_probe_decoder']:
        if hasattr(adapter, name) and getattr(adapter, name) is not None:
            for p in getattr(adapter, name).parameters():
                p.requires_grad = True
                trainable_params.append(p)
    
    # DataLoader (num_workers=0 to avoid issues with raw_spans list)
    num_workers = min(4, max(2, args.batch_size)) if torch.cuda.is_available() else 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 and torch.cuda.is_available() else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Validation DataLoader
    val_dataloader = None
    if val_dataset is not None:
        val_num_workers = min(2, max(1, args.batch_size)) if torch.cuda.is_available() else 0
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            drop_last=False,
            prefetch_factor=2 if val_num_workers > 0 and torch.cuda.is_available() else None,
            persistent_workers=True if val_num_workers > 0 else False,
        )
        print(f"[DataLoader] Validation: {len(val_dataloader)} batches")

    lm_val_dataloader = None
    if lm_val_dataset is not None:
        lm_val_dataloader = DataLoader(
            lm_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_lm_completion_fn,
            drop_last=False,
        )
        print(f"[DataLoader] HumanEval JSONL LM val: {len(lm_val_dataloader)} batches")
    
    # Optimizer
    if len(trainable_params) == 0:
        trainable_params = [p for p in adapter.parameters() if p.requires_grad]
    opt = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # Calculate total optimizer steps for scheduler
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_optimizer_steps = args.epochs * steps_per_epoch
    
    # LR Scheduler
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            opt, 
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=total_optimizer_steps,
            min_lr_ratio=args.min_lr_ratio
        )
        print(f"[setup] Using cosine LR scheduler: warmup={args.lr_warmup_steps}, total={total_optimizer_steps}")
    elif args.lr_scheduler == "linear":
        def linear_lambda(step):
            if step < args.lr_warmup_steps:
                return float(step) / float(max(1, args.lr_warmup_steps))
            return max(args.min_lr_ratio, 1.0 - (step - args.lr_warmup_steps) / float(max(1, total_optimizer_steps - args.lr_warmup_steps)))
        scheduler = LambdaLR(opt, linear_lambda)
        print(f"[setup] Using linear LR scheduler: warmup={args.lr_warmup_steps}, total={total_optimizer_steps}")
    
    # Logging
    writer = SummaryWriter(args.log_dir)
    
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_count = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    frozen_count = total_params - trainable_count
    
    print(f"[setup] total_params={total_params:,} trainable={trainable_count:,} frozen={frozen_count:,}")
    print(f"[setup] batch_size={args.batch_size}, lr={args.lr}, dtype={dtype}")
    print(f"[setup] Losses: lm_ce={adapter.lm_loss_weight}, node_recon={args.node_recon_weight}, boundary={args.boundary_weight}, latent_mse={args.latent_mse_weight}")
    print(f"[setup] KL=0, InfoNCE=0 (disabled)")
    print(f"[setup] boundary_include_singles={getattr(adapter, 'boundary_include_singles', False)}")
    
    writer.add_text("setup/config", str(vars(args)))
    writer.add_text("setup/trainable_params", str(trainable_count))
    
    # Validation function
    @torch.no_grad()
    def run_validation(val_loader, epoch_num):
        adapter.eval()
        val_sum = {
            'total': torch.zeros((), device=device, dtype=torch.float32),
            'lm_ce': torch.zeros((), device=device, dtype=torch.float32),
            'node_recon': torch.zeros((), device=device, dtype=torch.float32),
            'boundary': torch.zeros((), device=device, dtype=torch.float32),
            'latent_mse': torch.zeros((), device=device, dtype=torch.float32),
        }
        val_count = {'total': 0, 'lm_ce': 0, 'node_recon': 0, 'boundary': 0, 'latent_mse': 0}
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            span_metadata = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch['span_metadata'].items()}
            # IMPORTANT: mask padding for LM CE (otherwise loss is dominated by pad tokens)
            labels = input_ids.clone()
            labels = labels.masked_fill(attention_mask == 0, -100)
            outputs = adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_metadata=span_metadata,
                labels=labels
            )
            loss = torch.zeros((), device=device, dtype=dtype)
            if hasattr(outputs, 'lm_ce') and outputs.lm_ce is not None:
                loss = loss + adapter.lm_loss_weight * outputs.lm_ce
                val_sum['lm_ce'] = val_sum['lm_ce'] + outputs.lm_ce.detach().float()
                val_count['lm_ce'] += 1
            if hasattr(outputs, 'node_recon_loss') and outputs.node_recon_loss is not None:
                loss = loss + adapter.node_recon_loss_weight * outputs.node_recon_loss
                val_sum['node_recon'] = val_sum['node_recon'] + outputs.node_recon_loss.detach().float()
                val_count['node_recon'] += 1
            if hasattr(outputs, 'boundary_loss') and outputs.boundary_loss is not None:
                loss = loss + adapter.boundary_loss_weight * outputs.boundary_loss
                val_sum['boundary'] = val_sum['boundary'] + outputs.boundary_loss.detach().float()
                val_count['boundary'] += 1
            if hasattr(outputs, 'latent_mse') and outputs.latent_mse is not None:
                loss = loss + adapter.latent_mse_weight * outputs.latent_mse
                val_sum['latent_mse'] = val_sum['latent_mse'] + outputs.latent_mse.detach().float()
                val_count['latent_mse'] += 1
            val_sum['total'] = val_sum['total'] + loss.detach().float()
            val_count['total'] += 1
        adapter.train()

        avg_val = {}
        for k in val_sum.keys():
            denom = max(1, int(val_count.get(k, 0)))
            avg_val[k] = float((val_sum[k] / denom).item())
        print(f"\n[Val Epoch {epoch_num}] total={avg_val['total']:.4f}, lm_ce={avg_val['lm_ce']:.4f}, node_recon={avg_val['node_recon']:.4f}, boundary={avg_val['boundary']:.4f}, latent_mse={avg_val['latent_mse']:.4f}")
        return avg_val

    @torch.no_grad()
    def run_lm_completion_validation(val_loader, epoch_num):
        adapter.eval()
        total = torch.zeros((), device=device, dtype=torch.float32)
        count = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            out = adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if hasattr(out, "loss") and out.loss is not None:
                total = total + out.loss.detach().float()
                count += 1
        adapter.train()
        avg = float((total / max(1, count)).item())
        print(f"\n[Val Epoch {epoch_num}] HumanEval completion-only LM CE: {avg:.4f}")
        return avg
    
    # === Resume support (optimizer/scheduler/global_step) ===
    start_epoch = 0
    saved_global_step = None
    if args.resume_from and os.path.isdir(args.resume_from):
        match = re.search(r'epoch_(\d+)(?:\/|$)', args.resume_from)
        if match:
            start_epoch = int(match.group(1))
            print(f"[setup] Detected starting epoch: {start_epoch} from checkpoint path")
        optimizer_state_path = os.path.join(args.resume_from, "optimizer.pt")
        scheduler_state_path = os.path.join(args.resume_from, "scheduler.pt")
        training_state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.exists(optimizer_state_path):
            try:
                opt_state = torch.load(optimizer_state_path, map_location=device)
                opt.load_state_dict(opt_state)
                print(f"[setup] Loaded optimizer state from {optimizer_state_path}")
            except Exception as e:
                print(f"[setup] Warning: Could not load optimizer state: {e}")
        if scheduler is not None and os.path.exists(scheduler_state_path):
            try:
                scheduler_state = torch.load(scheduler_state_path, map_location=device)
                scheduler.load_state_dict(scheduler_state)
                print(f"[setup] Loaded scheduler state from {scheduler_state_path}")
            except Exception as e:
                print(f"[setup] Warning: Could not load scheduler state: {e}")
        if os.path.exists(training_state_path):
            try:
                training_state = torch.load(training_state_path, map_location="cpu")
                if "global_step" in training_state:
                    saved_global_step = int(training_state["global_step"])
                    print(f"[setup] Training state indicates global_step={saved_global_step}")
            except Exception as e:
                print(f"[setup] Warning: Could not load training state: {e}")

    # Training loop
    adapter.train()
    global_step = saved_global_step if saved_global_step is not None else start_epoch * steps_per_epoch
    last_logged_step = global_step
    accumulated_loss_tensor = torch.tensor(0.0, device=device, dtype=dtype)
    accumulated_optimizer_steps = 0
    print(f"[setup] Starting from epoch {start_epoch}, global_step={global_step}, steps_per_epoch={steps_per_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_sum = {
            'total': torch.zeros((), device=device, dtype=torch.float32),
            'lm_ce': torch.zeros((), device=device, dtype=torch.float32),
            'node_recon': torch.zeros((), device=device, dtype=torch.float32),
            'boundary': torch.zeros((), device=device, dtype=torch.float32),
            'latent_mse': torch.zeros((), device=device, dtype=torch.float32),
        }
        epoch_count = {'total': 0, 'lm_ce': 0, 'node_recon': 0, 'boundary': 0, 'latent_mse': 0}

        for batch_idx, batch in enumerate(dataloader):
            # Warmup schedule
            if global_step < args.warmup_steps:
                warmup_ratio = global_step / args.warmup_steps
                adapter.lm_loss_weight = args.warmup_lm_weight + warmup_ratio * (args.lm_weight - args.warmup_lm_weight)
                adapter.node_recon_loss_weight = args.warmup_node_weight + warmup_ratio * (args.node_recon_weight - args.warmup_node_weight)
                adapter.boundary_loss_weight = args.warmup_boundary_weight + warmup_ratio * (args.boundary_weight - args.warmup_boundary_weight)
                adapter.latent_mse_weight = args.warmup_mse_weight + warmup_ratio * (args.latent_mse_weight - args.warmup_mse_weight)
            else:
                adapter.lm_loss_weight = args.lm_weight
                adapter.node_recon_loss_weight = args.node_recon_weight
                adapter.boundary_loss_weight = args.boundary_weight
                adapter.latent_mse_weight = args.latent_mse_weight

            # Local-decoder self-conditioning schedule (train analogue of free-run generation).
            # Stored on the model and used inside `_compute_local_node_recon_loss`.
            try:
                adapter.local_decoder_train_mode = str(getattr(args, "local_decoder_train_mode", "teacher"))
                if adapter.local_decoder_train_mode == "self_condition":
                    p0 = float(getattr(args, "local_decoder_self_condition_p_start", 0.0))
                    p1 = float(getattr(args, "local_decoder_self_condition_p_end", 1.0))
                    frac = float(getattr(args, "local_decoder_self_condition_schedule_frac", 0.5))
                    ramp_steps = max(1, int(frac * max(1, int(total_optimizer_steps))))
                    prog = min(1.0, float(global_step) / float(ramp_steps))
                    adapter.local_decoder_self_condition_p = p0 + prog * (p1 - p0)
                else:
                    adapter.local_decoder_self_condition_p = 0.0
            except Exception:
                adapter.local_decoder_train_mode = "teacher"
                adapter.local_decoder_self_condition_p = 0.0

            # Local node recon chunk sizes (used inside `blt_adapter_model_v2.py`)
            try:
                adapter.node_ce_chunk_tokens = int(getattr(args, "node_ce_chunk_tokens", 1024))
            except Exception:
                adapter.node_ce_chunk_tokens = 1024
            try:
                adapter.node_argmax_chunk_tokens = int(getattr(args, "node_argmax_chunk_tokens", 1024))
            except Exception:
                adapter.node_argmax_chunk_tokens = 1024
            try:
                adapter.node_logits_mode = str(getattr(args, "node_logits_mode", "auto"))
            except Exception:
                adapter.node_logits_mode = "auto"
            try:
                adapter.node_full_logits_max_tokens = int(getattr(args, "node_full_logits_max_tokens", 4096))
            except Exception:
                adapter.node_full_logits_max_tokens = 4096
            try:
                adapter.node_compute_teacher_ce = bool(getattr(args, "node_compute_teacher_ce", False))
            except Exception:
                adapter.node_compute_teacher_ce = False
            try:
                adapter.node_teacher_ce_max_tokens = int(getattr(args, "node_teacher_ce_max_tokens", 1024))
            except Exception:
                adapter.node_teacher_ce_max_tokens = 1024
            
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            span_metadata = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch['span_metadata'].items()}
            labels = input_ids.clone()
            labels = labels.masked_fill(attention_mask == 0, -100)
            
            outputs = adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_metadata=span_metadata,
                labels=labels
            )
            
            loss = torch.zeros((), device=device, dtype=dtype)
            
            # Include global LM CE (NTP) if enabled
            if hasattr(outputs, 'lm_ce') and outputs.lm_ce is not None:
                loss = loss + adapter.lm_loss_weight * outputs.lm_ce
                epoch_sum['lm_ce'] = epoch_sum['lm_ce'] + outputs.lm_ce.detach().float()
                epoch_count['lm_ce'] += 1
            
            if hasattr(outputs, 'node_recon_loss') and outputs.node_recon_loss is not None:
                loss = loss + adapter.node_recon_loss_weight * outputs.node_recon_loss
                epoch_sum['node_recon'] = epoch_sum['node_recon'] + outputs.node_recon_loss.detach().float()
                epoch_count['node_recon'] += 1
            
            if hasattr(outputs, 'boundary_loss') and outputs.boundary_loss is not None:
                loss = loss + adapter.boundary_loss_weight * outputs.boundary_loss
                epoch_sum['boundary'] = epoch_sum['boundary'] + outputs.boundary_loss.detach().float()
                epoch_count['boundary'] += 1
            
            if hasattr(outputs, 'latent_mse') and outputs.latent_mse is not None:
                loss = loss + adapter.latent_mse_weight * outputs.latent_mse
                epoch_sum['latent_mse'] = epoch_sum['latent_mse'] + outputs.latent_mse.detach().float()
                epoch_count['latent_mse'] += 1
            
            epoch_sum['total'] = epoch_sum['total'] + loss.detach().float()
            epoch_count['total'] += 1
            accumulated_loss_tensor = accumulated_loss_tensor + loss.detach()
            
            # Backward (scaled for gradient accumulation)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Separate backward for probes only (monitoring only)
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
                
                # Step LR scheduler
                if scheduler is not None:
                    scheduler.step()
                
                global_step += 1
                accumulated_optimizer_steps += 1

                should_log = (global_step - last_logged_step >= 50)
                if should_log:
                    denom = max(1, int(accumulated_optimizer_steps) * int(args.gradient_accumulation_steps))
                    avg_accumulated_loss = float(accumulated_loss_tensor.item() / denom)
                    accumulated_loss_tensor = torch.tensor(0.0, device=device, dtype=dtype)
                    accumulated_optimizer_steps = 0

                    writer.add_scalar("loss/total", avg_accumulated_loss, global_step)
                    
                    # Log current learning rate
                    current_lr = opt.param_groups[0]['lr']
                    writer.add_scalar("lr/learning_rate", current_lr, global_step)
                    # Boundary metrics
                    if hasattr(outputs, 'boundary_acc'):
                        writer.add_scalar("acc/boundary", float(outputs.boundary_acc.item()), global_step)
                    if hasattr(outputs, 'boundary_start_recall'):
                        writer.add_scalar("acc/boundary_start_recall", float(outputs.boundary_start_recall.item()), global_step)
                    if hasattr(outputs, 'boundary_single_recall'):
                        writer.add_scalar("acc/boundary_single_recall", float(outputs.boundary_single_recall.item()), global_step)
                    if hasattr(outputs, 'boundary_pos_rate'):
                        writer.add_scalar("stat/boundary_pos_rate", float(outputs.boundary_pos_rate.item()), global_step)
                    if hasattr(outputs, 'boundary_pred_pos_rate'):
                        writer.add_scalar("stat/boundary_pred_pos_rate", float(outputs.boundary_pred_pos_rate.item()), global_step)
                    if hasattr(outputs, 'boundary_prob_mean'):
                        writer.add_scalar("stat/boundary_prob_mean", float(outputs.boundary_prob_mean.item()), global_step)
                    if hasattr(outputs, 'lm_ce') and outputs.lm_ce is not None:
                        writer.add_scalar("loss/lm_ce", float(outputs.lm_ce.item()), global_step)
                        writer.add_scalar("loss/lm_ce_weighted", float((adapter.lm_loss_weight * outputs.lm_ce).item()), global_step)
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
                    current_lr = opt.param_groups[0]['lr']
                    msg = f"epoch {epoch+1} step {global_step} lr {current_lr:.2e} | total {avg_accumulated_loss:.4f}"
                    if hasattr(outputs, 'lm_ce') and outputs.lm_ce is not None:
                        msg += f" | lm_ce {float(outputs.lm_ce.item()):.4f}"
                    if hasattr(outputs, 'node_recon_loss') and outputs.node_recon_loss is not None:
                        msg += f" | node_recon {float(outputs.node_recon_loss.item()):.4f}"
                    if hasattr(outputs, 'boundary_loss') and outputs.boundary_loss is not None:
                        msg += f" | boundary {float(outputs.boundary_loss.item()):.4f}"
                    if hasattr(outputs, 'latent_mse') and outputs.latent_mse is not None:
                        msg += f" | latent_mse {float(outputs.latent_mse.item()):.4f}"
                    if hasattr(outputs, 'boundary_acc'):
                        msg += f" | bnd_acc {float(outputs.boundary_acc.item()):.3f}"
                    if hasattr(outputs, 'boundary_start_recall'):
                        msg += f" | bnd_start_rec {float(outputs.boundary_start_recall.item()):.3f}"
                    if hasattr(outputs, 'boundary_single_recall'):
                        msg += f" | bnd_single_rec {float(outputs.boundary_single_recall.item()):.3f}"
                    if hasattr(outputs, 'type_probe_encoder_loss'):
                        msg += f" | probe_enc_ce {float(outputs.type_probe_encoder_loss.item()):.4f}"
                    if hasattr(outputs, 'type_probe_encoder_acc'):
                        msg += f" | probe_enc_acc {float(outputs.type_probe_encoder_acc.item()):.3f}"
                    if hasattr(outputs, 'type_probe_decoder_loss'):
                        msg += f" | probe_dec_ce {float(outputs.type_probe_decoder_loss.item()):.4f}"
                    if hasattr(outputs, 'type_probe_decoder_acc'):
                        msg += f" | probe_dec_acc {float(outputs.type_probe_decoder_acc.item()):.3f}"
                    print(msg)
                    last_logged_step = global_step
        
        def _avg(k: str) -> float:
            denom = max(1, int(epoch_count.get(k, 0)))
            return float((epoch_sum[k] / denom).item())
        avg_total = _avg('total')
        avg_lmce = _avg('lm_ce')
        avg_node = _avg('node_recon')
        avg_bnd = _avg('boundary')
        avg_mse = _avg('latent_mse')
        
        print(f"\n[Epoch {epoch+1}] Avg losses: total={avg_total:.4f}, lm_ce={avg_lmce:.4f}, node_recon={avg_node:.4f}, boundary={avg_bnd:.4f}, latent_mse={avg_mse:.4f}\n")
        
        writer.add_scalar("epoch/total_loss", avg_total, epoch)
        writer.add_scalar("epoch/lm_ce_loss", avg_lmce, epoch)
        writer.add_scalar("epoch/node_recon_loss", avg_node, epoch)
        writer.add_scalar("epoch/boundary_loss", avg_bnd, epoch)
        writer.add_scalar("epoch/latent_mse_loss", avg_mse, epoch)
        
        # Validation (if available)
        if val_dataloader is not None and (epoch + 1) % args.eval_every_n_epochs == 0:
            val_metrics = run_validation(val_dataloader, epoch + 1)
            writer.add_scalar("val/total_loss", val_metrics['total'], epoch)
            writer.add_scalar("val/lm_ce_loss", val_metrics['lm_ce'], epoch)
            writer.add_scalar("val/node_recon_loss", val_metrics['node_recon'], epoch)
            writer.add_scalar("val/boundary_loss", val_metrics['boundary'], epoch)
            writer.add_scalar("val/latent_mse_loss", val_metrics['latent_mse'], epoch)

        if lm_val_dataloader is not None and (epoch + 1) % args.eval_every_n_epochs == 0:
            lm_ce = run_lm_completion_validation(lm_val_dataloader, epoch + 1)
            writer.add_scalar("val_completion_only/lm_ce", lm_ce, epoch)
        
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
        # Save optimizer / scheduler / training state for resume
        try:
            torch.save(opt.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        except Exception as e:
            print(f"[save] Warning: Could not save optimizer state: {e}")
        if scheduler is not None:
            try:
                torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
            except Exception as e:
                print(f"[save] Warning: Could not save scheduler state: {e}")
        try:
            torch.save({"global_step": global_step, "epoch": epoch + 1}, os.path.join(save_dir, "training_state.pt"))
        except Exception as e:
            print(f"[save] Warning: Could not save training state: {e}")
        print(f"Saved checkpoint to {save_dir}")
    
    writer.add_text("training/status", "COMPLETED", global_step)
    writer.close()
    print(f"\nTraining complete! Final checkpoint: {args.output_dir}")


if __name__ == "__main__":
    train_focused()

