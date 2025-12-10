"""
Baseline LM-only fine-tuning for Qwen2.5 on plain code `content`.
Matches the default hyperparameters used in blt_focused_training.py.
"""

import argparse
import math
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine LR with linear warmup."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class ContentOnlyDataset(Dataset):
    """Parquet dataset that uses only the `content` column."""

    def __init__(self, parquet_path: str, tokenizer, max_length: int):
        super().__init__()
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet path not found: {parquet_path}")
        # Pandas can read a directory of shards directly.
        self.df = pd.read_parquet(parquet_path)
        content_filter = (self.df["content"].notna()) & (self.df["content"].str.strip() != "")
        self.df = self.df[content_filter]
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"[Dataset] Loaded {len(self.df)} rows from {parquet_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        content = row["content"]
        enc = self.tokenizer(
            content,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class JsonlContentDataset(Dataset):
    """JSONL dataset that concatenates prompt + canonical_solution for LM training/eval."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int):
        super().__init__()
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        import json

        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                prompt = obj.get("prompt", "")
                sol = obj.get("canonical_solution", "")
                text = f"{prompt}{sol}"
                if text.strip():
                    self.rows.append(text)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"[Dataset] Loaded {len(self.rows)} rows from {jsonl_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        content = self.rows[idx]
        enc = self.tokenizer(
            content,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def train_lm_baseline():
    parser = argparse.ArgumentParser(description="Baseline LM-only finetune for Qwen2.5")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding")
    parser.add_argument(
        "--parquet",
        type=str,
        default="/data/home/zhangsj/Data/more_big_code_language/python/baseline_training",
        help="Directory or file of parquet shards with `content` column.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=328)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--trial_name", type=str, default="baseline")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine", "linear"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--val_jsonl", type=str, default="/data/home/zhangsj/Data/HumanEval/human-eval-v2-20210705.jsonl")
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    args = parser.parse_args()

    trial_name = args.trial_name
    if not args.output_dir:
        args.output_dir = f"/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/{trial_name}"
    if not args.log_dir:
        args.log_dir = f"/data/home/zhangsj/AST_decoding/tensorboard_logs/{trial_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = ContentOnlyDataset(args.parquet, tokenizer, max_length=args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_dataloader = None
    if args.val_jsonl and os.path.exists(args.val_jsonl):
        val_dataset = JsonlContentDataset(args.val_jsonl, tokenizer, max_length=args.max_length)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            drop_last=False,
        )
        print(f"[DataLoader] Validation batches: {len(val_dataloader)}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported() and args.dtype in ("auto", "bf16")
                else torch.float16
                if args.dtype in ("auto", "fp16")
                else torch.float32
            )
        except Exception:
            dtype = torch.float16 if args.dtype in ("auto", "fp16") else torch.float32
        model = model.to(device=device, dtype=dtype)
    else:
        dtype = torch.float32
        model = model.to(device=device, dtype=dtype)

    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_optimizer_steps = max(1, args.epochs * steps_per_epoch)

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=total_optimizer_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        print(f"[setup] Using cosine LR scheduler: warmup={args.lr_warmup_steps}, total={total_optimizer_steps}")
    elif args.lr_scheduler == "linear":
        def linear_lambda(step):
            if step < args.lr_warmup_steps:
                return float(step) / float(max(1, args.lr_warmup_steps))
            return max(
                args.min_lr_ratio,
                1.0 - (step - args.lr_warmup_steps) / float(max(1, total_optimizer_steps - args.lr_warmup_steps)),
            )
        scheduler = LambdaLR(opt, linear_lambda)
        print(f"[setup] Using linear LR scheduler: warmup={args.lr_warmup_steps}, total={total_optimizer_steps}")

    writer = SummaryWriter(args.log_dir)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] total_params={total_params:,} trainable={trainable_count:,}, batch_size={args.batch_size}, lr={args.lr}, dtype={dtype}")
    writer.add_text("setup/config", str(vars(args)))
    writer.add_text("setup/trainable_params", str(trainable_count))

    model.train()
    global_step = 0
    accumulated_loss = 0.0

    @torch.no_grad()
    def run_validation(val_loader, epoch_num):
        model.eval()
        losses = []
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            if outputs.loss is not None:
                losses.append(float(outputs.loss.item()))
        model.train()
        if losses:
            avg = float(np.mean(losses))
            print(f"[Val Epoch {epoch_num}] avg_loss={avg:.4f}")
            return avg
        return 0.0

    for epoch in range(args.epochs):
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Accumulate
            unscaled_loss = float(loss.item())
            accumulated_loss += unscaled_loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                avg_accum_loss = accumulated_loss / args.gradient_accumulation_steps
                accumulated_loss = 0.0
                epoch_losses.append(avg_accum_loss)

                if global_step % 50 == 0:
                    writer.add_scalar("loss/total", avg_accum_loss, global_step)
                    writer.add_scalar("lr/learning_rate", opt.param_groups[0]["lr"], global_step)
                    if torch.cuda.is_available():
                        writer.add_scalar("mem/alloc_GB", torch.cuda.memory_allocated() / (1024 ** 3), global_step)
                    print(f"epoch {epoch+1} step {global_step} lr {opt.param_groups[0]['lr']:.2e} | loss {avg_accum_loss:.4f}")

        avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"[Epoch {epoch+1}] Avg loss: {avg_epoch_loss:.4f}")
        writer.add_scalar("epoch/total_loss", avg_epoch_loss, epoch)

        if val_dataloader is not None and ((epoch + 1) % max(1, args.eval_every_n_epochs) == 0):
            val_loss = run_validation(val_dataloader, epoch + 1)
            writer.add_scalar("val/total_loss", val_loss, epoch)

        # Save checkpoint each epoch
        save_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        writer.add_text("checkpoints/epoch", f"Saved checkpoint: {save_dir}", global_step)
        print(f"Saved checkpoint to {save_dir}")

    writer.add_text("training/status", "COMPLETED", global_step)
    writer.close()
    print(f"\nTraining complete! Final checkpoint: {args.output_dir}")


if __name__ == "__main__":
    train_lm_baseline()

