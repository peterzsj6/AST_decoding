import os
import sys
import argparse
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

# Make project root importable
PROJECT_ROOT = "/data/home/zhangsj"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from AST_decoding.blt_adapter_model import create_blt_adapter_model, BLTAdapterModel  # type: ignore
# Optional PEFT import (only needed if --peft_adapter is provided)
try:
    from peft import PeftModel  # type: ignore
except Exception:  # pragma: no cover
    PeftModel = None  # type: ignore


def select_device(preferred_device: str = "auto") -> str:
    if preferred_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred_device in {"cuda", "cpu"}:
        if preferred_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return preferred_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(device: str, preferred_dtype: str = "auto") -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if preferred_dtype == "bf16":
        return torch.bfloat16
    if preferred_dtype == "fp16":
        return torch.float16
    if preferred_dtype == "fp32":
        return torch.float32
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        return torch.float16


def load_adapter_and_tokenizer(checkpoint_path: Optional[str], model_path: str, device: str, dtype: torch.dtype, peft_adapter: Optional[str] = None) -> (BLTAdapterModel, Any):
    if checkpoint_path and os.path.isdir(checkpoint_path):
        # Load from saved checkpoint
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base = BLTAdapterModel.from_pretrained(checkpoint_path, torch_dtype=dtype)
        adapter = PeftModel.from_pretrained(base, peft_adapter) if (peft_adapter and os.path.isdir(peft_adapter)) else base
        try:
            adapter = adapter.to(device=device, dtype=dtype)
        except RuntimeError as e:
            # Graceful fallback to CPU on CUDA OOM
            if "out of memory" in str(e).lower() and device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                adapter = adapter.to(device="cpu", dtype=torch.float32)
            else:
                raise
    else:
        # Fresh adapter wrapping base model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base = create_blt_adapter_model(model_path)
        adapter = PeftModel.from_pretrained(base, peft_adapter) if (peft_adapter and os.path.isdir(peft_adapter)) else base
        try:
            adapter = adapter.to(device=device, dtype=dtype)
        except RuntimeError as e:
            # Graceful fallback to CPU on CUDA OOM
            if "out of memory" in str(e).lower() and device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                adapter = adapter.to(device="cpu", dtype=torch.float32)
            else:
                raise
    # Reduce memory during inference
    try:
        adapter.config.use_cache = False
    except Exception:
        pass
    adapter.eval()
    return adapter, tokenizer


def is_boundary_heuristic(tokenizer, token_id: int) -> bool:
    """
    Simple heuristic: boundary if decoded token ends with whitespace or punctuation.
    """
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=True)
    except Exception:
        return False
    if len(text) == 0:
        return False
    ch = text[-1]
    return ch.isspace() or ch in {':', ';', ',', '.', '(', ')', '{', '}', '[', ']', '-', '=', '+', '*', '/', '\\'}


def compute_entropy(logits: torch.Tensor) -> float:
    """
    logits: [V]
    """
    probs = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    ent = -torch.sum(probs * logp).item()
    return ent


def is_inside_docstring(tokenizer, input_ids: torch.Tensor) -> bool:
    """
    Heuristic: we're inside a triple-quoted docstring if the decoded text has an odd
    number of triple quotes (either \"\"\" or ''').
    """
    try:
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    except Exception:
        return False
    dq = text.count('\"\"\"')
    sq = text.count("'''")
    return (dq % 2 == 1) or (sq % 2 == 1)


@torch.no_grad()
def incremental_generate(
    model: BLTAdapterModel,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 128,
    patcher: str = "none",  # none|heuristic|entropy|learned
    entropy_threshold: float = 4.0,
    max_patch_len: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    boundary_threshold: float = 0.7,
    min_steps_between_patches: int = 8,
    disable_patching_in_docstring: bool = True,
) -> str:
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096
    )
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # Helper to append tokens
    def append_tokens(toks: List[int]):
        nonlocal input_ids, attention_mask
        add = torch.tensor([toks], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, add], dim=1)
        add_mask = torch.ones_like(add)
        attention_mask = torch.cat([attention_mask, add_mask], dim=1)

    # Simple sampler
    def sample_from_logits(logits: torch.Tensor) -> int:
        if temperature > 0.0:
            logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = (cumprobs > top_p).nonzero(as_tuple=False)
            if cutoff.numel() > 0:
                last_idx = cutoff[0, 1]
                sorted_logits = sorted_logits[:, :last_idx+1]
                sorted_indices = sorted_indices[:, :last_idx+1]
                probs = F.softmax(sorted_logits, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1)
                return int(sorted_indices.gather(1, sampled_idx).item())
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    new_tokens = 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else getattr(model.config, "eos_token_id", None)
    steps_since_patch = 1_000_000  # large to allow first eligible patch

    while new_tokens < max_new_tokens:
        # Global step: get logits and hidden state for last token
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        logits = outputs.logits[:, -1, :]  # [1, V]
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, H]  type: ignore[attr-defined]
        # Proactively free large hidden_states collection before branching
        try:
            del outputs
        except Exception:
            pass

        # Decide boundary
        do_patch = False
        boundary_confidence = None
        # Optional guard: avoid patching inside triple-quoted docstrings
        if not (disable_patching_in_docstring and is_inside_docstring(tokenizer, input_ids)):
            if patcher == "none":
                do_patch = False
            elif patcher == "heuristic":
                if input_ids.size(1) > 0:
                    prev_id = int(input_ids[0, -1].item())
                    do_patch = is_boundary_heuristic(tokenizer, prev_id)
            elif patcher == "entropy":
                ent = compute_entropy(logits[0])
                # High entropy -> start of a hard boundary (use local decoder after)
                do_patch = ent > entropy_threshold
            elif patcher == "learned":
                # Use boundary_head on last_hidden
                with torch.no_grad():
                    boundary_logits = model.boundary_head(last_hidden)  # [1, 2]
                    probs = torch.softmax(boundary_logits, dim=-1)
                    boundary_confidence = float(probs[0, 1].item())
                    # Patch only if confident enough AND we have spaced out recent patches
                    do_patch = (boundary_confidence >= boundary_threshold) and (steps_since_patch >= int(min_steps_between_patches))
            else:
                do_patch = False

        if do_patch:
            # Predict span latent and decode locally
            span_latent = model.latent_from_global(last_hidden)  # [1, H]
            patch_ids = model.generate_node_tokens(span_latent.squeeze(0), max_len=max_patch_len)  # [T]
            if patch_ids.numel() == 0:
                # Fallback to single token
                next_id = int(torch.argmax(logits, dim=-1).item())
                append_tokens([next_id])
                new_tokens += 1
                steps_since_patch += 1
            else:
                append_tokens(patch_ids.tolist())
                new_tokens += int(patch_ids.numel())
                steps_since_patch = 0
                if eos_id is not None and int(patch_ids[-1].item()) == eos_id:
                    break
        else:
            # Single global token step
            if temperature <= 0.0 and top_p >= 1.0:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                next_id = sample_from_logits(logits)
            append_tokens([next_id])
            new_tokens += 1
            steps_since_patch += 1
            if eos_id is not None and next_id == eos_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLT Adapter Inference with patchers")
    parser.add_argument("--checkpoint", type=str, default="/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/blt_adapter_frozen_global_transformer_with_span_boundary_and_type_loss/epoch_3", help="Path to saved adapter checkpoint (optional)")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding", help="Base Qwen2.5 path if no checkpoint provided")
    parser.add_argument("--input_file", type=str, required=True, help="Source code file (python)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--patcher", type=str, default="none", choices=["none", "heuristic", "entropy", "learned"])
    parser.add_argument("--entropy_threshold", type=float, default=4.0)
    parser.add_argument("--max_patch_len", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--peft_adapter", type=str, default="", help="Path to PEFT LoRA adapter directory (optional)")
    parser.add_argument("--boundary_threshold", type=float, default=0.7, help="Probability threshold for learned patcher to trigger a patch")
    parser.add_argument("--min_steps_between_patches", type=int, default=8, help="Minimum global steps between two patches for learned patcher")
    parser.add_argument("--disable_patching_in_docstring", action="store_true", help="If set, prevents patching while inside triple-quoted docstrings")
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    dtype = select_dtype(device, args.dtype)

    adapter, tokenizer = load_adapter_and_tokenizer(
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        peft_adapter=args.peft_adapter if args.peft_adapter else None
    )

    with open(args.input_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    output = incremental_generate(
        model=adapter,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=args.max_new_tokens,
        patcher=args.patcher,
        entropy_threshold=args.entropy_threshold,
        max_patch_len=args.max_patch_len,
        temperature=args.temperature,
        top_p=args.top_p,
        boundary_threshold=args.boundary_threshold,
        min_steps_between_patches=args.min_steps_between_patches,
        disable_patching_in_docstring=args.disable_patching_in_docstring
    )
    print(output)


if __name__ == "__main__":
    main()


