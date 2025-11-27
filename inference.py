import os
import sys
import argparse
import random
import numpy as np
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer


# Avoid side-effects in K_distillation_test1.py that set CUDA_VISIBLE_DEVICES
os.environ.setdefault("LOCAL_RANK", "0")

# Make project root importable
PROJECT_ROOT = "/data/home/zhangsj"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the custom model and helper module
# Note: importing this module executes some top-level code (e.g., tokenizer/config loads),
#       but we will override the module-level tokenizer before span processing to ensure alignment.
from AST_decoding.K_distillation_test1 import CustomQwen2Model  # type: ignore
import AST_decoding.K_distillation_test1 as kd  # type: ignore


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
    # auto
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        return torch.float16


def load_model_and_tokenizer(checkpoint_path: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = CustomQwen2Model.from_pretrained(
        checkpoint_path,
        dtype=dtype,                 # use new API, avoids deprecation
        low_cpu_mem_usage=False,     # ensure real tensors during __init__
        device_map=None              # avoid meta device mapping
    ).to(device)
    model.eval()
    return model, tokenizer


def build_span_metadata_from_code(code_text: str, language: str, tokenizer) -> Optional[Dict[str, Any]]:
    # Ensure the span builder in kd uses the same tokenizer as the checkpoint
    kd.tokenizer = tokenizer  # type: ignore[attr-defined]
    try:
        span_meta = kd.process_code_for_spans(code_text, language=language)  # type: ignore[attr-defined]
        if span_meta is None:
            return None
        # Ensure batch dimension presence and correct structure for generate()
        # process_code_for_spans already returns tensors with batch dim and raw_spans as list
        return span_meta
    except Exception as exc:
        print(f"[warn] Failed to build span metadata: {exc}")
        return None


def run_generation(
    model: CustomQwen2Model,
    tokenizer,
    prompt_text: str,
    language: str,
    use_spans: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    num_beams: int,
    only_new_tokens: bool,
    device: str
) -> str:
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,  # keep alignment with span token indices
        truncation=True,
        max_length=4096  # allow long prompts; model will clamp as needed
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    model_kwargs = {}
    if use_spans:
        span_metadata = build_span_metadata_from_code(prompt_text, language, tokenizer)
        if span_metadata is None:
            print("[info] Span parsing failed; falling back to token-only embeddings.")
        else:
            # raw_spans is a per-batch list; process_code_for_spans already returns with a batch dim
            model_kwargs["span_metadata"] = span_metadata

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else getattr(model.config, "eos_token_id", None)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if num_beams and num_beams > 1:
            gen_kwargs["num_beams"] = num_beams
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
            **model_kwargs
        )

    if only_new_tokens:
        new_tokens = output_ids[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def maybe_demo_node_decode(
    model: CustomQwen2Model,
    tokenizer,
    prompt_text: str,
    language: str,
    device: str,
    span_index: int,
    max_len: int
) -> Optional[str]:
    # Re-tokenize to compute embeddings once
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096
    )
    input_ids = enc["input_ids"].to(device)

    # Build spans
    span_metadata = build_span_metadata_from_code(prompt_text, language, tokenizer)
    if span_metadata is None or not isinstance(span_metadata.get("raw_spans"), list):
        return None

    with torch.no_grad():
        emb_out = model.model.embed_tokens(input_ids, span_metadata)  # type: ignore[attr-defined]
        inputs_embeds = emb_out["embeddings"]

        spans_list = span_metadata["raw_spans"][0] if len(span_metadata["raw_spans"]) > 0 else []
        if not spans_list:
            return None
        idx = max(0, min(span_index, len(spans_list) - 1))
        token_indices = spans_list[idx].get("token_indices", [])
        if not token_indices:
            return None
        token_indices_tensor = torch.tensor(token_indices, device=inputs_embeds.device, dtype=torch.long)
        node_tokens = model.decode_node_from_span(input_ids, inputs_embeds, token_indices_tensor, max_len=max_len)  # type: ignore[attr-defined]

    return tokenizer.decode(node_tokens, skip_special_tokens=True)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Span-aware inference for CustomQwen2Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/home/zhangsj/AST_decoding/checkpoints/11_17_mean_pooling/best_model",
        help="Path to the saved model checkpoint directory"
    )
    group_input = parser.add_mutually_exclusive_group(required=True)
    group_input.add_argument("--prompt", type=str, help="Inline source code prompt")
    group_input.add_argument("--input_file", type=str, help="Path to a file containing source code prompt")
    parser.add_argument("--language", type=str, default="python", help="Programming language for AST parsing")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Penalty >1.0 discourages repetition")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4, help="Disallow repeating ngrams of this size")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width (1 = disabled)")
    parser.add_argument("--only_new_tokens", action="store_true", help="Print only the continuation, omit the prompt echo")
    parser.add_argument("--no_spans", action="store_true", help="Disable span-aware embeddings (token-only)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Compute dtype")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--demo_node_decode", action="store_true", help="Also demo node-level local decoding on one span")
    parser.add_argument("--node_span_index", type=int, default=0, help="Index of span to decode with local decoder")
    parser.add_argument("--node_max_len", type=int, default=64, help="Max node tokens to decode with local decoder")
    return parser.parse_args()


def main():
    args = parse_args()

    device = select_device(args.device)
    dtype = select_dtype(device, args.dtype)

    # Seeding for reproducibility (when sampling)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device, dtype)

    prompt_text = read_text(args.input_file) if args.input_file else args.prompt
    if prompt_text is None:
        print("No prompt provided.")
        return

    print("=== Running span-aware generation ===")
    output = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        language=args.language,
        use_spans=not args.no_spans,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        num_beams=args.num_beams,
        only_new_tokens=args.only_new_tokens,
        device=device
    )
    print("\n--- Generated Output ---\n")
    print(output)

    if args.demo_node_decode and not args.no_spans:
        print("\n=== Demo: node-level local decoding (single span) ===")
        node_text = maybe_demo_node_decode(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            language=args.language,
            device=device,
            span_index=args.node_span_index,
            max_len=args.node_max_len
        )
        if node_text is None:
            print("Node-level demo not available (no spans parsed or invalid indices).")
        else:
            print("\n--- Node Decode ---\n")
            print(node_text)


if __name__ == "__main__":
    main()


