#!/usr/bin/env python3
"""
Quick sanity sampler for HumanEval(+): run incremental inference on a few tasks and
print the completions to stdout so we can inspect formatting.
"""

import argparse
import json
import os
import sys

DEFAULT_EVALPLUS_ROOT = "/data/home/zhangsj/qwen_coder_1.5b/evalplus/evalplus"
if DEFAULT_EVALPLUS_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_EVALPLUS_ROOT)

from evalplus.data import get_human_eval_plus  # type: ignore

import torch

from blt_inference import incremental_generate, load_adapter_and_tokenizer  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a few HumanEval tasks via BLT inference")
    parser.add_argument("--checkpoint", type=str, default="/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/focused_sep_embedding_global_kv_residual_LM_NTP/epoch_10", help="Adapter checkpoint (epoch folder)")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding")
    parser.add_argument("--num_tasks", type=int, default=3, help="Number of tasks to sample from HumanEval+")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index in sorted task list")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--patcher", type=str, default="learned")
    parser.add_argument("--boundary_threshold", type=float, default=0.65)
    parser.add_argument("--min_steps_between_patches", type=int, default=4)
    parser.add_argument("--disable_patching_in_docstring", action="store_true", help="Prevent patching in docstrings/comments")
    parser.add_argument("--use_local_decoder", action="store_true", default=True, help="Enable local decoder refinement")
    parser.add_argument("--max_patch_len", type=int, default=32, help="Maximum span length to rewrite")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    adapter, tokenizer = load_adapter_and_tokenizer(
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        peft_adapter=None,
    )

    dataset = get_human_eval_plus()
    task_items = sorted(dataset.items())
    slice_tasks = task_items[args.start_index : args.start_index + args.num_tasks]

    for task_id, task in slice_tasks:
        prompt = task["prompt"].strip() + "\n"
        completion = incremental_generate(
            model=adapter,
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_new_tokens=args.max_new_tokens,
            patcher=args.patcher,
            boundary_threshold=args.boundary_threshold,
            min_steps_between_patches=args.min_steps_between_patches,
            disable_patching_in_docstring=args.disable_patching_in_docstring,
            use_local_decoder=args.use_local_decoder,
            max_patch_len=args.max_patch_len,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        print("=" * 80)
        print(task_id)
        print("-" * 80)
        print(completion)


if __name__ == "__main__":
    main()

