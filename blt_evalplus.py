import os
import sys
import argparse
from typing import Dict, Any, List, Tuple

import torch

# Make project root importable
PROJECT_ROOT = "/data/home/zhangsj"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Make local EvalPlus repo importable (uninstalled source tree)
EVALPLUS_SRC = "/data/home/zhangsj/qwen_coder_1.5b/evalplus/evalplus"
if EVALPLUS_SRC not in sys.path:
    sys.path.insert(0, EVALPLUS_SRC)

# Reuse loader and generation utilities
from AST_decoding.blt_inference import (  # type: ignore
    select_device,
    select_dtype,
    load_adapter_and_tokenizer,
    incremental_generate,
)

# EvalPlus imports (from local repo)
from evalplus.data.humaneval import get_human_eval_plus  # type: ignore
from evalplus.data.mbpp import get_mbpp_plus  # type: ignore
from evalplus.data.utils import write_jsonl  # type: ignore
from evalplus.evaluate import evaluate as evalplus_evaluate  # type: ignore


def generate_solutions_for_tasks(
    model,
    tokenizer,
    problems: Dict[str, Dict[str, Any]],
    *,
    n_samples: int = 1,
    max_new_tokens: int = 256,
    patcher: str = "none",
    entropy_threshold: float = 4.0,
    max_patch_len: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> List[Dict[str, str]]:
    """
    Generate full solutions (prompt + completion) for each task, possibly with multiple samples.
    Returns a list of dicts with keys: task_id, solution.
    """
    samples: List[Dict[str, str]] = []
    task_ids: List[str] = sorted(list(problems.keys()))
    for task_id in task_ids:
        prompt_text: str = problems[task_id]["prompt"]
        for _ in range(max(1, n_samples)):
            generated_full: str = incremental_generate(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                patcher=patcher,
                entropy_threshold=entropy_threshold,
                max_patch_len=max_patch_len,
                temperature=temperature,
                top_p=top_p,
            )
            # Store as full solution to avoid any mismatch with prompt concatenation downstream
            samples.append({"task_id": task_id, "solution": generated_full})
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvalPlus on BLT adapter model")
    # Model loading
    parser.add_argument("--checkpoint", type=str, default="/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/blt_adapter_frozen_global_transformer_with_span_boundary_and_type_loss/epoch_3", help="Path to saved adapter checkpoint (optional)")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding", help="Base Qwen2.5 path if no checkpoint provided")
    parser.add_argument("--peft_adapter", type=str, default="", help="Optional PEFT LoRA adapter directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    # Dataset and generation
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per task")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--patcher", type=str, default="none", choices=["none", "heuristic", "entropy", "learned"])
    parser.add_argument("--entropy_threshold", type=float, default=4.0)
    parser.add_argument("--max_patch_len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help=">0 enables stochastic sampling")
    parser.add_argument("--top_p", type=float, default=1.0)

    # EvalPlus evaluation options
    parser.add_argument("--mini", action="store_true", help="Use mini dataset variant")
    parser.add_argument("--noextreme", action="store_true", help="Use no-extreme dataset variant")
    parser.add_argument("--base_only", action="store_true", help="Evaluate only base tests (no extra plus tests)")
    parser.add_argument("--parallel", type=int, default=0, help="Number of parallel workers for evaluation (0=auto)")
    parser.add_argument("--test_details", action="store_true", help="Record per-test details (slower)")
    parser.add_argument("--min_time_limit", type=float, default=0.0, help="Override minimum time limit (0 uses EvalPlus default)")
    parser.add_argument("--gt_time_limit_factor", type=float, default=0.0, help="Override GT time limit factor (0 uses EvalPlus default)")
    parser.add_argument("--version", type=str, default="default")

    # Outputs
    parser.add_argument("--output_dir", type=str, default="/data/home/zhangsj/AST_decoding/evalplus_runs")
    parser.add_argument("--samples_filename", type=str, default="", help="Optional custom filename for generated samples JSONL")
    parser.add_argument("--results_filename", type=str, default="", help="Optional path for eval results JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = select_device(args.device)
    dtype = select_dtype(device, args.dtype)

    adapter, tokenizer = load_adapter_and_tokenizer(
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        peft_adapter=args.peft_adapter if args.peft_adapter else None,
    )

    # Load dataset problems
    if args.dataset == "humaneval":
        problems = get_human_eval_plus(mini=args.mini, noextreme=args.noextreme, version=args.version)
    else:
        problems = get_mbpp_plus(mini=args.mini, noextreme=args.noextreme, version=args.version)

    # Generate solutions
    samples = generate_solutions_for_tasks(
        model=adapter,
        tokenizer=tokenizer,
        problems=problems,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        patcher=args.patcher,
        entropy_threshold=args.entropy_threshold,
        max_patch_len=args.max_patch_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Write samples JSONL
    samples_path = (
        args.samples_filename
        if args.samples_filename
        else os.path.join(args.output_dir, f"{args.dataset}_samples.jsonl")
    )
    write_jsonl(samples_path, samples, append=False, drop_builtin=True)
    print(f"Wrote {len(samples)} samples to {samples_path}")

    # Prepare evaluation kwargs
    eval_kwargs: Dict[str, Any] = {
        "dataset": args.dataset,
        "samples": samples_path,
        "base_only": bool(args.base_only),
        "parallel": (args.parallel if args.parallel > 0 else None),
        "i_just_wanna_run": False,
        "test_details": bool(args.test_details),
        "mini": bool(args.mini),
        "noextreme": bool(args.noextreme),
        "version": args.version,
    }
    if args.min_time_limit > 0.0:
        eval_kwargs["min_time_limit"] = float(args.min_time_limit)
    if args.gt_time_limit_factor > 0.0:
        eval_kwargs["gt_time_limit_factor"] = float(args.gt_time_limit_factor)
    if args.results_filename:
        eval_kwargs["output_file"] = args.results_filename

    # Run EvalPlus evaluation (prints pass@k and writes results JSON)
    evalplus_evaluate(**eval_kwargs)

    print("Evaluation completed.")


if __name__ == "__main__":
    main()


