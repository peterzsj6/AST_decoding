#!/usr/bin/env python3
"""
Run EvalPlus benchmarks (HumanEval+/MBPP+) using BLT adapter inference.
Includes timestamp-based output directories and all inference hyperparameters.
Supports base model evaluation by disabling local decoder/encoder.
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, List

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
from evalplus.sanitize import sanitize  # type: ignore


def generate_solutions_for_tasks(
    model,
    tokenizer,
    problems: Dict[str, Dict[str, Any]],
    *,
    n_samples: int = 1,
    max_new_tokens: int = 512,
    patcher: str = "learned",
    boundary_threshold: float = 0.65,
    min_steps_between_patches: int = 4,
    max_patch_len: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    disable_patching_in_docstring: bool = True,
    use_local_decoder: bool = True,
    sanitize_code: bool = True,
) -> List[Dict[str, str]]:
    """
    Generate full solutions (prompt + completion) for each task, possibly with multiple samples.
    Returns a list of dicts with keys: task_id, solution.
    If sanitize_code=True, applies evalplus.sanitize to clean up the generated code.
    """
    samples: List[Dict[str, str]] = []
    task_ids: List[str] = sorted(list(problems.keys()))
    for task_id in task_ids:
        prompt_text: str = problems[task_id]["prompt"]
        entry_point: str = problems[task_id].get("entry_point", "")
        for _ in range(max(1, n_samples)):
            generated_full: str = incremental_generate(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                patcher=patcher,
                boundary_threshold=boundary_threshold,
                min_steps_between_patches=min_steps_between_patches,
                max_patch_len=max_patch_len,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                disable_patching_in_docstring=disable_patching_in_docstring,
                use_local_decoder=use_local_decoder,
            )
            
            # Sanitize the generated code if requested
            if sanitize_code:
                try:
                    sanitized_solution = sanitize(code=generated_full, entrypoint=entry_point)
                    solution_to_store = sanitized_solution
                except Exception as e:
                    print(f"Warning: Failed to sanitize {task_id}: {e}. Using raw output.")
                    solution_to_store = generated_full
            else:
                solution_to_store = generated_full
            
            # Store as full solution to avoid any mismatch with prompt concatenation downstream
            samples.append({"task_id": task_id, "solution": solution_to_store})
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvalPlus on BLT adapter model")
    # Model loading
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved adapter checkpoint (epoch folder) or base model path")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding", help="Base Qwen2.5 path")
    parser.add_argument("--peft_adapter", type=str, default="", help="Optional PEFT LoRA adapter directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    # Dataset and generation
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per task")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    # BLT-specific inference hyperparameters
    parser.add_argument("--patcher", type=str, default="learned", choices=["none", "heuristic", "entropy", "learned"],
                       help="Patch strategy. 'none' disables all patching (base model only).")
    parser.add_argument("--boundary_threshold", type=float, default=0.65, help="Threshold for boundary head predictions")
    parser.add_argument("--min_steps_between_patches", type=int, default=4, help="Minimum steps between boundary patches")
    parser.add_argument("--max_patch_len", type=int, default=128, help="Maximum span length to rewrite")
    parser.add_argument("--disable_patching_in_docstring", action="store_true", default=True, help="Prevent patching in docstrings/comments")
    parser.add_argument("--use_local_decoder", action="store_true", default=True, help="Enable local decoder refinement")
    parser.add_argument("--disable_local_decoder", action="store_true", help="Disable local decoder/encoder (use only global transformer, for base model evaluation)")
    
    # Sampling hyperparameters
    parser.add_argument("--temperature", type=float, default=0.0, help=">0 enables stochastic sampling")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    # EvalPlus evaluation options
    parser.add_argument("--mini", action="store_true", help="Use mini dataset variant")
    parser.add_argument("--noextreme", action="store_true", help="Use no-extreme dataset variant")
    parser.add_argument("--base_only", action="store_true", help="Evaluate only base tests (no extra plus tests)")
    parser.add_argument("--parallel", type=int, default=0, help="Number of parallel workers for evaluation (0=auto)")
    parser.add_argument("--test_details", action="store_true", help="Record per-test details (slower)")
    parser.add_argument("--min_time_limit", type=float, default=0.0, help="Override minimum time limit (0 uses EvalPlus default)")
    parser.add_argument("--gt_time_limit_factor", type=float, default=0.0, help="Override GT time limit factor (0 uses EvalPlus default)")
    parser.add_argument("--version", type=str, default="default")
    
    # Code sanitization
    parser.add_argument("--no_sanitize", action="store_true", help="Skip code sanitization (use raw generated output)")

    # Outputs
    parser.add_argument("--output", type=str, default="", help="Output file path (if not provided, uses timestamp-based path)")
    parser.add_argument("--output_dir", type=str, default="/data/home/zhangsj/evalplus_results", help="Base output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle local decoder flag
    use_local_decoder = args.use_local_decoder and not args.disable_local_decoder
    
    # If disabling local decoder, also set patcher to "none" for base model evaluation
    if args.disable_local_decoder:
        if args.patcher != "none":
            print(f"Warning: --disable_local_decoder is set, but patcher is '{args.patcher}'. Setting patcher to 'none' for base model evaluation.")
            patcher = "none"
        else:
            patcher = "none"
    else:
        patcher = args.patcher

    # Generate timestamp-based output path if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = args.dataset
        model_type = "basemodel" if args.disable_local_decoder else "blt_adapter"
        output_filename = f"{model_type}_{timestamp}.jsonl"
        output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = args.output
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

    # Check if file exists and handle overwrite
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Output file {output_path} already exists. Use --overwrite to replace it.")
        sys.exit(1)

    device = select_device(args.device)
    dtype = select_dtype(device, args.dtype)

    print(f"Loading model from checkpoint: {args.checkpoint}")
    adapter, tokenizer = load_adapter_and_tokenizer(
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        peft_adapter=args.peft_adapter if args.peft_adapter else None,
    )

    # Load dataset problems
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "humaneval":
        problems = get_human_eval_plus(mini=args.mini, noextreme=args.noextreme, version=args.version)
    else:
        problems = get_mbpp_plus(mini=args.mini, noextreme=args.noextreme, version=args.version)

    print(f"Generating solutions for {len(problems)} tasks...")
    print(f"Inference hyperparameters:")
    print(f"  - patcher: {patcher}")
    print(f"  - use_local_decoder: {use_local_decoder}")
    if patcher == "learned":
        print(f"  - boundary_threshold: {args.boundary_threshold}")
        print(f"  - min_steps_between_patches: {args.min_steps_between_patches}")
        print(f"  - max_patch_len: {args.max_patch_len}")
    print(f"  - disable_patching_in_docstring: {args.disable_patching_in_docstring}")
    print(f"  - temperature: {args.temperature}")
    print(f"  - top_p: {args.top_p}")
    print(f"  - repetition_penalty: {args.repetition_penalty}")
    print(f"  - no_repeat_ngram_size: {args.no_repeat_ngram_size}")
    print(f"  - sanitize_code: {not args.no_sanitize}")

    # Generate solutions
    samples = generate_solutions_for_tasks(
        model=adapter,
        tokenizer=tokenizer,
        problems=problems,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        patcher=patcher,
        boundary_threshold=args.boundary_threshold,
        min_steps_between_patches=args.min_steps_between_patches,
        max_patch_len=args.max_patch_len,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        disable_patching_in_docstring=args.disable_patching_in_docstring,
        use_local_decoder=use_local_decoder,
        sanitize_code=not args.no_sanitize,
    )

    # Write samples JSONL
    write_jsonl(output_path, samples, append=False, drop_builtin=True)
    print(f"Wrote {len(samples)} samples to {output_path}")

    # Prepare evaluation kwargs
    eval_kwargs: Dict[str, Any] = {
        "dataset": args.dataset,
        "samples": output_path,
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

    # Run EvalPlus evaluation (prints pass@k and writes results JSON)
    print("\nRunning EvalPlus evaluation...")
    evalplus_evaluate(**eval_kwargs)

    print(f"\nEvaluation completed. Results saved alongside samples at: {output_path}")


if __name__ == "__main__":
    main()

