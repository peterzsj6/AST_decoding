#!/usr/bin/env python3
"""
Run EvalPlus benchmarks (HumanEval+/MBPP+) using BLT adapter inference.
Includes timestamp-based output directories and all inference hyperparameters.
Supports base model evaluation by disabling local decoder/encoder.
"""

import os
import sys

# Parse --gpu argument first (before importing torch) to set CUDA_VISIBLE_DEVICES
# This must be done BEFORE importing torch or any module that imports torch
_gpu_arg = None
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu_arg = sys.argv[i + 1]
        break
    elif arg.startswith("--gpu="):
        _gpu_arg = arg.split("=", 1)[1]
        break

# Set CUDA_VISIBLE_DEVICES before importing torch
# Priority: 1) --gpu argument, 2) existing env var, 3) default GPU 7
if _gpu_arg is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg
elif "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Default to GPU 7

import argparse
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List

# Make project root importable
PROJECT_ROOT = "/data/home/zhangsj"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Make local EvalPlus repo importable (uninstalled source tree)
EVALPLUS_SRC = "/data/home/zhangsj/qwen_coder_1.5b/evalplus/evalplus"
if EVALPLUS_SRC not in sys.path:
    sys.path.insert(0, EVALPLUS_SRC)

import torch

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
    collect_stats: bool = True,
) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Generate full solutions (prompt + completion) for each task, possibly with multiple samples.
    Returns a tuple of (samples, stats) where:
    - samples: list of dicts with keys: task_id, solution
    - stats: dict with aggregated statistics (fired_boundaries, total_tokens, etc.)
    If sanitize_code=True, applies evalplus.sanitize to clean up the generated code.
    """
    samples: List[Dict[str, str]] = []
    raw_samples: List[Dict[str, str]] = []
    task_ids: List[str] = sorted(list(problems.keys()))
    
    # Statistics aggregation
    total_fired_boundaries = 0
    total_tokens = 0
    total_tasks = 0
    boundary_stats_per_task: List[Dict[str, Any]] = []
    
    for task_id in task_ids:
        prompt_text: str = problems[task_id]["prompt"]
        entry_point: str = problems[task_id].get("entry_point", "")
        for sample_idx in range(max(1, n_samples)):
            result = incremental_generate(
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
                collect_stats=collect_stats,
            )
            
            # Handle return value (string or tuple with stats)
            if isinstance(result, tuple):
                generated_full, stats = result
                fired_boundaries = stats.get("fired_boundaries", 0)
                tokens = stats.get("total_tokens", 0)
                total_fired_boundaries += fired_boundaries
                total_tokens += tokens
                boundary_stats_per_task.append({
                    "task_id": task_id,
                    "sample": sample_idx,
                    "fired_boundaries": fired_boundaries,
                    "total_tokens": tokens,
                })
            else:
                generated_full = result
                fired_boundaries = 0
                tokens = 0
            
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
            # Always store raw output (pre-sanitization) for debugging
            raw_samples.append({"task_id": task_id, "solution": generated_full})
            total_tasks += 1
    
    # Aggregate statistics
    stats = {
        "total_tasks": total_tasks,
        "total_samples": len(samples),
        "total_fired_boundaries": total_fired_boundaries,
        "total_tokens": total_tokens,
        "avg_boundaries_per_task": total_fired_boundaries / total_tasks if total_tasks > 0 else 0.0,
        "avg_boundaries_per_token": total_fired_boundaries / total_tokens if total_tokens > 0 else 0.0,
        "boundary_stats_per_task": boundary_stats_per_task,
    }
    
    return samples, raw_samples, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvalPlus on BLT adapter model")
    # Model loading
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved adapter checkpoint (epoch folder) or base model path")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding", help="Base Qwen2.5 path")
    parser.add_argument("--peft_adapter", type=str, default="", help="Optional PEFT LoRA adapter directory")
    parser.add_argument("--gpu", type=int, default=5, help="GPU device ID to use (sets CUDA_VISIBLE_DEVICES before importing PyTorch)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use. GPU selection is handled by CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    # Dataset and generation
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per task")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--task_limit",
        type=int,
        default=0,
        help="If >0, only evaluate the first N tasks (sorted by task_id). Useful for quick debugging.",
    )
    
    # BLT-specific inference hyperparameters
    parser.add_argument("--patcher", type=str, default="learned", choices=["none", "heuristic", "entropy", "learned"],
                       help="Patch strategy. 'none' disables all patching (base model only).")
    parser.add_argument("--boundary_threshold", type=float, default=0.2, help="Threshold for boundary head predictions")
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
    
    # Show what GPU is being used (already set before torch import)
    current_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")

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
        
        # Extract meaningful checkpoint identifier from checkpoint path
        checkpoint_path = os.path.normpath(args.checkpoint)
        
        # Get the last meaningful directory name (e.g., "epoch_10" or model config name)
        checkpoint_name = os.path.basename(checkpoint_path)
        
        # If it's a generic name like "epoch_10", try to include parent directory for context
        # This helps identify the model configuration
        if checkpoint_name.startswith("epoch_") or checkpoint_name in ["", ".", ".."]:
            parent_dir = os.path.basename(os.path.dirname(checkpoint_path))
            if parent_dir and parent_dir not in ["", ".", "..", "checkpoints", "checkpoint"]:
                # Combine parent (config) and current (epoch) for better identification
                if checkpoint_name.startswith("epoch_"):
                    checkpoint_name = f"{parent_dir}_{checkpoint_name}"
                else:
                    checkpoint_name = parent_dir
            elif not checkpoint_name or checkpoint_name in ["", ".", ".."]:
                checkpoint_name = "checkpoint"
        
        # Clean up checkpoint name: remove any problematic characters for filenames
        checkpoint_name = checkpoint_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        
        output_filename = f"{model_type}_{checkpoint_name}_{timestamp}.jsonl"
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

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"Using device: {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"PyTorch sees {torch.cuda.device_count()} GPU(s), using GPU 0 (which corresponds to the GPU specified in CUDA_VISIBLE_DEVICES)")

    print(f"Loading model from checkpoint: {args.checkpoint}")
    print(f"[DEBUG] Checkpoint path (normalized): {os.path.normpath(args.checkpoint)}")
    print(f"[DEBUG] Checkpoint exists: {os.path.exists(args.checkpoint)}")
    adapter, tokenizer = load_adapter_and_tokenizer(
        checkpoint_path=args.checkpoint,
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        peft_adapter=args.peft_adapter if args.peft_adapter else None,
    )
    print(f"[DEBUG] Model loaded successfully from: {args.checkpoint}")

    # Load dataset problems
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "humaneval":
        problems = get_human_eval_plus(mini=args.mini, noextreme=args.noextreme, version=args.version)
    else:
        problems = get_mbpp_plus(mini=args.mini, noextreme=args.noextreme, version=args.version)

    # Optional task limiting for fast debugging
    if int(args.task_limit) > 0:
        # Prefer numeric ordering for HumanEval/0..HumanEval/163 style ids
        def _task_sort_key(tid: str):
            try:
                return int(str(tid).split("/")[-1])
            except Exception:
                return str(tid)

        task_ids_sorted = sorted(list(problems.keys()), key=_task_sort_key)
        keep = task_ids_sorted[: int(args.task_limit)]
        problems = {k: problems[k] for k in keep}
        print(f"[DEBUG] task_limit={args.task_limit} enabled: evaluating {len(problems)} task(s): {keep}")

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
    samples, raw_samples, stats = generate_solutions_for_tasks(
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
        collect_stats=True,
    )

    # Print boundary firing statistics
    print(f"\n{'='*80}")
    print("Boundary Firing Statistics")
    print(f"{'='*80}")
    print(f"Total tasks evaluated: {stats['total_tasks']}")
    print(f"Total samples generated: {stats['total_samples']}")
    print(f"Total boundaries fired: {stats['total_fired_boundaries']}")
    print(f"Total tokens generated: {stats['total_tokens']}")
    print(f"Average boundaries per task: {stats['avg_boundaries_per_task']:.2f}")
    print(f"Average boundaries per token: {stats['avg_boundaries_per_token']:.4f}")
    if stats['total_tokens'] > 0:
        boundary_rate = (stats['total_fired_boundaries'] / stats['total_tokens']) * 100
        print(f"Boundary firing rate: {boundary_rate:.2f}% of tokens")
    print(f"{'='*80}\n")

    # Write raw samples JSONL (pre-sanitization) alongside the sanitized one
    raw_output_path = output_path.replace(".jsonl", ".raw.jsonl")
    write_jsonl(raw_output_path, raw_samples, append=False, drop_builtin=True)
    print(f"Wrote {len(raw_samples)} raw samples to {raw_output_path}")

    # Write samples JSONL (sanitized unless --no_sanitize)
    write_jsonl(output_path, samples, append=False, drop_builtin=True)
    print(f"Wrote {len(samples)} samples to {output_path}")
    
    # Debug: Show sample file info
    if samples:
        print(f"[DEBUG] Sample file info:")
        print(f"[DEBUG]   Total samples: {len(samples)}")
        print(f"[DEBUG]   First task_id: {samples[0].get('task_id', 'N/A')}")
        first_solution = samples[0].get('solution', '')
        print(f"[DEBUG]   First solution length: {len(first_solution)} chars")
        print(f"[DEBUG]   First solution preview: {first_solution[:100]}...")
        # Compute a simple hash of first sample for comparison
        first_sample_hash = hashlib.md5(json.dumps(samples[0], sort_keys=True).encode()).hexdigest()[:8]
        print(f"[DEBUG]   First sample hash (MD5 first 8 chars): {first_sample_hash}")
    
    # Write statistics to a separate file
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote statistics to {stats_path}")

    # If we limited tasks for quick debugging, skip EvalPlus evaluation.
    # EvalPlus's evaluate() loads the full dataset internally and asserts that all tasks are present,
    # so evaluating a truncated samples file will raise "Missing problems in samples".
    if int(args.task_limit) > 0:
        print(f"[DEBUG] task_limit={args.task_limit}: skipping EvalPlus evaluation (samples file is intentionally incomplete).")
        return

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

    # Delete cached results files if --overwrite is set (to force fresh evaluation)
    if args.overwrite:
        # EvalPlus checks for both .eval_results.json and _eval_results.json (legacy)
        results_file_standard = output_path.replace(".jsonl", ".eval_results.json")
        results_file_legacy = output_path.replace(".jsonl", "_eval_results.json")
        
        print(f"\n[DEBUG] Checking for cached results files before deletion:")
        print(f"[DEBUG]   Standard results file: {results_file_standard}")
        print(f"[DEBUG]   Standard file exists: {os.path.exists(results_file_standard)}")
        print(f"[DEBUG]   Legacy results file: {results_file_legacy}")
        print(f"[DEBUG]   Legacy file exists: {os.path.exists(results_file_legacy)}")
        
        deleted_any = False
        if os.path.exists(results_file_standard):
            os.remove(results_file_standard)
            deleted_any = True
            print(f"[DEBUG] ✓ Deleted cached results file: {results_file_standard}")
            # Verify deletion
            if os.path.exists(results_file_standard):
                print(f"[ERROR] Failed to delete {results_file_standard} - file still exists!")
            else:
                print(f"[DEBUG] ✓ Verified deletion successful for {results_file_standard}")
        if os.path.exists(results_file_legacy):
            os.remove(results_file_legacy)
            deleted_any = True
            print(f"[DEBUG] ✓ Deleted cached results file (legacy): {results_file_legacy}")
            # Verify deletion
            if os.path.exists(results_file_legacy):
                print(f"[ERROR] Failed to delete {results_file_legacy} - file still exists!")
            else:
                print(f"[DEBUG] ✓ Verified deletion successful for {results_file_legacy}")
        
        if not deleted_any:
            print(f"[DEBUG] No cached results files found to delete")
        
        # Final check: ensure files don't exist before calling evalplus_evaluate
        print(f"\n[DEBUG] Final verification before evaluation:")
        print(f"[DEBUG]   Standard file exists: {os.path.exists(results_file_standard)}")
        print(f"[DEBUG]   Legacy file exists: {os.path.exists(results_file_legacy)}")
        print(f"[DEBUG]   EvalPlus will check for: {results_file_standard}")
        if os.path.exists(results_file_standard) or os.path.exists(results_file_legacy):
            print(f"[WARNING] Results files still exist! EvalPlus may load cached results!")

    # Run EvalPlus evaluation (prints pass@k and writes results JSON)
    print("\nRunning EvalPlus evaluation...")
    print(f"[DEBUG] Samples file: {eval_kwargs['samples']}")
    print(f"[DEBUG] EvalPlus will construct result path from samples file")
    evalplus_evaluate(**eval_kwargs)

    print(f"\nEvaluation completed. Results saved alongside samples at: {output_path}")


if __name__ == "__main__":
    main()

