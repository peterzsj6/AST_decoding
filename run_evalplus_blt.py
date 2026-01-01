#!/usr/bin/env python3
"""
Run EvalPlus benchmarks (HumanEval+/MBPP+) using BLT adapter inference.
Includes timestamp-based output directories and all inference hyperparameters.
Supports base model evaluation by disabling local decoder/encoder.

Also supports evaluating a plain HuggingFace base model (no BLT wrapper) via the
vendored EvalPlus HF provider by using `--backend hf`.
"""

import os
import sys

# Avoid HuggingFace tokenizer's "forked after parallelism" spam when EvalPlus uses multiprocessing.
# This must be set before any `transformers`/`tokenizers` code is imported/used.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Default to GPU 7

import argparse
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List

# Make project root importable
# Try to detect the project root dynamically based on script location
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure this directory is importable (so we can `import blt_inference` when running as a script)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# If script is at /data/AST_decoding/run_evalplus_blt.py, project root should be /data
# Check if we're in AST_decoding directory
if os.path.basename(_script_dir) == "AST_decoding":
    PROJECT_ROOT = os.path.dirname(_script_dir)  # Parent of AST_decoding
else:
    # Fallback to hardcoded path if structure is different
    PROJECT_ROOT = "/data/home/zhangsj"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Make local EvalPlus repo importable (uninstalled source tree).
# Prefer the vendored copy inside this repo (contains local patches / bugfixes),
# fallback to the older external path if needed.
EVALPLUS_SRC_VENDOR_LOCAL = os.path.join(_script_dir, "evalplus", "evalplus")
EVALPLUS_PARENT_VENDOR_LOCAL = os.path.join(_script_dir, "evalplus")
EVALPLUS_SRC_VENDOR = "/data/home/zhangsj/AST_decoding/evalplus/evalplus"
EVALPLUS_PARENT_VENDOR = "/data/home/zhangsj/AST_decoding/evalplus"
EVALPLUS_SRC_FALLBACK = "/data/home/zhangsj/qwen_coder_1.5b/evalplus/evalplus"
EVALPLUS_PARENT_FALLBACK = "/data/home/zhangsj/qwen_coder_1.5b/evalplus"

# Try local evalplus first, then fallback paths
if os.path.isdir(EVALPLUS_SRC_VENDOR_LOCAL):
    if EVALPLUS_SRC_VENDOR_LOCAL not in sys.path:
        sys.path.insert(0, EVALPLUS_SRC_VENDOR_LOCAL)
    if EVALPLUS_PARENT_VENDOR_LOCAL not in sys.path:
        sys.path.insert(0, EVALPLUS_PARENT_VENDOR_LOCAL)
elif os.path.isdir(EVALPLUS_SRC_VENDOR):
    if EVALPLUS_SRC_VENDOR not in sys.path:
        sys.path.insert(0, EVALPLUS_SRC_VENDOR)
    if EVALPLUS_PARENT_VENDOR not in sys.path:
        sys.path.insert(0, EVALPLUS_PARENT_VENDOR)
else:
    if EVALPLUS_SRC_FALLBACK not in sys.path:
        sys.path.insert(0, EVALPLUS_SRC_FALLBACK)
    if EVALPLUS_PARENT_FALLBACK not in sys.path:
        sys.path.insert(0, EVALPLUS_PARENT_FALLBACK)

import torch

# Reuse loader and generation utilities
try:
    # Works when running as `python -m AST_decoding.run_evalplus_blt` or when /data is on PYTHONPATH
    from AST_decoding.blt_inference import (  # type: ignore
        select_device,
        select_dtype,
        load_adapter_and_tokenizer,
        incremental_generate,
    )
except ModuleNotFoundError:
    # Works when running as `python /data/AST_decoding/run_evalplus_blt.py`
    from blt_inference import (  # type: ignore
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
    local_decoder_mode: str = "refine",
    disable_local_encoder_only: bool = False,
    min_rewrite_span_len: int = 8,
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
                local_decoder_mode=str(local_decoder_mode),
                disable_local_encoder_only=disable_local_encoder_only,
                min_rewrite_span_len=int(min_rewrite_span_len),
                collect_stats=collect_stats,
            )
            
            # Handle return value (string or tuple with stats)
            if isinstance(result, tuple):
                generated_full, stats = result
                fired_boundaries = int(stats.get("fired_boundaries", 0) or 0)
                tokens = int(stats.get("total_tokens", 0) or 0)
                boundary_rate = float(stats.get("boundary_rate", (fired_boundaries / tokens if tokens > 0 else 0.0)) or 0.0)
                total_fired_boundaries += fired_boundaries
                total_tokens += tokens
                boundary_stats_per_task.append({
                    "task_id": task_id,
                    "sample": sample_idx,
                    "fired_boundaries": fired_boundaries,
                    "total_tokens": tokens,
                    "boundary_rate": boundary_rate,
                })
            else:
                generated_full = result
                fired_boundaries = 0
                tokens = 0
                boundary_rate = 0.0
                stats = {}

            # Be robust: generation can sometimes return None (e.g., upstream failure paths).
            # EvalPlus expects string solutions; storing None can crash later steps.
            if generated_full is None:
                generated_full = ""
            
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
            samples.append({
                "task_id": task_id,
                "solution": solution_to_store,
                # Per-task boundary stats for analysis (same as raw file)
                "fired_boundaries": int(fired_boundaries),
                "total_tokens": int(tokens),
                "boundary_rate": float(boundary_rate),
                # Per-boundary trigger tokens (from incremental_generate stats)
                "boundary_trigger_tokens": stats.get("boundary_trigger_tokens", []),
                "boundary_trigger_token_ids": stats.get("boundary_trigger_token_ids", []),
                "boundary_trigger_confidences": stats.get("boundary_trigger_confidences", []),
                # Full boundary event objects (position, token, confidence, etc.)
                "boundary_events": stats.get("boundary_events", []),
            })
            # Always store raw output (pre-sanitization) for debugging
            raw_samples.append({
                "task_id": task_id,
                "solution": generated_full,
                # Per-task boundary stats for analysis
                "fired_boundaries": int(fired_boundaries),
                "total_tokens": int(tokens),
                "boundary_rate": float(boundary_rate),
                # Per-boundary trigger tokens (from incremental_generate stats)
                "boundary_trigger_tokens": stats.get("boundary_trigger_tokens", []),
                "boundary_trigger_token_ids": stats.get("boundary_trigger_token_ids", []),
                "boundary_trigger_confidences": stats.get("boundary_trigger_confidences", []),
                # Full boundary event objects (position, token, confidence, etc.)
                "boundary_events": stats.get("boundary_events", []),
            })
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
    parser.add_argument(
        "--backend",
        type=str,
        default="blt",
        choices=["blt", "hf"],
        help="Inference backend: 'blt' uses BLTAdapterModel wrapper; 'hf' evaluates a plain HF base model (no BLT wrapper).",
    )
    # Model loading
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved adapter checkpoint (epoch folder) or base model path")
    parser.add_argument("--model_path", type=str, default="/data/qwen2.5coder", help="Base Qwen2.5 path")
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
    # Default is resolved at runtime: if not provided, we try to use the checkpoint's own setting.
    parser.add_argument(
        "--boundary_threshold",
        type=float,
        default=None,
        help="Threshold for boundary head predictions. If omitted, uses checkpoint default (e.g. adapter.rewrite_boundary_threshold) when available.",
    )
    # Conservative default to avoid over-patching (patching too frequently tends to hurt pass@1).
    parser.add_argument("--min_steps_between_patches", type=int, default=4, help="Minimum steps between boundary patches")
    parser.add_argument("--max_patch_len", type=int, default=128, help="Maximum span length to rewrite")
    parser.add_argument("--disable_patching_in_docstring", action="store_true", default=True, help="Prevent patching in docstrings/comments")
    parser.add_argument(
        "--enable_patching_in_docstring",
        action="store_false",
        dest="disable_patching_in_docstring",
        help="Allow patching even inside docstrings/comments (NOT recommended)",
    )
    parser.add_argument("--use_local_decoder", action="store_true", default=True, help="Enable local decoder refinement")
    parser.add_argument(
        "--local_decoder_mode",
        type=str,
        default="refine",
        choices=["generate", "refine"],
        help="Local decoder behavior for span rewrite: 'refine' (teacher-forced denoise) or 'generate' (free-run from BOS).",
    )
    parser.add_argument("--disable_local_decoder", action="store_true", help="Disable local decoder/encoder (use only global transformer, for base model evaluation)")
    parser.add_argument(
        "--disable_local_encoder_only",
        action="store_true",
        help="Keep local decoder enabled but skip local encoder (span_memory=None). This aligns inference with the training student path (latent_from_global + global memory).",
    )
    parser.add_argument(
        "--min_rewrite_span_len",
        type=int,
        default=8,
        help="Minimum span length to actually apply a rewrite in inference. Boundaries can still trigger and finalize shorter spans, but no rewrite is applied.",
    )
    
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

    # HF backend-only knobs
    parser.add_argument(
        "--force_base_prompt",
        action="store_true",
        help="HF backend only: force direct completion prompt (ignore chat template). Recommended for base models.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="HF backend only: attention implementation passed to Transformers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Show what GPU is being used (already set before torch import)
    current_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")

    # --- UX guardrails for "base model" runs ---
    # Users often intend to evaluate a plain HF base model (e.g. /data/qwen2.5coder)
    # but forget to set `--backend hf` and instead pass `--disable_local_decoder`.
    # In that case, the BLT backend will try to run `incremental_generate()` with a
    # BLTAdapterModel wrapper, which can yield empty generations / misleading results.
    def _looks_like_hf_model_dir(p: str) -> bool:
        try:
            return bool(p) and os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json"))
        except Exception:
            return False

    if args.backend == "blt" and bool(args.disable_local_decoder):
        # Heuristic: if --checkpoint looks like an HF model directory while --model_path does not,
        # the user most likely wants HF backend base-model evaluation.
        if _looks_like_hf_model_dir(str(args.checkpoint)) and not _looks_like_hf_model_dir(str(args.model_path)):
            print(
                "[setup] Detected a base-model HF directory passed via --checkpoint together with --disable_local_decoder. "
                "Switching backend to 'hf' for plain base-model evaluation. "
                "Tip: you can explicitly pass `--backend hf` and omit `--disable_local_decoder`."
            )
            args.backend = "hf"

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

    # === Backend selection ===
    if args.backend == "hf":
        # Plain HF base model evaluation (no BLT wrapper).
        # This is useful to validate the evaluation pipeline on models like Qwen3, even if BLTAdapterModel
        # does not match the base architecture.
        from evalplus.codegen import codegen as evalplus_codegen  # type: ignore
        from evalplus.provider.hf import HuggingFaceDecoder  # type: ignore

        hf_dtype = "bfloat16"
        if args.dtype in ["auto", "bf16"]:
            hf_dtype = "bfloat16"
        elif args.dtype == "fp16":
            hf_dtype = "float16"
        elif args.dtype == "fp32":
            hf_dtype = "float32"

        print(f"HF backend enabled. Base model: {args.checkpoint}")
        print(f"Generating solutions for {len(problems)} tasks using HF backend...")
        model = HuggingFaceDecoder(
            name=args.checkpoint,
            dataset=args.dataset,
            force_base_prompt=bool(args.force_base_prompt),
            attn_implementation=str(args.attn_implementation),
            device_map=None,
            batch_size=1,
            temperature=float(args.temperature),
            max_new_tokens=int(args.max_new_tokens),
            dtype=hf_dtype,
            trust_remote_code=False,
        )

        # EvalPlus codegen writes both sanitized and raw outputs.
        # It will also resume if the jsonl already exists.
        evalplus_codegen(
            target_path=output_path,
            model=model,
            dataset=problems,
            greedy=(float(args.temperature) == 0.0),
            n_samples=int(args.n_samples),
            resume=True,
        )

        # HF backend doesn't have BLT boundary stats; write a minimal stats file for consistency.
        stats_path = output_path.replace(".jsonl", "_stats.json")
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "backend": "hf",
                    "total_tasks": int(len(problems) * max(1, int(args.n_samples))),
                    "total_samples": int(len(problems) * max(1, int(args.n_samples))),
                },
                f,
                indent=2,
            )
        print(f"Wrote statistics to {stats_path}")
    else:
        # BLT adapter inference backend (existing behavior)
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

        # If user did not set boundary_threshold, align inference with what the checkpoint was trained with (when available).
        boundary_threshold = args.boundary_threshold
        if boundary_threshold is None:
            # Training script uses --rewrite_boundary_threshold (stored on adapter as rewrite_boundary_threshold).
            # Fall back to inference default if missing.
            boundary_threshold = float(getattr(adapter, "rewrite_boundary_threshold", 0.65))
            print(f"[setup] boundary_threshold not provided; using checkpoint/default value: {boundary_threshold}")

        print(f"Generating solutions for {len(problems)} tasks...")
        print(f"Inference hyperparameters:")
        print(f"  - patcher: {patcher}")
        print(f"  - use_local_decoder: {use_local_decoder}")
        print(f"  - local_decoder_mode: {str(args.local_decoder_mode)}")
        print(f"  - disable_local_encoder_only: {bool(args.disable_local_encoder_only)}")
        print(f"  - min_rewrite_span_len: {int(args.min_rewrite_span_len)}")
        if patcher == "learned":
            print(f"  - boundary_threshold: {boundary_threshold}")
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
            boundary_threshold=float(boundary_threshold),
            min_steps_between_patches=args.min_steps_between_patches,
            max_patch_len=args.max_patch_len,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            disable_patching_in_docstring=args.disable_patching_in_docstring,
            use_local_decoder=use_local_decoder,
            local_decoder_mode=str(args.local_decoder_mode),
            disable_local_encoder_only=bool(args.disable_local_encoder_only),
            min_rewrite_span_len=int(args.min_rewrite_span_len),
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
            first_solution = samples[0].get("solution", "")
            # Be robust: some upstream generation/sanitization failures can yield None.
            if first_solution is None:
                first_solution = ""
            first_solution_str = str(first_solution)
            print(f"[DEBUG]   First solution length: {len(first_solution_str)} chars")
            print(f"[DEBUG]   First solution preview: {first_solution_str[:100]}...")
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

