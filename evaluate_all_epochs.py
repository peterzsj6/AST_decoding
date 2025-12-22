#!/usr/bin/env python3
"""
Evaluate all epochs in a checkpoint directory and compile pass@1 rates into a JSONL file.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Make project root importable
PROJECT_ROOT = "/data/home/zhangsj"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def find_epoch_directories(checkpoint_dir: str) -> List[str]:
    """Find all epoch directories in the checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    epochs = []
    for item in sorted(checkpoint_path.iterdir()):
        if item.is_dir() and item.name.startswith("epoch_"):
            try:
                epoch_num = int(item.name.split("_")[1])
                epochs.append((epoch_num, str(item)))
            except (ValueError, IndexError):
                continue
    
    # Sort by epoch number
    epochs.sort(key=lambda x: x[0])
    return [path for _, path in epochs]


def run_evaluation(
    checkpoint_path: str,
    model_path: str,
    output_dir: str,
    gpu: int = 0,
    dataset: str = "humaneval",
    **kwargs
) -> str:
    """Run evaluation for a single epoch and return the results file path."""
    # Extract epoch name from checkpoint path (e.g., "epoch_5" from full path)
    epoch_name = os.path.basename(os.path.normpath(checkpoint_path))
    
    # Generate predictable output path
    eval_output_dir = os.path.join(output_dir, dataset)
    os.makedirs(eval_output_dir, exist_ok=True)
    output_path = os.path.join(eval_output_dir, f"{epoch_name}.jsonl")
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "AST_decoding", "run_evalplus_blt.py"),
        "--checkpoint", checkpoint_path,
        "--model_path", model_path,
        "--gpu", str(gpu),
        "--dataset", dataset,
        "--output", output_path,
        "--overwrite",
    ]

    # Force global-transformer-only evaluation for consistent baseline scoring.
    # In run_evalplus_blt.py this also forces patcher="none".
    cmd.append("--disable_local_decoder")
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed for {checkpoint_path}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    # Construct the results file path (EvalPlus replaces .jsonl with .eval_results.json)
    results_file = output_path.replace('.jsonl', '.eval_results.json')
    
    # Verify the results file exists
    if not os.path.exists(results_file):
        print(f"WARNING: Results file not found at {results_file}")
        print(f"Expected results file based on output path: {output_path}")
        return None
    
    return results_file


def extract_pass_at_1(results_file: str) -> Dict[str, float]:
    """Extract pass@1 rates from EvalPlus results file."""
    if not results_file or not os.path.exists(results_file):
        return {"base_pass@1": None, "plus_pass@1": None}
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        pass_at_k = data.get("pass_at_k", {})
        base_pass = pass_at_k.get("base", {}).get("pass@1", None)
        plus_pass = pass_at_k.get("plus", {}).get("pass@1", None)
        
        return {
            "base_pass@1": base_pass,
            "plus_pass@1": plus_pass
        }
    except Exception as e:
        print(f"ERROR: Failed to parse results file {results_file}: {e}")
        return {"base_pass@1": None, "plus_pass@1": None}


def main():
    parser = argparse.ArgumentParser(description="Evaluate all epochs and compile pass@1 rates")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing epoch checkpoints")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding",
                       help="Base model path")
    parser.add_argument("--output_dir", type=str, default="/data/home/zhangsj/evalplus_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--output_jsonl", type=str, default="",
                       help="Output JSONL file path (default: checkpoint_dir/epoch_results.jsonl)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip epochs that already have results")
    
    # Pass through arguments for run_evalplus_blt.py
    # NOTE: We always add --disable_local_decoder in run_evaluation() above.
    # These patcher-related options are ignored in that mode (run_evalplus_blt forces patcher='none').
    parser.add_argument("--patcher", type=str, default="learned",
                       choices=["none", "heuristic", "entropy", "learned"])
    parser.add_argument("--boundary_threshold", type=float, default=0.65)
    parser.add_argument("--min_steps_between_patches", type=int, default=4)
    parser.add_argument("--max_patch_len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    args = parser.parse_args()
    
    # Find all epoch directories
    epoch_dirs = find_epoch_directories(args.checkpoint_dir)
    if not epoch_dirs:
        print(f"No epoch directories found in {args.checkpoint_dir}")
        return
    
    print(f"Found {len(epoch_dirs)} epochs to evaluate:")
    for epoch_dir in epoch_dirs:
        print(f"  - {epoch_dir}")
    
    # Determine output JSONL path
    if args.output_jsonl:
        output_jsonl = args.output_jsonl
    else:
        checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint_dir))
        output_jsonl = os.path.join(args.checkpoint_dir, f"{checkpoint_name}_epoch_results.jsonl")
    
    # Load existing results if output file exists
    existing_results = {}
    if os.path.exists(output_jsonl) and args.skip_existing:
        print(f"\nLoading existing results from {output_jsonl}...")
        with open(output_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    epoch = result.get("epoch")
                    if epoch:
                        existing_results[epoch] = result
    
    # Evaluate each epoch
    all_results = []
    eval_kwargs = {
        "patcher": args.patcher,
        "boundary_threshold": args.boundary_threshold,
        "min_steps_between_patches": args.min_steps_between_patches,
        "max_patch_len": args.max_patch_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }
    
    for epoch_dir in epoch_dirs:
        epoch_name = os.path.basename(epoch_dir)
        epoch_num = int(epoch_name.split("_")[1])
        
        # Check if we should skip this epoch
        if args.skip_existing and epoch_num in existing_results:
            print(f"\nSkipping {epoch_name} (already in results)")
            all_results.append(existing_results[epoch_num])
            continue
        
        # Run evaluation
        results_file = run_evaluation(
            checkpoint_path=epoch_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            gpu=args.gpu,
            dataset=args.dataset,
            **eval_kwargs
        )
        
        # Extract pass@1
        pass_at_1 = extract_pass_at_1(results_file)
        
        result_entry = {
            "epoch": epoch_num,
            "epoch_name": epoch_name,
            "checkpoint_path": epoch_dir,
            "results_file": results_file,
            **pass_at_1
        }
        
        all_results.append(result_entry)
        
        print(f"\n{epoch_name} Results:")
        print(f"  Base pass@1: {pass_at_1['base_pass@1']}")
        print(f"  Plus pass@1: {pass_at_1['plus_pass@1']}")
    
    # Write results to JSONL
    print(f"\n{'='*80}")
    print(f"Writing results to {output_jsonl}")
    print(f"{'='*80}\n")
    
    with open(output_jsonl, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Epoch':<10} {'Base pass@1':<15} {'Plus pass@1':<15}")
    print("-"*80)
    for result in sorted(all_results, key=lambda x: x["epoch"]):
        base = result.get("base_pass@1")
        plus = result.get("plus_pass@1")
        base_str = f"{base:.4f}" if base is not None else "N/A"
        plus_str = f"{plus:.4f}" if plus is not None else "N/A"
        print(f"{result['epoch_name']:<10} {base_str:<15} {plus_str:<15}")
    print("="*80)
    print(f"\nResults saved to: {output_jsonl}")


if __name__ == "__main__":
    main()

