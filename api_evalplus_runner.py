import os
import sys
import json
import argparse
import logging
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add evalplus to path (must be first to override installed package)
EVALPLUS_SRC = "/data/home/zhangsj/AST_decoding/evalplus/evalplus"
if EVALPLUS_SRC not in sys.path:
    sys.path.insert(0, EVALPLUS_SRC)

# Also add parent directory for evalplus imports
EVALPLUS_PARENT = "/data/home/zhangsj/AST_decoding/evalplus"
if EVALPLUS_PARENT not in sys.path:
    sys.path.insert(0, EVALPLUS_PARENT)

# API configuration (from api_worker.py)
from openai import OpenAI

API_KEY = "HLwaq5OfA0dhif7NLl7afm7kUFuP0toS"
BASE_URL = "https://api.deepinfra.com/v1/openai"

# Model pricing dictionary (from api_worker.py lines 17-28)
MODEL_PRICING = {
    "MiniMaxAI/MiniMax-M2": {"in": 0.27, "out": 1.15},
    "deepseek-ai/DeepSeek-V3.2-Exp": {"in": 0.27, "out": 0.40},
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {"in": 0.09, "out": 0.57},
    "moonshotai/Kimi-K2-Instruct-0905": {"in": 0.50, "out": 2.00},
    "openai/gpt-oss-120b": {"in": 0.05, "out": 0.24},
    "google/gemma-3-27b-it": {"in": 0.09, "out": 0.16},
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {"in": 0.075, "out": 0.20},
    "google/gemma-3-12b-it": {"in": 0.04, "out": 0.13},
    "deepseek-ai/DeepSeek-V3.1-Terminus": {"in": 0.21, "out": 0.79},
}

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Import EvalPlus modules
from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data.utils import write_jsonl
from evalplus.evaluate import evaluate as evalplus_evaluate
from evalplus.sanitize import sanitize
from evalplus.eval import PASS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def get_completion_with_tokens(prompt, model_name, prices):
    """Wrapper to get completion with token counts."""
    try:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = chat_completion.choices[0].message.content
        p_tokens = chat_completion.usage.prompt_tokens
        c_tokens = chat_completion.usage.completion_tokens
        
        # Calculate cost
        total_cost = (p_tokens * prices["in"] / 1000000) + (c_tokens * prices["out"] / 1000000)
        return content, total_cost, p_tokens, c_tokens
    except Exception as e:
        raise e


def safe_run_single_with_tokens(prompt, model_name, prices):
    """Retry logic with token tracking: 2s -> 10s -> Fail"""
    # First attempt
    try:
        return get_completion_with_tokens(prompt, model_name, prices)
    except Exception as e:
        logger.warning(f"[Retry 2s] {model_name}: {str(e)[:100]}")
        time.sleep(2)
    
    # Second attempt
    try:
        return get_completion_with_tokens(prompt, model_name, prices)
    except Exception as e:
        logger.warning(f"[Retry 10s] {model_name}: {str(e)[:100]}")
        time.sleep(10)
    
    # Third attempt
    try:
        return get_completion_with_tokens(prompt, model_name, prices)
    except Exception as e:
        logger.error(f"[FAILED] {model_name}: {str(e)[:100]}")
        return None, 0.0, 0, 0


def process_task(
    task_id: str,
    problem: Dict[str, Any],
    model_name: str,
    prices: Dict[str, float],
    dataset_name: str,
) -> Dict[str, Any]:
    """Process a single task: generate solution, sanitize, and track cost."""
    prompt = problem["prompt"]
    entry_point = problem.get("entry_point")
    
    # Generate solution via API with token tracking
    response, cost, p_tokens, c_tokens = safe_run_single_with_tokens(prompt, model_name, prices)
    
    if response is None:
        logger.warning(f"Failed to generate solution for {task_id}")
        return None
    
    # Sanitize the response
    try:
        sanitized_code = sanitize(code=response, entrypoint=entry_point)
    except Exception as e:
        logger.warning(f"Sanitization failed for {task_id}: {e}, using original response")
        sanitized_code = response
    
    return {
        "task_id": task_id,
        "solution": sanitized_code,
        "original_response": response,
        "cost": cost,
        "tokens_in": p_tokens,
        "tokens_out": c_tokens,
    }


def run_benchmark(
    model_name: str,
    dataset: str,
    output_dir: str,
    max_samples: int = None,
    concurrency: int = 5,
    test_mode: bool = False,
):
    """Run benchmark for a single model and dataset."""
    logger.info(f"Starting benchmark: Model={model_name}, Dataset={dataset}")
    
    # Get pricing
    prices = MODEL_PRICING.get(model_name, {"in": 0, "out": 0})
    if prices["in"] == 0 and prices["out"] == 0:
        logger.warning(f"No pricing found for {model_name}, cost tracking will be 0")
    
    # Load dataset
    if dataset == "humaneval":
        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    logger.info(f"Loaded {len(problems)} problems from {dataset}")
    
    # Create output directory
    safe_model_name = model_name.replace("/", "_")
    model_output_dir = os.path.join(output_dir, safe_model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    samples_file = os.path.join(model_output_dir, f"{dataset}_samples.jsonl")
    
    # Check for existing samples (resume capability)
    processed_task_ids = set()
    if os.path.exists(samples_file):
        logger.info(f"Found existing samples file: {samples_file}, checking for processed tasks...")
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_task_ids.add(record["task_id"])
                    except:
                        continue
        logger.info(f"Found {len(processed_task_ids)} already processed tasks")
    
    # Filter tasks
    task_ids = sorted(list(problems.keys()))
    if max_samples:
        task_ids = task_ids[:max_samples]
        logger.info(f"Limiting to {max_samples} samples for testing")
    
    remaining_tasks = [tid for tid in task_ids if tid not in processed_task_ids]
    logger.info(f"Processing {len(remaining_tasks)} tasks (skipping {len(processed_task_ids)} already processed)")
    
    if not remaining_tasks:
        logger.info("All tasks already processed, skipping generation")
        # Load existing cost data if available
        cost_data_file = os.path.join(model_output_dir, f"{dataset}_cost_data.json")
        if not os.path.exists(cost_data_file):
            # Create empty cost data file if it doesn't exist
            with open(cost_data_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        return samples_file, cost_data_file
    
    # Load existing cost data if resuming
    cost_data_file = os.path.join(model_output_dir, f"{dataset}_cost_data.json")
    cost_data = {}
    if os.path.exists(cost_data_file):
        with open(cost_data_file, 'r', encoding='utf-8') as f:
            cost_data = json.load(f)
    
    # Process tasks with concurrency
    total_cost = sum(cost_data.get(tid, {}).get("cost", 0.0) for tid in processed_task_ids)
    total_tokens_in = sum(cost_data.get(tid, {}).get("tokens_in", 0) for tid in processed_task_ids)
    total_tokens_out = sum(cost_data.get(tid, {}).get("tokens_out", 0) for tid in processed_task_ids)
    results = []
    
    with open(samples_file, 'a', encoding='utf-8', buffering=1) as out_f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    process_task,
                    task_id,
                    problems[task_id],
                    model_name,
                    prices,
                    dataset,
                ): task_id
                for task_id in remaining_tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result(timeout=120)
                    if result:
                        # Write to file immediately (for evalplus evaluation)
                        sample_record = {
                            "task_id": result["task_id"],
                            "solution": result["solution"],
                        }
                        out_f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        
                        # Store cost information
                        cost_data[result["task_id"]] = {
                            "cost": result["cost"],
                            "tokens_in": result.get("tokens_in", 0),
                            "tokens_out": result.get("tokens_out", 0),
                        }
                        
                        total_cost += result["cost"]
                        total_tokens_in += result.get("tokens_in", 0)
                        total_tokens_out += result.get("tokens_out", 0)
                        results.append(result)
                        
                        if test_mode:
                            logger.info(f"\n{'='*60}")
                            logger.info(f"Task ID: {task_id}")
                            logger.info(f"Entry Point: {problems[task_id].get('entry_point', 'N/A')}")
                            logger.info(f"\n--- Original Response (first 800 chars) ---")
                            logger.info(result['original_response'][:800])
                            logger.info(f"\n--- Sanitized Solution (first 800 chars) ---")
                            logger.info(result['solution'][:800])
                            logger.info(f"\nCost: ${result['cost']:.6f}, Tokens: {result.get('tokens_in', 0)} in, {result.get('tokens_out', 0)} out")
                            logger.info(f"{'='*60}\n")
                except Exception as e:
                    logger.error(f"Error processing {task_id}: {e}")
    
    logger.info(f"Generated {len(results)} solutions, total cost: ${total_cost:.6f}")
    logger.info(f"Total tokens: {total_tokens_in} in, {total_tokens_out} out")
    
    # Save cost data for later combination with evaluation results (append to existing)
    with open(cost_data_file, 'w', encoding='utf-8') as f:
        json.dump(cost_data, f, indent=2, ensure_ascii=False)
    
    # Save cost summary
    cost_summary = {
        "model": model_name,
        "dataset": dataset,
        "total_cost": total_cost,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "num_tasks": len(results),
    }
    cost_file = os.path.join(model_output_dir, f"{dataset}_cost_summary.json")
    with open(cost_file, 'w', encoding='utf-8') as f:
        json.dump(cost_summary, f, indent=2, ensure_ascii=False)
    
    return samples_file, cost_data_file


def evaluate_samples(samples_file: str, dataset: str, output_dir: str, model_name: str, cost_data_file: str):
    """Run EvalPlus evaluation on the samples and combine with cost data."""
    logger.info(f"Running EvalPlus evaluation on {samples_file}")
    
    safe_model_name = model_name.replace("/", "_")
    model_output_dir = os.path.join(output_dir, safe_model_name)
    results_file = os.path.join(model_output_dir, f"{dataset}_results.json")
    
    # Run evaluation with default options
    # Use the local evalplus version which supports output_file
    eval_kwargs = {
        "dataset": dataset,
        "samples": samples_file,
        "base_only": False,
        "parallel": None,
        "i_just_wanna_run": False,
        "test_details": False,
        "mini": False,
        "noextreme": False,
        "version": "default",
    }
    # Try with output_file first (local version supports it)
    try:
        eval_kwargs["output_file"] = results_file
        evalplus_evaluate(**eval_kwargs)
    except TypeError:
        # Fallback if output_file not supported
        logger.warning("output_file parameter not supported, using default location")
        eval_kwargs.pop("output_file", None)
        evalplus_evaluate(**eval_kwargs)
        # Find and rename the default result file
        default_result = samples_file.replace(".jsonl", ".eval_results.json")
        if os.path.exists(default_result):
            import shutil
            shutil.move(default_result, results_file)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    # Load evaluation results and cost data
    with open(results_file, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)
    
    with open(cost_data_file, 'r', encoding='utf-8') as f:
        cost_data = json.load(f)
    
    # Load samples to get solutions
    samples_dict = {}
    with open(samples_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                samples_dict[sample["task_id"]] = sample["solution"]
    
    # Combine evaluation results with cost data
    combined_output_file = os.path.join(model_output_dir, f"{dataset}_combined_results.jsonl")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        for task_id, task_results in eval_results.get("eval", {}).items():
            # For each solution (usually just one per task)
            for res in task_results:
                solution = res.get("solution", samples_dict.get(task_id, ""))
                base_status = res.get("base_status", "unknown")
                plus_status = res.get("plus_status", "unknown")
                
                # Determine overall pass/fail
                # Pass if both base and plus pass (or just base if plus_status is None)
                if plus_status is None:
                    passed = (base_status == PASS)
                else:
                    passed = (base_status == PASS and plus_status == PASS)
                
                # Get cost data
                cost_info = cost_data.get(task_id, {"cost": 0.0, "tokens_in": 0, "tokens_out": 0})
                
                # Write combined record
                combined_record = {
                    "task_id": task_id,
                    "solution": solution,
                    "passed": passed,
                    "base_status": base_status,
                    "plus_status": plus_status,
                    "cost": cost_info["cost"],
                    "tokens_in": cost_info["tokens_in"],
                    "tokens_out": cost_info["tokens_out"],
                }
                f.write(json.dumps(combined_record, ensure_ascii=False) + "\n")
    
    logger.info(f"Combined results (solution + pass/fail + cost) saved to {combined_output_file}")
    return combined_output_file


def main():
    parser = argparse.ArgumentParser(description="Run EvalPlus benchmarks using API")
    parser.add_argument("--model", type=str, help="Model name (if not specified, runs all models)")
    parser.add_argument("--dataset", type=str, choices=["humaneval", "mbpp", "both"], default="both",
                       help="Dataset to run (default: both)")
    parser.add_argument("--output_dir", type=str, default="evalplus_api_results",
                       help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--concurrency", type=int, default=5,
                       help="Number of parallel API requests")
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode: show sanitized output for verification")
    parser.add_argument("--skip_eval", action="store_true",
                       help="Skip evaluation, only generate samples")
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.model:
        if args.model not in MODEL_PRICING:
            logger.error(f"Model {args.model} not found in MODEL_PRICING")
            logger.info(f"Available models: {list(MODEL_PRICING.keys())}")
            sys.exit(1)
        models = [args.model]
    else:
        models = list(MODEL_PRICING.keys())
    
    # Determine which datasets to run
    if args.dataset == "both":
        datasets = ["humaneval", "mbpp"]
    else:
        datasets = [args.dataset]
    
    logger.info(f"Running benchmarks for {len(models)} model(s) on {len(datasets)} dataset(s)")
    
    # Run benchmarks
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*60}")
        
        for dataset in datasets:
            try:
                # Generate samples
                samples_file, cost_data_file = run_benchmark(
                    model_name=model_name,
                    dataset=dataset,
                    output_dir=args.output_dir,
                    max_samples=args.max_samples,
                    concurrency=args.concurrency,
                    test_mode=args.test_mode,
                )
                
                # Run evaluation (unless skipped)
                if not args.skip_eval:
                    evaluate_samples(samples_file, dataset, args.output_dir, model_name, cost_data_file)
                
            except Exception as e:
                logger.error(f"Error processing {model_name} on {dataset}: {e}", exc_info=True)
                continue
    
    logger.info("\nAll benchmarks completed!")


if __name__ == "__main__":
    main()

