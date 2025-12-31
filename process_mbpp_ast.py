#!/usr/bin/env python3
"""
Process MBPP dataset and generate AST spans similar to HumanEval format.
"""

import json
import pandas as pd
import sys
import os
from pathlib import Path
from contextlib import contextmanager
from io import StringIO

# Add the AST_parsing folder to path
sys.path.insert(0, str(Path(__file__).parent / "ast_parsing_folder"))

from AST_parsing import (
    parse_to_ast_with_fallback,
    get_ast_leaf_nodes_for_spans,
    preprocess_code_for_parsing
)

def load_mbpp_data(jsonl_path):
    """Load MBPP dataset from JSONL file."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def process_code_to_ast_spans(code_text, tokenizer_path="/data/home/zhangsj/AST_decoding", verbose=False):
    """
    Process code text and generate AST spans.
    
    Args:
        code_text: Python code string
        tokenizer_path: Path to tokenizer directory (contains tokenizer.json and tokenizer_config.json)
    
    Returns:
        List of AST span dictionaries or None if parsing fails
    """
    if not code_text or not isinstance(code_text, str):
        return None
    
    # Preprocess code
    preprocessed_code = preprocess_code_for_parsing(code_text, ps_language="python")
    
    if not preprocessed_code.strip():
        print(f"Warning: Code became empty after preprocessing")
        return None
    
    # Parse to AST with fallback strategies
    ast_root, strategy = parse_to_ast_with_fallback(
        preprocessed_code, 
        ps_language="python", 
        verbose_errors=False
    )
    
    if ast_root is None:
        print(f"Warning: Failed to parse code with all strategies")
        return None
    
    # Get AST spans using token-aligned method
    try:
        # Directly use get_token_aligned_ast_spans to have control over tokenizer path
        from AST_parsing import get_token_aligned_ast_spans
        
        # Suppress verbose output unless verbose is True
        if verbose:
            ast_spans = get_token_aligned_ast_spans(
                preprocessed_code,
                ast_root,
                include_positions=True,
                tokenizer_path=tokenizer_path
            )
        else:
            with suppress_stdout():
                ast_spans = get_token_aligned_ast_spans(
                    preprocessed_code,
                    ast_root,
                    include_positions=True,
                    tokenizer_path=tokenizer_path
                )
        
        return ast_spans
        
    except Exception as e:
        print(f"Error generating AST spans: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_mbpp_dataset(input_jsonl, output_parquet, tokenizer_path="/data/home/zhangsj/AST_decoding"):
    """
    Process MBPP dataset and generate parquet file with AST spans.
    
    Args:
        input_jsonl: Path to MBPP JSONL file
        output_parquet: Path to output parquet file
        tokenizer_path: Path to tokenizer directory
    """
    print(f"Loading MBPP dataset from {input_jsonl}...")
    sys.stdout.flush()
    mbpp_data = load_mbpp_data(input_jsonl)
    print(f"Loaded {len(mbpp_data)} MBPP tasks")
    sys.stdout.flush()
    
    results = []
    success_count = 0
    error_count = 0
    
    for idx, item in enumerate(mbpp_data):
        task_id = f"MBPP/{item.get('task_id', idx)}"
        code = item.get('code', '')
        text = item.get('text', '')
        
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"Processing task {idx+1}/{len(mbpp_data)}: {task_id}")
        
        if not code:
            if (idx + 1) % 50 == 0:
                print(f"  Warning: No code found for task {task_id}")
            error_count += 1
            continue
        
        # Process code to get AST spans
        ast_spans = process_code_to_ast_spans(code, tokenizer_path)
        
        if ast_spans is None or len(ast_spans) == 0:
            if (idx + 1) % 50 == 0:
                print(f"  Warning: Failed to generate AST spans for task {task_id}")
            error_count += 1
            # Still add the record but with empty AST_span
            ast_spans = []
        else:
            success_count += 1
            if (idx + 1) % 50 == 0:
                print(f"  Success: Generated {len(ast_spans)} AST spans")
        
        # Convert AST spans to JSON string format (same as HumanEval)
        ast_span_json = json.dumps(ast_spans)
        
        # Create result record similar to HumanEval format
        result = {
            'task_id': task_id,
            'content': code,  # Full code content
            'prompt': text,   # Task description as prompt
            'canonical_solution': code,  # For MBPP, code is the solution
            'entry_point': '',  # MBPP doesn't have entry_point like HumanEval
            'AST_span': ast_span_json
        }
        
        results.append(result)
        
        # Progress update
        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx+1}/{len(mbpp_data)} processed (Success: {success_count}, Errors: {error_count})")
            sys.stdout.flush()
    
    # Create DataFrame
    print(f"\nCreating DataFrame...")
    sys.stdout.flush()
    df = pd.DataFrame(results)
    
    # Save to parquet
    print(f"Saving to {output_parquet}...")
    sys.stdout.flush()
    df.to_parquet(output_parquet, index=False)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Total tasks: {len(mbpp_data)}")
    print(f"  Successfully parsed: {success_count}")
    print(f"  Failed to parse: {error_count}")
    print(f"  Success rate: {success_count/len(mbpp_data)*100:.1f}%")
    print(f"  Output saved to: {output_parquet}")
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process MBPP dataset and generate AST spans")
    parser.add_argument(
        "--input",
        type=str,
        default="/data/home/zhangsj/Data/MBPP/mbpp.jsonl",
        help="Path to MBPP JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/home/zhangsj/Data/MBPP/mbpp_ast_parsed.parquet",
        help="Path to output parquet file"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/data/home/zhangsj/AST_decoding",
        help="Path to tokenizer directory"
    )
    
    args = parser.parse_args()
    
    process_mbpp_dataset(args.input, args.output, args.tokenizer)

