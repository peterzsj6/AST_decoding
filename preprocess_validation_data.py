"""
Preprocess validation datasets to add AST span information.

Processes:
1. BigCode validation parquet: train-00058-of-00059.parquet
2. HumanEval JSONL: human-eval-v2-20210705.jsonl

Output: Parquet files with AST_span column ready for validation.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
from multiprocessing import Pool, cpu_count

# Add the AST parsing module to path
sys.path.insert(0, '/data/home/zhangsj/AST_decoding/ast_parsing_folder')
from AST_parsing import (
    parse_to_ast,
    parse_to_ast_with_fallback,
    get_ast_leaf_nodes_for_spans,
    validate_comprehensive_coverage,
)


def parse_code_to_ast_spans(code: str, language: str = "python", verbose: bool = False) -> Optional[str]:
    """
    Parse code and extract AST spans with token indices.
    
    Args:
        code: Source code string
        language: Programming language
        verbose: Print detailed info
    
    Returns:
        JSON string of AST spans, or None if parsing fails
    """
    if not code or not isinstance(code, str) or len(code.strip()) < 10:
        return None
    
    try:
        # Parse with fallback strategies for robustness
        ast_root, strategy = parse_to_ast_with_fallback(code, ps_language=language, verbose_errors=verbose)
        
        if ast_root is None:
            return None
        
        # Get token-aligned AST spans
        spans = get_ast_leaf_nodes_for_spans(ast_root, include_positions=True)
        
        if not spans:
            return None
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_spans = []
        for span in spans:
            span_dict = {
                'type': span.get('type', 'unknown'),
                'text': span.get('text', ''),
            }
            
            # Handle token_indices (may be numpy array or list)
            token_indices = span.get('token_indices', [])
            if hasattr(token_indices, 'tolist'):
                token_indices = token_indices.tolist()
            span_dict['token_indices'] = token_indices
            
            # Add position info if available
            if 'start_line' in span:
                span_dict['start_line'] = int(span['start_line'])
                span_dict['start_column'] = int(span['start_column'])
                span_dict['end_line'] = int(span['end_line'])
                span_dict['end_column'] = int(span['end_column'])
            
            serializable_spans.append(span_dict)
        
        return json.dumps(serializable_spans)
        
    except Exception as e:
        if verbose:
            print(f"Error parsing code: {e}")
        return None


def process_single_row(args):
    """Process a single row (for multiprocessing)."""
    idx, content, language, verbose = args
    ast_span = parse_code_to_ast_spans(content, language, verbose)
    return idx, ast_span


def process_parquet_file(
    input_path: str,
    output_path: str,
    content_column: str = "content",
    language: str = "python",
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    verbose: bool = False,
):
    """
    Process a parquet file and add AST_span column.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        content_column: Name of the column containing code
        language: Programming language
        max_samples: Maximum number of samples to process (None = all)
        num_workers: Number of parallel workers
        verbose: Print detailed info
    """
    print(f"\n{'='*60}")
    print(f"Processing parquet: {input_path}")
    print(f"{'='*60}")
    
    # Load parquet
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    if content_column not in df.columns:
        raise ValueError(f"Content column '{content_column}' not found. Available: {list(df.columns)}")
    
    # Limit samples if requested
    if max_samples and max_samples < len(df):
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")
    
    # Filter out empty/invalid content
    valid_mask = df[content_column].notna() & (df[content_column].str.strip() != '')
    df = df[valid_mask].copy()
    print(f"After filtering empty: {len(df)} samples")
    
    # Prepare arguments for multiprocessing
    args_list = [
        (idx, row[content_column], language, verbose)
        for idx, row in df.iterrows()
    ]
    
    # Process with multiprocessing
    print(f"\nParsing AST spans with {num_workers} workers...")
    ast_spans = {}
    
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_row, args_list),
                total=len(args_list),
                desc="Parsing"
            ))
        for idx, ast_span in results:
            ast_spans[idx] = ast_span
    else:
        for args in tqdm(args_list, desc="Parsing"):
            idx, ast_span = process_single_row(args)
            ast_spans[idx] = ast_span
    
    # Add AST_span column
    df['AST_span'] = df.index.map(ast_spans)
    
    # Calculate statistics
    success_count = df['AST_span'].notna().sum()
    total_count = len(df)
    success_rate = success_count / total_count * 100
    
    print(f"\nParsing Results:")
    print(f"  Total samples: {total_count}")
    print(f"  Successfully parsed: {success_count}")
    print(f"  Failed: {total_count - success_count}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    # Filter to only successfully parsed samples
    df_success = df[df['AST_span'].notna()].copy()
    print(f"\nKeeping {len(df_success)} successfully parsed samples")
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_success.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    return df_success


def process_humaneval_jsonl(
    input_path: str,
    output_path: str,
    language: str = "python",
    verbose: bool = False,
):
    """
    Process HumanEval JSONL file and save as parquet with AST_span.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output parquet file
        language: Programming language
        verbose: Print detailed info
    """
    print(f"\n{'='*60}")
    print(f"Processing HumanEval: {input_path}")
    print(f"{'='*60}")
    
    # Load JSONL
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                samples.append(item)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(samples)} HumanEval problems")
    
    # Process each sample
    processed = []
    success_count = 0
    
    for item in tqdm(samples, desc="Parsing"):
        task_id = item.get('task_id', '')
        prompt = item.get('prompt', '')
        canonical_solution = item.get('canonical_solution', '')
        entry_point = item.get('entry_point', '')
        
        # Combine prompt + solution for full code
        full_code = prompt + canonical_solution
        
        # Parse AST
        ast_span = parse_code_to_ast_spans(full_code, language, verbose)
        
        if ast_span:
            success_count += 1
        
        processed.append({
            'task_id': task_id,
            'content': full_code,
            'prompt': prompt,
            'canonical_solution': canonical_solution,
            'entry_point': entry_point,
            'AST_span': ast_span,
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(processed)
    
    # Statistics
    total_count = len(df)
    success_rate = success_count / total_count * 100
    
    print(f"\nParsing Results:")
    print(f"  Total problems: {total_count}")
    print(f"  Successfully parsed: {success_count}")
    print(f"  Failed: {total_count - success_count}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    # Filter to successfully parsed
    df_success = df[df['AST_span'].notna()].copy()
    print(f"\nKeeping {len(df_success)} successfully parsed problems")
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_success.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    return df_success


def validate_output(parquet_path: str, num_samples: int = 5):
    """
    Validate the output parquet file.
    """
    print(f"\n{'='*60}")
    print(f"Validating: {parquet_path}")
    print(f"{'='*60}")
    
    df = pd.read_parquet(parquet_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check AST_span column
    if 'AST_span' not in df.columns:
        print("❌ AST_span column missing!")
        return False
    
    ast_valid = df['AST_span'].notna().sum()
    print(f"Valid AST_span: {ast_valid}/{len(df)}")
    
    # Sample validation
    print(f"\nSample spans from first {num_samples} rows:")
    for i in range(min(num_samples, len(df))):
        ast_span_str = df.iloc[i]['AST_span']
        try:
            spans = json.loads(ast_span_str)
            print(f"  Row {i}: {len(spans)} spans")
            if spans:
                # Show first span
                first_span = spans[0]
                print(f"    First span: type='{first_span.get('type')}', "
                      f"text='{first_span.get('text', '')[:30]}...', "
                      f"tokens={len(first_span.get('token_indices', []))}")
        except Exception as e:
            print(f"  Row {i}: Error parsing - {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess validation data with AST spans")
    
    parser.add_argument("--bigcode_input", type=str,
                        default="/data/home/zhangsj/Data/more_big_code_language/python/validation/train-00058-of-00059.parquet",
                        help="Path to BigCode validation parquet")
    parser.add_argument("--bigcode_output", type=str,
                        default="/data/home/zhangsj/Data/more_big_code_language/python/validation/bigcode_val_ast_parsed.parquet",
                        help="Output path for BigCode validation")
    
    parser.add_argument("--humaneval_input", type=str,
                        default="/data/home/zhangsj/Data/HumanEval/human-eval-v2-20210705.jsonl",
                        help="Path to HumanEval JSONL")
    parser.add_argument("--humaneval_output", type=str,
                        default="/data/home/zhangsj/Data/HumanEval/humaneval_ast_parsed.parquet",
                        help="Output path for HumanEval")
    
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process (for testing)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--skip_bigcode", action="store_true",
                        help="Skip BigCode processing")
    parser.add_argument("--skip_humaneval", action="store_true",
                        help="Skip HumanEval processing")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate existing output files")
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Just validate existing files
        if os.path.exists(args.bigcode_output):
            validate_output(args.bigcode_output)
        if os.path.exists(args.humaneval_output):
            validate_output(args.humaneval_output)
        return
    
    # Process BigCode validation
    if not args.skip_bigcode and os.path.exists(args.bigcode_input):
        process_parquet_file(
            input_path=args.bigcode_input,
            output_path=args.bigcode_output,
            content_column="content",
            language="python",
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            verbose=args.verbose,
        )
        validate_output(args.bigcode_output)
    else:
        print(f"Skipping BigCode: {'--skip_bigcode flag' if args.skip_bigcode else 'file not found'}")
    
    # Process HumanEval
    if not args.skip_humaneval and os.path.exists(args.humaneval_input):
        process_humaneval_jsonl(
            input_path=args.humaneval_input,
            output_path=args.humaneval_output,
            language="python",
            verbose=args.verbose,
        )
        validate_output(args.humaneval_output)
    else:
        print(f"Skipping HumanEval: {'--skip_humaneval flag' if args.skip_humaneval else 'file not found'}")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nOutput files:")
    if not args.skip_bigcode:
        print(f"  BigCode val: {args.bigcode_output}")
    if not args.skip_humaneval:
        print(f"  HumanEval:   {args.humaneval_output}")
    
    print(f"\nTo use in training:")
    print(f"  python blt_focused_training.py \\")
    print(f"      --val_parquet {args.bigcode_output} \\")
    print(f"      --humaneval_path {args.humaneval_output}")


if __name__ == "__main__":
    main()


