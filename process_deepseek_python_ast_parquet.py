#!/usr/bin/env python3
"""
DeepSeek-tokenizer AST span preprocessing for Python parquet shards.

Reads baseline training parquet(s), parses Python AST with tree-sitter, then
creates token-aligned AST spans using the DeepSeek tokenizer's offset mapping.

Output schema keeps input columns and adds/overwrites:
  - AST_span: JSON string of spans with token_indices (DeepSeek token positions)
  - total_spans: number of spans
  - coverage_percentage: token coverage over the truncated tokenization
  - error: optional error string
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import os
import sys

import pandas as pd


def _import_ast_parsing() -> Any:
    # Ensure we can import AST_parsing.py as a module.
    ast_dir = os.path.join(os.path.dirname(__file__), "ast_parsing_folder")
    if ast_dir not in sys.path:
        sys.path.insert(0, ast_dir)
    import AST_parsing as ap  # type: ignore
    return ap


def _load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # If pad_token is missing, set it to eos to avoid issues in downstream padding-based code.
    if getattr(tok, "pad_token", None) is None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass
    return tok


def deepseek_token_aligned_ast_spans(
    *,
    code_text: str,
    ast_root: Any,
    tokenizer: Any,
    ap: Any,
    include_positions: bool = True,
    max_length: int = 328,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Token-first span generation with guaranteed coverage over the *truncated* tokenization
    (truncation=True, max_length=max_length).

    Returns: (spans, coverage_percentage)
    """
    if not code_text:
        return [], 0.0

    # Tokenize with offset mapping for precise alignment (truncate to training max_length).
    encoding = tokenizer(
        code_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors=None,
    )

    offset_mapping = encoding.get("offset_mapping", None)
    if offset_mapping is None:
        return [], 0.0

    # Build AST node lookup for semantic information.
    char_to_node_map = ap.build_char_to_ast_node_map(code_text, ast_root)

    # Create individual token spans first.
    token_spans: List[Dict[str, Any]] = []
    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
        # HF tokenizers sometimes return None for special cases; skip.
        if token_start is None or token_end is None:
            continue
        token_text = code_text[int(token_start) : int(token_end)]

        semantic_info = ap.get_best_ast_node_for_range(char_to_node_map, int(token_start), int(token_end))
        semantic_type = semantic_info.get("type", "unknown")

        token_span = ap.create_span_from_tokens(
            [int(token_idx)],
            token_text,
            semantic_type,
            offset_mapping,
            code_text,
            include_positions,
        )
        token_spans.append(token_span)

    # Merge adjacent tokens of the same semantic type.
    merged_spans: List[Dict[str, Any]] = []
    current_group: List[Dict[str, Any]] = []
    current_type: Optional[str] = None

    for sp in token_spans:
        if (
            current_type == sp.get("type")
            and current_group
            and sp["token_indices"][0] == current_group[-1]["token_indices"][-1] + 1
        ):
            current_group.append(sp)
        else:
            if current_group:
                merged_spans.append(ap.merge_token_spans(current_group, code_text, offset_mapping, include_positions))
            current_group = [sp]
            current_type = sp.get("type")

    if current_group:
        merged_spans.append(ap.merge_token_spans(current_group, code_text, offset_mapping, include_positions))

    # Coverage over the truncated tokenization
    num_tokens = len(offset_mapping)
    covered: set[int] = set()
    for sp in merged_spans:
        for ti in sp.get("token_indices", []) or []:
            if isinstance(ti, int):
                covered.add(ti)
    cov = 100.0 * (len(covered) / max(1, num_tokens))
    return merged_spans, float(cov)


@dataclass
class ProcessArgs:
    input_paths: List[str]
    tokenizer_path: str
    output_path: str
    language: str = "python"
    max_length: int = 328
    include_positions: bool = True
    batch_size: int = 256
    max_rows: Optional[int] = None


def _iter_input_rows(df: pd.DataFrame, max_rows: Optional[int] = None):
    n = len(df) if max_rows is None else min(len(df), int(max_rows))
    for i in range(n):
        yield df.iloc[i]


def run(args: ProcessArgs) -> None:
    ap = _import_ast_parsing()
    tokenizer = _load_tokenizer(args.tokenizer_path)

    # Determine output columns as union of all input columns + added ones.
    input_cols: List[str] = []
    for p in args.input_paths:
        df0 = pd.read_parquet(p, engine="pyarrow")
        for c in df0.columns.tolist():
            if c not in input_cols:
                input_cols.append(c)

    extra_cols = ["AST_span", "coverage_percentage", "total_spans", "language", "error"]
    out_cols = input_cols + [c for c in extra_cols if c not in input_cols]

    # Stream-write parquet to avoid holding everything in memory.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("pyarrow is required for streaming parquet writes") from e

    writer: Optional[pq.ParquetWriter] = None
    total_written = 0

    def _flush_batch(rows: List[Dict[str, Any]]) -> None:
        nonlocal writer, total_written
        if not rows:
            return
        batch_df = pd.DataFrame(rows)
        # Ensure all columns exist.
        for c in out_cols:
            if c not in batch_df.columns:
                batch_df[c] = None
        batch_df = batch_df[out_cols]
        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        if writer is None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            writer = pq.ParquetWriter(args.output_path, table.schema, compression="zstd")
        writer.write_table(table)
        total_written += len(rows)
        rows.clear()

    rows_buf: List[Dict[str, Any]] = []

    for path in args.input_paths:
        df = pd.read_parquet(path, engine="pyarrow")
        for row in _iter_input_rows(df, args.max_rows):
            content = row.get("content", "")
            if content is None or (isinstance(content, float) and pd.isna(content)):
                continue
            content = str(content)
            if not content.strip():
                continue

            out_row: Dict[str, Any] = {c: row.get(c, None) for c in input_cols}
            out_row["language"] = args.language

            try:
                # Preprocess to improve parse success.
                try:
                    content_for_parse = ap.preprocess_code_for_parsing(content, ps_language=args.language)
                except Exception:
                    content_for_parse = content

                ast_root = ap.parse_to_ast(content_for_parse, ps_language=args.language)
                if ast_root is None:
                    spans = []
                    cov = 0.0
                    out_row["error"] = "Failed to parse AST"
                else:
                    spans, cov = deepseek_token_aligned_ast_spans(
                        code_text=content_for_parse,
                        ast_root=ast_root,
                        tokenizer=tokenizer,
                        ap=ap,
                        include_positions=bool(args.include_positions),
                        max_length=int(args.max_length),
                    )
                    out_row["error"] = None

                out_row["AST_span"] = json.dumps(spans, ensure_ascii=False)
                out_row["coverage_percentage"] = float(cov)
                out_row["total_spans"] = int(len(spans))

            except Exception as e:
                out_row["AST_span"] = json.dumps([], ensure_ascii=False)
                out_row["coverage_percentage"] = 0.0
                out_row["total_spans"] = 0
                out_row["error"] = f"{type(e).__name__}: {e}"

            rows_buf.append(out_row)
            if len(rows_buf) >= int(args.batch_size):
                _flush_batch(rows_buf)

        # If max_rows is set, apply it per-file and stop after first file for smoke runs.
        if args.max_rows is not None:
            break

    _flush_batch(rows_buf)
    if writer is not None:
        writer.close()

    print(f"[done] wrote {total_written} rows to {args.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek tokenizer-aligned AST span preprocessing (Python)")
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="+",
        default=[
            "/data/home/zhangsj/Data/more_big_code_language/python/baseline_training/train-00000-of-00059.parquet",
            "/data/home/zhangsj/Data/more_big_code_language/python/baseline_training/train-00001-of-00059.parquet",
            "/data/home/zhangsj/Data/more_big_code_language/python/baseline_training/train-00002-of-00059.parquet",
        ],
        help="Input parquet shard paths.",
    )
    parser.add_argument("--tokenizer_path", type=str, default="/data/home/zhangsj/deepseek_qwen1.5_distill")
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/home/zhangsj/Data/more_big_code_language/python/deepseek_qwen1.5_python_ast.Parquet",
    )
    parser.add_argument("--max_length", type=int, default=328, help="Tokenizer truncation length for span token indices.")
    parser.add_argument("--batch_size", type=int, default=256, help="Rows per write_table flush.")
    parser.add_argument("--max_rows", type=int, default=None, help="If set, only process first N rows (smoke test).")
    parser.add_argument("--include_positions", action="store_true", default=True, help="Include line/column info in spans.")

    a = parser.parse_args()
    run(
        ProcessArgs(
            input_paths=list(a.input_paths),
            tokenizer_path=str(a.tokenizer_path),
            output_path=str(a.output_path),
            max_length=int(a.max_length),
            include_positions=bool(a.include_positions),
            batch_size=int(a.batch_size),
            max_rows=(int(a.max_rows) if a.max_rows is not None else None),
        )
    )


if __name__ == "__main__":
    main()



