#!/usr/bin/env python3
"""
Batch-run EvalPlus (HumanEval+/MBPP+) for a fixed list of checkpoints using the existing
`/data/AST_decoding/run_evalplus_blt.py` entrypoint, then summarize base/plus pass@1 into
an independent Markdown file.

Key requirements this script enforces:
- Runs 10 checkpoints × 2 datasets (humaneval, mbpp) with consistent hyperparameters.
- Forces overwrite of any cached EvalPlus results (`--overwrite`).
- Ensures output filenames include the FULL checkpoint path (sanitized) so you can
  distinguish different checkpoints by filename alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


RUN_EVALPLUS_BLT = "/data/AST_decoding/run_evalplus_blt.py"

CHECKPOINTS: List[str] = [
    "/data/unfreeze_2layers/epoch_1",
    "/data/unfreeze_2layers/epoch_1_25pct",
    "/data/unfreeze_2layers/epoch_1_50pct",
    "/data/unfreeze_2layers/epoch_1_75pct",
    "/data/unfreeze_2layers/epoch_2",
    "/data/unfreeze_3layers/epoch_1",
    "/data/unfreeze_3layers/epoch_1_25pct",
    "/data/unfreeze_3layers/epoch_1_50pct",
    "/data/unfreeze_3layers/epoch_1_75pct",
    "/data/unfreeze_3layers/epoch_2",
]

DATASETS: List[str] = ["humaneval", "mbpp"]

try:
    # Optional dependency; script will gracefully fall back if not installed.
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _sanitize_ckpt_tag(checkpoint: str) -> str:
    """
    Convert a checkpoint path into a filename-safe tag that still contains the entire path.
    Example: /data/unfreeze_2layers/epoch_1 -> data__unfreeze_2layers__epoch_1
    """
    tag = checkpoint.strip("/")
    tag = tag.replace("/", "__")
    # Replace any remaining unsafe characters conservatively.
    tag = re.sub(r"[^A-Za-z0-9._=-]+", "_", tag)
    return tag


def _compute_pass1_from_eval_results(eval_results_json: str) -> Tuple[float, float, int]:
    """
    Compute base/plus pass@1 from EvalPlus per-task results.

    Expected structure (confirmed in this repo's existing results):
      {
        "eval": {
          "HumanEval/0": [
            {"base_status": "pass|fail", "plus_status": "pass|fail", ...},
            ...
          ],
          ...
        },
        "pass_at_k": ...
      }

    We compute pass@1 as the fraction of tasks whose first sample passes.
    To be robust to n_samples > 1, we treat a task as passed if ANY sample passes.
    """
    with open(eval_results_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    ev = obj.get("eval", {})
    if not isinstance(ev, dict):
        raise ValueError(f"Unexpected eval_results format: 'eval' is {type(ev).__name__}")

    total_tasks = 0
    base_pass_tasks = 0
    plus_pass_tasks = 0

    for _task_id, entries in ev.items():
        if not isinstance(entries, list):
            continue
        if not entries:
            continue

        total_tasks += 1

        def _status_is_pass(v: Any) -> bool:
            return str(v).lower() == "pass"

        base_ok = any(_status_is_pass(e.get("base_status")) for e in entries if isinstance(e, dict))
        plus_ok = any(_status_is_pass(e.get("plus_status")) for e in entries if isinstance(e, dict))

        if base_ok:
            base_pass_tasks += 1
        if plus_ok:
            plus_pass_tasks += 1

    if total_tasks <= 0:
        raise ValueError(f"No tasks found in eval results: {eval_results_json}")

    return base_pass_tasks / total_tasks, plus_pass_tasks / total_tasks, total_tasks


@dataclass
class OneRunResult:
    checkpoint: str
    dataset: str
    ok: bool
    base_pass_at_1: Optional[float]
    plus_pass_at_1: Optional[float]
    num_tasks: Optional[int]
    samples_jsonl: str
    eval_results_json: str
    error: Optional[str]


def _write_markdown_summary(
    out_path: str,
    results: List[OneRunResult],
    *,
    run_dir: str,
) -> None:
    by_ckpt: Dict[str, Dict[str, OneRunResult]] = {}
    for r in results:
        by_ckpt.setdefault(r.checkpoint, {})[r.dataset] = r

    lines: List[str] = []
    lines.append("# EvalPlus sweep summary (pass@1)\n")
    lines.append(f"- Run directory: `{run_dir}`\n")
    lines.append("- Metrics computed from per-task `base_status` / `plus_status` in `*.eval_results.json`.\n")
    lines.append("\n")

    lines.append("| checkpoint | humaneval base_pass@1 | humaneval plus_pass@1 | mbpp base_pass@1 | mbpp plus_pass@1 | |\n")
    lines.append("|---|---:|---:|---:|---:|---|\n")

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "NA"
        return f"{v:.4f}"

    for ckpt in CHECKPOINTS:
        he = by_ckpt.get(ckpt, {}).get("humaneval")
        mb = by_ckpt.get(ckpt, {}).get("mbpp")

        he_base = _fmt(he.base_pass_at_1) if he else "NA"
        he_plus = _fmt(he.plus_pass_at_1) if he else "NA"
        mb_base = _fmt(mb.base_pass_at_1) if mb else "NA"
        mb_plus = _fmt(mb.plus_pass_at_1) if mb else "NA"

        note_parts: List[str] = []
        if he and not he.ok:
            note_parts.append("humaneval:ERROR")
        if mb and not mb.ok:
            note_parts.append("mbpp:ERROR")
        note = " ".join(note_parts) if note_parts else ""

        lines.append(f"| `{ckpt}` | {he_base} | {he_plus} | {mb_base} | {mb_plus} | {note} |\n")

    lines.append("\n")
    lines.append("## Artifacts\n\n")
    lines.append("Each run writes files whose names include the full checkpoint path tag:\n\n")
    lines.append("- samples: `*.jsonl`\n")
    lines.append("- eval results: `*.eval_results.json`\n")
    lines.append("- raw samples: `*.raw.jsonl`\n")
    lines.append("- boundary stats (BLT path): `*_stats.json`\n")
    lines.append("\n")

    # Optionally include per-run file pointers for quick debugging.
    lines.append("## Per-run file paths\n\n")
    for r in results:
        status = "OK" if r.ok else f"ERROR: {r.error}"
        lines.append(f"- `{r.checkpoint}` `{r.dataset}`: {status}\n")
        lines.append(f"  - samples: `{r.samples_jsonl}`\n")
        lines.append(f"  - eval: `{r.eval_results_json}`\n")
        if r.ok:
            lines.append(f"  - base_pass@1: {r.base_pass_at_1:.4f} (tasks={r.num_tasks})\n")
            lines.append(f"  - plus_pass@1: {r.plus_pass_at_1:.4f} (tasks={r.num_tasks})\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep EvalPlus across multiple checkpoints and summarize pass@1.")
    p.add_argument("--gpu", type=int, default=0, help="GPU device ID passed to run_evalplus_blt.py (--gpu).")
    p.add_argument("--model_path", type=str, default="/data/AST_decoding", help="Passed to run_evalplus_blt.py (--model_path).")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--output_root", type=str, default="/data/evalplus_sweeps", help="Base output directory for this sweep run.")
    p.add_argument("--dry_run", action="store_true", help="Print commands but do not execute.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(RUN_EVALPLUS_BLT):
        print(f"[FATAL] Missing entrypoint: {RUN_EVALPLUS_BLT}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    all_results: List[OneRunResult] = []

    run_jobs: List[Tuple[str, str]] = [(ckpt, dataset) for ckpt in CHECKPOINTS for dataset in DATASETS]

    if tqdm is not None:
        iterator = tqdm(run_jobs, total=len(run_jobs), desc="EvalPlus sweep", unit="run")
    else:
        iterator = iter(run_jobs)

    for job_idx, (ckpt, dataset) in enumerate(iterator, start=1):
        if tqdm is None:
            print(f"\n[progress] {job_idx}/{len(run_jobs)} runs")

        ckpt_tag = _sanitize_ckpt_tag(ckpt)
        output_jsonl = os.path.join(
            run_dir,
            f"{dataset}__ckpt={ckpt_tag}__max_new_tokens={int(args.max_new_tokens)}__patcher=none__disable_local_decoder=true.jsonl",
        )
        eval_results_json = output_jsonl.replace(".jsonl", ".eval_results.json")

        cmd = [
            sys.executable,
            RUN_EVALPLUS_BLT,
            "--checkpoint",
            ckpt,
            "--model_path",
            args.model_path,
            "--dataset",
            dataset,
            "--n_samples",
            str(int(args.n_samples)),
            "--max_new_tokens",
            str(int(args.max_new_tokens)),
            "--disable_local_decoder",
            "--patcher",
            "none",
            "--gpu",
            str(int(args.gpu)),
            "--output",
            output_jsonl,
            "--overwrite",
        ]

        print("\n" + "=" * 100)
        print(f"[RUN] checkpoint={ckpt} dataset={dataset}")
        print("[CMD] " + " ".join(cmd))

        if args.dry_run:
            if tqdm is not None:
                iterator.set_postfix_str("dry_run")
            all_results.append(
                OneRunResult(
                    checkpoint=ckpt,
                    dataset=dataset,
                    ok=False,
                    base_pass_at_1=None,
                    plus_pass_at_1=None,
                    num_tasks=None,
                    samples_jsonl=output_jsonl,
                    eval_results_json=eval_results_json,
                    error="dry_run",
                )
            )
            continue

        try:
            subprocess.run(cmd, check=True)
            base_p1, plus_p1, n_tasks = _compute_pass1_from_eval_results(eval_results_json)
            all_results.append(
                OneRunResult(
                    checkpoint=ckpt,
                    dataset=dataset,
                    ok=True,
                    base_pass_at_1=base_p1,
                    plus_pass_at_1=plus_p1,
                    num_tasks=n_tasks,
                    samples_jsonl=output_jsonl,
                    eval_results_json=eval_results_json,
                    error=None,
                )
            )
            print(f"[OK] base_pass@1={base_p1:.4f} plus_pass@1={plus_p1:.4f} tasks={n_tasks}")
            if tqdm is not None:
                iterator.set_postfix_str(f"{dataset} OK")
        except Exception as e:
            all_results.append(
                OneRunResult(
                    checkpoint=ckpt,
                    dataset=dataset,
                    ok=False,
                    base_pass_at_1=None,
                    plus_pass_at_1=None,
                    num_tasks=None,
                    samples_jsonl=output_jsonl,
                    eval_results_json=eval_results_json,
                    error=str(e),
                )
            )
            print(f"[ERROR] {e}", file=sys.stderr)
            if tqdm is not None:
                iterator.set_postfix_str(f"{dataset} ERROR")

    summary_md = os.path.join(run_dir, "pass1_summary.md")
    _write_markdown_summary(summary_md, all_results, run_dir=run_dir)
    print("\n" + "=" * 100)
    print(f"[DONE] Wrote Markdown summary: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


