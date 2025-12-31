#!/usr/bin/env python3
"""Evaluate all DeepSeek BLT v2 checkpoints in a directory using EvalPlus (base-only).

For each checkpoint subdir (e.g., epoch_1, epoch_2, ...), run:
  - HumanEval+ base-only (disable local decoder, patcher=none)
  - MBPP+ base-only (disable local decoder, patcher=none)

Writes a summary JSON with pass@1 per (checkpoint, dataset).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Result:
    checkpoint: str
    dataset: str
    pass_at_1: Optional[float]
    samples_jsonl: str
    eval_results_json: Optional[str]
    ok: bool
    error: Optional[str]


def _sorted_checkpoints(root: Path) -> List[Path]:
    items = [p for p in root.iterdir() if p.is_dir()]

    def key(p: Path) -> Tuple[int, int, str]:
        m = re.match(r"^epoch_(\d+)(?:_(.*))?$", p.name)
        if not m:
            return (10**9, 10**9, p.name)
        ep = int(m.group(1))
        suffix = m.group(2) or ""
        return (ep, 0 if suffix == "" else 1, suffix)

    return sorted(items, key=key)


def _run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return int(p.returncode), str(p.stdout)


def _load_pass1(eval_results_json: Path) -> Optional[float]:
    try:
        obj = json.loads(eval_results_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(obj, dict):
        if "pass@1" in obj and isinstance(obj["pass@1"], (int, float)):
            return float(obj["pass@1"])
        scores = obj.get("scores")
        if isinstance(scores, dict) and "pass@1" in scores and isinstance(scores["pass@1"], (int, float)):
            return float(scores["pass@1"])
    return None


def _write_summary(summary_path: Path, results: List[Result]) -> None:
    summary_obj: Dict[str, Dict[str, Dict[str, object]]] = {}
    for r in results:
        summary_obj.setdefault(r.checkpoint, {})
        summary_obj[r.checkpoint][r.dataset] = {
            "pass@1": r.pass_at_1,
            "ok": r.ok,
            "samples_jsonl": r.samples_jsonl,
            "eval_results_json": r.eval_results_json,
            "error": r.error,
        }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval DeepSeek BLT v2 checkpoints (base-only) on EvalPlus")
    ap.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/deepseek_distill_blt_v2",
    )
    ap.add_argument(
        "--runner",
        type=str,
        default="/data/home/zhangsj/AST_decoding/run_evalplus_blt_deepseek15.py",
    )
    ap.add_argument("--model_path", type=str, default="/data/home/zhangsj/deepseek_qwen1.5_distill")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true", default=True)
    ap.add_argument("--output_dir", type=str, default="/data/home/zhangsj/AST_decoding/deepseek_passat1")
    ap.add_argument(
        "--summary_path",
        type=str,
        default="/data/home/zhangsj/AST_decoding/deepseek_passat1/deepseek_result_summary.json",
    )

    args = ap.parse_args()

    ckpt_root = Path(args.checkpoints_dir)
    runner = Path(args.runner)
    if not runner.exists():
        raise SystemExit(f"Runner not found: {runner}")
    if not ckpt_root.is_dir():
        raise SystemExit(f"Checkpoints dir not found: {ckpt_root}")

    checkpoints = _sorted_checkpoints(ckpt_root)
    if not checkpoints:
        raise SystemExit(f"No checkpoint directories found in {ckpt_root}")

    results: List[Result] = []
    summary_path = Path(args.summary_path)

    for ckpt in checkpoints:
        for dataset in ["humaneval", "mbpp"]:
            out_dir = Path(args.output_dir) / dataset
            out_dir.mkdir(parents=True, exist_ok=True)

            samples_jsonl = out_dir / f"basemodel_{ckpt.name}.jsonl"

            cmd = [
                sys.executable,
                str(runner),
                "--checkpoint",
                str(ckpt),
                "--model_path",
                str(args.model_path),
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
                str(samples_jsonl),
            ]
            if bool(args.overwrite):
                cmd.append("--overwrite")

            code, out = _run(cmd)

            eval_json = samples_jsonl.with_suffix(".eval_results.json")
            eval_json_legacy = Path(str(samples_jsonl).replace(".jsonl", "_eval_results.json"))
            eval_path = eval_json if eval_json.exists() else eval_json_legacy if eval_json_legacy.exists() else None

            pass1 = _load_pass1(eval_path) if eval_path is not None else None
            ok = (code == 0) and (pass1 is not None)

            results.append(
                Result(
                    checkpoint=str(ckpt),
                    dataset=dataset,
                    pass_at_1=pass1,
                    samples_jsonl=str(samples_jsonl),
                    eval_results_json=str(eval_path) if eval_path is not None else None,
                    ok=ok,
                    error=None if code == 0 else out[-4000:],
                )
            )

            _write_summary(summary_path, results)

    print(f"[done] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
