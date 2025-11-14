#!/usr/bin/env python3
"""
Generate EvalPlus-formatted samples (.jsonl) using the PEFT span-aware model,
then you can run evalplus.evaluate on the generated file.

This script mirrors EvalPlus's codegen flow but uses your PEFT adapter and
custom span-aware embedding from evaluate_peft_model.py.

Example:

  python peft_evalplus_codegen.py \
    --adapter_dir /data/home/zhangsj/qwen_coder_1.5b/best_peft_lora_span_aware_peft_lora_v1_mean_pooling \
    --base_model_dir /data/home/zhangsj/qwen_coder_1.5b \
    --dataset humaneval \
    --root /data/home/zhangsj/qwen_coder_1.5b/evalplus_results \
    --greedy --use_ast --max_new_tokens 256

Then evaluate:

  PYTHONPATH=$PYTHONPATH:/data/home/zhangsj/qwen_coder_1.5b/evalplus \
  python -m evalplus.evaluate --dataset humaneval --samples \
    /data/home/zhangsj/qwen_coder_1.5b/evalplus_results/humaneval/<identifier>.jsonl
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import torch


def _ensure_evalplus_on_path():
    try:
        import evalplus  # noqa: F401
        return
    except Exception:
        # Try to add local evalplus repo (package root) to sys.path
        local_evalplus_root = "/data/home/zhangsj/qwen_coder_1.5b/evalplus"
        if os.path.isdir(local_evalplus_root):
            sys.path.append(local_evalplus_root)
        # Retry import
        import evalplus  # noqa: F401


_ensure_evalplus_on_path()

from evalplus.data import get_human_eval_plus, get_mbpp_plus  # type: ignore  # noqa: E402
from evalplus.sanitize import sanitize  # type: ignore  # noqa: E402

# Reuse your PEFT loader and generator (span-aware)
from evaluate_peft_model import (  # type: ignore  # noqa: E402
    load_peft_span_aware_model,
    generate_with_optional_ast,
)


def _default_identifier(adapter_dir: str, backend_tag: str, temperature: float) -> str:
    # Mimic EvalPlus identifier style
    normalized = adapter_dir.strip("./").replace("/", "--")
    return f"{normalized}_{backend_tag}_temp_{temperature}"


def _load_dataset(dataset: str) -> Dict:
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported dataset: {dataset}")


def _maybe_read_existing_counts(jsonl_path: str) -> Dict[str, int]:
    counts = {}
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                task_id = obj.get("task_id")
                if task_id:
                    counts[task_id] = counts.get(task_id, 0) + 1
    return counts


def _write_jsonl_pair(jsonl_path: str, raw_jsonl_path: str, records: List[Dict]):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a") as f_out, open(raw_jsonl_path, "a") as f_raw:
        for rec in records:
            f_out.write(json.dumps(rec["out_rec"]) + "\n")
            f_raw.write(json.dumps(rec["raw_rec"]) + "\n")


def _extract_code_from_markdown(text: str, entrypoint: str = None) -> str:
    """Extract code from markdown fenced blocks when present.
    Prefer the block that contains the entrypoint if available.
    Fallback to full text when no fence found.
    """
    matches = list(re.finditer(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE))
    if not matches:
        return text
    # choose the first block with entrypoint, else the first block
    if entrypoint:
        for m in matches:
            block = m.group(1).strip()
            if f"def {entrypoint}" in block:
                return block
    return matches[0].group(1).strip()


def run_codegen(
    adapter_dir: str,
    base_model_dir: str,
    dataset: str,
    root: str,
    greedy: bool,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    resume: bool,
    id_range: Optional[List[int]],
    use_ast: bool,
    language: str,
    seed: int,
    top_p: float,
    identifier: Optional[str],
    fmt: str,
):
    torch.manual_seed(int(seed))

    # Load model/tokenizer with PEFT LoRA + span-aware embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_peft_span_aware_model(
        adapter_dir=adapter_dir, base_model_dir=base_model_dir
    )
    model = model.to(device)

    # Dataset
    dataset_dict = _load_dataset(dataset)

    # Resolve identifier and output paths
    backend_tag = "peft"
    if greedy:
        temperature = 0.0
        n_samples = 1
    ident = identifier or _default_identifier(adapter_dir, backend_tag, temperature)
    target_dir = os.path.join(root, dataset)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"{ident}.jsonl")
    raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")

    # Resume counts if needed
    task2count = _maybe_read_existing_counts(target_path) if resume else {}

    # Iterate tasks
    for task_id, task in dataset_dict.items():
        if id_range is not None:
            try:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    continue
            except Exception:
                pass

        already = int(task2count.get(task_id, 0))
        need = max(0, int(n_samples) - already)
        if need == 0:
            continue

        prompt = task["prompt"].strip() + "\n"
        pending: List[Dict] = []

        sidx = already
        while sidx < int(n_samples):
            # Generate one sample at a time (to leverage AST per sample)
            out_text = generate_with_optional_ast(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                use_ast=bool(use_ast),
                language=str(language),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                device=device,
            )
            # Extract potential fenced code for cleaner downstream use
            extracted = _extract_code_from_markdown(out_text, entrypoint=task["entry_point"])  # type: ignore

            if fmt == "solution":
                raw_solution = prompt + extracted
                sanitized_solution = sanitize(raw_solution, entrypoint=task["entry_point"])  # type: ignore
                out_rec = {"task_id": task_id, "solution": sanitized_solution}
                raw_rec = {"task_id": task_id, "solution": raw_solution}
            else:  # completion format (preferred)
                completion = extracted
                # keep raw completion as the un-extracted text for debugging
                out_rec = {"task_id": task_id, "completion": completion}
                raw_rec = {"task_id": task_id, "completion": out_text}

            pending.append({"task_id": task_id, "out_rec": out_rec, "raw_rec": raw_rec})
            sidx += 1

        if pending:
            _write_jsonl_pair(target_path, raw_target_path, pending)

    return target_path


def parse_args():
    p = argparse.ArgumentParser(description="PEFT span-aware codegen for EvalPlus")
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--base_model_dir", type=str, default="/data/home/zhangsj/qwen_coder_1.5b")
    p.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], required=True)
    p.add_argument("--root", type=str, default="evalplus_results")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--id_range", type=int, nargs=2, default=None, help="Optional [low high) numeric id range for quick runs")
    p.add_argument("--use_ast", action="store_true")
    p.add_argument("--language", type=str, default="python")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--identifier", type=str, default=None, help="Override output identifier name")
    p.add_argument("--fmt", type=str, choices=["completion", "solution"], default="completion", help="Output JSONL format: completion (recommended) or solution")
    p.set_defaults(resume=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = run_codegen(
        adapter_dir=args.adapter_dir,
        base_model_dir=args.base_model_dir,
        dataset=args.dataset,
        root=args.root,
        greedy=bool(args.greedy),
        n_samples=int(args.n_samples),
        temperature=float(args.temperature),
        max_new_tokens=int(args.max_new_tokens),
        resume=bool(args.resume),
        id_range=args.id_range,
        use_ast=bool(args.use_ast),
        language=str(args.language),
        seed=int(args.seed),
        top_p=float(args.top_p),
        identifier=args.identifier,
        fmt=str(args.fmt),
    )
    print(f"Saved sanitized samples to: {out_path}")
    print(f"Raw outputs: {out_path.replace('.jsonl', '.raw.jsonl')}")
    print("Now run evaluation, e.g.:")
    print(
        f"  PYTHONPATH=$PYTHONPATH:/data/home/zhangsj/qwen_coder_1.5b/evalplus python -m evalplus.evaluate --dataset {args.dataset} --samples {out_path}"
    )


if __name__ == "__main__":
    main()


