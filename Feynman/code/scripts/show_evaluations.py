#!/usr/bin/env python3
"""Pretty-print saved VERL rollout Q&A transcripts."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("train", "validation"), default="validation")
    parser.add_argument("--step", type=int)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    root = Path("outputs/evaluations") / args.kind
    files = sorted(root.glob("*.jsonl"), key=lambda path: int(path.stem))
    if not files:
        raise SystemExit(f"No {args.kind} evaluation transcripts found in {root}")
    path = root / f"{args.step}.jsonl" if args.step is not None else files[-1]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not 0 <= args.index < len(rows):
        raise SystemExit(f"--index must be between 0 and {len(rows) - 1}")

    row = rows[args.index]
    print(f"File: {path} | sample {args.index + 1}/{len(rows)} | score: {row.get('score')}")
    print("\n=== QUESTION / INPUT ===\n")
    print(row.get("input", ""))
    print("\n=== MODEL ANSWER ===\n")
    print(row.get("output", ""))
    details = {k: v for k, v in row.items() if k not in {"input", "output", "gts"}}
    print("\n=== EVALUATION DETAILS ===\n")
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
