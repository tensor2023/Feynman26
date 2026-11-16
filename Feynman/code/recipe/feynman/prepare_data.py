"""Validate JSONL scenarios and emit OdysSim-compatible parquet files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.feynman.schema import Scenario


def convert(source: Path, destination: Path) -> int:
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            scenario = Scenario.model_validate_json(line)
            payload = scenario.model_dump(mode="json")
            payload["index"] = len(rows)
            rows.append(
                {
                    "data_source": "feynman",
                    "prompt": [{"role": "user", "content": scenario.user_message}],
                    "ability": "source_grounded_simulation",
                    "reward_model": {"style": "model", "ground_truth": ""},
                    "extra_info": payload,
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(destination))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = convert(args.source, args.destination)
    print(f"Wrote {count} validated scenarios to {args.destination}")


if __name__ == "__main__":
    main()
