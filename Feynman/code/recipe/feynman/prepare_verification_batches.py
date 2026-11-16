"""Join source episodes with generated cards and prepare an independent pass."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


INSTRUCTION = """You are the independent verification pass for a source-grounded dataset.
Judge each generated card only against its supplied private primary-source episode.

Set verified=true only when ALL conditions hold:
1. Every required fact and fatal error is directly supported or directly contradicted by the source as appropriate.
2. The episode is sufficiently complete and intelligible to serve as a private reference.
3. The user prompt is natural, answerable from the episode, does not name the historical speaker,
   and does not leak distinctive source phrasing or invite quotation/impersonation.
4. Reasoning moves and communication traits genuinely occur in the episode.
5. No material outside knowledge, invented detail, or repaired OCR has entered the card.

Scores are 0..1. copying_risk measures leakage/copying risk, so lower is better.
Reject when source_support < 0.90, prompt_independence < 0.85,
episode_completeness < 0.80, or copying_risk > 0.25. List concrete issues concisely.
Return exactly one result per input item in the same order and only the requested JSON object.
"""


def load_jsonl(path: Path) -> dict[str, dict]:
    return {row["episode_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_episodes", type=Path)
    parser.add_argument("additional_episodes", type=Path)
    parser.add_argument("generation_glob")
    parser.add_argument("additional_generation_glob")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    episodes = load_jsonl(args.base_episodes)
    episodes.update(load_jsonl(args.additional_episodes))
    cards = []
    for pattern in (args.generation_glob, args.additional_generation_glob):
        for path in sorted(glob.glob(pattern)):
            cards.extend(item for item in json.load(open(path, encoding="utf-8"))["items"] if item["accepted"])
    joined = [{"source_episode": episodes[card["episode_id"]], "generated_card": card} for card in cards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(joined), args.batch_size):
        batch = joined[start : start + args.batch_size]
        number = start // args.batch_size + 1
        text = INSTRUCTION + "\nITEMS TO VERIFY:\n" + json.dumps(batch, ensure_ascii=False, indent=2)
        (args.output_dir / f"verification_{number:02d}.txt").write_text(text, encoding="utf-8")
    print(f"Prepared {len(joined)} accepted cards in {(len(joined)+args.batch_size-1)//args.batch_size} verification batches")


if __name__ == "__main__":
    main()
