"""Create cached Codex generation prompts from extracted source episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


INSTRUCTION = """You are the generation pass for a source-grounded behavioral-simulation dataset.
For every supplied primary-source episode, either reject it or build one independent task.

Accept only an episode containing a reasonably complete explanation, correction, estimate,
argument, or scientific reasoning move. Reject broken OCR, equation-dominated fragments,
administrative text, incomplete fragments, and passages whose central claim cannot be
understood from the supplied text alone.

For accepted episodes:
- Infer a natural question or situation that could elicit the reasoning.
- The prompt must not name Richard Feynman, request impersonation, quote the passage, or leak
  distinctive wording. It must be answerable using the episode.
- Paraphrase 2-4 required facts supported directly by the episode.
- Extract 2-4 reasoning moves and 2-4 communication traits.
- State safe variations and concrete fatal errors.
- Record at most three distinctive source phrases, each 4-10 words, solely for private copying detection.
- Do not add facts from outside the supplied episode.

For rejected episodes, preserve episode_id, set accepted=false, give a concise rejection_reason,
and use empty strings/lists for all remaining fields. Return exactly one item per input episode,
in the same order. Output only the requested JSON object.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.episodes.read_text(encoding="utf-8").splitlines() if line]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        number = start // args.batch_size + 1
        payload = [
            {
                "episode_id": row["episode_id"],
                "source_id": row["source_id"],
                "title": row["title"],
                "source_year": row["source_year"],
                "page": row["page"],
                "source_text_private": row["text"],
            }
            for row in batch
        ]
        prompt = INSTRUCTION + "\nINPUT EPISODES:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        (args.output_dir / f"generation_{number:02d}.txt").write_text(prompt, encoding="utf-8")
    print(f"Prepared {(len(rows) + args.batch_size - 1) // args.batch_size} generation batches")


if __name__ == "__main__":
    main()
