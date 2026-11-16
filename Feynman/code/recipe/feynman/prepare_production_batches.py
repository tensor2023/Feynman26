"""Select 500 parents, freeze source splits, and prepare 1,500-task Codex batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FAMILIES = ("direct_application", "behavioral_transfer", "misconception_diagnosis")

INSTRUCTION = """You generate source-grounded GRPO scenarios. Each input includes a private primary-source episode and exactly three requested derivatives. Return one output per requested derivative.

Rules:
- Ground every field only in the private episode; add no outside factual claims.
- Make each question independently understandable and answerable without seeing the source.
- Never name or impersonate Richard Feynman, quote the source, or leak distinctive wording.
- direct_application applies the episode's principle to a nearby concrete situation.
- behavioral_transfer transfers the same observable reasoning move to a different domain without requiring specialized outside facts.
- misconception_diagnosis presents a plausible mistaken assumption and asks for diagnosis and repair.
- The three prompts must be substantively different, not paraphrases.
- Give 2-4 supported required facts, reasoning moves, communication traits, allowed variations, and specific fatal errors.
- Distinctive phrases are private copying-detector strings copied from the source, 4-10 words each, at most three.
- Reject only a derivative whose source is incomplete, corrupt, or incapable of supporting that family; explain why.
Output only the schema-conforming JSON object, preserving requested task_id values and order."""


def stable_split(source_id: str) -> str:
    bucket = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 7 else "development" if bucket < 9 else "test"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_episodes", type=Path)
    parser.add_argument("transcript_episodes", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--parents", type=int, default=500)
    parser.add_argument("--transcript-parents", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--base-offset", type=int, default=0)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--batch-offset", type=int, default=0)
    args = parser.parse_args()

    transcript = read_jsonl(args.transcript_episodes)
    base = read_jsonl(args.base_episodes)
    # Favor strong, clean transcript Q&A, then diversify the established pool by source.
    transcript.sort(key=lambda r: (-r.get("word_count", 0), r["episode_id"]))
    base.sort(key=lambda r: (-r.get("quality_score", 0), r["source_id"], r["episode_id"]))
    selected = transcript[: args.transcript_parents]
    needed = args.parents - len(selected)
    selected.extend(base[args.base_offset : args.base_offset + needed])
    if len(selected) != args.parents:
        raise RuntimeError(f"needed {args.parents} parent episodes, found {len(selected)}")

    requests = []
    manifest_rows = []
    for parent_index, row in enumerate(selected, 1):
        split = stable_split(row["source_id"])
        manifest_rows.append({"parent_episode_id": row["episode_id"], "source_id": row["source_id"], "split": split})
        for family_index, family in enumerate(FAMILIES, 1):
            requests.append({
                "task_id": f"FEYN-GRPO-V2-{args.task_offset + (parent_index - 1) * 3 + family_index:06d}",
                "parent_episode_id": row["episode_id"],
                "template_family": family,
                "split": split,
                "source": {
                    "source_id": row["source_id"], "title": row["title"],
                    "source_year": row["source_year"], "location": row["page"],
                    "source_text_private": row["text"], "source_sha256": row["source_sha256"],
                    "local_path": row["local_path"],
                },
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(requests), args.batch_size * 3):
        batch = requests[start : start + args.batch_size * 3]
        number = args.batch_offset + start // (args.batch_size * 3) + 1
        prompt = INSTRUCTION + "\n\nINPUT REQUESTS:\n" + json.dumps(batch, ensure_ascii=False, indent=2)
        (args.output_dir / f"generation_{number:03d}.txt").write_text(prompt, encoding="utf-8")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(requests)} tasks from {len(selected)} parents in {(len(requests) + args.batch_size * 3 - 1) // (args.batch_size * 3)} batches")


if __name__ == "__main__":
    main()
