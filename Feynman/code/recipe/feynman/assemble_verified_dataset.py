"""Assemble the top verified source-grounded scenarios into final JSONL."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.feynman.schema import Scenario


def load_jsonl(path: Path) -> dict[str, dict]:
    return {row["episode_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def load_items(patterns: list[str], key: str) -> dict[str, dict]:
    rows = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for item in json.load(open(path, encoding="utf-8"))["items"]:
                rows[item[key]] = item
    return rows


def score(item: dict) -> float:
    return (
        0.40 * item["source_support"]
        + 0.25 * item["prompt_independence"]
        + 0.25 * item["episode_completeness"]
        + 0.10 * (1.0 - item["copying_risk"])
    )


def assemble(episodes: dict[str, dict], cards: dict[str, dict], checks: dict[str, dict], limit: int) -> list[Scenario]:
    verified = [item for item in checks.values() if item["verified"]]
    verified.sort(key=lambda item: (-score(item), item["episode_id"]))
    scenarios = []
    for index, check in enumerate(verified[:limit], 1):
        episode = episodes[check["episode_id"]]
        card = cards[check["episode_id"]]
        card_hash = hashlib.sha256(json.dumps(card, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        scenario = Scenario.model_validate(
            {
                "task_id": f"FEYN-RL-{index:06d}",
                "mode": "historical_simulation",
                "date": f'{episode["source_year"]}-12-31',
                "audience": card["audience"],
                "scenario": "Answer the question using concrete, mechanistic reasoning grounded in the documented source episode.",
                "user_message": card["prompt"],
                "persona_card": {
                    "identity": "anonymous_persona_x",
                    "documented_principles": list(dict.fromkeys(card["reasoning_moves"] + card["communication_traits"])),
                    "knowledge_cutoff": f'{episode["source_year"]}-12-31',
                },
                "evidence": [
                    {
                        "source_id": episode["source_id"],
                        "type": "private_primary_source_episode",
                        "content": "Private reference episode available to the reward judge; not exposed to the policy.",
                        "allowed_for_answer": False,
                        "source_date": f'{episode["source_year"]}-01-01',
                        "title": episode["title"],
                        "location": f'page {episode["page"]}',
                    }
                ],
                "forbidden_shortcuts": [
                    "invented quotation",
                    "long verbatim source reproduction",
                    "claim to literally be the historical person",
                    *card["fatal_errors"],
                ],
                "evaluation_tags": [
                    "scientific_correctness",
                    "source_grounding",
                    "reasoning_fidelity",
                    "epistemic_honesty",
                    "non_recitation",
                    "temporal_honesty",
                ],
                "reference_facts": card["required_facts"],
                "source_passages": [episode["text"]],
                "reference_card": {
                    "reference_text_private": episode["text"],
                    "required_facts": card["required_facts"],
                    "reasoning_moves": card["reasoning_moves"],
                    "communication_traits": card["communication_traits"],
                    "allowed_variation": card["allowed_variation"],
                    "fatal_errors": card["fatal_errors"],
                    "distinctive_phrases": card["distinctive_phrases"],
                },
                "source_metadata": episode,
                "generation_metadata": {
                    "generator": "Codex CLI generation pass",
                    "model": "gpt-5.6-sol",
                    "prompt_version": "feynman-generation-v1",
                    "output_hash": card_hash,
                },
                "verification": {
                    "verifier": "separate Codex CLI verification pass",
                    "model": "gpt-5.6-sol",
                    "prompt_version": "feynman-verification-v1",
                    **{key: check[key] for key in ("source_support", "prompt_independence", "episode_completeness", "copying_risk", "issues")},
                },
                "split": "train",
            }
        )
        scenarios.append(scenario)
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    episodes = load_jsonl(Path("data/feynman/work/source_episodes.jsonl"))
    episodes.update(load_jsonl(Path("data/feynman/work/source_episodes_additional.jsonl")))
    cards = load_items(["data/feynman/work/generation_*.json", "data/feynman/work/additional_generation_*.json"], "episode_id")
    checks = load_items(["data/feynman/work/verification_*.json"], "episode_id")
    rows = assemble(episodes, cards, checks, args.limit)
    if len(rows) != args.limit:
        raise RuntimeError(f"needed {args.limit} verified rows, found {len(rows)}")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
    print(f"Wrote {len(rows)} verified scenarios to {args.destination}")


if __name__ == "__main__":
    main()
