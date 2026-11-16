"""Join cached production generations with private sources and emit audited JSONL/report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


# The initial stable hash left this small source registry without a lockbox group.
# Freeze one complete source group into the lockbox rather than splitting rows.
SPLIT_OVERRIDES = {"FA1987W": "test"}


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def load_requests(prompt_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    marker = "INPUT REQUESTS:\n"
    for path in sorted(prompt_dir.glob("generation_*.txt")):
        text = path.read_text(encoding="utf-8")
        for row in json.loads(text.split(marker, 1)[1]):
            rows[row["task_id"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=1500)
    args = parser.parse_args()
    requests = load_requests(args.prompt_dir)
    cards: dict[str, dict] = {}
    for path in sorted(args.result_dir.glob("generation_*.json")):
        for item in json.loads(path.read_text(encoding="utf-8"))["items"]:
            if item["task_id"] in cards:
                raise ValueError(f'duplicate task id: {item["task_id"]}')
            cards[item["task_id"]] = item
    if len(requests) != args.expected or len(cards) != args.expected:
        raise RuntimeError(f"expected {args.expected}; requests={len(requests)}, generated={len(cards)}")

    prompts = Counter(normalize(card["prompt"]) for card in cards.values() if card["accepted"])
    rows, rejected, issues = [], [], []
    for task_id in sorted(requests):
        request, card = requests[task_id], cards[task_id]
        source = request["source"]
        split = SPLIT_OVERRIDES.get(source["source_id"], request["split"])
        row_issues: list[str] = []
        if not card["accepted"]:
            rejected.append({"task_id": task_id, "reason": card["rejection_reason"]})
            continue
        if prompts[normalize(card["prompt"])] > 1:
            row_issues.append("exact normalized duplicate prompt")
        if re.search(r"\b(?:Richard\s+)?Feynman\b", card["prompt"], re.I):
            row_issues.append("policy prompt names Feynman")
        leaked = [p for p in card["distinctive_phrases"] if normalize(p) and normalize(p) in normalize(card["prompt"])]
        if leaked:
            row_issues.append("distinctive phrase leaked into prompt")
        if not all(card[key] for key in ("required_facts", "reasoning_moves", "fatal_errors")):
            row_issues.append("required private card field is empty")
        if row_issues:
            issues.append({"task_id": task_id, "issues": row_issues})
            continue
        card_hash = hashlib.sha256(json.dumps(card, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        year = int(source["source_year"])
        rows.append({
            "task_id": task_id,
            "parent_episode_id": request["parent_episode_id"],
            "template_family": request["template_family"],
            "mode": "historical_simulation",
            "date": f"{year}-12-31",
            "audience": card["audience"],
            "scenario": "Answer with concrete, mechanistic reasoning grounded in the documented episode.",
            "user_message": card["prompt"],
            "persona_card": {"identity": "anonymous_persona_x", "documented_principles": list(dict.fromkeys(card["reasoning_moves"] + card["communication_traits"])), "knowledge_cutoff": f"{year}-12-31"},
            "evidence": [{"source_id": source["source_id"], "type": "private_primary_source_episode", "content": "Private reference available only to the reward judge.", "allowed_for_answer": False, "source_date": f"{year}-01-01", "title": source["title"], "location": f'location {source["location"]}'}],
            "forbidden_shortcuts": ["invented quotation", "long verbatim source reproduction", "claim to literally be the historical person", *card["fatal_errors"]],
            "evaluation_tags": ["scientific_correctness", "source_grounding", "reasoning_fidelity", "epistemic_honesty", "non_recitation", request["template_family"]],
            "reference_facts": card["required_facts"],
            "source_passages": [source["source_text_private"]],
            "reference_card": {"reference_text_private": source["source_text_private"], "required_facts": card["required_facts"], "reasoning_moves": card["reasoning_moves"], "communication_traits": card["communication_traits"], "allowed_variation": card["allowed_variation"], "fatal_errors": card["fatal_errors"], "distinctive_phrases": card["distinctive_phrases"]},
            "source_metadata": {"episode_id": request["parent_episode_id"], "source_id": source["source_id"], "title": source["title"], "source_year": year, "page": int(source["location"]), "local_path": source["local_path"], "source_sha256": source["source_sha256"]},
            "generation_metadata": {"generator": "Codex CLI production generation pass", "model": "gpt-5.6-sol", "prompt_version": "feynman-production-v2", "output_hash": card_hash},
            "split": split,
            "dataset_version": "feynman26-grpo-v2-1500",
        })
        if len(rows) == args.limit:
            break
    if len(rows) != args.limit:
        raise RuntimeError(f"needed {args.limit} clean accepted rows, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "requested": len(requests), "accepted": len(rows), "rejected": len(rejected),
        "deterministic_issue_count": len(issues), "issues": issues, "rejections": rejected,
        "by_family": Counter(r["template_family"] for r in rows),
        "by_split": Counter(r["split"] for r in rows),
        "unique_parents": len({r["parent_episode_id"] for r in rows}),
        "unique_prompts": len({normalize(r["user_message"]) for r in rows}),
        "transcript_rows": sum(r["source_metadata"]["source_id"].startswith("TXT-") for r in rows),
        "source_distribution": Counter(r["source_metadata"]["source_id"] for r in rows),
    }
    report["source_dominance_warnings"] = [
        {"source_id": source_id, "rows": count, "share": round(count / len(rows), 6)}
        for source_id, count in report["source_distribution"].items() if count / len(rows) > 0.10
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"issues", "rejections"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
