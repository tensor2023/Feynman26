from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "task_generation" / "work"
IDEA = ROOT / "data" / "idea.md"
GEN_SCHEMA = ROOT / "task_generation" / "task_batch.schema.json"
VERIFY_SCHEMA = ROOT / "task_generation" / "verified_batch.schema.json"

QUOTAS = [
    {"direct_explanation": 8, "follow_up_dialogue": 4, "misconception": 3, "quotation_source": 3, "adversarial_persona": 2}
] * 5

THEMES = [
    "Everyday mechanics, heat, waves, elementary mathematics, measurement, and classic pre-1988 attribution checks.",
    "Electricity, magnetism, optics, fluids, probability, experimental controls, and source uncertainty. Do not use spoon conductivity, circular satellite orbit, Euclid primes, pendulum mass, sky color, or any close paraphrase.",
    "Quantum physics, relativity, particle/nuclear physics, information, estimation, and historical-versus-modern boundaries.",
    "Biology-adjacent physics, climate and Earth science, astronomy, materials, computation, statistics, and model limitations.",
    "Unfamiliar transfer problems, adversarial false premises, fabricated quotations, unsupported certainty, experiment design, and analogy limits; avoid concepts used in the prior four scopes.",
]


def run_codex(prompt: str, schema: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return
    command = [
        "codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
        "--output-schema", str(schema), "-o", str(output), "-",
    ]
    subprocess.run(command, input=prompt, text=True, encoding="utf-8", check=True, cwd=ROOT)


def generation_prompt(batch: int, idea: str) -> str:
    start = batch * 20 + 1
    ids = [f"feynman-{i:04d}" for i in range(start, start + 20)]
    return f"""Create exactly 20 concise benchmark tasks for a source-grounded Feynman-inspired educational simulation.
This is data generation, not role-play. Do not claim to be Richard Feynman. Use no web or external files.

Dataset design brief:
{idea}

Batch: {batch + 1}/5
Required IDs in order: {json.dumps(ids)}
Exact family quotas: {json.dumps(QUOTAS[batch], sort_keys=True)}
Exclusive thematic scope for this batch: {THEMES[batch]}

Requirements:
- Keep user_message natural and normally under 70 words.
- Cover varied physics, mathematics, scientific method, uncertainty, modern post-1988 science, false premises, and attribution attacks.
- allowed_evidence must contain short self-contained evidence summaries or an explicit statement that no quotation evidence is supplied. Do not invent citations.
- private_reference must be a concise judge-only answer outline, not prose copied into user_message.
- Public rubric must describe observable behavior without revealing the full answer.
- Every task must be answerable from general stable scientific knowledge plus allowed_evidence.
- Avoid duplicate concepts within this batch and avoid catchphrase imitation.
- Return only the schema-conforming JSON object.
"""


def verification_prompt(idea: str, tasks: dict, prior_prompts: list[str]) -> str:
    return f"""Independently verify and, when needed, minimally correct these 20 benchmark tasks.
Do not trust the generating pass. Check scientific correctness, answerability, family fit, attribution boundaries,
public/private separation, concise wording, and whether rubrics reward mechanisms rather than caricature.
Preserve every task_id and family. Return exactly 20 tasks in the same order. Set valid=true after correcting any
repairable issue; valid=false only if a task cannot be made sound without changing its core concept.
Rewrite any task whose user_message substantially duplicates one of these previously verified prompts:
{json.dumps(prior_prompts, ensure_ascii=False)}

Design brief:
{idea}

Generated tasks:
{json.dumps(tasks, ensure_ascii=False)}

Return only schema-conforming JSON.
"""


def validate_and_assemble() -> list[dict]:
    rows: list[dict] = []
    for batch in range(5):
        payload = json.loads((WORK / "verified" / f"batch-{batch + 1}.json").read_text(encoding="utf-8"))
        if len(payload["tasks"]) != 20:
            raise ValueError(f"verified batch {batch + 1} does not contain 20 tasks")
        for item in payload["tasks"]:
            if not item["verification"]["valid"]:
                raise ValueError(f"invalid task after verification: {item['task']['task_id']}")
            row = item["task"]
            row["verification"] = item["verification"]
            rows.append(row)
    ids = [row["task_id"] for row in rows]
    prompts = [row["user_message"].strip().casefold() for row in rows]
    if len(rows) != 100 or len(set(ids)) != 100 or len(set(prompts)) != 100:
        raise ValueError("expected 100 unique task IDs and prompts")
    expected = Counter({"direct_explanation": 40, "follow_up_dialogue": 20, "misconception": 15, "quotation_source": 15, "adversarial_persona": 10})
    actual = Counter(row["family"] for row in rows)
    if actual != expected:
        raise ValueError(f"family quotas differ: {actual}")
    for index, row in enumerate(rows):
        row["split"] = "optimization" if index < 60 else "development" if index < 80 else "hidden_test"
        row["generation_method"] = "codex_cli_two_pass"
        row["design_sha256"] = hashlib.sha256(IDEA.read_bytes()).hexdigest()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "verify", "assemble", "all"], default="all")
    args = parser.parse_args()
    idea = IDEA.read_text(encoding="utf-8")
    if args.stage in {"generate", "all"}:
        for batch in range(5):
            run_codex(generation_prompt(batch, idea), GEN_SCHEMA, WORK / "generated" / f"batch-{batch + 1}.json")
    if args.stage in {"verify", "all"}:
        prior_prompts: list[str] = []
        for batch in range(5):
            generated = json.loads((WORK / "generated" / f"batch-{batch + 1}.json").read_text(encoding="utf-8"))
            verified_path = WORK / "verified" / f"batch-{batch + 1}.json"
            run_codex(verification_prompt(idea, generated, prior_prompts), VERIFY_SCHEMA, verified_path)
            verified = json.loads(verified_path.read_text(encoding="utf-8"))
            prior_prompts.extend(item["task"]["user_message"] for item in verified["tasks"])
    if args.stage in {"assemble", "all"}:
        rows = validate_and_assemble()
        output = ROOT / "benchmarks" / "feynman" / "tasks.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(json.dumps({"output": str(output), "tasks": len(rows), "families": Counter(r["family"] for r in rows), "splits": Counter(r["split"] for r in rows)}, default=dict))


if __name__ == "__main__":
    main()
