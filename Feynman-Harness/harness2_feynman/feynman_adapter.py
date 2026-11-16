from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness2.core.proposer import Proposer, Reply
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec
from harness2.core.tree import SurfaceSpec


ROOT = Path(__file__).resolve().parents[1]
TASKS_FILE = ROOT / "benchmarks" / "feynman" / "tasks.jsonl"
HARNESS_FILES = (
    "systemprompt.md", "character-card.json", "retrieval-policy.md", "response-policy.md",
    "verifier-policy.md", "modern-knowledge-policy.md", "memory-policy.md",
)


def _api_key() -> str:
    if value := os.environ.get("DEEPSEEK_API_KEY"):
        return value
    credential_file = Path("/mnt/c/2026/AGI/Feynman/code/Feynman26/api.md")
    text = credential_file.read_text(encoding="utf-8")
    match = re.search(r"^OPENAI_API_KEY=(\S+)$", text, re.MULTILINE)
    if not match:
        raise RuntimeError("OPENAI_API_KEY entry not found in the configured credential file")
    return match.group(1)


def api_call(messages: list[dict[str, str]], *, json_mode: bool = False, max_tokens: int = 5000, thinking: bool = True) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if not thinking:
        body["thinking"] = {"type": "disabled"}
    last_error: Exception | None = None
    for attempt in range(1, 4):
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["choices"][0]["message"].get("content", ""), payload.get("usage", {})
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(3 * attempt)
    raise RuntimeError(f"API call failed after 3 attempts: {last_error}")


def parse_or_repair_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        repaired, _ = api_call(
            [
                {"role": "system", "content": "Repair the supplied malformed JSON. Preserve its meaning and return only one valid JSON object."},
                {"role": "user", "content": cleaned},
            ],
            json_mode=True,
            max_tokens=6000,
            thinking=False,
        )
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            value, _ = json.JSONDecoder().raw_decode(repaired.lstrip())
            if not isinstance(value, dict):
                raise ValueError("repaired JSON is not an object")
            return value


class FeynmanSubstrate:
    name = "feynman-api"
    _surface = SurfaceSpec(
        name="opencode",
        always_on=HARNESS_FILES,
        always_on_max=40_000,
        always_on_cap_reason="the compact production behavior surface must remain reviewable",
        skill_dir="skills",
        skill_name_rule="must_match",
        trigger_hint="",
        autotrigger_file="systemprompt.md",
        trigger_line="",
        subagent_requires_mode=False,
        plugin_dir=None,
        hooks_file=None,
        reject_empty_always_on=True,
    )

    def surface(self) -> SurfaceSpec:
        return self._surface


class DeepSeekProposer(Proposer):
    """Harness² proposer/composer backed directly by the DeepSeek API."""

    def call(self, mission, context_dir, label, *, system="", timeout=None, max_attempts=None, harvest=None) -> Reply:
        started = time.time()
        context = Path(context_dir) if context_dir else None
        context_text = []
        if context and context.is_dir():
            for path in sorted(context.rglob("*")):
                if path.is_file() and path.stat().st_size < 80_000:
                    rel = path.relative_to(context)
                    context_text.append(f"\n--- FILE {rel} ---\n{path.read_text(encoding='utf-8', errors='replace')}")
        packed = "".join(context_text)
        workspace = None
        if harvest:
            source = context / "harness"
            workspace = Path(self.audit_dir or ROOT / "runs" / "proposer-audit") / "workspaces" / re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            if workspace.exists():
                shutil.rmtree(workspace)
            shutil.copytree(source, workspace)
            prompt = mission + "\n\nReturn JSON with keys 'files' and 'notes'. 'files' maps at least one changed relative harness path to its complete replacement text. Make a real, generalizable change even when the current harness is already strong. Do not add task IDs or task-specific answers.\n" + packed
            raw, usage = api_call([{"role": "system", "content": system or "Edit the bounded harness carefully."}, {"role": "user", "content": prompt}], json_mode=True, thinking=False)
            payload = parse_or_repair_json(raw)
            edits = payload.get("files", {})
            if not isinstance(edits, dict) or not any(rel in HARNESS_FILES for rel in edits):
                retry = "The previous proposal made no usable bounded edit. Return JSON with files containing one or more of these exact paths: " + ", ".join(HARNESS_FILES) + ". Each value must be the complete replacement file. Make the smallest generalizable improvement.\n\n" + packed
                raw, usage = api_call([{"role": "system", "content": "You must edit at least one allowed harness file."}, {"role": "user", "content": retry}], json_mode=True, thinking=False)
                payload = parse_or_repair_json(raw)
                edits = payload.get("files", {})
            if not isinstance(edits, dict):
                current = (workspace / "response-policy.md").read_text(encoding="utf-8")
                edits = {"response-policy.md": current.rstrip() + "\nBefore sending, ensure the final answer is nonempty, directly addresses the question, and includes a concrete check when useful.\n"}
            for rel, content in edits.items():
                if rel not in HARNESS_FILES:
                    continue
                (workspace / rel).write_text(str(content), encoding="utf-8")
            text = f"<notes>{payload.get('notes', 'Applied bounded harness edits.')}</notes>"
        else:
            prompt = mission + "\n\nUse the following context. Follow the requested output format exactly.\n" + packed
            text, usage = api_call([{"role": "system", "content": system or "Follow the Harness² mission exactly."}, {"role": "user", "content": prompt}], max_tokens=3000, thinking=False)
            if label.startswith("select.") and "```choice" not in text:
                candidates_dir = context / "candidates" if context else None
                valid = sorted(path.stem for path in candidates_dir.glob("*.md")) if candidates_dir and candidates_dir.is_dir() else []
                repair_prompt = (
                    "Convert the prior selection into the exact required format. Choose exactly one valid ID. "
                    f"Valid IDs: {valid}. Return only:\n```choice\n<ID>\n```\n\nPrior response:\n{text}"
                )
                text, usage = api_call([{"role": "system", "content": "Return exactly one fenced choice block and nothing else."}, {"role": "user", "content": repair_prompt}], max_tokens=100, thinking=False)
                if "```choice" not in text and valid:
                    text = f"```choice\n{valid[0]}\n```"
        reply = Reply(label=label, text=text, ok=True, attempts=1, wall_s=round(time.time() - started, 1), workspace=workspace, channel="deepseek-api")
        self._calls.append(reply)
        return reply


class FeynmanAdapter:
    def __init__(self, root: Path = ROOT, base_harness_dir: Path | None = None):
        self.root = root
        self.base_harness_dir = Path(base_harness_dir) if base_harness_dir else root / "feynman-harness"
        self.substrate = FeynmanSubstrate()
        channel = ChannelSpec("opencode", "deepseek/deepseek-flash", effort="low", steps=20, timeout=600)
        self.spec = BenchSpec(
            name="feynman",
            runs_root=root / "runs",
            install_mode="workspace_config",
            prompts=BenchPrompts(
                work="produce a concise, source-grounded scientific explanation that teaches through mechanisms, checks assumptions, and handles historical boundaries honestly",
                domain_word="scientific topic",
                inv_levers="clarity, uncertainty handling, evidence discipline, response economy",
                dom_levers="mechanistic explanation, conservation checks, analogy limits, experiment design, historical-versus-modern separation",
                yardstick="an accurate explanation a careful learner could test or use",
                deliverable_word="answer",
            ),
            channels={"proposer": channel, "selector": channel},
            leak_globs=("private-task.json", "judge-feedback.json"),
            obs_trunc=50_000,
        )
        self.tasks_by_id = {row["task_id"]: row for row in self._load_tasks()}

    def _load_tasks(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in TASKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

    def tasks(self, split: str = "all") -> list[str]:
        rows = self.tasks_by_id.values()
        return [r["task_id"] for r in rows if split == "all" or r["split"] == split]

    def domain_of(self, task_id: str) -> str:
        return self.tasks_by_id[task_id]["family"]

    def task_prompt(self, task_id: str) -> str:
        return self.tasks_by_id[task_id]["user_message"]

    def task_files(self, task_id: str) -> Path:
        directory = self.root / "benchmarks" / "feynman" / "public" / task_id
        directory.mkdir(parents=True, exist_ok=True)
        task = self.tasks_by_id[task_id]
        public = {k: task[k] for k in ("user_message", "allowed_evidence", "public_rubric")}
        (directory / "task.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
        return directory

    def base_harness(self) -> Path:
        return self.base_harness_dir

    def validate_tree(self, candidate: dict[str, str]) -> None:
        missing = [name for name in HARNESS_FILES if not candidate.get(name, "").strip()]
        if missing:
            raise ValueError(f"missing or empty required files: {missing}")
        json.loads(candidate["character-card.json"])
        combined = "\n".join(candidate.values())
        if len(combined) > 40_000:
            raise ValueError("harness exceeds the 40,000-character prompt budget")
        if re.search(r"\b(?:I am|I'm) Richard Feynman\b", combined, re.IGNORECASE):
            raise ValueError("literal historical identity claims are forbidden")
        if re.search(r"feynman-\d{4}", combined, re.IGNORECASE):
            raise ValueError("candidate contains benchmark task identifiers")

    def _judge(self, task: dict[str, Any], answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
        private = {k: task[k] for k in ("private_reference", "required_facts", "fatal_errors", "reasoning_moves", "communication_traits", "attribution_boundary")}
        prompt = json.dumps({"question": task["user_message"], "allowed_evidence": task["allowed_evidence"], "answer": answer, "citations": citations, "judge_only": private}, ensure_ascii=False)
        system = "Return strict JSON with numeric fields scientific_correctness, explanatory_quality, feynman_behavior, modern_grounding, robustness (0..1), hard_penalties (0..1), arrays issues and suggested_changes. Do not copy judge-only reference prose into feedback."
        required = set(DIMENSIONS := ("scientific_correctness", "explanatory_quality", "feynman_behavior", "modern_grounding", "robustness", "hard_penalties", "issues", "suggested_changes"))
        result: dict[str, Any] = {}
        usage: dict[str, Any] = {}
        def normalize(value: dict[str, Any]) -> dict[str, Any]:
            if required.issubset(value):
                return value
            for nested in value.values():
                if isinstance(nested, dict):
                    found = normalize(nested)
                    if required.issubset(found):
                        return found
            return value

        for attempt in range(4):
            suffix = "" if attempt == 0 else "\nYour previous JSON omitted required fields. Include every required field exactly."
            raw, usage = api_call([{"role": "system", "content": system + suffix}, {"role": "user", "content": prompt}], json_mode=True, max_tokens=1800, thinking=attempt < 2)
            result = normalize(parse_or_repair_json(raw))
            if required.issubset(result):
                break
        if not required.issubset(result):
            raise RuntimeError(f"judge omitted required fields: {sorted(required - set(result))}")
        weighted = 0.30 * result["scientific_correctness"] + 0.20 * result["explanatory_quality"] + 0.20 * result["feynman_behavior"] + 0.15 * result["modern_grounding"] + 0.15 * result["robustness"]
        result["raw_score"] = max(0.0, weighted - result["hard_penalties"])
        result["usage"] = usage
        return result

    def execute(self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None) -> Path:
        run_dir = self.root / "runs" / "executions" / run_id
        answer_path = run_dir / "answer.md"
        if answer_path.is_file() and answer_path.read_text(encoding="utf-8", errors="replace").strip():
            return run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        harness_root = harness_dir or self.base_harness()
        harness = {name: (harness_root / name).read_text(encoding="utf-8") for name in HARNESS_FILES}
        self.validate_tree(harness)
        task = self.tasks_by_id[task_id]
        evidence = "\n".join(f"[E{i}] {item}" for i, item in enumerate(task["allowed_evidence"], 1))
        system = "\n\n".join(harness[name] for name in HARNESS_FILES)
        user = f"Question:\n{task['user_message']}\n\nPermitted evidence:\n{evidence}\n\nPublic success criteria:\n- " + "\n- ".join(task["public_rubric"])
        started = time.time()
        answer, usage = api_call([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=1800)
        if not answer.strip():
            answer, usage = api_call([{"role": "system", "content": system}, {"role": "user", "content": user + "\n\nYou must return a nonempty final answer."}], max_tokens=1800)
        if not answer.strip():
            raise RuntimeError("model returned an empty answer twice")
        answer_path.write_text(answer.strip() + "\n", encoding="utf-8")
        citations = [{"evidence_id": f"E{i}", "text": item} for i, item in enumerate(task["allowed_evidence"], 1) if f"[E{i}]" in answer]
        (run_dir / "citations.json").write_text(json.dumps(citations, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "retrieval-log.json").write_text(json.dumps({"permitted_items": len(task["allowed_evidence"]), "retrieved": [c["evidence_id"] for c in citations]}, indent=2), encoding="utf-8")
        (run_dir / "conversation.json").write_text(json.dumps({"user": user, "assistant": answer}, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "token-usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
        (run_dir / "runtime-metadata.json").write_text(json.dumps({"model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"), "temperature": 0.2, "wall_seconds": round(time.time() - started, 2)}, indent=2), encoding="utf-8")
        trace = f"[ACTION] Load bounded Feynman harness and {len(task['allowed_evidence'])} public evidence items\n[OBSERVATION]\n{answer.strip()}\n"
        if task["split"] == "optimization":
            feedback = self._judge(task, answer, citations)
            (run_dir / "judge-feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8")
            safe_feedback = {k: feedback[k] for k in ("scientific_correctness", "explanatory_quality", "feynman_behavior", "modern_grounding", "robustness", "hard_penalties", "issues", "suggested_changes")}
            trace += "\n[DEVELOPMENT JUDGE FEEDBACK]\n" + json.dumps(safe_feedback, ensure_ascii=False, indent=2)
        (run_dir / "trajectory.md").write_text(trace, encoding="utf-8")
        return run_dir

    def delivered(self, run_dir: Path) -> bool:
        path = run_dir / "answer.md"
        return path.is_file() and bool(path.read_text(encoding="utf-8", errors="replace").strip())

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        return (run_dir / "trajectory.md").read_text(encoding="utf-8")[-obs_trunc:]

    def deliverable_text(self, run_dir: Path) -> str:
        return (run_dir / "answer.md").read_text(encoding="utf-8")

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        return {"answer.md": self.deliverable_text(run_dir), "citations.json": (run_dir / "citations.json").read_text(encoding="utf-8")}

    def score(self, run_dir: Path, task_id: str) -> dict[str, Any] | None:
        feedback = run_dir / "judge-feedback.json"
        if feedback.is_file():
            return json.loads(feedback.read_text(encoding="utf-8"))
        citations = json.loads((run_dir / "citations.json").read_text(encoding="utf-8"))
        return self._judge(self.tasks_by_id[task_id], self.deliverable_text(run_dir), citations)
