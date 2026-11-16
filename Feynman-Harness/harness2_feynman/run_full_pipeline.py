from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from harness2.core import library as library_store
from harness2.core.schedule import run_task

from feynman_adapter import DeepSeekProposer, FeynmanAdapter, HARNESS_FILES, ROOT


PIPELINE = ROOT / "full_pipeline"
STATE_PATH = PIPELINE / "state.json"
FAMILIES = ("direct_explanation", "follow_up_dialogue", "misconception", "quotation_source", "adversarial_persona")
PHASE_SPECS = {
    "phase_a": {"k": 1, "seed": 101, "tag": "phase-a", "quotas": (1, 1, 1, 1, 1)},
    "phase_b": {"k": 2, "seed": 102, "tag": "phase-b", "quotas": (3, 2, 2, 2, 1)},
    "phase_c": {"k": 3, "seed": 103, "tag": "phase-c", "quotas": (8, 4, 3, 3, 2)},
}
DIMS = ("scientific_correctness", "explanatory_quality", "feynman_behavior", "modern_grounding", "robustness")


def load_state() -> dict[str, Any]:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"completed": [], "artifacts": {}}


def save_state(state: dict[str, Any]) -> None:
    PIPELINE.mkdir(parents=True, exist_ok=True)
    temp = STATE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE_PATH)


def task_plan(adapter: FeynmanAdapter) -> dict[str, list[str]]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for task_id in adapter.tasks("optimization"):
        by_family[adapter.domain_of(task_id)].append(task_id)
    offsets = {family: 0 for family in FAMILIES}
    plan: dict[str, list[str]] = {}
    for phase in ("phase_a", "phase_b", "phase_c"):
        selected: list[str] = []
        for family, count in zip(FAMILIES, PHASE_SPECS[phase]["quotas"]):
            start = offsets[family]
            selected.extend(by_family[family][start : start + count])
            offsets[family] += count
        plan[phase] = selected
    used = {task for tasks in plan.values() for task in tasks}
    plan["remaining"] = [task for task in adapter.tasks("optimization") if task not in used]
    return plan


def summary_path(adapter: FeynmanAdapter, task_id: str, k: int, seed: int, tag: str) -> Path:
    return adapter.spec.runs_root / f"feynman-feynman-api-agg{k}-s{seed}-{tag}" / task_id / "summary.json"


def selected_harness(adapter: FeynmanAdapter, task_id: str, k: int, seed: int, tag: str) -> tuple[Path, Path]:
    summary = json.loads(summary_path(adapter, task_id, k, seed, tag).read_text(encoding="utf-8"))
    selection = summary["selections"]["seq"]
    if not selection.get("ok") or not selection.get("chosen_run_id") or not selection.get("chosen_run_dir"):
        raise ValueError(f"{task_id} has no usable selected artifact")
    run_id = Path(selection["chosen_run_id"])
    harness = adapter.spec.runs_root / run_id.parent / "harnesses" / selection["chosen_label"]
    run_dir = Path(selection["chosen_run_dir"])
    return harness, run_dir


def run_optimization_phase(name: str, adapter: FeynmanAdapter, tasks: list[str], state: dict[str, Any], *, k: int, seed: int, tag: str, allow_failures: int = 0) -> None:
    proposer = DeepSeekProposer(adapter.spec.channels["proposer"], audit_dir=PIPELINE / "proposer-audit" / name)
    libraries = {
        family: library_store.Library(root=PIPELINE / "libraries" / name / family, domain=family, surface=adapter.substrate.surface())
        for family in FAMILIES
    }
    results = []
    for index, task_id in enumerate(tasks, 1):
        print(f"[{name}] {index}/{len(tasks)} {task_id}", flush=True)
        result = run_task(
            task_id=task_id,
            k=k,
            adapter=adapter,
            prop=proposer,
            selector_prop=proposer,
            runs_root=adapter.spec.runs_root,
            modes=("seq",),
            seed=seed,
            tag=tag,
            library=libraries[adapter.domain_of(task_id)],
        )
        results.append({"task_id": task_id, "status": result.status, "errors": result.errors})
        state["artifacts"][name] = results
        save_state(state)
    failures = [row for row in results if row["status"] not in ("ok", "skipped")]
    if len(failures) > allow_failures:
        raise RuntimeError(f"{name} has failed tasks: {failures}")
    if failures:
        state["artifacts"][f"{name}_accepted_failures"] = failures
    state["completed"].append(name)
    save_state(state)


def judge_for_run(adapter: FeynmanAdapter, task_id: str, run_dir: Path) -> dict[str, Any]:
    path = run_dir / "judge-feedback.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    score = adapter.score(run_dir, task_id)
    path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    return score


def collect_candidates(adapter: FeynmanAdapter, tasks: list[str], *, k: int, seed: int, tag: str, phase: str) -> list[dict[str, Any]]:
    candidates = []
    destination = PIPELINE / "candidates" / phase
    destination.mkdir(parents=True, exist_ok=True)
    for task_id in tasks:
        try:
            harness, run_dir = selected_harness(adapter, task_id, k, seed, tag)
        except (KeyError, OSError, TypeError, ValueError):
            continue
        score = judge_for_run(adapter, task_id, run_dir)
        fingerprint = json.dumps({name: (harness / name).read_text(encoding="utf-8") for name in HARNESS_FILES}, sort_keys=True)
        target = destination / task_id
        if not target.exists():
            shutil.copytree(harness, target)
        candidates.append({"task_id": task_id, "harness": str(target), "optimization_score": score["raw_score"], "fingerprint": hash(fingerprint)})
    candidates.sort(key=lambda row: row["optimization_score"], reverse=True)
    unique = []
    seen = set()
    for row in candidates:
        if row["fingerprint"] not in seen:
            seen.add(row["fingerprint"])
            unique.append(row)
    return unique


def evaluate_harness(adapter: FeynmanAdapter, harness: Path, task_ids: list[str], label: str, state: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for index, task_id in enumerate(task_ids, 1):
        print(f"[evaluate:{label}] {index}/{len(task_ids)} {task_id}", flush=True)
        run_id = f"full-evaluation/{label}/{task_id}"
        last_error = None
        for attempt in range(3):
            try:
                run_dir = adapter.execute(task_id, harness, run_id)
                last_error = None
                break
            except RuntimeError as exc:
                last_error = exc
                print(f"  retry {attempt + 1}/3 after execution error: {exc}", flush=True)
        if last_error is not None:
            score = {dim: 0.0 for dim in DIMS}
            score.update({"raw_score": 0.0, "hard_penalties": 1, "issues": [str(last_error)], "execution_failed": True})
            rows.append({"task_id": task_id, "score": score})
            state["artifacts"].setdefault("evaluations", {})[label] = rows
            save_state(state)
            continue
        score_path = run_dir / "final-score.json"
        if score_path.is_file():
            score = json.loads(score_path.read_text(encoding="utf-8"))
        else:
            score = adapter.score(run_dir, task_id)
            score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"task_id": task_id, "score": score})
        state["artifacts"].setdefault("evaluations", {})[label] = rows
        save_state(state)
    means = {dim: statistics.fmean(row["score"][dim] for row in rows) for dim in DIMS}
    result = {
        "label": label,
        "harness": str(harness),
        "n": len(rows),
        "aggregate": statistics.fmean(row["score"]["raw_score"] for row in rows),
        "dimensions": means,
        "worst_dimension": min(means.values()),
        "failures": sum(row["score"].get("hard_penalties", 0) > 0 for row in rows),
        "rows": rows,
    }
    return result


def phase_b_review(adapter: FeynmanAdapter, tasks: list[str], state: dict[str, Any]) -> None:
    reports = []
    for task_id in tasks:
        summary = json.loads(summary_path(adapter, task_id, 2, 102, "phase-b").read_text(encoding="utf-8"))
        selection = summary["selections"]["seq"]
        harness, run_dir = selected_harness(adapter, task_id, 2, 102, "phase-b")
        score = judge_for_run(adapter, task_id, run_dir)
        reports.append({"task_id": task_id, "selected": selection["chosen_label"], "score": score["raw_score"], "issues": score["issues"], "hard_penalties": score["hard_penalties"], "harness": str(harness)})
    (PIPELINE / "phase_b_review.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    state["artifacts"]["phase_b_review"] = str(PIPELINE / "phase_b_review.json")
    state["completed"].append("phase_b_review")
    save_state(state)


def choose_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(results, key=lambda row: (row["aggregate"], row["worst_dimension"], -row["failures"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-after", choices=["phase_a", "phase_b", "phase_c", "phase_d", "phase_e", "phase_f"])
    args = parser.parse_args()
    state = load_state()
    initial = FeynmanAdapter()
    plan = task_plan(initial)
    (PIPELINE / "task_plan.json").parent.mkdir(parents=True, exist_ok=True)
    (PIPELINE / "task_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    for phase in ("phase_a", "phase_b", "phase_c"):
        if phase not in state["completed"]:
            spec = PHASE_SPECS[phase]
            run_optimization_phase(phase, initial, plan[phase], state, k=spec["k"], seed=spec["seed"], tag=spec["tag"])
        if phase == "phase_b" and "phase_b_review" not in state["completed"]:
            phase_b_review(initial, plan["phase_b"], state)
        if args.stop_after == phase:
            return

    if "phase_d" not in state["completed"]:
        candidates = collect_candidates(initial, plan["phase_c"], k=3, seed=103, tag="phase-c", phase="phase_c")[:3]
        results = [evaluate_harness(initial, Path(row["harness"]), initial.tasks("development"), f"phase-d-{row['task_id']}", state) for row in candidates]
        winner = choose_best(results)
        target = PIPELINE / "selected" / "phase_d"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(winner["harness"], target)
        state["artifacts"]["phase_d"] = {"candidates": results, "winner": winner, "selected_harness": str(target)}
        state["completed"].append("phase_d")
        save_state(state)
    if args.stop_after == "phase_d":
        return

    phase_d_harness = Path(state["artifacts"]["phase_d"]["selected_harness"])
    if "phase_e" not in state["completed"]:
        weakest = min(state["artifacts"]["phase_d"]["winner"]["dimensions"], key=state["artifacts"]["phase_d"]["winner"]["dimensions"].get)
        targeted = [task for task in plan["remaining"] if weakest in initial.tasks_by_id[task]["capabilities"]]
        for task in plan["remaining"]:
            if task not in targeted and len(targeted) < 15:
                targeted.append(task)
        targeted = targeted[:15]
        phase_e_adapter = FeynmanAdapter(base_harness_dir=phase_d_harness)
        if "phase_e_runs" not in state["completed"]:
            run_optimization_phase("phase_e_runs", phase_e_adapter, targeted, state, k=2, seed=105, tag="phase-e", allow_failures=1)
        candidates = collect_candidates(phase_e_adapter, targeted, k=2, seed=105, tag="phase-e", phase="phase_e")[:3]
        candidates.append({"task_id": "phase-d-winner", "harness": str(phase_d_harness), "optimization_score": state["artifacts"]["phase_d"]["winner"]["aggregate"]})
        results = [evaluate_harness(initial, Path(row["harness"]), initial.tasks("development"), f"phase-e-{row['task_id']}", state) for row in candidates]
        winner = choose_best(results)
        frozen = PIPELINE / "selected" / "frozen_final"
        if frozen.exists():
            shutil.rmtree(frozen)
        shutil.copytree(winner["harness"], frozen)
        state["artifacts"]["phase_e"] = {"weakest_dimension": weakest, "targeted_tasks": targeted, "candidates": results, "winner": winner, "frozen_harness": str(frozen)}
        state["completed"].append("phase_e")
        save_state(state)
    if args.stop_after == "phase_e":
        return

    if "phase_f" not in state["completed"]:
        frozen = Path(state["artifacts"]["phase_e"]["frozen_harness"])
        hidden = initial.tasks("hidden_test")
        baselines = {
            "minimal": ROOT / "baselines" / "minimal",
            "original": ROOT / "feynman-harness",
            "optimized": frozen,
        }
        comparisons = [evaluate_harness(initial, harness, hidden, f"hidden-{name}", state) for name, harness in baselines.items()]
        robustness_subset = [task for task in hidden if "robustness" in initial.tasks_by_id[task]["capabilities"]][:5]
        repeat = evaluate_harness(initial, frozen, robustness_subset, "hidden-optimized-repeat", state)
        report = {"baselines": comparisons, "robustness_repeat": repeat, "frozen_harness": str(frozen)}
        (PIPELINE / "final_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        state["artifacts"]["phase_f"] = report
        state["completed"].append("phase_f")
        save_state(state)


if __name__ == "__main__":
    main()
