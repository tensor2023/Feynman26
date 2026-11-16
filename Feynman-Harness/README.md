# Feynman-Harness: Constrained Autonomous Behavior Optimization

Feynman-Harness studies a bounded form of recursive self-improvement: an optimizer may revise a small behavioral harness, observe rollout and judge feedback, and select a candidate, but it cannot modify credentials, evaluation code, hidden tests, persistence, or unrelated application code.

## Optimization surface

The editable harness contains seven behaviorally relevant files:

- `systemprompt.md`
- `character-card.json`
- `retrieval-policy.md`
- `response-policy.md`
- `verifier-policy.md`
- `modern-knowledge-policy.md`
- `memory-policy.md`

Tree validation enforces required files, JSON validity, prompt budgets, non-impersonation rules, and reference-leakage restrictions.

## Evaluation design

The benchmark contains 100 independently verified tasks separated into:

- 60 optimization tasks with structured development feedback;
- 20 development tasks for candidate and policy selection;
- 20 hidden-test tasks for final generalization measurement.

Tasks cover direct explanation, follow-up dialogue, misconception diagnosis, quotation/source verification, and adversarial persona requests. Scoring is separated into scientific correctness, explanatory quality, Feynman-like reasoning behavior, modern grounding, and robustness, with explicit hard penalties.

## Pipeline

The staged pipeline performs:

1. candidate proposal over a bounded harness;
2. isolated task rollout and trajectory capture;
3. structured multi-dimensional judging;
4. development-set selection and weak-dimension targeting;
5. freezing of the selected harness;
6. independent hidden-test evaluation before promotion.

The local experiment produced hidden-test aggregate scores of 0.7838 for a minimal baseline, 0.9487 for the original harness, and 0.9018 for the auto-optimized candidate. Because the optimized candidate regressed by 4.69 percentage points relative to the original harness, the evaluation correctly rejected the claim of generalized improvement. This negative result is retained as evidence for the necessity of held-out promotion gates.

## Code map

- [`harness2_feynman/feynman_adapter.py`](./harness2_feynman/feynman_adapter.py): benchmark adapter, bounded substrate, rollout execution, and structured judge.
- [`harness2_feynman/run_full_pipeline.py`](./harness2_feynman/run_full_pipeline.py): staged optimization, development selection, freezing, hidden testing, and resumable state handling.
- [`task_generation`](./task_generation): resumable task-generation and independent-verification utilities. Generated work products are excluded.
- [`objective_and_benchmark.md`](./objective_and_benchmark.md): benchmark design, split policy, task schema, and evaluation methodology. Generated benchmark records are excluded from the public release.
- [`feynman-harness`](./feynman-harness): original bounded behavioral surface.

## Public-repository exclusions

API configuration, credentials, private data, generated task workspaces, rollout trajectories, judge outputs, selected experimental candidates, caches, and other experiment artifacts are excluded from this repository.

## Upstream foundations

The project integrates Harness² as an optimization framework. Retained upstream license and provenance files apply to vendored components.

