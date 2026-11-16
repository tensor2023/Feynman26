# Designing a Harness² Pipeline for a Feynman-Inspired DSH Character

## 1. Objective

The goal is to improve a DSH/NextTavern character so that it behaves like a careful, source-grounded educational simulation inspired by Richard Feynman.

The system must not merely reproduce recognizable phrases. It must demonstrate five separate capabilities:

1. **Scientific correctness:** The response is factually and mathematically sound.
2. **Explanatory quality:** It builds an understandable mechanism or mental model.
3. **Feynman-like behavior:** It reasons concretely, questions assumptions, tests ideas, and admits uncertainty without becoming a caricature.
4. **Modern grounding:** It can explain developments after Feynman's death without pretending that the historical Feynman knew them.
5. **Robustness:** It resists false premises, invented citations, empty stylistic imitation, and attempts to game the evaluator.

The product should identify itself as a **Feynman-inspired simulation**, not as the historical person.

## 2. System roles

The pipeline has four distinct components:

```text
100 benchmark tasks
        |
        v
Harness² proposer/composer ----> editable Feynman harness
        |                                  |
        |                                  v
        |                         DSH + NextTavern runtime
        |                                  |
        |                                  v
        +<---- outputs, trajectories, and development feedback
                                           |
                                           v
                                independent final evaluation
```

### NextTavern

NextTavern remains the roleplay runtime. It owns:

- Character-card presentation and author rules
- Long conversations and worldlines
- Frozen persona settings
- Character and private memory
- Source retrieval and worldbooks
- Model routing and character-agent execution
- The user-facing roleplay interface

### Harness²

Harness² is an offline optimization system. It does not replace NextTavern and is not the production chat interface. It repeatedly edits a bounded set of prompts and policies, runs the same tasks through DSH, observes the consequences, and proposes improved versions.

### Development judge

The development judge supplies structured feedback for optimization tasks. This is necessary because subtle explanatory and persona fidelity cannot reliably be inferred from an answer alone.

### Final judge

The final judge evaluates untouched tasks without returning feedback to Harness². This is the credible measurement of generalization.

## 3. Execution environment

Run Harness² in **WSL2/Linux**, not native Windows. Its implementation uses POSIX functionality such as `fcntl`.

Prefer keeping active Linux repositories and run directories inside the WSL filesystem:

```text
~/agi/harness2
~/agi/dsh-nexttavern
~/agi/feynman-evaluation
~/agi/runs
```

Files may be copied back to `C:\2026\AGI\...` for editing or archival, but repeated model runs and dependency operations are generally more reliable and faster inside WSL.

## 4. The editable harness

Do not expose the entire NextTavern repository to the Harness² proposer. That would create a huge search space and risk unrelated or unsafe edits.

Create a small projection containing only behaviorally relevant files:

```text
feynman-harness/
  systemprompt.md
  character-card.json
  retrieval-policy.md
  response-policy.md
  verifier-policy.md
  modern-knowledge-policy.md
  memory-policy.md
```

The proposer may edit these files. It must not edit:

- DSH credentials
- NextTavern session history
- Benchmark tasks or references
- Evaluation code
- Hidden-test data
- NextTavern persistence or security code

`validate_tree()` should reject missing required files, invalid character-card JSON, excessive prompt size, forbidden claims of literal identity, and attempts to add benchmark answers.

## 5. The 100-task benchmark

Assume 100 completed tasks already exist. Data generation is outside this design.

Use the following split:

| Split | Tasks | Visible information | Purpose |
|---|---:|---|---|
| Optimization | 60 | Prompt, allowed sources, rubric, structured judge feedback | Generate and improve harness candidates |
| Development | 20 | Prompt and allowed sources; scores used for configuration selection | Detect overfitting and choose policies |
| Hidden test | 20 | Visible only to the final evaluator | Report final generalization |

The 100 tasks should cover all five evaluation capabilities and include three broad task families:

- Direct scientific explanation
- Misconception diagnosis
- Behavioral transfer to an unfamiliar problem

The hidden set must include adversarial cases involving false quotations, false premises, post-1988 discoveries, requests for unsupported certainty, and prompts that reward superficial catchphrases.

## 6. What one task contains

Each task needs a stable identifier and separate public and private fields:

```json
{
  "task_id": "feynman-0042",
  "split": "optimization",
  "user_message": "Why does a bouncing ball eventually stop?",
  "allowed_evidence": ["source passages available to DSH"],
  "public_rubric": [
    "Explain transfer from coordinated to microscopic motion",
    "Use a concrete causal sequence",
    "Do not claim that energy disappears"
  ],
  "private_reference": "judge-only reference material",
  "required_facts": ["bulk motion becomes microscopic motion and heat"],
  "fatal_errors": ["energy vanishes"],
  "reasoning_moves": ["track successive collisions"],
  "communication_traits": ["build a particle-level picture"]
}
```

The DSH agent may receive the user message, permitted evidence, and production persona. The development or final judge may receive private references. Hidden private material must never enter proposer context.

## 7. The custom Harness² adapter

Harness² does not provide a native Feynman benchmark. Implement a custom `BenchmarkAdapter`.

```python
class FeynmanAdapter:
    spec: BenchSpec
    substrate: SubstrateLike

    def tasks(self, split="all") -> list[str]:
        ...

    def domain_of(self, task_id: str) -> str:
        return "source-grounded-feynman-simulation"

    def task_prompt(self, task_id: str) -> str:
        return self.tasks_by_id[task_id]["user_message"]

    def task_files(self, task_id: str) -> Path | None:
        return self.materialize_public_task_files(task_id)

    def base_harness(self) -> Path:
        return self.base_harness_directory

    def validate_tree(self, candidate: dict[str, str]) -> None:
        validate_required_files(candidate)
        validate_character_card(candidate["character-card.json"])
        reject_identity_deception(candidate)
        reject_reference_leakage(candidate)
        enforce_prompt_budget(candidate)

    def execute(self, task_id, harness_dir, run_id, *, timeout=None) -> Path:
        ...

    def delivered(self, run_dir: Path) -> bool:
        return (run_dir / "answer.md").is_file()

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        ...

    def deliverable_text(self, run_dir: Path) -> str:
        return (run_dir / "answer.md").read_text(encoding="utf-8")

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        return {
            "answer.md": self.deliverable_text(run_dir),
            "citations.json": (run_dir / "citations.json").read_text(encoding="utf-8"),
        }

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        ...
```

The most important methods are `execute()`, `trajectory()`, and `score()`.

## 8. How `execute()` runs DSH

Every rollout must be isolated and reproducible:

1. Create a fresh temporary DSH profile and workspace.
2. Install or mount the fixed NextTavern runtime.
3. Apply the candidate files from `harness_dir`.
4. Load only the public task evidence.
5. Start a fresh roleplay session.
6. Send the task's user message.
7. Wait for the final response and maintenance tasks to settle.
8. Save the response, citations, retrieval trace, model usage, and execution trajectory.
9. For optimization tasks, run the development judge and save its structured feedback.
10. Return the rollout directory.

Recommended rollout layout:

```text
run-dir/
  answer.md
  citations.json
  retrieval-log.json
  conversation.json
  trajectory.md
  judge-feedback.json       # optimization tasks only
  token-usage.json
  runtime-metadata.json
```

Pin the responding model, temperature, output budget, retrieval configuration, and random seed where supported. Otherwise the harness comparison will be confounded by changing runtime conditions.

## 9. How Harness² improves without gradients

Harness² performs non-differentiable search. It does not update model weights.

With sequential mode and `k=3`:

```text
base
  -> propose inv1 and dom1
  -> execute both
  -> inspect their outputs and trajectories
  -> compose c1
  -> execute c1

c1
  -> propose inv2 and dom2
  -> execute both
  -> compose c2
  -> execute c2

c2
  -> propose inv3 and dom3
  -> execute both
  -> compose c3
  -> execute c3

[c1, c2, c3] -> selector -> selected harness
```

`INV` proposes a generally useful improvement, such as requiring uncertainty labels. `DOM` proposes a domain-specific improvement, such as checking physical analogies against conservation laws. The composer observes the consequences of both and produces the next harness.

The standard Harness² proposer does not automatically see `adapter.score()`, private references, or final judge output. It learns from the task, current harness, candidate deliverables, trajectories, earlier decisions, and any permitted experience library.

## 10. Injecting development feedback

For Feynman-like behavior, raw outputs are not sufficient. A proposer may not reliably notice subtle scientific or historical errors. Run a development judge inside `execute()` for optimization tasks.

```python
def execute(self, task_id, harness_dir, run_id, *, timeout=None):
    run_dir = run_dsh_isolated(task_id, harness_dir, run_id, timeout)

    if self.split_of(task_id) == "optimization":
        feedback = self.development_judge.evaluate(
            task=self.private_task(task_id),
            answer=(run_dir / "answer.md").read_text(encoding="utf-8"),
            citations=read_json(run_dir / "citations.json"),
        )
        write_json(run_dir / "judge-feedback.json", feedback)

    return run_dir
```

Append the feedback to the trajectory returned to Harness²:

```python
def trajectory(self, run_dir, *, obs_trunc):
    trace = (run_dir / "trajectory.md").read_text(encoding="utf-8")
    feedback_path = run_dir / "judge-feedback.json"
    if feedback_path.is_file():
        feedback = feedback_path.read_text(encoding="utf-8")
        trace += f"\n\n[DEVELOPMENT JUDGE FEEDBACK]\n{feedback}"
    return trace
```

This creates the learning signal:

```text
candidate edit
  -> DSH response
  -> structured critique
  -> critique appears in candidate trajectory
  -> composer revises the next harness
```

Use behavioral feedback, not copied reference prose:

```json
{
  "scientific_correctness": 0.92,
  "explanatory_quality": 0.64,
  "feynman_behavior": 0.58,
  "modern_grounding": 1.0,
  "robustness": 0.9,
  "issues": [
    "The response introduced terminology before constructing the mechanism.",
    "The analogy was not followed by a statement of its limitations.",
    "The answer did not propose an observable check."
  ],
  "suggested_changes": [
    "Begin with the observable event and track its causal sequence.",
    "After each analogy, state where the analogy fails.",
    "Offer a simple experiment or limiting-case check."
  ]
}
```

Do not include private reference sentences, hidden expected answers, or exact quotations solely available to the judge.

## 11. Scoring

Use a five-part score aligned with the objective:

| Dimension | Weight |
|---|---:|
| Scientific correctness | 30% |
| Explanatory quality | 20% |
| Feynman-like behavior | 20% |
| Modern grounding | 15% |
| Robustness | 15% |

Apply hard penalties independently:

- Invented quotation
- Unsupported attribution
- Claim to literally be Richard Feynman
- Treating post-1988 knowledge as Feynman's personal historical knowledge
- Contradicting a required scientific fact
- Reproducing private reference text
- Ignoring an explicit uncertainty or source limitation

Example:

```python
weighted = (
    0.30 * scientific_correctness
    + 0.20 * explanatory_quality
    + 0.20 * feynman_behavior
    + 0.15 * modern_grounding
    + 0.15 * robustness
)

final_score = max(0.0, weighted - hard_penalties)
```

`adapter.score()` should return both the aggregate and sub-scores:

```python
return {
    "raw_score": final_score,
    "scientific_correctness": scientific_correctness,
    "explanatory_quality": explanatory_quality,
    "feynman_behavior": feynman_behavior,
    "modern_grounding": modern_grounding,
    "robustness": robustness,
    "hard_penalties": hard_penalties,
}
```

In standard Harness², this score is applied after candidate selection and is primarily a final measurement. It does not automatically guide `inv`, `dom`, or `compose`; the injected development feedback supplies that role.

## 12. Judge design

Use a judge model different from the DSH response model when possible. A single model judging itself can amplify shared preferences and blind spots.

The judge should receive:

- User prompt
- Candidate answer
- Permitted citations and retrieval record
- Required facts
- Fatal errors
- Reasoning moves
- Communication traits
- Historical knowledge cutoff
- Private reference material, when authorized for that split

The judge must produce strict JSON and quote short spans from the candidate answer as evidence for its assessment. It should never reveal private reference prose in feedback.

Combine the LLM judge with deterministic checks:

- Citation identifiers exist and were retrieved
- Claimed quotations match an approved quotation ledger
- Historical dates respect the knowledge cutoff
- Required response fields are present
- The response does not contain forbidden identity claims
- Excessive phrase matching against private references is detected

Run a small portion of outputs through human review to calibrate the judge. The target is agreement with expert judgment, not merely internal consistency.

## 13. Leakage and overfitting controls

Three information boundaries are required:

```text
Optimization:
  public evidence + public rubric + structured critique

Development:
  public evidence + scores used to select configuration
  no reference answer copied into proposer context

Hidden test:
  no feedback, no proposer access, final aggregate metrics only
```

Further safeguards:

- Put private references and expected results under `leak_globs`.
- Ensure `task_prompt()` strips private grading material.
- Never place hidden-test feedback in the experience library.
- Search candidate harnesses for task identifiers and distinctive reference phrases.
- Reject edits that contain answers for individual benchmark questions.
- Rotate paraphrased and transfer tasks so success requires behavioral generalization.
- Report performance separately for the three task families and five dimensions.

## 14. Running Harness²

A single optimization task can be run as follows:

```python
result = run_task(
    task_id="feynman-0042",
    k=3,
    adapter=adapter,
    prop=proposer,
    selector_prop=selector,
    runs_root=adapter.spec.runs_root,
    modes=("seq",),
)
```

Argument meanings:

- `task_id` selects one benchmark task; it is not the task prompt itself.
- `k=3` requests three recursive refinement rounds, not three answers or top-three sampling.
- `adapter` supplies the task, editable base harness, DSH executor, artifacts, and scoring.
- `prop` is the model-backed harness editor used for invariant, domain, and composition stages.
- `selector_prop` chooses among the final composed candidates.
- `runs_root` stores rollouts, harness snapshots, contexts, and summaries.
- `modes=("seq",)` makes each composed candidate the starting harness for the next round.

For multiple tasks, use `run_domain()` so useful experience can be accumulated across optimization tasks. Each task should still begin from a controlled harness revision; experience should guide proposals rather than silently rewrite the benchmark base.

## 15. Practical optimization schedule

Do not run all 60 optimization tasks with `k=3` immediately.

### Phase A: integration smoke test

- 5 optimization tasks
- `k=1`
- DeepSeek Flash
- Verify isolation, artifact capture, judging, feedback injection, and resume behavior

### Phase B: pipeline calibration

- 10 diverse optimization tasks
- `k=2`
- Review every candidate and judge report manually
- Adjust the rubric and deterministic checks

### Phase C: first real optimization

- 20 diverse optimization tasks
- `k=3`
- Sequential mode
- Use the experience library only after verifying it contains no reference leakage

### Phase D: development selection

- Evaluate leading harnesses on all 20 development tasks
- No development feedback is injected into candidate generation
- Choose the harness using aggregate score, worst-category score, and failure count

### Phase E: broader optimization

- Use the remaining optimization tasks selectively to address demonstrated weaknesses
- Do not repeatedly optimize against the full development set

### Phase F: final evaluation

- Freeze the harness and all runtime parameters
- Run the 20 hidden-test tasks once
- Report aggregate and per-dimension results
- Compare against the original NextTavern Feynman harness under identical conditions

## 16. Baselines and acceptance criteria

Always measure at least three systems:

1. Base model with a minimal Feynman instruction
2. Original NextTavern Feynman harness
3. Harness²-optimized NextTavern Feynman harness

Suggested acceptance criteria:

- At least a meaningful improvement over original NextTavern on hidden aggregate score
- No reduction in scientific correctness
- Zero invented quotations in the hidden set
- Zero literal historical-identity claims
- Stronger or equal performance in every task family
- No abnormal phrase overlap with private references
- Stable results across at least two repeated runs on a small robustness subset

Do not accept a harness merely because its average style score rises. A candidate that sounds more recognizable but becomes less accurate, more deceptive, or more verbose has failed the goal.

## 17. Cost control

Harness² performs many calls per task: base rollouts, invariant and domain proposals, candidate executions, compositions, selection, and optional judges.

Control cost by:

- Using DeepSeek Flash for proposer and development-judge calls
- Reserving a stronger model for final DSH responses or disputed judgments
- Running during off-peak pricing periods
- Limiting optimization to diverse, diagnostic task subsets
- Reusing deterministic checks before invoking an LLM judge
- Recording API usage in every rollout
- Stopping a candidate early when validation or hard checks fail

After a five-task smoke test, estimate the full run from measured usage rather than theoretical token counts:

```text
estimated phase cost = pilot cost / pilot tasks * planned tasks
```

## 18. Final production architecture

The optimized production system should remain simpler than the training pipeline:

```text
User
  -> NextTavern/DSH
       -> frozen Feynman-inspired persona
       -> source retrieval
       -> private conversational memory
       -> physics/source verifier when needed
       -> response with clear historical and modern-knowledge boundaries
```

Harness² and private benchmark references stay offline. Only the selected harness files are promoted into production.

## 19. Central design principle

The pipeline should optimize a reproducible behavior:

> Explain accurately from concrete mechanisms, challenge weak assumptions, expose uncertainty, use documented sources responsibly, and distinguish historical Feynman from a modern Feynman-inspired simulation.

The target is not verbal impersonation. It is a source-grounded educational character whose observable reasoning and teaching behavior resemble the strongest documented aspects of Feynman's public explanations.
