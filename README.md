# Feynman26

Feynman26 is a research repository for studying whether language models can acquire a scientist's reasoning behavior rather than merely imitate surface-level style.

The repository contains two related projects:

- [`Feynman`](./Feynman): GRPO post-training, structured LLM-as-a-Judge rewards, checkpoint recovery, and held-out evaluation for a Feynman-inspired scientific reasoning agent.
- [`Feynman-Harness`](./Feynman-Harness): constrained autonomous optimization of prompts and behavioral policies through a propose-rollout-judge-select-freeze loop.

## Research questions

1. Can post-training transfer first-principles explanation, assumption checking, experimental reasoning, and calibrated uncertainty without reducing the task to stylistic imitation?
2. Can an agent improve its own bounded behavioral harness while remaining auditable and resistant to benchmark leakage?
3. How should optimization, development, and hidden-test evidence be separated so that regressions do not get promoted?

## Repository policy

This public repository intentionally excludes datasets, private research notes, API configuration, model weights, checkpoints, experiment outputs, and runtime logs. Reproducing the reported experiments therefore requires users to provide their own legally obtained data, model access, and evaluation credentials.

## Projects

| Project | Focus | Main components |
| --- | --- | --- |
| Feynman | Reinforcement learning and behavioral distillation | VERL, GRPO, FSDP, LoRA, structured judge rewards |
| Feynman-Harness | Constrained recursive improvement research | Harness² adapter, bounded policy surface, staged evaluation, hidden-test promotion |

## Safety and evaluation

The work treats scientific correctness, explanatory quality, Feynman-like reasoning behavior, modern-knowledge grounding, and robustness as separate evaluation dimensions. A candidate is not considered improved solely because it performs better on optimization tasks.

## License

Some code under the project directories is adapted from or integrates with upstream open-source projects. Their original license and attribution files remain authoritative. No model weights or third-party datasets are distributed here.

