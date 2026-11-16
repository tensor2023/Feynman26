# Feynman: GRPO Post-training for Scientific Reasoning Behavior

This project investigates whether post-training can teach a small language model to reproduce useful reasoning behaviors associated with Richard Feynman without impersonating the historical person or relying on superficial catchphrases.

## Research objective

The target is a Feynman-inspired scientific reasoning agent that:

- prioritizes scientific and mathematical correctness;
- builds concrete mechanisms and mental models;
- questions hidden assumptions and proposes tests;
- distinguishes historical knowledge from modern explanations;
- admits uncertainty and rejects fabricated quotations or unsupported attribution.

These capabilities are evaluated separately rather than collapsed into a single "persona similarity" score.

## Training design

The completed pilot uses an LFM3-1.7B policy with VERL, FSDP, LoRA, and task-specific GRPO. The prepared 1,500-example corpus was isolated by its authoritative split labels into:

- 1,360 training examples;
- 33 development examples;
- 107 held-out test examples that were excluded from training and validation.

The single-GPU configuration used an NVIDIA A800 80GB, a batch of 64 prompts, eight rollouts per prompt, and 32 concurrent reward workers. The run reached 98.9% observed VRAM utilization and completed 21 steps without CUDA OOM before an external judge billing failure stopped the experiment.

## Reward model

The structured LLM-as-a-Judge reward evaluates multiple dimensions, including scientific correctness, source fidelity, reasoning behavior, historical consistency, robustness, and anti-memorization checks. Judge responses are parsed into a typed schema and cached to support reproducible restart behavior.

Relevant custom components are under [`code/agents/feynman`](./code/agents/feynman). Training launchers and recovery utilities are under [`code/scripts`](./code/scripts).

## Reliability work

The project includes engineering for:

- synchronous Hugging Face rollout fallback when vLLM is incompatible with the model architecture;
- structured judge concurrency and provider-specific JSON handling;
- durable model, optimizer, scheduler, RNG, and dataloader checkpoint restoration;
- recovery of 306 LoRA tensors from FSDP actor checkpoints;
- preflight judge availability checks and a watchdog for OOM, fatal trainer errors, billing failures, and missing training sessions.

## Public-repository exclusions

Training data, evaluation transcripts, model weights, LoRA adapters, checkpoints, judge caches, credentials, and experiment logs are intentionally excluded. Paths and launch configurations in the code are examples and must be adapted to the user's environment.

## Upstream foundations

The training system builds on VERL and related open-source libraries. Retained upstream license files and notices apply to their corresponding code.

