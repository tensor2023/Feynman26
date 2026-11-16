"""Gated Feynman26 reward from the design document."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import JudgeResult


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    base_reward: float
    quality: float
    hard_failure: bool
    reasons: tuple[str, ...]


def compute_gated_reward(judge: JudgeResult, copy_penalty: float) -> RewardBreakdown:
    reasons: list[str] = []
    if judge.fatal_error or judge.central_scientific_error:
        reasons.append("central scientific error")
    if judge.fake_quotation:
        reasons.append("fabricated exact quotation")
    if judge.historical_boundary_violation:
        reasons.append("historical boundary violation")
    if copy_penalty >= 0.85:
        reasons.append("long source copying")
    if judge.judge_manipulation >= 0.85:
        reasons.append("judge manipulation")
    if reasons:
        return RewardBreakdown(0.0, 0.0, 0.0, True, tuple(reasons))

    honesty = (judge.epistemic_honesty + judge.temporal_honesty) / 2
    quality = (
        0.25 * judge.required_content
        + 0.25 * judge.reasoning_alignment
        + 0.20 * judge.explanation_quality
        + 0.20 * honesty
        + 0.10 * judge.behavioral_style
    )
    base = judge.scientific_correctness * (0.35 + 0.65 * quality)
    reward = base - 0.35 * copy_penalty - 0.25 * judge.judge_manipulation
    return RewardBreakdown(max(0.0, min(1.0, reward)), base, quality, False, ())
