"""VERL synchronous-rollout adapter for the Feynman judge reward."""

from __future__ import annotations

import asyncio

from agents.feynman.judge import judge_response
from agents.feynman.overlap import detect_copy
from agents.feynman.parsing import parse_action
from agents.feynman.reward import compute_gated_reward
from agents.feynman.schema import Scenario


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, ground_truth
    scenario = Scenario.model_validate(extra_info or {})
    evidence_ids, response = parse_action(solution_str)
    sources = list(scenario.source_passages)
    if scenario.reference_card is not None:
        sources.append(scenario.reference_card.reference_text_private)
        sources.extend(scenario.reference_card.distinctive_phrases)
    sources.extend(e.content for e in scenario.evidence)
    overlap = detect_copy(response, sources)
    judged = asyncio.run(
        judge_response(scenario, response, evidence_ids, overlap.as_dict())
    )
    allowed_ids = {e.source_id for e in scenario.evidence if e.allowed_for_answer}
    unknown_ids = sorted(set(evidence_ids) - allowed_ids)
    if unknown_ids:
        judged.required_content = 0.0
        judged.unsupported_claims.append(
            f"Unknown evidence IDs: {', '.join(unknown_ids)}"
        )
    reward = compute_gated_reward(judged, overlap.copy_penalty)
    return {
        "score": reward.reward,
        "scientific_correctness": judged.scientific_correctness,
        "required_content": judged.required_content,
        "reasoning_alignment": judged.reasoning_alignment,
        "copy_penalty": overlap.copy_penalty,
        "hard_failure": float(reward.hard_failure),
    }
