"""OdysSim rollout loop for the Feynman26 task."""

from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path

from agents.utils import Agent, process_post_chat

from .judge import JUDGE_PROMPT_VERSION, judge_response
from .overlap import detect_copy
from .parsing import parse_action
from .prompts import build_policy_chat
from .reward import compute_gated_reward
from .schema import Scenario


def _write_evaluation_record(scenario, response, judged, reward, overlap, context):
    """Write a policy-safe, human-readable validation record (no private reference)."""
    experiment = context.config.trainer.experiment_name
    root = Path(context.config.trainer.default_local_dir) / "evaluation" / f"step_{context.global_step:06d}"
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": scenario.task_id,
        "step": context.global_step,
        "mode": scenario.mode,
        "date": scenario.date.isoformat(),
        "audience": scenario.audience,
        "scenario": scenario.scenario,
        "user_message": scenario.user_message,
        "documented_principles": scenario.persona_card.documented_principles,
        "response": response,
        "reward": reward.reward,
        "judge": judged.model_dump(mode="json"),
        "overlap": overlap.as_dict(),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "experiment": experiment,
    }
    filename = f"{scenario.task_id}_{os.getpid()}_{uuid.uuid4().hex}.json"
    (root / filename).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


async def _rollout(scenario: Scenario, context, feedback: str | None = None):
    actor = Agent(context.llm_client, build_policy_chat(scenario, feedback), context.tokenizer, context.config, prompt_turn=2)
    raw = await actor.step()
    evidence_ids, response = parse_action(raw)
    allowed_ids = {e.source_id for e in scenario.evidence if e.allowed_for_answer}
    unknown_ids = sorted(set(evidence_ids) - allowed_ids)
    private_card = scenario.reference_card
    sources = list(scenario.source_passages)
    if private_card is not None:
        sources.append(private_card.reference_text_private)
        sources.extend(private_card.distinctive_phrases)
    # Evidence content may also originate from the source corpus. Include it in
    # copying checks even when it was policy-visible; visibility is not a license
    # for long reproduction.
    sources.extend(e.content for e in scenario.evidence)
    overlap = detect_copy(response, sources)
    judged = await judge_response(scenario, response, evidence_ids, overlap.as_dict())
    if unknown_ids:
        judged.required_content = 0.0
        judged.unsupported_claims.append(f"Unknown evidence IDs: {', '.join(unknown_ids)}")
    reward = compute_gated_reward(judged, overlap.copy_penalty)
    metrics = {
        "feynman/reward": reward.reward,
        "feynman/scientific_correctness": judged.scientific_correctness,
        "feynman/required_content": judged.required_content,
        "feynman/reasoning_alignment": judged.reasoning_alignment,
        "feynman/explanation_quality": judged.explanation_quality,
        "feynman/epistemic_honesty": judged.epistemic_honesty,
        "feynman/temporal_honesty": judged.temporal_honesty,
        "feynman/behavioral_style": judged.behavioral_style,
        "feynman/copy_penalty": overlap.copy_penalty,
        "feynman/hard_failure": int(reward.hard_failure),
        "all/score": reward.reward,
        "all/score_v1": reward.reward,
    }
    output = await actor.get_agent_output(reward.reward, extra_info=metrics)
    return actor, output, judged, reward, response, overlap


async def agent_loop(data: dict, context):
    scenario = Scenario.model_validate(data["extra_info"])
    actor, output, judged, reward, response, overlap = await _rollout(scenario, context)

    if not context.is_train:
        _write_evaluation_record(scenario, response, judged, reward, overlap, context)

    use_feedback = getattr(context.config.algorithm, "agent_version", None) == "copy" and context.is_train
    if use_feedback and judged.feedback:
        refined_actor, refined_output, refined_judged, refined_reward, _, _ = await _rollout(
            scenario, context, judged.feedback
        )
        # OdysSim's copy recipe expects: original, refined conditioned on the original
        # prompt, and refined conditioned on prompt+feedback.
        copied = copy.deepcopy(refined_output)
        copied.prompt_ids = copy.deepcopy(output.prompt_ids)
        copied.extra_fields["gen_uid"] = str(uuid.uuid4())
        copied.extra_fields["agent_role"] = "refined_without_feedback"
        refined_output.extra_fields["agent_role"] = "feedback_agent"
        output = [output, copied, refined_output]
        extra = {
            "feedback": judged.feedback,
            "original_reward": reward.reward,
            "refined_reward": refined_reward.reward,
            "refined_judge": refined_judged.model_dump(),
        }
    else:
        extra = {
            "judge": judged.model_dump(),
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "hard_failure_reasons": reward.reasons,
        }

    await process_post_chat(data, context, actor.chat, output, extra=extra)
    return output
