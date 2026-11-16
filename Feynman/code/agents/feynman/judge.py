"""Independent structured judge orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from agents.utils import call_openai_parse, get_judge_model, get_judge_reasoning

from .prompts import build_judge_prompt
from .schema import JudgeResult, Scenario

JUDGE_PROMPT_VERSION = "feynman-grpo-judge-v1"


def _cache_path(prompt: str, model: str, reasoning_effort: str) -> Path:
    payload = json.dumps(
        {
            "prompt_version": JUDGE_PROMPT_VERSION,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt": prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    root = Path(os.getenv("FEYNMAN_JUDGE_CACHE", "outputs/feynman_judge_cache"))
    return root / f"{key}.json"


async def judge_response(scenario: Scenario, response: str, evidence_ids: list[str], overlap: dict) -> JudgeResult:
    model = os.getenv("FEYNMAN_JUDGE_MODEL")
    if not model:
        raise RuntimeError("FEYNMAN_JUDGE_MODEL must name a fixed judge model independent of the policy")
    reasoning_effort = get_judge_reasoning(os.getenv("FEYNMAN_JUDGE_REASONING", "low"))
    prompt = build_judge_prompt(scenario, response, evidence_ids, overlap)
    cache_path = _cache_path(prompt, model, reasoning_effort)
    if cache_path.is_file():
        cached = await asyncio.to_thread(cache_path.read_text, encoding="utf-8")
        return JudgeResult.model_validate_json(cached)

    parsed = await call_openai_parse(
        prompt,
        JudgeResult,
        model=get_judge_model(model),
        reasoning_effort=reasoning_effort,
    )
    if parsed is None:
        # Judge outages must never accidentally create a positive training signal.
        result = JudgeResult(
            scientific_correctness=0,
            required_content=0,
            reasoning_alignment=0,
            explanation_quality=0,
            epistemic_honesty=0,
            behavioral_style=0,
            temporal_honesty=0,
            fatal_error=True,
            feedback="The judge failed. Re-check every scientific claim and use only permitted evidence.",
        )
    else:
        result = JudgeResult.model_validate(parsed)
    await asyncio.to_thread(cache_path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(cache_path.write_text, result.model_dump_json(indent=2), encoding="utf-8")
    return result
