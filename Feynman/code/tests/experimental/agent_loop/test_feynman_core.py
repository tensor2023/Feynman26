import pytest

from agents.feynman.overlap import detect_copy
from agents.feynman.parsing import parse_action
from agents.feynman.reward import compute_gated_reward
from agents.feynman.schema import JudgeResult


def _judge(**overrides):
    values = dict(
        scientific_correctness=1.0,
        required_content=1.0,
        reasoning_alignment=1.0,
        explanation_quality=1.0,
        epistemic_honesty=1.0,
        behavioral_style=1.0,
        temporal_honesty=1.0,
        feedback="Be concrete.",
    )
    values.update(overrides)
    return JudgeResult(**values)


def test_gated_reward_formula():
    result = compute_gated_reward(_judge(), copy_penalty=0.0)
    assert result.reward == pytest.approx(1.0)
    assert not result.hard_failure


def test_voice_cannot_rescue_bad_science():
    result = compute_gated_reward(_judge(scientific_correctness=0.0), copy_penalty=0.0)
    assert result.reward == 0.0


def test_plain_correct_answer_beats_stylish_wrong_answer():
    plain = compute_gated_reward(
        _judge(reasoning_alignment=0.6, explanation_quality=0.6, behavioral_style=0.2), copy_penalty=0.0
    )
    stylish_wrong = compute_gated_reward(
        _judge(scientific_correctness=0.2, behavioral_style=1.0), copy_penalty=0.0
    )
    assert plain.reward > stylish_wrong.reward


def test_independent_answer_beats_source_copy():
    independent = compute_gated_reward(_judge(), copy_penalty=0.0)
    copied = compute_gated_reward(_judge(), copy_penalty=1.0)
    assert independent.reward > copied.reward


def test_fake_quote_is_hard_failure():
    result = compute_gated_reward(_judge(fake_quotation=True), copy_penalty=0.0)
    assert result.reward == 0.0
    assert result.hard_failure


def test_copy_detector_distinguishes_recitation():
    source = "The experiment forces us to ask what is actually measured before assigning a classical path to the electron."
    copied = detect_copy(source, [source])
    original = detect_copy("A detector can reveal a route, but adding it changes the setup and its outcomes.", [source])
    assert copied.copy_penalty > 0.85
    assert original.copy_penalty < copied.copy_penalty


def test_action_parser():
    ids, response = parse_action('{"evidence_ids":["F004"],"response":"Test the setup."}')
    assert ids == ["F004"]
    assert response == "Test the setup."
