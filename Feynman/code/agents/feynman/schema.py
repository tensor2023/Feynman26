"""Schemas and validation for Feynman26 scenarios and judge results."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Evidence(BaseModel):
    source_id: str
    type: str = "paraphrased_principle"
    content: str
    allowed_for_answer: bool = True
    source_date: date | None = None
    title: str | None = None
    location: str | None = None


class PersonaCard(BaseModel):
    identity: str = "anonymous_persona_x"
    documented_principles: list[str] = Field(min_length=1)
    knowledge_cutoff: date


class ReferenceCard(BaseModel):
    reference_text_private: str
    required_facts: list[str] = Field(min_length=1)
    reasoning_moves: list[str] = Field(min_length=1)
    communication_traits: list[str] = Field(default_factory=list)
    allowed_variation: list[str] = Field(default_factory=list)
    fatal_errors: list[str] = Field(default_factory=list)
    distinctive_phrases: list[str] = Field(default_factory=list)


class SourceMetadata(BaseModel):
    episode_id: str
    source_id: str
    title: str
    source_year: int
    page: int
    local_path: str
    source_sha256: str


class GenerationMetadata(BaseModel):
    generator: str
    model: str
    prompt_version: str
    output_hash: str


class VerificationMetadata(BaseModel):
    verifier: str
    model: str
    prompt_version: str
    source_support: float = Field(ge=0, le=1)
    prompt_independence: float = Field(ge=0, le=1)
    episode_completeness: float = Field(ge=0, le=1)
    copying_risk: float = Field(ge=0, le=1)
    issues: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    task_id: str
    parent_episode_id: str | None = None
    template_family: Literal["direct_application", "behavioral_transfer", "misconception_diagnosis"] | None = None
    mode: Literal["archive", "historical_simulation", "modern_feynman_inspired"]
    date: date
    audience: str
    scenario: str
    user_message: str
    persona_card: PersonaCard
    evidence: list[Evidence] = Field(default_factory=list)
    forbidden_shortcuts: list[str] = Field(default_factory=list)
    evaluation_tags: list[str] = Field(default_factory=list)
    reference_facts: list[str] = Field(default_factory=list)
    source_passages: list[str] = Field(default_factory=list)
    reference_card: ReferenceCard | None = None
    source_metadata: SourceMetadata | None = None
    generation_metadata: GenerationMetadata | None = None
    verification: VerificationMetadata | None = None
    split: Literal["train", "development", "test"] = "train"
    dataset_version: str | None = None

    @field_validator("persona_card")
    @classmethod
    def cutoff_must_match_task_date(cls, card: PersonaCard, info):
        task_date = info.data.get("date")
        if task_date and card.knowledge_cutoff > task_date:
            raise ValueError("persona knowledge_cutoff cannot be after scenario date")
        return card


class JudgeResult(BaseModel):
    scientific_correctness: float = Field(ge=0, le=1)
    required_content: float = Field(ge=0, le=1)
    reasoning_alignment: float = Field(ge=0, le=1)
    explanation_quality: float = Field(ge=0, le=1)
    epistemic_honesty: float = Field(ge=0, le=1)
    behavioral_style: float = Field(ge=0, le=1)
    temporal_honesty: float = Field(ge=0, le=1)
    score_evidence: dict[str, str] = Field(default_factory=dict)
    major_errors: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    fatal_error: bool = False
    fake_quotation: bool = False
    historical_boundary_violation: bool = False
    central_scientific_error: bool = False
    judge_manipulation: float = Field(default=0, ge=0, le=1)
    feedback: str
