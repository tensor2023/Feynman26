"""Deterministic anti-copy metrics; intentionally independent of an LLM judge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def _longest_common_run(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    best = 0
    for left in a:
        current = [0]
        for j, right in enumerate(b, 1):
            value = previous[j - 1] + 1 if left == right else 0
            current.append(value)
            best = max(best, value)
        previous = current
    return best


@dataclass(frozen=True)
class OverlapReport:
    copy_penalty: float
    longest_common_words: int
    max_sequence_similarity: float
    max_ngram_overlap: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "copy_penalty": self.copy_penalty,
            "longest_common_words": self.longest_common_words,
            "max_sequence_similarity": self.max_sequence_similarity,
            "max_ngram_overlap": self.max_ngram_overlap,
        }


def detect_copy(response: str, sources: list[str], ngram_size: int = 6) -> OverlapReport:
    response_words = _words(response)
    response_grams = _ngrams(response_words, ngram_size)
    max_run = 0
    max_similarity = 0.0
    max_ngram = 0.0
    for source in sources:
        source_words = _words(source)
        max_run = max(max_run, _longest_common_run(response_words, source_words))
        max_similarity = max(max_similarity, SequenceMatcher(None, response_words, source_words).ratio())
        source_grams = _ngrams(source_words, ngram_size)
        if response_grams and source_grams:
            max_ngram = max(max_ngram, len(response_grams & source_grams) / len(response_grams))

    # Short technical phrases are tolerated; distinctive 12+ word runs are not.
    run_score = max(0.0, min(1.0, (max_run - 8) / 16))
    similarity_score = max(0.0, min(1.0, (max_similarity - 0.35) / 0.55))
    ngram_score = max(0.0, min(1.0, (max_ngram - 0.10) / 0.70))
    penalty = max(run_score, similarity_score, ngram_score)
    return OverlapReport(penalty, max_run, max_similarity, max_ngram)
