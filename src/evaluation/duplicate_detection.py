"""Deterministic duplicate-risk detection for generated questions."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Protocol

from src.evaluation.normalization import normalize_for_duplicate_detection, tokenize
from src.models.question_schemas import FillBlankQuestion, GeneratedQuestion, MCQQuestion


class SemanticSimilarityProvider(Protocol):
    """Optional provider-independent semantic similarity extension point."""

    async def similarity(self, left: str, right: str) -> float: ...


def duplicate_representation(question: GeneratedQuestion) -> str:
    """Build the primary duplicate key from question text and correct answer.

    Explanations are intentionally excluded because different questions may
    legitimately share similar post-answer explanations.
    """

    if isinstance(question, FillBlankQuestion):
        answer = question.answer
    else:
        answer = _mcq_correct_answer_text(question)
    return f"{question.question}\n{answer}"


def token_jaccard_similarity(first_text: str, second_text: str, *, language: str | None = None) -> float:
    """Return token-set Jaccard similarity for two text values."""

    first_tokens = tokenize(first_text, language=language)
    second_tokens = tokenize(second_text, language=language)
    if not first_tokens and not second_tokens:
        return 1.0
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def fuzzy_lexical_similarity(first_text: str, second_text: str, *, language: str | None = None) -> float:
    """Return a standard-library fuzzy lexical similarity score."""

    first_normalized = normalize_for_duplicate_detection(first_text, language=language)
    second_normalized = normalize_for_duplicate_detection(second_text, language=language)
    return SequenceMatcher(None, first_normalized, second_normalized).ratio()


def duplicate_risk_score(
    question: GeneratedQuestion,
    previous_questions: list[GeneratedQuestion],
    *,
    language: str | None = None,
) -> float:
    """Estimate duplicate risk against already accepted questions.

    Exact normalized representation matches score ``1.0``. Otherwise the score
    is the maximum of token Jaccard and fuzzy lexical similarity. The fuzzy
    threshold is intentionally kept in evaluation policy because it is an
    initial product assumption that must be calibrated using real evaluation
    data, not a universal scientific constant.
    """

    if not previous_questions:
        return 0.0

    candidate = duplicate_representation(question)
    normalized_candidate = normalize_for_duplicate_detection(candidate, language=language)
    scores: list[float] = []
    for previous_question in previous_questions:
        previous = duplicate_representation(previous_question)
        if normalize_for_duplicate_detection(previous, language=language) == normalized_candidate:
            scores.append(1.0)
        else:
            scores.append(
                max(
                    token_jaccard_similarity(candidate, previous, language=language),
                    fuzzy_lexical_similarity(candidate, previous, language=language),
                )
            )

    return max(scores)


def _mcq_correct_answer_text(question: MCQQuestion) -> str:
    return next(option.text for option in question.options if option.id == question.correct_option_id)
