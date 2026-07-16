"""Small deterministic helpers for Streamlit-facing UI code."""

from __future__ import annotations

from collections.abc import Hashable
from uuid import UUID

import streamlit as st

from src.common.exceptions import (
    ApplicationConfigurationError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMInvalidRequestError,
    LLMModelUnavailableError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseParsingError,
    LLMTimeoutError,
    QuestionEvaluationError,
    QuestionGenerationError,
    QuestionRegenerationExhaustedError,
    QuestionSetGenerationError,
    StudyBuddyException,
)
from src.models.question_schemas import DifficultyLevel, QuestionType


_QUESTION_TYPE_LABELS: dict[str, dict[QuestionType, str]] = {
    "en": {
        QuestionType.MCQ: "MCQ",
        QuestionType.FILL_BLANK: "Fill blank",
    },
    "tr": {
        QuestionType.MCQ: "Çoktan seçmeli",
        QuestionType.FILL_BLANK: "Boşluk doldurma",
    },
}

_DIFFICULTY_LABELS: dict[str, dict[DifficultyLevel, str]] = {
    "en": {
        DifficultyLevel.EASY: "Easy",
        DifficultyLevel.MEDIUM: "Medium",
        DifficultyLevel.HARD: "Hard",
    },
    "tr": {
        DifficultyLevel.EASY: "Kolay",
        DifficultyLevel.MEDIUM: "Orta",
        DifficultyLevel.HARD: "Zor",
    },
}


def build_widget_key(
    component: str,
    *,
    session_id: UUID | str,
    question_id: UUID | str,
    position: int,
    suffix: Hashable | None = None,
) -> str:
    """Build a stable Streamlit widget key without learner answers or secrets."""

    normalized_component = component.strip()
    if not normalized_component:
        raise ValueError("component name must not be empty")
    parts = [
        _key_part(normalized_component),
        _key_part(session_id),
        _key_part(question_id),
        str(position),
    ]
    if suffix is not None:
        parts.append(_key_part(suffix))
    return ":".join(parts)


def request_rerun() -> None:
    """Centralize rerun requests for call sites that need an indirection."""

    st.rerun()


def format_question_type_label(question_type: QuestionType, *, language: str = "en") -> str:
    labels = _QUESTION_TYPE_LABELS.get(_language_code(language), _QUESTION_TYPE_LABELS["en"])
    return labels[question_type]


def format_difficulty_label(difficulty: DifficultyLevel, *, language: str = "en") -> str:
    labels = _DIFFICULTY_LABELS.get(_language_code(language), _DIFFICULTY_LABELS["en"])
    return labels[difficulty]


def safe_error_message(exc: Exception) -> str:
    """Map technical exceptions to short learner-facing messages."""

    if isinstance(exc, (ApplicationConfigurationError, LLMAuthenticationError)):
        return "Question generation is not configured correctly yet."
    if isinstance(exc, LLMRateLimitError):
        return "The AI service is busy right now. Please try again soon."
    if isinstance(exc, LLMTimeoutError):
        return "The AI service took too long to respond. Please try again."
    if isinstance(exc, LLMConnectionError):
        return "The AI service could not be reached. Please try again."
    if isinstance(exc, LLMModelUnavailableError):
        return "The selected AI model is currently unavailable."
    if isinstance(exc, QuestionRegenerationExhaustedError):
        return "Generated questions did not meet the quality bar. Try a narrower topic or fewer questions."
    if isinstance(exc, QuestionSetGenerationError):
        return "The question set could not be generated reliably. Please try again."
    if isinstance(
        exc,
        (
            LLMEmptyResponseError,
            LLMInvalidRequestError,
            LLMProviderError,
            LLMResponseParsingError,
            QuestionGenerationError,
            QuestionEvaluationError,
        ),
    ):
        return "The AI question service is temporarily unavailable."
    if isinstance(exc, StudyBuddyException):
        return "Something went wrong while preparing your study session."
    return "Something went wrong. Please try again."


def _language_code(language: str) -> str:
    normalized = language.strip().lower()
    if normalized.startswith("tr") or normalized in {"turkish", "türkçe", "turkce"}:
        return "tr"
    if normalized.startswith("en") or normalized == "english":
        return "en"
    return normalized


def _key_part(value: object) -> str:
    return str(value).strip().replace(":", "_")
