"""Tests for small UI helper functions."""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.common.exceptions import (
    ApplicationConfigurationError,
    LLMAuthenticationError,
    LLMModelUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
    QuestionRegenerationExhaustedError,
    QuestionSetGenerationError,
)
from src.models.question_schemas import DifficultyLevel, QuestionType
from src.ui.helpers import (
    build_widget_key,
    format_difficulty_label,
    format_question_type_label,
    safe_error_message,
)


def test_widget_keys_are_deterministic_and_stable_across_reruns() -> None:
    session_id = uuid4()
    question_id = uuid4()

    first = build_widget_key(
        "answer-form",
        session_id=session_id,
        question_id=question_id,
        position=2,
        suffix="confidence",
    )
    second = build_widget_key(
        "answer-form",
        session_id=session_id,
        question_id=question_id,
        position=2,
        suffix="confidence",
    )

    assert first == second
    assert "confidence" in first


def test_widget_key_collision_prevention_uses_component_position_and_suffix() -> None:
    session_id = uuid4()
    question_id = uuid4()

    keys = {
        build_widget_key("unknown", session_id=session_id, question_id=question_id, position=1),
        build_widget_key("answer", session_id=session_id, question_id=question_id, position=1),
        build_widget_key("answer", session_id=session_id, question_id=question_id, position=2),
        build_widget_key("answer", session_id=session_id, question_id=question_id, position=1, suffix="x"),
    }

    assert len(keys) == 4


def test_widget_key_rejects_empty_component_names() -> None:
    with pytest.raises(ValueError, match="component"):
        build_widget_key("  ", session_id=uuid4(), question_id=uuid4(), position=1)


def test_labels_support_english_turkish_and_fallback() -> None:
    assert format_question_type_label(QuestionType.MCQ, language="English") == "MCQ"
    assert format_question_type_label(QuestionType.FILL_BLANK, language="tr") == "Boşluk doldurma"
    assert format_question_type_label(QuestionType.FILL_BLANK, language="German") == "Fill blank"
    assert format_difficulty_label(DifficultyLevel.EASY, language="en") == "Easy"
    assert format_difficulty_label(DifficultyLevel.MEDIUM, language="Türkçe") == "Orta"
    assert format_difficulty_label(DifficultyLevel.HARD, language="Spanish") == "Hard"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ApplicationConfigurationError("GROQ_API_KEY=secret"), "configured correctly"),
        (LLMAuthenticationError("bad api key sk-secret"), "configured correctly"),
        (LLMRateLimitError("raw provider rate text"), "busy right now"),
        (LLMTimeoutError("stack trace timeout"), "too long"),
        (LLMModelUnavailableError("model qwen-secret unavailable"), "model is currently unavailable"),
        (
            QuestionRegenerationExhaustedError(
                total_attempts=3,
                failed_dimensions=[],
                issue_codes=["contains-secret-prompt"],
            ),
            "quality bar",
        ),
        (
            QuestionSetGenerationError(
                failed_position=1,
                question_type="mcq",
                total_attempts=3,
                failed_dimensions=[],
                issue_codes=["raw-provider-message"],
            ),
            "question set",
        ),
        (RuntimeError("Traceback sk-secret prompt text"), "Something went wrong"),
    ],
)
def test_safe_error_messages_map_typed_errors_without_secret_leakage(
    exc: Exception,
    expected: str,
) -> None:
    message = safe_error_message(exc)

    assert expected in message
    assert "secret" not in message.lower()
    assert "traceback" not in message.lower()
    assert "prompt" not in message.lower()
    assert repr(exc) not in message
