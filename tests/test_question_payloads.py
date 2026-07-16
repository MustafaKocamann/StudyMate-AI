from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.question_payloads import (
    GeneratedFillBlankPayload,
    GeneratedMCQPayload,
    GeneratedQuestionPayload,
)
from src.models.question_schemas import ANSWER_MAX_LENGTH, QUESTION_MAX_LENGTH, QuestionType


def mcq_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "mcq",
        "position": 1,
        "question": "Which language is commonly used for data science?",
        "difficulty": "easy",
        "explanation": "Python is correct because its ecosystem includes common data tools.",
        "options": [
            {"id": "A", "text": "Python"},
            {"id": "B", "text": "HTML"},
            {"id": "C", "text": "CSS"},
            {"id": "D", "text": "SQL"},
        ],
        "correct_option_id": "A",
    }
    payload.update(overrides)
    return payload


def fill_blank_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "fill_blank",
        "position": 1,
        "question": "Python'da __init__ metodunun görevi ___.",
        "difficulty": "medium",
        "explanation": "Başlatıcı metot, nesne oluşturulurken ilk durumun kurulmasını sağlar.",
        "answer": "nesneyi başlatmak",
    }
    payload.update(overrides)
    return payload


def test_valid_mcq_payload_has_no_uuid() -> None:
    payload = GeneratedMCQPayload.model_validate(mcq_payload())

    assert payload.type is QuestionType.MCQ
    assert payload.correct_option_id == "A"
    assert [option.id for option in payload.options] == ["A", "B", "C", "D"]
    assert "id" not in payload.model_dump()


def test_valid_fill_blank_payload_has_no_uuid_and_allows_technical_identifiers() -> None:
    payload = GeneratedFillBlankPayload.model_validate(fill_blank_payload())

    assert payload.type is QuestionType.FILL_BLANK
    assert payload.question == "Python'da __init__ metodunun görevi ___."
    assert "id" not in payload.model_dump()


@pytest.mark.parametrize(
    "extra_field",
    ["id", "created_at", "generation_id", "prompt_version", "provider", "model_name", "metadata"],
)
def test_payloads_reject_identity_and_metadata_fields(extra_field: str) -> None:
    with pytest.raises(ValidationError):
        GeneratedMCQPayload.model_validate(mcq_payload(**{extra_field: str(uuid4())}))


@pytest.mark.parametrize(
    "options",
    [
        [
            {"id": "A", "text": "Python"},
            {"id": "B", "text": "HTML"},
            {"id": "C", "text": "CSS"},
        ],
        [
            {"id": "A", "text": "Python"},
            {"id": "B", "text": "HTML"},
            {"id": "C", "text": "CSS"},
            {"id": "D", "text": "SQL"},
            {"id": "A", "text": "Java"},
        ],
        [
            {"id": "A", "text": "Python"},
            {"id": "A", "text": "HTML"},
            {"id": "C", "text": "CSS"},
            {"id": "D", "text": "SQL"},
        ],
        [
            {"id": "A", "text": "Python"},
            {"id": "B", "text": "python"},
            {"id": "C", "text": "CSS"},
            {"id": "D", "text": "SQL"},
        ],
    ],
)
def test_invalid_mcq_payload_options_fail(options: list[dict[str, str]]) -> None:
    with pytest.raises(ValidationError):
        GeneratedMCQPayload.model_validate(mcq_payload(options=options))


def test_invalid_mcq_correct_option_reference_fails() -> None:
    options = [
        {"id": "A", "text": "Python"},
        {"id": "B", "text": "HTML"},
        {"id": "C", "text": "CSS"},
        {"id": "C", "text": "SQL"},
    ]

    with pytest.raises(ValidationError):
        GeneratedMCQPayload.model_validate(mcq_payload(options=options, correct_option_id="D"))


@pytest.mark.parametrize(
    "question",
    [
        "No placeholder here.",
        "___ and ___",
        "The answer is __.",
        "The answer is ____.",
        "The answer is _____.",
        "The answer is value___name.",
    ],
)
def test_invalid_fill_blank_payload_markers_fail(question: str) -> None:
    with pytest.raises(ValidationError):
        GeneratedFillBlankPayload.model_validate(fill_blank_payload(question=question))


@pytest.mark.parametrize(
    "payload",
    [
        mcq_payload(question="Q" * (QUESTION_MAX_LENGTH + 1)),
        mcq_payload(type="essay"),
        mcq_payload(topic="Python"),
        mcq_payload(question=["What is Python?"]),
    ],
)
def test_invalid_payload_structure_fails(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GeneratedMCQPayload.model_validate(payload)


def test_fill_blank_payload_rejects_excessive_answer_length() -> None:
    with pytest.raises(ValidationError):
        GeneratedFillBlankPayload.model_validate(
            fill_blank_payload(answer="A" * (ANSWER_MAX_LENGTH + 1))
        )


def test_fill_blank_assignment_validation_rejects_invalid_answer() -> None:
    payload = GeneratedFillBlankPayload.model_validate(fill_blank_payload())

    with pytest.raises(ValidationError):
        payload.answer = " "


def test_discriminated_payload_union_parses_supported_types() -> None:
    adapter = TypeAdapter(GeneratedQuestionPayload)

    mcq = adapter.validate_python(mcq_payload())
    fill_blank = adapter.validate_python(fill_blank_payload())

    assert isinstance(mcq, GeneratedMCQPayload)
    assert isinstance(fill_blank, GeneratedFillBlankPayload)


def test_discriminated_payload_union_rejects_unsupported_type() -> None:
    adapter = TypeAdapter(GeneratedQuestionPayload)

    with pytest.raises(ValidationError):
        adapter.validate_python(mcq_payload(type="essay"))
