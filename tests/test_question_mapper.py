from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.question_mapper import question_from_payload
from src.models.question_payloads import GeneratedFillBlankPayload, GeneratedMCQPayload
from src.models.question_schemas import FillBlankQuestion, MCQQuestion, QuestionSet


def mcq_payload_dict(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "mcq",
        "position": 1,
        "question": "Which process allows plants to convert light energy into chemical energy?",
        "difficulty": "easy",
        "explanation": "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "options": [
            {"id": "A", "text": "Respiration"},
            {"id": "B", "text": "Photosynthesis"},
            {"id": "C", "text": "Transpiration"},
            {"id": "D", "text": "Fermentation"},
        ],
        "correct_option_id": "B",
    }
    payload.update(overrides)
    return payload


def fill_blank_payload_dict(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "fill_blank",
        "position": 2,
        "question": "Plants convert light energy into chemical energy through ___.",
        "difficulty": "easy",
        "explanation": "Photosynthesis is the process that stores light energy in chemical form.",
        "answer": "photosynthesis",
    }
    payload.update(overrides)
    return payload


def test_mcq_payload_maps_to_domain_question_with_generated_uuid() -> None:
    payload = GeneratedMCQPayload.model_validate(mcq_payload_dict())
    before_mapping = deepcopy(payload.model_dump())

    question = question_from_payload(payload)

    assert isinstance(question, MCQQuestion)
    assert isinstance(question.id, UUID)
    assert "id" not in payload.model_dump()
    assert payload.model_dump() == before_mapping
    assert question.type == payload.type
    assert question.position == payload.position
    assert question.question == payload.question
    assert question.difficulty == payload.difficulty
    assert question.explanation == payload.explanation
    assert [option.model_dump() for option in question.options] == [
        option.model_dump() for option in payload.options
    ]
    assert question.correct_option_id == payload.correct_option_id


def test_fill_blank_payload_maps_to_domain_question_with_generated_uuid() -> None:
    payload = GeneratedFillBlankPayload.model_validate(fill_blank_payload_dict())
    before_mapping = deepcopy(payload.model_dump())

    question = question_from_payload(payload)

    assert isinstance(question, FillBlankQuestion)
    assert isinstance(question.id, UUID)
    assert "id" not in payload.model_dump()
    assert payload.model_dump() == before_mapping
    assert question.type == payload.type
    assert question.position == payload.position
    assert question.question == payload.question
    assert question.difficulty == payload.difficulty
    assert question.explanation == payload.explanation
    assert question.answer == payload.answer


def test_mapping_same_payload_twice_creates_distinct_domain_uuids() -> None:
    payload = GeneratedMCQPayload.model_validate(mcq_payload_dict())

    first_question = question_from_payload(payload)
    second_question = question_from_payload(payload)

    assert first_question.id != second_question.id
    first_content = first_question.model_dump(exclude={"id"})
    second_content = second_question.model_dump(exclude={"id"})
    assert first_content == second_content


def test_external_id_cannot_enter_through_payload() -> None:
    with pytest.raises(ValidationError):
        GeneratedMCQPayload.model_validate(mcq_payload_dict(id="00000000-0000-0000-0000-000000000000"))


def test_mapped_questions_can_be_placed_in_question_set() -> None:
    mcq = question_from_payload(GeneratedMCQPayload.model_validate(mcq_payload_dict(position=1)))
    fill_blank = question_from_payload(
        GeneratedFillBlankPayload.model_validate(fill_blank_payload_dict(position=2))
    )

    question_set = QuestionSet.model_validate({"questions": [mcq, fill_blank]})

    assert [question.position for question in question_set.questions] == [1, 2]


def test_question_set_still_rejects_invalid_positions_after_mapping() -> None:
    mcq = question_from_payload(GeneratedMCQPayload.model_validate(mcq_payload_dict(position=2)))
    fill_blank = question_from_payload(
        GeneratedFillBlankPayload.model_validate(fill_blank_payload_dict(position=1))
    )

    with pytest.raises(ValidationError):
        QuestionSet.model_validate({"questions": [mcq, fill_blank]})


def test_mapper_rejects_raw_dictionary_as_main_api() -> None:
    with pytest.raises(TypeError):
        question_from_payload(mcq_payload_dict())  # type: ignore[arg-type]


def test_prompt_to_domain_pipeline_for_mcq() -> None:
    payload = GeneratedMCQPayload.model_validate(mcq_payload_dict())
    question = question_from_payload(payload)

    assert isinstance(question, MCQQuestion)
    assert isinstance(question.id, UUID)
    assert "id" not in payload.model_dump()
    assert question.correct_option_id == "B"


def test_prompt_to_domain_pipeline_for_fill_blank() -> None:
    payload = GeneratedFillBlankPayload.model_validate(fill_blank_payload_dict())
    question = question_from_payload(payload)

    assert isinstance(question, FillBlankQuestion)
    assert isinstance(question.id, UUID)
    assert "id" not in payload.model_dump()
    assert question.answer == "photosynthesis"
