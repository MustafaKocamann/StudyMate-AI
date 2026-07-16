"""Pure mapping functions from raw payload schemas to domain question schemas."""

from __future__ import annotations

from typing import NoReturn, overload

from src.models.question_payloads import GeneratedFillBlankPayload, GeneratedMCQPayload
from src.models.question_schemas import FillBlankQuestion, MCQQuestion


@overload
def question_from_payload(payload: GeneratedMCQPayload) -> MCQQuestion: ...


@overload
def question_from_payload(payload: GeneratedFillBlankPayload) -> FillBlankQuestion: ...


def question_from_payload(
    payload: GeneratedMCQPayload | GeneratedFillBlankPayload,
) -> MCQQuestion | FillBlankQuestion:
    """Convert a validated raw LLM payload into an application domain question.

    The payload layer has no application identity, so the mapper deliberately
    does not read or copy an ID. Constructing the domain model lets the UUID
    default factory create exactly one application-owned identifier while all
    learner-facing content, position, difficulty, options, answers, and answer
    references are preserved.
    """

    if isinstance(payload, GeneratedMCQPayload):
        return MCQQuestion(
            type=payload.type,
            position=payload.position,
            question=payload.question,
            difficulty=payload.difficulty,
            explanation=payload.explanation,
            options=[option.model_dump() for option in payload.options],
            correct_option_id=payload.correct_option_id,
        )

    if isinstance(payload, GeneratedFillBlankPayload):
        return FillBlankQuestion(
            type=payload.type,
            position=payload.position,
            question=payload.question,
            difficulty=payload.difficulty,
            explanation=payload.explanation,
            answer=payload.answer,
        )

    _raise_unsupported_payload(payload)


def _raise_unsupported_payload(payload: object) -> NoReturn:
    raise TypeError(f"unsupported generated question payload: {type(payload).__name__}")
