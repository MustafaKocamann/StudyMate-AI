"""Strict schemas for raw LLM-generated question payloads."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.question_constraints import (
    ANSWER_MAX_LENGTH,
    EXPLANATION_MAX_LENGTH,
    QUESTION_MAX_LENGTH,
    AnswerText,
    ExplanationText,
    OptionId,
    QuestionText,
    validate_mcq_options_and_answer,
    validate_single_standalone_blank_placeholder,
)
from src.models.question_schemas import DifficultyLevel, QuestionOption, QuestionType


class StrictPayloadModel(BaseModel):
    """Shared configuration for raw LLM payload validation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class BaseGeneratedQuestionPayload(StrictPayloadModel):
    """Common fields expected directly from question-generation prompts.

    Payload models represent untrusted raw LLM JSON before application identity
    exists. They intentionally omit UUIDs, persistence metadata, provider
    metadata, tracing data, and operational fields; those belong to later
    application layers.
    """

    type: QuestionType = Field(description="Question type discriminator supplied by the LLM.")
    position: int = Field(ge=1, description="One-based deterministic order supplied by the prompt.")
    question: QuestionText = Field(
        description=f"Non-empty generated question text. Maximum {QUESTION_MAX_LENGTH} characters."
    )
    difficulty: DifficultyLevel = Field(description="Generated difficulty enum value.")
    explanation: ExplanationText = Field(
        description=(
            "Non-empty learner-facing explanation. "
            f"Maximum {EXPLANATION_MAX_LENGTH} characters."
        )
    )


class GeneratedMCQPayload(BaseGeneratedQuestionPayload):
    """Raw LLM payload for one multiple-choice question."""

    type: Literal[QuestionType.MCQ] = Field(description='Question type. Must be "mcq".')
    options: list[QuestionOption] = Field(
        min_length=4,
        max_length=4,
        description="Exactly four options with IDs A, B, C, and D.",
    )
    correct_option_id: OptionId = Field(description="ID of the single correct option.")

    @model_validator(mode="after")
    def validate_options_and_answer(self) -> GeneratedMCQPayload:
        validate_mcq_options_and_answer(self.options, self.correct_option_id)
        return self


class GeneratedFillBlankPayload(BaseGeneratedQuestionPayload):
    """Raw LLM payload for one fill-in-the-blank question."""

    type: Literal[QuestionType.FILL_BLANK] = Field(
        description='Question type. Must be "fill_blank".'
    )
    answer: AnswerText = Field(
        description=f"Non-empty generated answer. Maximum {ANSWER_MAX_LENGTH} characters."
    )

    @model_validator(mode="after")
    def validate_single_blank_placeholder(self) -> GeneratedFillBlankPayload:
        validate_single_standalone_blank_placeholder(self.question)
        return self


GeneratedQuestionPayload = Annotated[
    GeneratedMCQPayload | GeneratedFillBlankPayload,
    Field(discriminator="type"),
]
"""Discriminated union of supported raw LLM question payloads."""
