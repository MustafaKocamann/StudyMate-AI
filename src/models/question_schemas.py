"""Strict Pydantic schemas for generated educational questions."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.question_constraints import (
    ANSWER_MAX_LENGTH,
    EXPLANATION_MAX_LENGTH,
    OPTION_TEXT_MAX_LENGTH,
    QUESTION_MAX_LENGTH,
    AnswerText,
    ExplanationText,
    OptionId,
    OptionText,
    QuestionText,
    validate_mcq_options_and_answer,
    validate_single_standalone_blank_placeholder,
)

class StrictSchemaModel(BaseModel):
    """Shared configuration for schemas parsed from LLM structured output."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class QuestionType(StrEnum):
    """Supported generated question kinds."""

    MCQ = "mcq"
    FILL_BLANK = "fill_blank"


class DifficultyLevel(StrEnum):
    """Supported learner difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionOption(StrictSchemaModel):
    """One answer option for a multiple-choice question."""

    id: OptionId = Field(description="Option identifier. Must be exactly one of A, B, C, or D.")
    text: OptionText = Field(
        description=f"Non-empty option text shown to the learner. Maximum {OPTION_TEXT_MAX_LENGTH} characters."
    )


class BaseQuestion(StrictSchemaModel):
    """Common fields shared by generated questions.

    The base model keeps question content independent from storage, API, and LLM
    provider concerns. Concrete subclasses constrain ``type`` so that
    discriminated unions can parse mixed question sets deterministically.
    """

    id: UUID = Field(default_factory=uuid4, description="Server-generated UUID for the question.")
    type: QuestionType = Field(description="Question type discriminator used to parse the schema.")
    position: int = Field(ge=1, description="One-based deterministic order within the question set.")
    question: QuestionText = Field(
        description=f"Non-empty learner-facing question text. Maximum {QUESTION_MAX_LENGTH} characters."
    )
    difficulty: DifficultyLevel = Field(description="Difficulty level: easy, medium, or hard.")
    explanation: ExplanationText = Field(
        description=(
            "Non-empty explanation of why the answer is correct. "
            f"Maximum {EXPLANATION_MAX_LENGTH} characters."
        )
    )

    @field_validator("question", mode="before")
    @classmethod
    def accept_documented_legacy_question_shape(cls, value: object) -> object:
        """Allow only the documented legacy dictionary shape for question text.

        Older prompts sometimes returned ``{"description": "question text"}``.
        Keeping this single shape lets existing prompt output migrate safely while
        still rejecting arbitrary dictionaries, lists, numbers, or nested objects
        that would otherwise be hidden by broad string coercion.
        """

        if isinstance(value, dict):
            if set(value) == {"description"} and isinstance(value["description"], str):
                return value["description"]
            raise ValueError('question dictionaries must have exactly one string "description" field')
        return value


class MCQQuestion(BaseQuestion):
    """Multiple-choice question with exactly four options and one correct option.

    Options must be the canonical A-D set, option labels and normalized option
    texts must be unique, and ``correct_option_id`` must reference one of the
    provided options.
    """

    type: Literal[QuestionType.MCQ] = Field(description='Question type. Must be "mcq".')
    options: list[QuestionOption] = Field(
        min_length=4,
        max_length=4,
        description="Exactly four options with IDs A, B, C, and D.",
    )
    correct_option_id: OptionId = Field(description="ID of the correct option.")

    @model_validator(mode="after")
    def validate_options_and_answer(self) -> MCQQuestion:
        """Enforce cross-field MCQ invariants before services consume the model.

        This catches malformed LLM responses where an option is duplicated, a
        required label is missing, the same answer text appears in a disguised
        form, or the declared correct option is not one of the options. After
        validation, scoring code can rely on a complete A-D option set and a
        correct option reference that is locally valid.
        """

        validate_mcq_options_and_answer(self.options, self.correct_option_id)
        return self


class FillBlankQuestion(BaseQuestion):
    """Fill-in-the-blank question with exactly one supported placeholder.

    The question text must contain one standalone ``___`` placeholder. Similar
    underscore runs such as ``__`` or ``____`` are rejected so renderers and
    answer checkers do not need to infer author intent.
    """

    type: Literal[QuestionType.FILL_BLANK] = Field(
        description='Question type. Must be "fill_blank".'
    )
    answer: AnswerText = Field(
        description=f"Non-empty correct answer for the blank. Maximum {ANSWER_MAX_LENGTH} characters."
    )

    @model_validator(mode="after")
    def validate_single_blank_placeholder(self) -> FillBlankQuestion:
        """Reject ambiguous or missing blank placeholders in generated text.

        The shared helper allows technical identifiers while rejecting malformed
        blank markers, so downstream UI code can render exactly one blank
        without losing programming or data-field examples.
        """

        validate_single_standalone_blank_placeholder(self.question)
        return self


GeneratedQuestion = Annotated[
    MCQQuestion | FillBlankQuestion,
    Field(discriminator="type"),
]
"""Discriminated union of supported generated question models."""


class QuestionSet(StrictSchemaModel):
    """Ordered collection of generated questions.

    The list uses the ``type`` discriminator to parse each item into the correct
    concrete schema. Positions must be one-based, contiguous, unique, and match
    physical list order, preserving deterministic generation order for services
    that score, render, or persist the set.
    """

    questions: list[GeneratedQuestion] = Field(
        min_length=1,
        description="Non-empty ordered list of generated questions.",
    )

    @model_validator(mode="after")
    def validate_question_set_invariants(self) -> QuestionSet:
        """Validate ordering and identity invariants for contained questions.

        LLMs can return plausible-looking lists with missing, duplicated, or
        shuffled positions. This validator rejects those cases before the data
        reaches UI pagination, scoring, or persistence layers, allowing those
        layers to treat list order and ``position`` as the same deterministic
        sequence.

        UUID uniqueness is checked explicitly even though fresh ``uuid4``
        collisions are extremely unlikely. Domain models may be reconstructed
        from persistence, callers may explicitly supply IDs, external services
        may return duplicate IDs, and deterministic fixtures may accidentally
        reuse IDs. Downstream code can assume every question in a set has a
        distinct application identity.
        """

        question_ids = [question.id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question ids must be unique within a question set")

        positions = [question.position for question in self.questions]
        expected_positions = list(range(1, len(self.questions) + 1))

        if len(set(positions)) != len(positions):
            raise ValueError("question positions must be unique")
        if positions != expected_positions:
            raise ValueError("question positions must start at 1, be contiguous, and match list order")

        return self
