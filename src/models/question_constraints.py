"""Shared constraints and invariants for question schemas."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Protocol, Sequence

from pydantic import StringConstraints


QUESTION_MAX_LENGTH = 500
OPTION_TEXT_MAX_LENGTH = 200
ANSWER_MAX_LENGTH = 200
EXPLANATION_MAX_LENGTH = 1_200

OptionId = Literal["A", "B", "C", "D"]
"""Supported multiple-choice option identifiers."""

QuestionText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=QUESTION_MAX_LENGTH, strict=True),
]
OptionText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=OPTION_TEXT_MAX_LENGTH, strict=True),
]
AnswerText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=ANSWER_MAX_LENGTH, strict=True),
]
ExplanationText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=EXPLANATION_MAX_LENGTH, strict=True),
]

VALID_OPTION_IDS: frozenset[OptionId] = frozenset(("A", "B", "C", "D"))
BLANK_PLACEHOLDER = "___"
STANDALONE_BLANK_PATTERN = re.compile(r"(?<!\w)___(?!\w)")
UNDERSCORE_SEQUENCE_PATTERN = re.compile(r"_+")


class OptionLike(Protocol):
    """Minimal shape needed by shared MCQ invariant validation."""

    id: OptionId
    text: str


def normalize_option_text_for_comparison(text: str) -> str:
    """Return a comparison key for detecting duplicated option text."""

    return " ".join(text.split()).casefold()


def validate_mcq_options_and_answer(
    options: Sequence[OptionLike],
    correct_option_id: OptionId,
) -> None:
    """Enforce shared MCQ option and answer invariants.

    Raw payload models and identified domain models must reject the same
    malformed multiple-choice structures: duplicated or missing labels,
    disguised duplicate option text, and answer references that do not point to
    a supplied option. Keeping the cross-field logic here prevents prompt-facing
    payload schemas and application domain schemas from drifting apart.
    """

    option_ids = [option.id for option in options]
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("option ids must be unique")

    option_id_set = set(option_ids)
    if option_id_set != VALID_OPTION_IDS:
        raise ValueError("options must contain exactly the IDs A, B, C, and D")

    normalized_texts = [normalize_option_text_for_comparison(option.text) for option in options]
    if len(set(normalized_texts)) != len(normalized_texts):
        raise ValueError("option text values must be unique after normalization")

    if correct_option_id not in option_id_set:
        raise ValueError("correct_option_id must reference an option in options")


def _is_identifier_adjacent(text: str, start: int, end: int) -> bool:
    previous_char = text[start - 1] if start > 0 else ""
    next_char = text[end] if end < len(text) else ""
    return previous_char.isalnum() or next_char.isalnum()


def validate_single_standalone_blank_placeholder(question: str) -> None:
    """Validate that fill-blank text contains one real blank and safe identifiers.

    A simple substring count is not enough: ``____`` contains ``___`` as a
    substring, and technical content often contains underscores that are not
    blanks at all, such as ``__init__``, ``__name__``, ``user_id``, or
    ``snake_case``. This validator first identifies standalone exact ``___``
    placeholders that are not embedded inside word-like identifiers, masks the
    one real placeholder, then inspects remaining underscore runs.

    It rejects malformed LLM outputs such as missing blanks, multiple blanks,
    standalone ``__``/``____``/``_____`` markers, and embedded three-underscore
    runs like ``value___name``. Downstream UI and scoring code may therefore
    assume there is exactly one renderable blank while still allowing legitimate
    programming or technical identifiers in the question text.
    """

    placeholders = list(STANDALONE_BLANK_PATTERN.finditer(question))
    if len(placeholders) != 1:
        raise ValueError('fill-blank questions must contain exactly one standalone "___" placeholder')

    placeholder = placeholders[0]
    masked_question = (
        question[: placeholder.start()]
        + (" " * len(BLANK_PLACEHOLDER))
        + question[placeholder.end() :]
    )

    for underscore_sequence in UNDERSCORE_SEQUENCE_PATTERN.finditer(masked_question):
        sequence = underscore_sequence.group()
        if len(sequence) >= len(BLANK_PLACEHOLDER):
            raise ValueError("underscore runs of three or more characters are only allowed for the blank")
        if len(sequence) == 2 and not _is_identifier_adjacent(
            masked_question,
            underscore_sequence.start(),
            underscore_sequence.end(),
        ):
            raise ValueError('standalone "__" is not a supported blank placeholder')
