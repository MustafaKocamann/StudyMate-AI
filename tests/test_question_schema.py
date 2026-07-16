from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.question_schemas import (
    ANSWER_MAX_LENGTH,
    DifficultyLevel,
    EXPLANATION_MAX_LENGTH,
    FillBlankQuestion,
    MCQQuestion,
    OPTION_TEXT_MAX_LENGTH,
    QUESTION_MAX_LENGTH,
    QuestionSet,
    QuestionType,
)


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
        "question": "The capital of France is ___.",
        "difficulty": "easy",
        "explanation": "Paris is correct because it is the capital city of France.",
        "answer": "Paris",
    }
    payload.update(overrides)
    return payload


def test_valid_mcq_generates_uuid_and_validates_answer_reference() -> None:
    question = MCQQuestion.model_validate(mcq_payload())

    assert isinstance(question.id, UUID)
    assert question.type is QuestionType.MCQ
    assert question.difficulty is DifficultyLevel.EASY
    assert question.correct_option_id == "A"


@pytest.mark.parametrize("option_count", [3, 5])
def test_invalid_mcq_option_count(option_count: int) -> None:
    options = [
        {"id": "A", "text": "Alpha"},
        {"id": "B", "text": "Beta"},
        {"id": "C", "text": "Gamma"},
        {"id": "D", "text": "Delta"},
        {"id": "A", "text": "Extra"},
    ][:option_count]

    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(options=options))


def test_duplicate_mcq_option_ids_fail() -> None:
    options = [
        {"id": "A", "text": "Alpha"},
        {"id": "B", "text": "Beta"},
        {"id": "C", "text": "Gamma"},
        {"id": "C", "text": "Delta"},
    ]

    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(options=options))


def test_missing_option_id_fails() -> None:
    options = [
        {"id": "A", "text": "Alpha"},
        {"id": "B", "text": "Beta"},
        {"id": "C", "text": "Gamma"},
        {"id": "C", "text": "Delta"},
    ]

    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(options=options, correct_option_id="D"))


@pytest.mark.parametrize(
    ("first_text", "second_text"),
    [
        ("Python", "Python"),
        ("Python", "PYTHON"),
        ("Python", " python "),
        ("Data   Science", "data science"),
    ],
)
def test_duplicate_option_text_after_normalization_fails(
    first_text: str,
    second_text: str,
) -> None:
    options = [
        {"id": "A", "text": first_text},
        {"id": "B", "text": second_text},
        {"id": "C", "text": "Java"},
        {"id": "D", "text": "SQL"},
    ]

    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(options=options))


def test_invalid_correct_option_fails() -> None:
    options = [
        {"id": "A", "text": "Alpha"},
        {"id": "B", "text": "Beta"},
        {"id": "C", "text": "Gamma"},
        {"id": "C", "text": "Delta"},
    ]

    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(options=options, correct_option_id="D"))


def test_valid_fill_blank_question() -> None:
    question = FillBlankQuestion.model_validate(fill_blank_payload())

    assert question.type is QuestionType.FILL_BLANK
    assert question.answer == "Paris"


@pytest.mark.parametrize(
    "question_text",
    [
        "Python'da __init__ metodunun görevi ___.",
        "Bir kullanıcı kimliği çoğunlukla user_id alanında tutulur ve ___ olarak kullanılır.",
        "Python'da snake_case çoğunlukla ___ için kullanılır.",
        "__name__ değişkeninin değeri doğrudan çalıştırılan modülde ___ olur.",
    ],
)
def test_valid_fill_blank_question_allows_technical_identifiers(question_text: str) -> None:
    question = FillBlankQuestion.model_validate(fill_blank_payload(question=question_text))

    assert question.question == question_text


@pytest.mark.parametrize(
    "question_text",
    [
        "The capital of France is Paris.",
        "___ is the capital of ___.",
        "The capital is __.",
        "The capital is ____.",
        "The capital is _____.",
        "The answer is value___name.",
        "Use ___ instead of __.",
    ],
)
def test_invalid_fill_blank_placeholder(question_text: str) -> None:
    with pytest.raises(ValidationError):
        FillBlankQuestion.model_validate(fill_blank_payload(question=question_text))


@pytest.mark.parametrize("answer", ["", "   "])
def test_invalid_fill_blank_answer(answer: str) -> None:
    with pytest.raises(ValidationError):
        FillBlankQuestion.model_validate(fill_blank_payload(answer=answer))


def test_question_accepts_exact_max_length() -> None:
    question = MCQQuestion.model_validate(mcq_payload(question="Q" * QUESTION_MAX_LENGTH))

    assert len(question.question) == QUESTION_MAX_LENGTH


def test_question_rejects_over_max_length() -> None:
    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(question="Q" * (QUESTION_MAX_LENGTH + 1)))


def test_option_text_accepts_exact_max_length() -> None:
    options = [
        {"id": "A", "text": "A" * OPTION_TEXT_MAX_LENGTH},
        {"id": "B", "text": "Beta"},
        {"id": "C", "text": "Gamma"},
        {"id": "D", "text": "Delta"},
    ]

    question = MCQQuestion.model_validate(mcq_payload(options=options))

    assert len(question.options[0].text) == OPTION_TEXT_MAX_LENGTH


def test_option_text_rejects_over_max_length() -> None:
    options = [
        {"id": "A", "text": "A" * (OPTION_TEXT_MAX_LENGTH + 1)},
        {"id": "B", "text": "Beta"},
        {"id": "C", "text": "Gamma"},
        {"id": "D", "text": "Delta"},
    ]

    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(options=options))


def test_answer_accepts_exact_max_length() -> None:
    question = FillBlankQuestion.model_validate(fill_blank_payload(answer="A" * ANSWER_MAX_LENGTH))

    assert len(question.answer) == ANSWER_MAX_LENGTH


def test_answer_rejects_over_max_length() -> None:
    with pytest.raises(ValidationError):
        FillBlankQuestion.model_validate(fill_blank_payload(answer="A" * (ANSWER_MAX_LENGTH + 1)))


def test_explanation_accepts_exact_max_length() -> None:
    question = MCQQuestion.model_validate(mcq_payload(explanation="E" * EXPLANATION_MAX_LENGTH))

    assert len(question.explanation) == EXPLANATION_MAX_LENGTH


def test_explanation_rejects_over_max_length() -> None:
    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(explanation="E" * (EXPLANATION_MAX_LENGTH + 1)))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("question", "   "),
        ("explanation", "   "),
    ],
)
def test_shared_text_fields_reject_whitespace_only(field_name: str, value: str) -> None:
    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(**{field_name: value}))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("question", ["What is Python?"]),
        ("explanation", 42),
    ],
)
def test_shared_text_fields_reject_unsupported_non_string_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(**{field_name: value}))


@pytest.mark.parametrize("option_text", ["   ", 123, ["Python"]])
def test_option_text_rejects_invalid_text_values(option_text: object) -> None:
    options = [
        {"id": "A", "text": option_text},
        {"id": "B", "text": "Beta"},
        {"id": "C", "text": "Gamma"},
        {"id": "D", "text": "Delta"},
    ]

    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(options=options))


@pytest.mark.parametrize("answer", [123, ["Paris"]])
def test_answer_rejects_unsupported_non_string_values(answer: object) -> None:
    with pytest.raises(ValidationError):
        FillBlankQuestion.model_validate(fill_blank_payload(answer=answer))


def test_invalid_question_types_fail() -> None:
    with pytest.raises(ValidationError):
        QuestionSet.model_validate({"questions": [mcq_payload(type="essay")]})


def test_unexpected_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(topic="Python"))


@pytest.mark.parametrize(
    "question_value",
    [
        ["What is Python?"],
        42,
        {"text": "What is Python?"},
        {"description": {"nested": "What is Python?"}},
        object(),
    ],
)
def test_invalid_question_input_values_are_not_stringified(question_value: object) -> None:
    with pytest.raises(ValidationError):
        MCQQuestion.model_validate(mcq_payload(question=question_value))


def test_documented_legacy_question_dictionary_is_supported() -> None:
    question = MCQQuestion.model_validate(
        mcq_payload(question={"description": " Which option is Python? "})
    )

    assert question.question == "Which option is Python?"


def test_valid_mixed_question_set() -> None:
    question_set = QuestionSet.model_validate(
        {
            "questions": [
                mcq_payload(position=1),
                fill_blank_payload(position=2),
            ]
        }
    )

    assert [question.position for question in question_set.questions] == [1, 2]
    assert [question.type for question in question_set.questions] == [
        QuestionType.MCQ,
        QuestionType.FILL_BLANK,
    ]


def test_question_set_allows_automatically_generated_unique_uuids() -> None:
    question_set = QuestionSet.model_validate(
        {
            "questions": [
                mcq_payload(position=1),
                fill_blank_payload(position=2),
            ]
        }
    )

    assert len({question.id for question in question_set.questions}) == 2


def test_question_set_allows_explicit_distinct_uuids() -> None:
    first_id = uuid4()
    second_id = uuid4()

    question_set = QuestionSet.model_validate(
        {
            "questions": [
                mcq_payload(id=first_id, position=1),
                fill_blank_payload(id=second_id, position=2),
            ]
        }
    )

    assert [question.id for question in question_set.questions] == [first_id, second_id]


def test_question_set_rejects_duplicate_uuids() -> None:
    duplicate_id = uuid4()

    with pytest.raises(ValidationError, match="question ids must be unique within a question set"):
        QuestionSet.model_validate(
            {
                "questions": [
                    mcq_payload(id=duplicate_id, position=1),
                    fill_blank_payload(id=duplicate_id, position=2),
                ]
            }
        )


@pytest.mark.parametrize(
    "positions",
    [
        [],
        [1, 1],
        [1, 3],
        [2, 3],
        [0, 1],
        [2, 1],
    ],
)
def test_question_set_position_validation(positions: list[int]) -> None:
    questions = []
    for index, position in enumerate(positions):
        if index % 2 == 0:
            questions.append(mcq_payload(position=position))
        else:
            questions.append(fill_blank_payload(position=position))

    with pytest.raises(ValidationError):
        QuestionSet.model_validate({"questions": questions})


def test_assignment_validation_rejects_invalid_values() -> None:
    question = FillBlankQuestion.model_validate(fill_blank_payload())

    with pytest.raises(ValidationError):
        question.answer = "   "


def test_assignment_validation_rejects_over_limit_values() -> None:
    question = FillBlankQuestion.model_validate(fill_blank_payload())

    with pytest.raises(ValidationError):
        question.answer = "A" * (ANSWER_MAX_LENGTH + 1)


def test_question_set_assignment_rejects_duplicate_uuids() -> None:
    question_set = QuestionSet.model_validate(
        {
            "questions": [
                mcq_payload(position=1),
                fill_blank_payload(position=2),
            ]
        }
    )
    duplicate_id = uuid4()

    with pytest.raises(ValidationError, match="question ids must be unique within a question set"):
        question_set.questions = [
            MCQQuestion.model_validate(mcq_payload(id=duplicate_id, position=1)),
            FillBlankQuestion.model_validate(fill_blank_payload(id=duplicate_id, position=2)),
        ]


def test_json_mode_serialization_uses_enum_values_uuid_strings_and_discriminators() -> None:
    question_set = QuestionSet.model_validate(
        {
            "questions": [
                mcq_payload(position=1),
                fill_blank_payload(position=2),
            ]
        }
    )

    serialized = question_set.model_dump(mode="json")

    assert isinstance(serialized["questions"][0]["id"], str)
    assert serialized["questions"][0]["type"] == "mcq"
    assert serialized["questions"][1]["type"] == "fill_blank"
