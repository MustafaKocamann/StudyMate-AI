from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.question_service import (
    QualityGatedStudyQuestionService,
    QuestionSetGenerationRequest,
    question_type_blueprint,
)
from src.common.exceptions import (
    LLMModelUnavailableError,
    LLMTimeoutError,
    QuestionGenerationResponseValidationError,
    QuestionJudgeError,
    QuestionRegenerationExhaustedError,
    QuestionSetGenerationError,
)
from src.evaluation.models import (
    QUALITY_EVALUATION_VERSION,
    QualityDecision,
    QualityDimension,
    QualityDimensionResult,
    QualityIssue,
    QualityStatus,
    QuestionQualityReport,
)
from src.generator.regeneration import QualityGatedGenerationResult, QuestionGenerationRequest
from src.models.question_schemas import (
    DifficultyLevel,
    FillBlankQuestion,
    GeneratedQuestion,
    MCQQuestion,
    QuestionOption,
    QuestionType,
)
from src.models.study_session import StudyQuestionMode


def mcq_question(*, position: int) -> MCQQuestion:
    return MCQQuestion(
        type=QuestionType.MCQ,
        position=position,
        question=f"Which process converts light energy into chemical energy at position {position}?",
        difficulty=DifficultyLevel.EASY,
        explanation="Photosynthesis converts light energy into chemical energy stored in glucose.",
        options=[
            QuestionOption(id="A", text=f"Respiration {position}"),
            QuestionOption(id="B", text=f"Photosynthesis {position}"),
            QuestionOption(id="C", text=f"Transpiration {position}"),
            QuestionOption(id="D", text=f"Fermentation {position}"),
        ],
        correct_option_id="B",
    )


def fill_blank_question(*, position: int) -> FillBlankQuestion:
    return FillBlankQuestion(
        type=QuestionType.FILL_BLANK,
        position=position,
        question=f"Plants convert light energy at position {position} through ___.",
        difficulty=DifficultyLevel.EASY,
        explanation="Photosynthesis stores light energy in chemical form.",
        answer=f"photosynthesis {position}",
    )


def dimension_result(dimension: QualityDimension) -> QualityDimensionResult:
    return QualityDimensionResult(
        dimension=dimension,
        status=QualityStatus.PASSED,
        passed=True,
        score=0.9,
        reason=f"{dimension.value} passed.",
        issues=[],
    )


def quality_report(question: GeneratedQuestion) -> QuestionQualityReport:
    distractor_quality = (
        dimension_result(QualityDimension.DISTRACTOR_QUALITY)
        if question.type is QuestionType.MCQ
        else QualityDimensionResult(
            dimension=QualityDimension.DISTRACTOR_QUALITY,
            status=QualityStatus.NOT_APPLICABLE,
            passed=None,
            score=None,
            reason="Distractors do not apply.",
        )
    )
    return QuestionQualityReport(
        evaluation_version=QUALITY_EVALUATION_VERSION,
        question_id=question.id,
        question_type=question.type,
        overall_passed=True,
        answer_validity=dimension_result(QualityDimension.ANSWER_VALIDITY),
        distractor_quality=distractor_quality,
        explanation_quality=dimension_result(QualityDimension.EXPLANATION_QUALITY),
        difficulty_alignment=dimension_result(QualityDimension.DIFFICULTY_ALIGNMENT),
        context_alignment=dimension_result(QualityDimension.CONTEXT_ALIGNMENT),
        answer_leakage=dimension_result(QualityDimension.ANSWER_LEAKAGE),
        duplicate_risk=dimension_result(QualityDimension.DUPLICATE_RISK),
        decision=QualityDecision.ACCEPT,
        repair_feedback=[],
    )


class FakeQualityGatedGenerator:
    def __init__(
        self,
        *,
        repaired_positions: set[int] | None = None,
        fail_position: int | None = None,
        exc_by_position: dict[int, Exception] | None = None,
    ) -> None:
        self.repaired_positions = repaired_positions or set()
        self.fail_position = fail_position
        self.exc_by_position = exc_by_position or {}
        self.calls: list[tuple[QuestionGenerationRequest, list[GeneratedQuestion]]] = []

    async def generate(self, request, *, existing_questions=None):  # noqa: ANN001
        previous_questions = list(existing_questions or [])
        self.calls.append((request, previous_questions))
        if request.position in self.exc_by_position:
            raise self.exc_by_position[request.position]
        if request.position == self.fail_position:
            raise QuestionRegenerationExhaustedError(
                total_attempts=3,
                failed_dimensions=[QualityDimension.ANSWER_LEAKAGE],
                issue_codes=["answer_revealed_in_question"],
            )

        question = (
            mcq_question(position=request.position)
            if request.question_type is QuestionType.MCQ
            else fill_blank_question(position=request.position)
        )
        repaired = request.position in self.repaired_positions
        return QualityGatedGenerationResult(
            question=question,
            quality_report=quality_report(question),
            attempts_used=2 if repaired else 1,
            repaired=repaired,
        )


def test_question_set_request_is_strict_application_input() -> None:
    request = QuestionSetGenerationRequest(
        topic="photosynthesis",
        source_content="Plants use photosynthesis.",
        difficulty="easy",
        language="English",
        question_count=3,
        question_mode="mixed",
    )

    assert request.question_count == 3
    assert request.question_mode is StudyQuestionMode.MIXED

    with pytest.raises(ValidationError):
        QuestionSetGenerationRequest(
            topic=" ",
            difficulty="easy",
            language="English",
            question_count=1,
            question_mode="mcq",
        )

    with pytest.raises(ValidationError):
        QuestionSetGenerationRequest(
            topic="photosynthesis",
            source_content=" ",
            difficulty="easy",
            language="English",
            question_count=1,
            question_mode="mcq",
        )

    with pytest.raises(ValidationError):
        QuestionSetGenerationRequest(
            topic="photosynthesis",
            difficulty="easy",
            language="English",
            question_count=21,
            question_mode="mcq",
            model="provider-model",
        )


def test_question_type_blueprint_is_deterministic() -> None:
    assert question_type_blueprint(StudyQuestionMode.MCQ, 1) == [QuestionType.MCQ]
    assert question_type_blueprint(StudyQuestionMode.MCQ, 3) == [
        QuestionType.MCQ,
        QuestionType.MCQ,
        QuestionType.MCQ,
    ]
    assert question_type_blueprint(StudyQuestionMode.FILL_BLANK, 1) == [QuestionType.FILL_BLANK]
    assert question_type_blueprint(StudyQuestionMode.FILL_BLANK, 2) == [
        QuestionType.FILL_BLANK,
        QuestionType.FILL_BLANK,
    ]
    assert question_type_blueprint(StudyQuestionMode.MIXED, 5) == [
        QuestionType.MCQ,
        QuestionType.FILL_BLANK,
        QuestionType.MCQ,
        QuestionType.FILL_BLANK,
        QuestionType.MCQ,
    ]

    with pytest.raises(ValueError):
        question_type_blueprint(StudyQuestionMode.MIXED, 0)


def test_generate_question_delegates_to_quality_gated_generator() -> None:
    generator = FakeQualityGatedGenerator()
    service = QualityGatedStudyQuestionService(generator)  # type: ignore[arg-type]
    existing = [mcq_question(position=1)]
    request = QuestionGenerationRequest(
        topic="photosynthesis",
        difficulty="easy",
        question_type="fill_blank",
        position=2,
        language="English",
    )

    result = asyncio.run(service.generate_question(request, existing_questions=existing))

    assert result.question.type is QuestionType.FILL_BLANK
    assert generator.calls == [(request, existing)]
    delegated_request = generator.calls[0][0]
    assert delegated_request.topic == "photosynthesis"
    assert delegated_request.source_content is None
    assert delegated_request.difficulty is DifficultyLevel.EASY
    assert delegated_request.position == 2
    assert delegated_request.language == "English"


def test_generate_mcq_and_fill_blank_are_thin_wrappers() -> None:
    generator = FakeQualityGatedGenerator()
    service = QualityGatedStudyQuestionService(generator)  # type: ignore[arg-type]

    mcq = asyncio.run(
        service.generate_mcq(topic="photosynthesis", difficulty=DifficultyLevel.EASY, language="English")
    )
    fill_blank = asyncio.run(
        service.generate_fill_blank(
            topic="photosynthesis",
            difficulty=DifficultyLevel.EASY,
            language="English",
            position=2,
        )
    )

    assert mcq.question.type is QuestionType.MCQ
    assert fill_blank.question.type is QuestionType.FILL_BLANK
    assert [call[0].question_type for call in generator.calls] == [
        QuestionType.MCQ,
        QuestionType.FILL_BLANK,
    ]


def test_generate_question_set_returns_ordered_result_and_reports() -> None:
    generator = FakeQualityGatedGenerator(repaired_positions={2})
    service = QualityGatedStudyQuestionService(generator)  # type: ignore[arg-type]
    request = QuestionSetGenerationRequest(
        topic="photosynthesis",
        difficulty=DifficultyLevel.EASY,
        language="English",
        question_count=3,
        question_mode=StudyQuestionMode.MIXED,
    )

    result = asyncio.run(service.generate_question_set(request))

    assert [question.type for question in result.question_set.questions] == [
        QuestionType.MCQ,
        QuestionType.FILL_BLANK,
        QuestionType.MCQ,
    ]
    assert [question.position for question in result.question_set.questions] == [1, 2, 3]
    assert len(result.generation_reports) == 3
    assert result.total_attempts == 4
    assert result.repaired_question_count == 1
    assert [len(existing) for _, existing in generator.calls] == [0, 1, 2]
    assert len({question.id for question in result.question_set.questions}) == 3
    assert [
        report.question_id for report in result.generation_reports
    ] == [question.id for question in result.question_set.questions]


def test_generate_question_set_supports_all_single_mode_sets() -> None:
    mcq_generator = FakeQualityGatedGenerator()
    mcq_service = QualityGatedStudyQuestionService(mcq_generator)  # type: ignore[arg-type]
    fill_generator = FakeQualityGatedGenerator()
    fill_service = QualityGatedStudyQuestionService(fill_generator)  # type: ignore[arg-type]

    mcq_result = asyncio.run(
        mcq_service.generate_question_set(
            QuestionSetGenerationRequest(
                topic="photosynthesis",
                difficulty=DifficultyLevel.EASY,
                language="English",
                question_count=2,
                question_mode=StudyQuestionMode.MCQ,
            )
        )
    )
    fill_result = asyncio.run(
        fill_service.generate_question_set(
            QuestionSetGenerationRequest(
                topic="photosynthesis",
                difficulty=DifficultyLevel.EASY,
                language="English",
                question_count=2,
                question_mode=StudyQuestionMode.FILL_BLANK,
            )
        )
    )

    assert [question.type for question in mcq_result.question_set.questions] == [
        QuestionType.MCQ,
        QuestionType.MCQ,
    ]
    assert [question.type for question in fill_result.question_set.questions] == [
        QuestionType.FILL_BLANK,
        QuestionType.FILL_BLANK,
    ]


def test_generate_questions_compatibility_returns_question_set() -> None:
    generator = FakeQualityGatedGenerator()
    service = QualityGatedStudyQuestionService(generator)  # type: ignore[arg-type]

    question_set = asyncio.run(
        service.generate_questions(
            topic="photosynthesis",
            difficulty=DifficultyLevel.EASY,
            question_mode=StudyQuestionMode.MIXED,
            count=2,
            language="English",
        )
    )

    assert [question.type for question in question_set.questions] == [
        QuestionType.MCQ,
        QuestionType.FILL_BLANK,
    ]


def test_question_set_generation_fails_closed_on_regeneration_exhaustion() -> None:
    generator = FakeQualityGatedGenerator(fail_position=2)
    service = QualityGatedStudyQuestionService(generator)  # type: ignore[arg-type]
    request = QuestionSetGenerationRequest(
        topic="photosynthesis",
        difficulty=DifficultyLevel.EASY,
        language="English",
        question_count=3,
        question_mode=StudyQuestionMode.MIXED,
    )

    with pytest.raises(QuestionSetGenerationError) as exc_info:
        asyncio.run(service.generate_question_set(request))

    exc = exc_info.value
    assert exc.failed_position == 2
    assert exc.question_type == "fill_blank"
    assert exc.total_attempts == 4
    assert exc.issue_codes == ["answer_revealed_in_question"]
    assert len(generator.calls) == 2


def test_question_set_generation_fails_closed_at_first_position() -> None:
    generator = FakeQualityGatedGenerator(fail_position=1)
    service = QualityGatedStudyQuestionService(generator)  # type: ignore[arg-type]
    request = QuestionSetGenerationRequest(
        topic="photosynthesis",
        difficulty=DifficultyLevel.EASY,
        language="English",
        question_count=3,
        question_mode=StudyQuestionMode.MIXED,
    )

    with pytest.raises(QuestionSetGenerationError) as exc_info:
        asyncio.run(service.generate_question_set(request))

    assert exc_info.value.failed_position == 1
    assert exc_info.value.__cause__ is not None
    assert len(generator.calls) == 1


@pytest.mark.parametrize(
    "exc",
    [
        LLMTimeoutError("provider timeout", provider="groq", model="model", error_category="timeout"),
        LLMModelUnavailableError(
            "model unavailable",
            provider="groq",
            model="missing-model",
            error_category="model_unavailable",
        ),
        QuestionGenerationResponseValidationError("payload malformed"),
        QuestionJudgeError("quality evaluator failed"),
    ],
)
def test_question_set_generation_preserves_typed_technical_failures(exc: Exception) -> None:
    generator = FakeQualityGatedGenerator(exc_by_position={2: exc})
    service = QualityGatedStudyQuestionService(generator)  # type: ignore[arg-type]
    request = QuestionSetGenerationRequest(
        topic="sensitive source text should not appear in exceptions",
        source_content="private source content should not appear in exceptions",
        difficulty=DifficultyLevel.EASY,
        language="English",
        question_count=3,
        question_mode=StudyQuestionMode.MIXED,
    )

    with pytest.raises(type(exc)) as exc_info:
        asyncio.run(service.generate_question_set(request))

    assert exc_info.value is exc
    assert "private source content" not in str(exc_info.value)
    assert "API key" not in str(exc_info.value)
    assert len(generator.calls) == 2
