from __future__ import annotations

import sys
import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.exceptions import QuestionJudgeError, QuestionRegenerationExhaustedError
from src.evaluation.models import (
    QUALITY_EVALUATION_VERSION,
    QualityDecision,
    QualityDimension,
    QualityDimensionResult,
    QualityIssue,
    QualityStatus,
    QuestionQualityReport,
)
from src.generator.regeneration import (
    GenerationAttemptPolicy,
    QualityGatedQuestionGenerator,
    QuestionGenerationRequest,
)
from src.generator.repair_prompts import (
    QUESTION_REPAIR_PROMPT_VERSION,
    fill_blank_repair_prompt_template,
    mcq_repair_prompt_template,
)
from src.models.question_payloads import GeneratedQuestionPayload
from src.models.question_schemas import GeneratedQuestion, QuestionType


def mcq_payload(question: str = "Which process lets plants convert light energy into chemical energy?") -> dict[str, object]:
    return {
        "type": "mcq",
        "position": 1,
        "question": question,
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


def fill_blank_payload() -> dict[str, object]:
    return {
        "type": "fill_blank",
        "position": 1,
        "question": "Plants convert light energy into chemical energy through ___.",
        "difficulty": "easy",
        "explanation": "Photosynthesis is the process that stores light energy in chemical form.",
        "answer": "photosynthesis",
    }


def generation_request(question_type: QuestionType = QuestionType.MCQ) -> QuestionGenerationRequest:
    return QuestionGenerationRequest(
        topic="plants light energy photosynthesis",
        source_content="Plants use photosynthesis to convert light energy into chemical energy.",
        difficulty="easy",
        question_type=question_type,
        position=1,
        language="English",
    )


def dimension_result(
    dimension: QualityDimension,
    *,
    passed: bool = True,
) -> QualityDimensionResult:
    if passed:
        return QualityDimensionResult(
            dimension=dimension,
            status=QualityStatus.PASSED,
            passed=True,
            score=0.9,
            reason=f"{dimension.value} passed.",
            issues=[],
        )
    return QualityDimensionResult(
        dimension=dimension,
        status=QualityStatus.FAILED,
        passed=False,
        score=0.1,
        reason=f"{dimension.value} failed.",
        issues=[
            QualityIssue(
                code="answer_revealed_in_question",
                message="The question reveals the answer.",
            )
        ],
    )


def quality_report(question: GeneratedQuestion, *, passed: bool) -> QuestionQualityReport:
    failed_dimension = QualityDimension.ANSWER_LEAKAGE
    return QuestionQualityReport(
        evaluation_version=QUALITY_EVALUATION_VERSION,
        question_id=question.id,
        question_type=question.type,
        overall_passed=passed,
        answer_validity=dimension_result(QualityDimension.ANSWER_VALIDITY),
        distractor_quality=(
            dimension_result(QualityDimension.DISTRACTOR_QUALITY)
            if question.type is QuestionType.MCQ
            else QualityDimensionResult(
                dimension=QualityDimension.DISTRACTOR_QUALITY,
                status=QualityStatus.NOT_APPLICABLE,
                passed=None,
                score=None,
                reason="Distractors do not apply.",
            )
        ),
        explanation_quality=dimension_result(QualityDimension.EXPLANATION_QUALITY),
        difficulty_alignment=dimension_result(QualityDimension.DIFFICULTY_ALIGNMENT),
        context_alignment=dimension_result(QualityDimension.CONTEXT_ALIGNMENT),
        answer_leakage=dimension_result(failed_dimension, passed=passed),
        duplicate_risk=dimension_result(QualityDimension.DUPLICATE_RISK),
        decision=QualityDecision.ACCEPT if passed else QualityDecision.REGENERATE,
        repair_feedback=[] if passed else ["answer_leakage: rewrite the question"],
    )


class FakeGenerator:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[GeneratedQuestionPayload | None, object | None]] = []

    async def generate(self, request, *, previous_payload=None, repair_feedback=None):  # noqa: ANN001
        self.calls.append((previous_payload, repair_feedback))
        return self.payloads[len(self.calls) - 1]


class RaisingGenerator:
    async def generate(self, request, *, previous_payload=None, repair_feedback=None):  # noqa: ANN001
        raise RuntimeError("provider failed")


class FakeQualityEvaluator:
    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = outcomes
        self.contexts = []

    async def evaluate(self, context):  # noqa: ANN001
        self.contexts.append(context)
        return quality_report(context.question, passed=self.outcomes[len(self.contexts) - 1])


class RaisingQualityEvaluator:
    async def evaluate(self, context):  # noqa: ANN001
        raise QuestionJudgeError("judge transport failed")


def test_repair_prompt_templates_render_with_expected_variables() -> None:
    for template in [mcq_repair_prompt_template, fill_blank_repair_prompt_template]:
        assert isinstance(template, ChatPromptTemplate)
        assert set(template.input_variables) == {
            "original_generation_request",
            "previous_payload",
            "repair_feedback",
            "target_output_schema",
        }
        messages = template.format_messages(
            original_generation_request='{"topic": "photosynthesis"}',
            previous_payload='{"type": "mcq"}',
            repair_feedback='{"failed_dimensions": ["answer_leakage"]}',
            target_output_schema='{"type": "mcq"}',
        )
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        system_text = str(messages[0].content)
        human_text = str(messages[1].content)
        assert "Return exactly one valid JSON object" in system_text
        assert "Do not include Markdown" in system_text
        assert "Do not include" in system_text
        assert "UUID" in system_text
        assert "<previous_validated_payload>" in human_text
        assert "{previous_payload}" not in human_text


def test_repair_prompt_version() -> None:
    assert QUESTION_REPAIR_PROMPT_VERSION == "question-repair-v1"


def test_generation_request_validation() -> None:
    request = generation_request()
    assert request.position == 1

    with pytest.raises(Exception):
        QuestionGenerationRequest(
            topic=" ",
            difficulty="easy",
            question_type="mcq",
            position=1,
            language="English",
        )

    with pytest.raises(Exception):
        QuestionGenerationRequest(
            topic="plants",
            source_content=" ",
            difficulty="easy",
            question_type="mcq",
            position=1,
            language="English",
        )


def test_invalid_attempt_policy_values_fail() -> None:
    with pytest.raises(Exception):
        GenerationAttemptPolicy(maximum_attempts=0)

    with pytest.raises(Exception):
        GenerationAttemptPolicy(maximum_attempts=11)


def test_first_attempt_passes() -> None:
    generator = FakeGenerator([mcq_payload()])
    evaluator = FakeQualityEvaluator([True])

    result = asyncio.run(async_generate(generator, evaluator, generation_request()))

    assert len(generator.calls) == 1
    assert len(evaluator.contexts) == 1
    assert result.attempts_used == 1
    assert not result.repaired
    assert result.quality_report.overall_passed


async def async_generate(generator, evaluator, request, existing_questions=None, attempts=3):  # noqa: ANN001
    return await QualityGatedQuestionGenerator(
        generator=generator,
        quality_evaluator=evaluator,
        attempt_policy=GenerationAttemptPolicy(maximum_attempts=attempts),
    ).generate(request, existing_questions=existing_questions)


def test_first_fails_second_passes_receives_repair_feedback() -> None:
    generator = FakeGenerator([mcq_payload("Photosynthesis is ___?"), mcq_payload()])
    evaluator = FakeQualityEvaluator([False, True])

    result = asyncio.run(async_generate(generator, evaluator, generation_request()))

    assert len(generator.calls) == 2
    assert len(evaluator.contexts) == 2
    assert generator.calls[0] == (None, None)
    assert generator.calls[1][0] is not None
    assert generator.calls[1][1] is not None
    assert result.attempts_used == 2
    assert result.repaired


def test_first_two_fail_third_passes() -> None:
    generator = FakeGenerator([mcq_payload("bad one"), mcq_payload("bad two"), mcq_payload()])
    evaluator = FakeQualityEvaluator([False, False, True])

    result = asyncio.run(async_generate(generator, evaluator, generation_request()))

    assert len(generator.calls) == 3
    assert result.attempts_used == 3
    assert result.repaired


def test_all_attempts_fail_exhaustion_contains_stable_details() -> None:
    generator = FakeGenerator([mcq_payload("bad one"), mcq_payload("bad two"), mcq_payload("bad three")])
    evaluator = FakeQualityEvaluator([False, False, False])

    with pytest.raises(QuestionRegenerationExhaustedError) as exc_info:
        asyncio.run(async_generate(generator, evaluator, generation_request()))

    exc = exc_info.value
    assert len(generator.calls) == 3
    assert exc.total_attempts == 3
    assert QualityDimension.ANSWER_LEAKAGE in exc.failed_dimensions
    assert "answer_revealed_in_question" in exc.issue_codes
    assert "API" not in str(exc)


def test_retry_payload_is_validated_mapped_and_evaluated() -> None:
    generator = FakeGenerator([mcq_payload("bad one"), fill_blank_payload()])
    evaluator = FakeQualityEvaluator([False, True])

    result = asyncio.run(async_generate(generator, evaluator, generation_request(QuestionType.FILL_BLANK)))

    assert result.question.type is QuestionType.FILL_BLANK
    assert len(evaluator.contexts) == 2


def test_same_payload_regenerated_unchanged_remains_rejectable() -> None:
    generator = FakeGenerator([mcq_payload("bad"), mcq_payload("bad"), mcq_payload("bad")])
    evaluator = FakeQualityEvaluator([False, False, False])

    with pytest.raises(QuestionRegenerationExhaustedError):
        asyncio.run(async_generate(generator, evaluator, generation_request()))


def test_feedback_from_immediately_previous_attempt_is_used() -> None:
    generator = FakeGenerator([mcq_payload("bad one"), mcq_payload("bad two"), mcq_payload()])
    evaluator = FakeQualityEvaluator([False, False, True])

    asyncio.run(async_generate(generator, evaluator, generation_request()))

    second_feedback = generator.calls[1][1]
    third_feedback = generator.calls[2][1]
    assert second_feedback is not third_feedback


def test_existing_questions_are_supplied_to_evaluator_context() -> None:
    existing_generator = FakeGenerator([mcq_payload()])
    existing_evaluator = FakeQualityEvaluator([True])
    existing = asyncio.run(async_generate(existing_generator, existing_evaluator, generation_request())).question

    generator = FakeGenerator([mcq_payload("another question")])
    evaluator = FakeQualityEvaluator([True])
    asyncio.run(async_generate(generator, evaluator, generation_request(), existing_questions=[existing]))

    assert evaluator.contexts[0].existing_questions == [existing]


def test_generator_technical_exception_is_not_quality_failure() -> None:
    with pytest.raises(RuntimeError):
        asyncio.run(async_generate(RaisingGenerator(), FakeQualityEvaluator([True]), generation_request()))


def test_quality_evaluator_technical_exception_is_not_quality_failure() -> None:
    with pytest.raises(QuestionJudgeError):
        asyncio.run(async_generate(FakeGenerator([mcq_payload()]), RaisingQualityEvaluator(), generation_request()))


def test_valid_fill_blank_accepted_on_first_attempt() -> None:
    generator = FakeGenerator([fill_blank_payload()])
    evaluator = FakeQualityEvaluator([True])

    result = asyncio.run(async_generate(generator, evaluator, generation_request(QuestionType.FILL_BLANK)))

    assert result.question.type is QuestionType.FILL_BLANK
    assert result.attempts_used == 1
    assert "id" not in fill_blank_payload()
