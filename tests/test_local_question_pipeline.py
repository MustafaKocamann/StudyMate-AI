from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.question_service import QualityGatedStudyQuestionService, QuestionSetGenerationRequest
from src.common.exceptions import LLMResponseParsingError, LLMTimeoutError
from src.evaluation.models import (
    LLMJudgeReport,
    QualityDimension,
    QualityDimensionResult,
    QualityStatus,
    QuestionEvaluationContext,
)
from src.evaluation.service import QuestionQualityEvaluator
from src.generator.regeneration import GenerationAttemptPolicy, QualityGatedQuestionGenerator
from src.llm.groq_question_generator import GroqQuestionGenerator
from src.llm.models import ChatMessage, CompletionProfile, LLMCompletionResult
from src.models.question_schemas import DifficultyLevel, QuestionType
from src.models.study_session import StudyQuestionMode


class SequencedGateway:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls: list[list[ChatMessage]] = []

    async def complete(self, *, messages, profile):  # noqa: ANN001
        self.calls.append(list(messages))
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        return LLMCompletionResult(content=response, model=profile.model)


class PassingJudge:
    async def evaluate(
        self,
        context: QuestionEvaluationContext,
        deterministic_findings: list[QualityDimensionResult],
    ) -> LLMJudgeReport:
        return LLMJudgeReport(
            answer_validity=_passed(QualityDimension.ANSWER_VALIDITY),
            distractor_quality=(
                _passed(QualityDimension.DISTRACTOR_QUALITY)
                if context.question.type is QuestionType.MCQ
                else QualityDimensionResult(
                    dimension=QualityDimension.DISTRACTOR_QUALITY,
                    status=QualityStatus.NOT_APPLICABLE,
                    passed=None,
                    score=None,
                    reason="Distractors do not apply.",
                )
            ),
            explanation_quality=_passed(QualityDimension.EXPLANATION_QUALITY),
            difficulty_alignment=_passed(QualityDimension.DIFFICULTY_ALIGNMENT),
            context_alignment=_passed(QualityDimension.CONTEXT_ALIGNMENT),
            answer_leakage=_passed(QualityDimension.ANSWER_LEAKAGE),
            overall_score=0.9,
            confidence=0.9,
            requires_secondary_review=False,
            feedback="Acceptable.",
        )


def _passed(dimension: QualityDimension) -> QualityDimensionResult:
    details = {}
    if dimension is QualityDimension.DIFFICULTY_ALIGNMENT:
        details = {"requested_difficulty": "easy", "estimated_difficulty": "easy"}
    if dimension is QualityDimension.CONTEXT_ALIGNMENT:
        details = {"context_alignment_mode": "topic_relevance"}
    return QualityDimensionResult(
        dimension=dimension,
        status=QualityStatus.PASSED,
        passed=True,
        score=0.9,
        reason=f"{dimension.value} passed.",
        issues=[],
        **details,
    )


def profile() -> CompletionProfile:
    return CompletionProfile(
        model="fake-model",
        temperature=0.7,
        max_completion_tokens=512,
        timeout_seconds=5.0,
        json_mode=True,
        reasoning_effort="none",
    )


def mcq_json(*, position: int, question: str, correct_text: str = "Photosynthesis") -> str:
    return json.dumps(
        {
            "type": "mcq",
            "position": position,
            "question": question,
            "difficulty": "easy",
            "explanation": "Photosynthesis converts light energy into chemical energy stored in glucose.",
            "options": [
                {"id": "A", "text": "Respiration"},
                {"id": "B", "text": correct_text},
                {"id": "C", "text": "Transpiration"},
                {"id": "D", "text": "Fermentation"},
            ],
            "correct_option_id": "B",
        }
    )


def fill_json(*, position: int, question: str = "Plants store light energy using ___.") -> str:
    return json.dumps(
        {
            "type": "fill_blank",
            "position": position,
            "question": question,
            "difficulty": "easy",
            "explanation": "Photosynthesis stores light energy in chemical form.",
            "answer": "photosynthesis",
        }
    )


def service_for_gateway(gateway: SequencedGateway) -> QualityGatedStudyQuestionService:
    generator = GroqQuestionGenerator(gateway=gateway, generation_profile=profile())
    evaluator = QuestionQualityEvaluator(primary_judge=PassingJudge())
    quality_gated = QualityGatedQuestionGenerator(
        generator=generator,
        quality_evaluator=evaluator,
        attempt_policy=GenerationAttemptPolicy(maximum_attempts=3),
    )
    return QualityGatedStudyQuestionService(quality_gated)


def test_local_pipeline_accepts_mixed_set_without_network() -> None:
    gateway = SequencedGateway(
        [
            mcq_json(position=1, question="Which process lets plants convert light energy?"),
            fill_json(position=2),
        ]
    )
    service = service_for_gateway(gateway)

    result = asyncio.run(
        service.generate_question_set(
            QuestionSetGenerationRequest(
                topic="plants light energy photosynthesis",
                difficulty=DifficultyLevel.EASY,
                language="English",
                question_count=2,
                question_mode=StudyQuestionMode.MIXED,
            )
        )
    )

    assert [question.type for question in result.question_set.questions] == [
        QuestionType.MCQ,
        QuestionType.FILL_BLANK,
    ]
    assert all(question.id for question in result.question_set.questions)
    assert result.total_attempts == 2
    assert len(gateway.calls) == 2


def test_local_pipeline_repairs_duplicate_candidate_before_acceptance() -> None:
    duplicate_question = "Which process lets plants convert light energy?"
    gateway = SequencedGateway(
        [
            mcq_json(position=1, question=duplicate_question),
            mcq_json(position=2, question=duplicate_question),
            mcq_json(position=2, question="Which plant process stores light energy as glucose?"),
        ]
    )
    service = service_for_gateway(gateway)

    result = asyncio.run(
        service.generate_question_set(
            QuestionSetGenerationRequest(
                topic="plants light energy photosynthesis",
                difficulty=DifficultyLevel.EASY,
                language="English",
                question_count=2,
                question_mode=StudyQuestionMode.MCQ,
            )
        )
    )

    assert result.repaired_question_count == 1
    assert result.total_attempts == 3
    assert result.question_set.questions[1].question != duplicate_question


def test_local_pipeline_malformed_response_fails_safely() -> None:
    gateway = SequencedGateway(["not json"])
    service = service_for_gateway(gateway)

    with pytest.raises(LLMResponseParsingError):
        asyncio.run(
            service.generate_question_set(
                QuestionSetGenerationRequest(
                    topic="plants light energy photosynthesis",
                    difficulty=DifficultyLevel.EASY,
                    language="English",
                    question_count=1,
                    question_mode=StudyQuestionMode.MCQ,
                )
            )
        )


def test_local_pipeline_provider_timeout_remains_typed_failure() -> None:
    timeout = LLMTimeoutError("provider timeout", provider="groq", model="fake-model", error_category="timeout")
    gateway = SequencedGateway([timeout])
    service = service_for_gateway(gateway)

    with pytest.raises(LLMTimeoutError) as exc_info:
        asyncio.run(
            service.generate_question_set(
                QuestionSetGenerationRequest(
                    topic="plants light energy photosynthesis",
                    difficulty=DifficultyLevel.EASY,
                    language="English",
                    question_count=1,
                    question_mode=StudyQuestionMode.MCQ,
                )
            )
        )

    assert exc_info.value.error_category == "timeout"
