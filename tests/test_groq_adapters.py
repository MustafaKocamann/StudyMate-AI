"""Tests for Groq question generator, judge, and dependency construction."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.dependencies import build_application_container
from src.common.exceptions import (
    ApplicationConfigurationError,
    LLMResponseParsingError,
    QuestionGenerationResponseValidationError,
    QuestionJudgeResponseValidationError,
)
from src.config.settings import Settings
from src.evaluation.models import (
    QualityDimension,
    QualityDimensionResult,
    QualityStatus,
    QuestionEvaluationContext,
)
from src.generator.regeneration import QuestionGenerationRequest
from src.generator.repair_prompts import QuestionRepairFeedback
from src.llm.groq_question_generator import GroqQuestionGenerator
from src.llm.groq_question_judge import GroqQuestionJudge
from src.llm.models import ChatMessage, CompletionProfile, LLMCompletionResult
from src.models.question_mapper import question_from_payload
from src.models.question_schemas import DifficultyLevel, QuestionType


MCQ_JSON = """{
  "type": "mcq",
  "position": 1,
  "question": "Which process lets plants convert light energy into chemical energy?",
  "difficulty": "easy",
  "explanation": "Photosynthesis converts light energy into chemical energy stored in glucose.",
  "options": [
    {"id": "A", "text": "Respiration"},
    {"id": "B", "text": "Photosynthesis"},
    {"id": "C", "text": "Transpiration"},
    {"id": "D", "text": "Fermentation"}
  ],
  "correct_option_id": "B"
}"""

FILL_BLANK_JSON = """{
  "type": "fill_blank",
  "position": 1,
  "question": "Plants convert light energy into chemical energy through ___.",
  "difficulty": "easy",
  "explanation": "Photosynthesis stores light energy in chemical form.",
  "answer": "photosynthesis"
}"""

JUDGE_JSON = """{
  "answer_validity": {
    "dimension": "answer_validity",
    "status": "passed",
    "passed": true,
    "score": 0.9,
    "reason": "The answer is correct.",
    "issues": []
  },
  "distractor_quality": {
    "dimension": "distractor_quality",
    "status": "passed",
    "passed": true,
    "score": 0.9,
    "reason": "Distractors are plausible.",
    "issues": []
  },
  "explanation_quality": {
    "dimension": "explanation_quality",
    "status": "passed",
    "passed": true,
    "score": 0.9,
    "reason": "The explanation teaches the concept.",
    "issues": []
  },
  "difficulty_alignment": {
    "dimension": "difficulty_alignment",
    "status": "passed",
    "passed": true,
    "score": 0.9,
    "reason": "Difficulty matches.",
    "issues": [],
    "requested_difficulty": "easy",
    "estimated_difficulty": "easy"
  },
  "context_alignment": {
    "dimension": "context_alignment",
    "status": "passed",
    "passed": true,
    "score": 0.9,
    "reason": "The question matches the topic.",
    "issues": [],
    "context_alignment_mode": "topic_relevance"
  },
  "answer_leakage": {
    "dimension": "answer_leakage",
    "status": "passed",
    "passed": true,
    "score": 0.9,
    "reason": "No leakage is present.",
    "issues": []
  },
  "overall_score": 0.9,
  "confidence": 0.9,
  "requires_secondary_review": false,
  "feedback": "Acceptable question."
}"""


class FakeGateway:
    def __init__(self, content: str = MCQ_JSON) -> None:
        self.content = content
        self.calls: list[tuple[list[ChatMessage], CompletionProfile]] = []

    async def complete(self, *, messages, profile):  # noqa: ANN001
        self.calls.append((list(messages), profile))
        return LLMCompletionResult(content=self.content, model=profile.model)


def generation_request(question_type: QuestionType = QuestionType.MCQ) -> QuestionGenerationRequest:
    return QuestionGenerationRequest(
        topic="plants light energy photosynthesis",
        difficulty=DifficultyLevel.EASY,
        question_type=question_type,
        position=1,
        language="English",
    )


def generation_profile() -> CompletionProfile:
    return CompletionProfile(
        model="qwen/qwen3.6-27b",
        temperature=0.7,
        max_completion_tokens=2048,
        timeout_seconds=30.0,
        json_mode=True,
        reasoning_effort="none",
    )


def judge_profile() -> CompletionProfile:
    return CompletionProfile(
        model="qwen/qwen3.6-27b",
        temperature=0.0,
        max_completion_tokens=2048,
        timeout_seconds=30.0,
        json_mode=True,
        reasoning_effort="none",
    )


def test_groq_question_generator_returns_valid_mcq_payload() -> None:
    gateway = FakeGateway(MCQ_JSON)
    generator = GroqQuestionGenerator(gateway=gateway, generation_profile=generation_profile())

    payload = asyncio.run(generator.generate(generation_request()))

    assert payload.type is QuestionType.MCQ
    assert payload.correct_option_id == "B"
    messages, profile = gateway.calls[0]
    assert profile.temperature == 0.7
    assert profile.json_mode
    assert any("plants light energy photosynthesis" in message.content for message in messages)


def test_groq_question_generator_returns_valid_fill_blank_payload() -> None:
    gateway = FakeGateway(FILL_BLANK_JSON)
    generator = GroqQuestionGenerator(gateway=gateway, generation_profile=generation_profile())

    payload = asyncio.run(generator.generate(generation_request(QuestionType.FILL_BLANK)))

    assert payload.type is QuestionType.FILL_BLANK
    assert payload.answer == "photosynthesis"


def test_groq_question_generator_rejects_invalid_payload_shape() -> None:
    gateway = FakeGateway('{"type":"mcq","id":"bad","correct_answer":"B"}')
    generator = GroqQuestionGenerator(gateway=gateway, generation_profile=generation_profile())

    with pytest.raises(QuestionGenerationResponseValidationError):
        asyncio.run(generator.generate(generation_request()))


def test_groq_question_generator_malformed_json_remains_parsing_error() -> None:
    gateway = FakeGateway("```json\n{}\n```")
    generator = GroqQuestionGenerator(gateway=gateway, generation_profile=generation_profile())

    with pytest.raises(LLMResponseParsingError):
        asyncio.run(generator.generate(generation_request()))


def test_groq_question_generator_repair_prompt_uses_feedback() -> None:
    gateway = FakeGateway(MCQ_JSON)
    generator = GroqQuestionGenerator(gateway=gateway, generation_profile=generation_profile())
    previous = asyncio.run(generator.generate(generation_request()))
    feedback = QuestionRepairFeedback(revision_instructions=["Remove answer leakage."])

    repaired = asyncio.run(
        generator.generate(
            generation_request(),
            previous_payload=previous,
            repair_feedback=feedback,
        )
    )

    assert repaired.type is QuestionType.MCQ
    repair_messages, _ = gateway.calls[1]
    assert any("Remove answer leakage." in message.content for message in repair_messages)


def test_groq_question_judge_returns_strict_report() -> None:
    gateway = FakeGateway(JUDGE_JSON)
    judge = GroqQuestionJudge(gateway=gateway, judge_profile=judge_profile())
    payload = asyncio.run(GroqQuestionGenerator(gateway=FakeGateway(MCQ_JSON), generation_profile=generation_profile()).generate(generation_request()))
    context = QuestionEvaluationContext(
        question=question_from_payload(payload),
        topic="photosynthesis",
        requested_difficulty=DifficultyLevel.EASY,
        language="English",
    )

    report = asyncio.run(judge.evaluate(context, [_dimension(QualityDimension.ANSWER_VALIDITY)]))

    assert report.overall_score == 0.9
    assert "chain_of_thought" not in report.model_dump()
    _, profile = gateway.calls[0]
    assert profile.temperature == 0.0
    assert profile.json_mode


def test_groq_question_judge_malformed_json_is_validation_error() -> None:
    judge = GroqQuestionJudge(gateway=FakeGateway("not json"), judge_profile=judge_profile())
    payload = asyncio.run(GroqQuestionGenerator(gateway=FakeGateway(MCQ_JSON), generation_profile=generation_profile()).generate(generation_request()))
    context = QuestionEvaluationContext(
        question=question_from_payload(payload),
        topic="photosynthesis",
        requested_difficulty=DifficultyLevel.EASY,
        language="English",
    )

    with pytest.raises(QuestionJudgeResponseValidationError):
        asyncio.run(judge.evaluate(context, [_dimension(QualityDimension.ANSWER_VALIDITY)]))


def test_application_container_reuses_gateway_and_separates_profiles() -> None:
    gateway = FakeGateway()
    container = build_application_container(app_settings=Settings(), gateway=gateway)

    assert container.question_generator.gateway is gateway
    assert container.question_judge.gateway is gateway
    assert container.generation_profile is not container.judge_profile
    assert container.generation_profile.temperature == 0.7
    assert container.judge_profile.temperature == 0.0
    assert container.quality_evaluator.primary_judge is container.question_judge
    assert container.quality_gated_generator.generator is container.question_generator


def test_application_container_missing_key_fails_safely_without_fake_gateway() -> None:
    settings = Settings()
    if settings.groq.api_key is not None or settings.gemini.api_key is not None:
        pytest.skip("developer environment has an LLM API key configured")

    with pytest.raises(ApplicationConfigurationError):
        build_application_container(app_settings=settings)


def _dimension(dimension: QualityDimension) -> QualityDimensionResult:
    return QualityDimensionResult(
        dimension=dimension,
        status=QualityStatus.PASSED,
        passed=True,
        score=0.9,
        reason="passed",
        issues=[],
    )
