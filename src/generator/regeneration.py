"""Controlled regeneration orchestration based on quality reports."""

from __future__ import annotations

from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

from src.common.exceptions import QuestionRegenerationExhaustedError
from src.evaluation.models import QuestionEvaluationContext, QuestionQualityReport
from src.evaluation.service import QuestionQualityEvaluator
from src.generator.repair_prompts import QuestionRepairFeedback, build_question_repair_feedback
from src.models.question_mapper import question_from_payload
from src.models.question_payloads import GeneratedQuestionPayload
from src.models.question_schemas import DifficultyLevel, GeneratedQuestion, QuestionType


GenerationText = Annotated[str, StringConstraints(min_length=1, strict=True)]


class QuestionGenerationRequest(BaseModel):
    """Provider-independent request for one generated question."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    topic: GenerationText
    source_content: GenerationText | None = None
    difficulty: DifficultyLevel
    question_type: QuestionType
    position: int = Field(ge=1)
    language: GenerationText


class GenerationAttemptPolicy(BaseModel):
    """Bounded retry policy for quality-gated generation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    maximum_attempts: int = Field(default=3, ge=1, le=10)


class QualityGatedGenerationResult(BaseModel):
    """Successful output of controlled quality-gated generation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    question: GeneratedQuestion
    quality_report: QuestionQualityReport
    attempts_used: int = Field(ge=1)
    repaired: bool


class QuestionGenerator(Protocol):
    """Asynchronous provider-independent generated-payload source."""

    async def generate(
        self,
        request: QuestionGenerationRequest,
        *,
        previous_payload: GeneratedQuestionPayload | None = None,
        repair_feedback: QuestionRepairFeedback | None = None,
    ) -> GeneratedQuestionPayload: ...


class QualityEvaluator(Protocol):
    """Minimal async evaluator shape required by controlled regeneration."""

    async def evaluate(self, context: QuestionEvaluationContext) -> QuestionQualityReport: ...


class RegenerationPlan(BaseModel):
    """Pure orchestration output for later generation services."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    should_regenerate: bool
    should_escalate_to_secondary_judge: bool
    feedback: list[str] = Field(default_factory=list)


def build_regeneration_plan(report: QuestionQualityReport) -> RegenerationPlan:
    """Convert a quality report into a controlled next-step plan."""

    return RegenerationPlan(
        should_regenerate=not report.overall_passed,
        should_escalate_to_secondary_judge=(
            report.decision.value == "escalate_to_secondary_judge"
        ),
        feedback=report.repair_feedback,
    )


class QualityGatedQuestionGenerator:
    """Generate, validate, evaluate, and repair questions with bounded attempts.

    The orchestrator owns retry control only. It does not call provider SDKs
    directly, does not build hidden prompts, and never accepts a final failed
    question merely because the retry limit was reached. Each attempt validates
    the payload, maps to a fresh domain question UUID, and evaluates quality.
    """

    def __init__(
        self,
        *,
        generator: QuestionGenerator,
        quality_evaluator: QualityEvaluator | QuestionQualityEvaluator,
        attempt_policy: GenerationAttemptPolicy | None = None,
    ) -> None:
        self.generator = generator
        self.quality_evaluator = quality_evaluator
        self.attempt_policy = attempt_policy or GenerationAttemptPolicy()
        self._payload_adapter = TypeAdapter(GeneratedQuestionPayload)

    async def generate(
        self,
        request: QuestionGenerationRequest,
        *,
        existing_questions: list[GeneratedQuestion] | None = None,
    ) -> QualityGatedGenerationResult:
        previous_payload: GeneratedQuestionPayload | None = None
        repair_feedback: QuestionRepairFeedback | None = None
        failed_report: QuestionQualityReport | None = None

        for attempt_number in range(1, self.attempt_policy.maximum_attempts + 1):
            raw_payload = await self.generator.generate(
                request,
                previous_payload=previous_payload,
                repair_feedback=repair_feedback,
            )
            payload = self._payload_adapter.validate_python(raw_payload)
            question = question_from_payload(payload)
            context = QuestionEvaluationContext(
                question=question,
                topic=request.topic,
                requested_difficulty=request.difficulty,
                language=request.language,
                source_content=request.source_content,
                existing_questions=existing_questions or [],
            )
            report = await self.quality_evaluator.evaluate(context)
            if report.overall_passed:
                return QualityGatedGenerationResult(
                    question=question,
                    quality_report=report,
                    attempts_used=attempt_number,
                    repaired=attempt_number > 1,
                )

            failed_report = report
            previous_payload = payload
            repair_feedback = build_question_repair_feedback(report)

        assert failed_report is not None
        failed_dimensions = [check.dimension for check in failed_report.checks if check.passed is False]
        issue_codes = [
            issue.code
            for check in failed_report.checks
            if check.passed is False
            for issue in check.issues
        ]
        raise QuestionRegenerationExhaustedError(
            total_attempts=self.attempt_policy.maximum_attempts,
            failed_dimensions=failed_dimensions,
            issue_codes=issue_codes,
        )
