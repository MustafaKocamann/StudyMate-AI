"""Contracts for educational question-quality evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.models.question_constraints import OptionId
from src.models.question_schemas import DifficultyLevel, GeneratedQuestion, QuestionType


QUALITY_EVALUATION_VERSION = "question-quality-v1"


class EvaluationModel(BaseModel):
    """Shared strict configuration for evaluation-layer data contracts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


IssueCode = Annotated[str, StringConstraints(min_length=1, max_length=120, strict=True)]
IssueMessage = Annotated[str, StringConstraints(min_length=1, max_length=1_000, strict=True)]
ReasonText = Annotated[str, StringConstraints(min_length=1, max_length=2_000, strict=True)]
FeedbackText = Annotated[str, StringConstraints(min_length=1, max_length=2_000, strict=True)]
ContextText = Annotated[str, StringConstraints(min_length=1, strict=True)]


class QualityDimension(StrEnum):
    """Supported educational quality dimensions."""

    ANSWER_VALIDITY = "answer_validity"
    DISTRACTOR_QUALITY = "distractor_quality"
    EXPLANATION_QUALITY = "explanation_quality"
    DIFFICULTY_ALIGNMENT = "difficulty_alignment"
    CONTEXT_ALIGNMENT = "context_alignment"
    ANSWER_LEAKAGE = "answer_leakage"
    DUPLICATE_RISK = "duplicate_risk"


class QualityStatus(StrEnum):
    """Status for an evaluated quality dimension."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"


class QualityDecision(StrEnum):
    """Final acceptance decision for a generated question."""

    ACCEPT = "accept"
    REGENERATE = "regenerate"
    ESCALATE_TO_SECONDARY_JUDGE = "escalate_to_secondary_judge"


class ContextAlignmentMode(StrEnum):
    """Context-alignment mode based on available evidence."""

    TOPIC_RELEVANCE = "topic_relevance"
    SOURCE_GROUNDEDNESS = "source_groundedness"


class QualityIssue(EvaluationModel):
    """Structured, safe quality issue for feedback and metrics."""

    code: IssueCode
    message: IssueMessage
    affected_option_ids: list[OptionId] = Field(default_factory=list)


class QuestionEvaluationContext(EvaluationModel):
    """Input context for evaluating one already-validated domain question.

    The context carries learner/request information needed by deterministic
    checks and judges, but it deliberately excludes provider configuration,
    credentials, raw prompts, and operational metadata.
    """

    question: GeneratedQuestion
    topic: ContextText
    requested_difficulty: DifficultyLevel
    language: ContextText
    source_content: ContextText | None = None
    existing_questions: list[GeneratedQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidate_not_in_existing_questions(self) -> QuestionEvaluationContext:
        existing_ids = {question.id for question in self.existing_questions}
        if self.question.id in existing_ids:
            raise ValueError("candidate question must not be included in existing_questions")
        return self


class QualityDimensionResult(EvaluationModel):
    """Result for one educational quality dimension.

    The ``passed`` field is intentionally nullable. Dimensions that do not
    apply, such as distractor quality for fill-in-the-blank questions, are not
    treated as successful evaluations.
    """

    dimension: QualityDimension
    status: QualityStatus
    passed: bool | None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: ReasonText
    issues: list[QualityIssue] = Field(default_factory=list)
    context_alignment_mode: ContextAlignmentMode | None = None
    requested_difficulty: DifficultyLevel | None = None
    estimated_difficulty: DifficultyLevel | None = None

    @model_validator(mode="after")
    def validate_status_consistency(self) -> QualityDimensionResult:
        if self.status is QualityStatus.PASSED and self.passed is not True:
            raise ValueError('passed must be True when status is "passed"')
        if self.status is QualityStatus.FAILED and self.passed is not False:
            raise ValueError('passed must be False when status is "failed"')
        if self.status in {QualityStatus.NOT_APPLICABLE, QualityStatus.NOT_EVALUATED}:
            if self.passed is not None:
                raise ValueError("passed must be None when a dimension is not applicable or not evaluated")
        if self.status is QualityStatus.FAILED and not self.issues:
            raise ValueError("failed quality dimension results must include at least one issue")
        if self.dimension is not QualityDimension.CONTEXT_ALIGNMENT and self.context_alignment_mode is not None:
            raise ValueError("context_alignment_mode is only valid for the context_alignment dimension")
        has_difficulty_details = self.requested_difficulty is not None or self.estimated_difficulty is not None
        if self.dimension is not QualityDimension.DIFFICULTY_ALIGNMENT and has_difficulty_details:
            raise ValueError("difficulty details are only valid for the difficulty_alignment dimension")
        return self


class LLMJudgeReport(EvaluationModel):
    """Semantic judge report produced by an injected judge implementation."""

    answer_validity: QualityDimensionResult
    distractor_quality: QualityDimensionResult
    explanation_quality: QualityDimensionResult
    difficulty_alignment: QualityDimensionResult
    context_alignment: QualityDimensionResult
    answer_leakage: QualityDimensionResult
    overall_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_secondary_review: bool
    feedback: FeedbackText

    @model_validator(mode="after")
    def validate_dimension_fields(self) -> LLMJudgeReport:
        expected_dimensions = {
            "answer_validity": QualityDimension.ANSWER_VALIDITY,
            "distractor_quality": QualityDimension.DISTRACTOR_QUALITY,
            "explanation_quality": QualityDimension.EXPLANATION_QUALITY,
            "difficulty_alignment": QualityDimension.DIFFICULTY_ALIGNMENT,
            "context_alignment": QualityDimension.CONTEXT_ALIGNMENT,
            "answer_leakage": QualityDimension.ANSWER_LEAKAGE,
        }
        for field_name, expected_dimension in expected_dimensions.items():
            result = getattr(self, field_name)
            if result.dimension is not expected_dimension:
                raise ValueError(f"{field_name} must use dimension {expected_dimension.value}")
        return self

    @property
    def checks(self) -> list[QualityDimensionResult]:
        return [
            self.answer_validity,
            self.distractor_quality,
            self.explanation_quality,
            self.difficulty_alignment,
            self.context_alignment,
            self.answer_leakage,
        ]


JudgeEvaluation = LLMJudgeReport


class QuestionQualityReport(EvaluationModel):
    """Serializable quality report for one generated domain question."""

    evaluation_version: str = Field(default=QUALITY_EVALUATION_VERSION)
    question_id: UUID
    question_type: QuestionType
    overall_passed: bool
    answer_validity: QualityDimensionResult
    distractor_quality: QualityDimensionResult
    explanation_quality: QualityDimensionResult
    difficulty_alignment: QualityDimensionResult
    context_alignment: QualityDimensionResult
    answer_leakage: QualityDimensionResult
    duplicate_risk: QualityDimensionResult
    judge_evaluation: LLMJudgeReport | None = None
    secondary_judge_evaluation: LLMJudgeReport | None = None
    decision: QualityDecision
    repair_feedback: list[FeedbackText] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.overall_passed

    @property
    def checks(self) -> list[QualityDimensionResult]:
        return [
            self.answer_validity,
            self.distractor_quality,
            self.explanation_quality,
            self.difficulty_alignment,
            self.context_alignment,
            self.answer_leakage,
            self.duplicate_risk,
        ]

    @property
    def duplicate_risk_score(self) -> float:
        return 1.0 - (self.duplicate_risk.score or 0.0)

    @property
    def context_alignment_mode(self) -> ContextAlignmentMode | None:
        return self.context_alignment.context_alignment_mode


QualityReport = QuestionQualityReport
QualityCheckResult = QualityDimensionResult


class QualityPolicyConfig(EvaluationModel):
    """Thresholds used to convert quality signals into a final decision."""

    minimum_dimension_score: float = Field(default=0.70, ge=0.0, le=1.0)
    minimum_judge_score: float = Field(default=0.75, ge=0.0, le=1.0)
    minimum_judge_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    secondary_judge_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    duplicate_rejection_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    duplicate_fuzzy_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
