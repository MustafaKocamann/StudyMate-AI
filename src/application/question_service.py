"""Question-generation facade used by the Streamlit learning experience."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.exceptions import (
    QuestionRegenerationExhaustedError,
    QuestionSetGenerationError,
    StudyBuddyException,
)
from src.common.logger import get_logger
from src.evaluation.models import QUALITY_EVALUATION_VERSION, QuestionQualityReport
from src.generator.regeneration import (
    GenerationText,
    QualityGatedGenerationResult,
    QualityGatedQuestionGenerator,
    QuestionGenerationRequest,
)
from src.models.question_schemas import DifficultyLevel, GeneratedQuestion, QuestionSet, QuestionType
from src.models.study_session import StudyQuestionMode
from src.prompts.question_prompts import PROMPT_VERSION


logger = get_logger(__name__)


class QuestionSetGenerationRequest(BaseModel):
    """Validated application input for an ordered question set.

    This model is intentionally provider-neutral: it carries learner-facing
    study intent and sequencing requirements, not API keys, model IDs, retry
    settings, or SDK clients. Downstream generation can rely on non-empty
    study text, a bounded question count, and a deterministic mode value.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    topic: GenerationText
    source_content: GenerationText | None = None
    difficulty: DifficultyLevel
    language: GenerationText
    question_count: int = Field(ge=1, le=20)
    question_mode: StudyQuestionMode


class QuestionSetGenerationResult(BaseModel):
    """Application-safe result for accepted set generation.

    The result exposes only domain questions and public quality metadata. Raw
    prompts, provider JSON, judge internals, credentials, and answer text are
    kept out of this contract so UI and persistence code can consume it without
    crossing LLM infrastructure boundaries.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    question_set: QuestionSet
    generation_reports: list[QuestionQualityReport]
    total_attempts: int = Field(ge=1)
    repaired_question_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result_consistency(self) -> QuestionSetGenerationResult:
        question_count = len(self.question_set.questions)
        if len(self.generation_reports) != question_count:
            raise ValueError("generation_reports must match the generated question count")
        if self.total_attempts < question_count:
            raise ValueError("total_attempts cannot be less than the generated question count")
        if self.repaired_question_count > question_count:
            raise ValueError("repaired_question_count cannot exceed the generated question count")
        return self


class StudyQuestionService(Protocol):
    """Single application-facing question generation boundary.

    Streamlit and future delivery surfaces should depend on this protocol
    instead of prompt renderers, payload schemas, Groq clients, or regeneration
    internals. Implementations return only accepted domain questions or typed
    exceptions from the existing hierarchy.
    """

    async def generate_question(
        self,
        request: QuestionGenerationRequest,
        *,
        existing_questions: Sequence[GeneratedQuestion] = (),
    ) -> QualityGatedGenerationResult: ...

    async def generate_question_set(
        self,
        request: QuestionSetGenerationRequest,
    ) -> QuestionSetGenerationResult: ...

    async def generate_questions(
        self,
        *,
        topic: str,
        difficulty: DifficultyLevel,
        question_mode: StudyQuestionMode,
        count: int,
        language: str,
    ) -> QuestionSet: ...


class QualityGatedStudyQuestionService:
    """Coordinate accepted question generation for the application layer.

    The service owns application concerns: single-question delegation,
    deterministic set blueprints, one-based positions, duplicate-detection
    context propagation, safe result metadata, and structured lifecycle logs.
    It deliberately reuses the quality-gated generator for payload validation,
    domain mapping, judging, and controlled regeneration.
    """

    def __init__(self, generator: QualityGatedQuestionGenerator) -> None:
        self.generator = generator

    async def generate_question(
        self,
        request: QuestionGenerationRequest,
        *,
        existing_questions: Sequence[GeneratedQuestion] = (),
    ) -> QualityGatedGenerationResult:
        """Generate exactly one accepted domain question through the quality gate."""

        return await self._generate_question(
            request,
            existing_questions=existing_questions,
            log_failure=True,
        )

    async def _generate_question(
        self,
        request: QuestionGenerationRequest,
        *,
        existing_questions: Sequence[GeneratedQuestion],
        log_failure: bool,
    ) -> QualityGatedGenerationResult:
        """Internal single-question path with caller-controlled failure logging."""

        started_at = time.monotonic()
        logger.info(
            "question_generation_started",
            extra={
                "event": "question_generation_started",
                "question_type": request.question_type.value,
                "position": request.position,
                "requested_difficulty": request.difficulty.value,
                "language": request.language,
                "prompt_version": PROMPT_VERSION,
            },
        )
        try:
            result = await self.generator.generate(
                request,
                existing_questions=list(existing_questions),
            )
        except StudyBuddyException as exc:
            if log_failure:
                _log_question_generation_failed(request, started_at, exc)
            raise
        except Exception as exc:
            if log_failure:
                _log_question_generation_failed(request, started_at, exc)
            raise

        logger.info(
            "question_generation_succeeded",
            extra={
                "event": "question_generation_succeeded",
                "question_type": request.question_type.value,
                "position": request.position,
                "requested_difficulty": request.difficulty.value,
                "language": request.language,
                "attempts_used": result.attempts_used,
                "repaired": result.repaired,
                "quality_passed": result.quality_report.overall_passed,
                "failed_dimension_names": _failed_dimension_names(result.quality_report),
                "prompt_version": PROMPT_VERSION,
                "evaluation_version": result.quality_report.evaluation_version,
                "latency_ms": _elapsed_ms(started_at),
            },
        )
        return result

    async def generate_mcq(
        self,
        *,
        topic: str,
        difficulty: DifficultyLevel,
        language: str,
        source_content: str | None = None,
        position: int = 1,
        existing_questions: Sequence[GeneratedQuestion] = (),
    ) -> QualityGatedGenerationResult:
        """Thin compatibility wrapper for one MCQ generation."""

        return await self.generate_question(
            QuestionGenerationRequest(
                topic=topic,
                source_content=source_content,
                difficulty=difficulty,
                question_type=QuestionType.MCQ,
                position=position,
                language=language,
            ),
            existing_questions=existing_questions,
        )

    async def generate_fill_blank(
        self,
        *,
        topic: str,
        difficulty: DifficultyLevel,
        language: str,
        source_content: str | None = None,
        position: int = 1,
        existing_questions: Sequence[GeneratedQuestion] = (),
    ) -> QualityGatedGenerationResult:
        """Thin compatibility wrapper for one fill-in-the-blank generation."""

        return await self.generate_question(
            QuestionGenerationRequest(
                topic=topic,
                source_content=source_content,
                difficulty=difficulty,
                question_type=QuestionType.FILL_BLANK,
                position=position,
                language=language,
            ),
            existing_questions=existing_questions,
        )

    async def generate_question_set(
        self,
        request: QuestionSetGenerationRequest,
    ) -> QuestionSetGenerationResult:
        """Generate an ordered set from a deterministic sequential MVP blueprint.

        Question-set items are intentionally generated sequentially. Accepted
        prior questions feed duplicate detection for later positions, ordering
        stays deterministic, provider rate-limit behavior remains conservative,
        and controlled regeneration is simpler to reason about.
        """

        started_at = time.monotonic()
        logger.info(
            "question_set_generation_started",
            extra={
                "event": "question_set_generation_started",
                "question_count": request.question_count,
                "requested_difficulty": request.difficulty.value,
                "language": request.language,
                "prompt_version": PROMPT_VERSION,
            },
        )
        generated: list[GeneratedQuestion] = []
        reports: list[QuestionQualityReport] = []
        total_attempts = 0
        repaired_count = 0

        for position, question_type in enumerate(
            question_type_blueprint(request.question_mode, request.question_count),
            start=1,
        ):
            generation_request = QuestionGenerationRequest(
                topic=request.topic,
                source_content=request.source_content,
                difficulty=request.difficulty,
                question_type=question_type,
                position=position,
                language=request.language,
            )
            try:
                result = await self._generate_question(
                    generation_request,
                    existing_questions=generated,
                    log_failure=False,
                )
            except QuestionRegenerationExhaustedError as exc:
                _log_question_set_generation_failed(
                    request=request,
                    started_at=started_at,
                    position=position,
                    question_type=question_type,
                    total_attempts=total_attempts + exc.total_attempts,
                    exc=exc,
                )
                raise QuestionSetGenerationError(
                    failed_position=position,
                    question_type=question_type.value,
                    total_attempts=total_attempts + exc.total_attempts,
                    failed_dimensions=exc.failed_dimensions,
                    issue_codes=exc.issue_codes,
                ) from exc
            except StudyBuddyException as exc:
                _log_question_set_generation_failed(
                    request=request,
                    started_at=started_at,
                    position=position,
                    question_type=question_type,
                    total_attempts=total_attempts,
                    exc=exc,
                )
                raise
            except Exception as exc:
                _log_question_set_generation_failed(
                    request=request,
                    started_at=started_at,
                    position=position,
                    question_type=question_type,
                    total_attempts=total_attempts,
                    exc=exc,
                )
                raise

            generated.append(result.question)
            reports.append(result.quality_report)
            total_attempts += result.attempts_used
            repaired_count += int(result.repaired)

        question_set = QuestionSet(questions=generated)
        set_result = QuestionSetGenerationResult(
            question_set=question_set,
            generation_reports=reports,
            total_attempts=total_attempts,
            repaired_question_count=repaired_count,
        )
        logger.info(
            "question_set_generation_succeeded",
            extra={
                "event": "question_set_generation_succeeded",
                "question_count": request.question_count,
                "requested_difficulty": request.difficulty.value,
                "language": request.language,
                "total_attempts": total_attempts,
                "repaired_question_count": repaired_count,
                "quality_passed": True,
                "failed_dimension_names": [],
                "prompt_version": PROMPT_VERSION,
                "evaluation_version": QUALITY_EVALUATION_VERSION,
                "latency_ms": _elapsed_ms(started_at),
            },
        )
        return set_result

    async def generate_questions(
        self,
        *,
        topic: str,
        difficulty: DifficultyLevel,
        question_mode: StudyQuestionMode,
        count: int,
        language: str,
    ) -> QuestionSet:
        result = await self.generate_question_set(
            QuestionSetGenerationRequest(
                topic=topic,
                difficulty=difficulty,
                language=language,
                question_count=count,
                question_mode=question_mode,
            )
        )
        return result.question_set


def question_type_blueprint(mode: StudyQuestionMode, question_count: int) -> list[QuestionType]:
    """Return the deterministic question-type plan for one set request."""

    if question_count < 1 or question_count > 20:
        raise ValueError("question count must be between 1 and 20")
    return [_question_type_for_position(mode, position) for position in range(1, question_count + 1)]


def _question_type_for_position(mode: StudyQuestionMode, position: int) -> QuestionType:
    if mode == StudyQuestionMode.MCQ:
        return QuestionType.MCQ
    if mode == StudyQuestionMode.FILL_BLANK:
        return QuestionType.FILL_BLANK
    return QuestionType.MCQ if position % 2 == 1 else QuestionType.FILL_BLANK


def _failed_dimension_names(report: QuestionQualityReport) -> list[str]:
    return [check.dimension.value for check in report.checks if check.passed is False]


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def _log_question_generation_failed(
    request: QuestionGenerationRequest,
    started_at: float,
    exc: Exception,
) -> None:
    logger.warning(
        "question_generation_failed",
        extra={
            "event": "question_generation_failed",
            "question_type": request.question_type.value,
            "position": request.position,
            "requested_difficulty": request.difficulty.value,
            "language": request.language,
            "failed_dimension_names": _exception_failed_dimension_names(exc),
            "prompt_version": PROMPT_VERSION,
            "evaluation_version": QUALITY_EVALUATION_VERSION,
            "latency_ms": _elapsed_ms(started_at),
            "error_type": type(exc).__name__,
            "error_category": getattr(exc, "error_category", None),
        },
    )


def _log_question_set_generation_failed(
    *,
    request: QuestionSetGenerationRequest,
    started_at: float,
    position: int,
    question_type: QuestionType,
    total_attempts: int,
    exc: Exception,
) -> None:
    logger.warning(
        "question_set_generation_failed",
        extra={
            "event": "question_set_generation_failed",
            "position": position,
            "question_type": question_type.value,
            "question_count": request.question_count,
            "requested_difficulty": request.difficulty.value,
            "language": request.language,
            "total_attempts": total_attempts,
            "failed_dimension_names": _exception_failed_dimension_names(exc),
            "prompt_version": PROMPT_VERSION,
            "evaluation_version": QUALITY_EVALUATION_VERSION,
            "latency_ms": _elapsed_ms(started_at),
            "error_type": type(exc).__name__,
            "error_category": getattr(exc, "error_category", None),
        },
    )


def _exception_failed_dimension_names(exc: Exception) -> list[str]:
    failed_dimensions = getattr(exc, "failed_dimensions", [])
    return [dimension.value for dimension in failed_dimensions]
