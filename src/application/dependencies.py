"""Application dependency construction for Study Buddy AI."""

from __future__ import annotations

from dataclasses import dataclass

from src.application.question_service import QualityGatedStudyQuestionService
from src.common.exceptions import ApplicationConfigurationError
from src.config.settings import Settings, settings
from src.evaluation.service import QuestionQualityEvaluator
from src.generator.regeneration import GenerationAttemptPolicy, QualityGatedQuestionGenerator
from src.llm.gemini_client import GeminiClient
from src.llm.groq_client import GroqClient
from src.llm.groq_question_generator import GroqQuestionGenerator
from src.llm.groq_question_judge import GroqQuestionJudge
from src.llm.gateway import LLMGateway
from src.llm.models import CompletionProfile, generation_profile_from_settings, judge_profile_from_settings


@dataclass(frozen=True)
class ApplicationContainer:
    """Constructed application graph shared by UI and tests."""

    settings: Settings
    llm_gateway: LLMGateway
    generation_profile: CompletionProfile
    judge_profile: CompletionProfile
    question_generator: GroqQuestionGenerator
    question_judge: GroqQuestionJudge | None
    quality_evaluator: QuestionQualityEvaluator
    quality_gated_generator: QualityGatedQuestionGenerator
    study_question_service: QualityGatedStudyQuestionService

    @property
    def groq_gateway(self) -> LLMGateway:
        """Compatibility alias for older callers; use ``llm_gateway``."""

        return self.llm_gateway


def build_application_container(
    *,
    app_settings: Settings | None = None,
    gateway: LLMGateway | None = None,
) -> ApplicationContainer:
    """Build the explicit production dependency graph.

    This factory is the only place that wires provider-backed question
    generation by default. It creates or accepts one low-level gateway, derives
    separate immutable completion profiles for generation and judging, and then
    composes prompt rendering, judging, deterministic checks, controlled
    regeneration, and the application service in order.

    SDK retries stay inside the selected gateway as transport behavior. Controlled
    regeneration stays in ``QualityGatedQuestionGenerator`` as educational
    quality behavior, so provider failures are not confused with rejected
    questions.
    """

    resolved_settings = app_settings or settings
    groq_settings = resolved_settings.groq
    gemini_settings = resolved_settings.gemini
    if gateway is not None:
        provider_settings = groq_settings
        llm_gateway = gateway
    elif gemini_settings.api_key is not None:
        provider_settings = gemini_settings
        llm_gateway = GeminiClient(gemini_settings)
    elif groq_settings.api_key is not None:
        provider_settings = groq_settings
        llm_gateway = GroqClient(groq_settings)
    else:
        raise ApplicationConfigurationError(
            "The AI question service is not configured. Set GEMINI_API_KEY or GROQ_API_KEY."
        )

    generation_profile = generation_profile_from_settings(provider_settings)
    judge_profile = judge_profile_from_settings(provider_settings)
    question_generator = GroqQuestionGenerator(
        gateway=llm_gateway,
        generation_profile=generation_profile,
    )
    question_judge = (
        GroqQuestionJudge(gateway=llm_gateway, judge_profile=judge_profile)
        if provider_settings.enable_quality_judge
        else None
    )
    quality_evaluator = QuestionQualityEvaluator(primary_judge=question_judge)
    quality_gated_generator = QualityGatedQuestionGenerator(
        generator=question_generator,
        quality_evaluator=quality_evaluator,
        attempt_policy=GenerationAttemptPolicy(
            maximum_attempts=provider_settings.generation_max_attempts,
        ),
    )
    study_question_service = QualityGatedStudyQuestionService(quality_gated_generator)
    return ApplicationContainer(
        settings=resolved_settings,
        llm_gateway=llm_gateway,
        generation_profile=generation_profile,
        judge_profile=judge_profile,
        question_generator=question_generator,
        question_judge=question_judge,
        quality_evaluator=quality_evaluator,
        quality_gated_generator=quality_gated_generator,
        study_question_service=study_question_service,
    )
