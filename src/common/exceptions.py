"""Application exception hierarchy for technical failures."""

from __future__ import annotations

from src.evaluation.models import QualityDimension


class StudyBuddyException(Exception):
    """Base exception for Study Buddy AI technical failures."""


class ApplicationConfigurationError(StudyBuddyException):
    """Raised when required runtime configuration is missing or invalid."""


class LLMProviderError(StudyBuddyException):
    """Base exception for provider-level LLM failures."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        error_category: str | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.error_category = error_category
        self.provider_request_id = provider_request_id
        super().__init__(message)


class LLMAuthenticationError(LLMProviderError):
    """Raised when provider authentication or authorization fails."""


class LLMRateLimitError(LLMProviderError):
    """Raised when the provider rejects a request due to rate limits."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM provider request times out."""


class LLMConnectionError(LLMProviderError):
    """Raised when the provider cannot be reached."""


class LLMInvalidRequestError(LLMProviderError):
    """Raised when the provider rejects request configuration or payload."""


class LLMModelUnavailableError(LLMInvalidRequestError):
    """Raised when the configured provider model is invalid or unavailable."""


class LLMEmptyResponseError(LLMProviderError):
    """Raised when a successful provider response has no usable content."""


class LLMUnexpectedProviderError(LLMProviderError):
    """Raised for unrecognized provider SDK failures."""


class LLMResponseParsingError(StudyBuddyException):
    """Raised when provider text cannot be parsed as the expected JSON object."""

    def __init__(
        self,
        message: str,
        *,
        error_category: str | None = None,
    ) -> None:
        self.error_category = error_category
        super().__init__(message)


class LLMMessageAdapterError(StudyBuddyException):
    """Raised when prompt messages cannot be adapted to provider-neutral chat messages."""


class QuestionGenerationError(StudyBuddyException):
    """Base exception for technical question-generation adapter failures."""


class QuestionGenerationPromptError(QuestionGenerationError):
    """Raised when a generation or repair prompt cannot be rendered."""


class QuestionGenerationResponseValidationError(QuestionGenerationError):
    """Raised when generated JSON does not match the requested payload schema."""


class QuestionEvaluationError(StudyBuddyException):
    """Base exception for question quality evaluation technical failures."""


class QuestionJudgeError(QuestionEvaluationError):
    """Raised when a judge cannot complete evaluation for technical reasons."""


class QuestionJudgeResponseValidationError(QuestionJudgeError):
    """Raised when parsed judge output does not match the judge schema."""


class QuestionRegenerationExhaustedError(QuestionEvaluationError):
    """Raised when controlled regeneration exhausts configured attempts."""

    def __init__(
        self,
        *,
        total_attempts: int,
        failed_dimensions: list[QualityDimension],
        issue_codes: list[str],
        final_report_id: str | None = None,
    ) -> None:
        self.total_attempts = total_attempts
        self.failed_dimensions = failed_dimensions
        self.issue_codes = issue_codes
        self.final_report_id = final_report_id
        super().__init__(
            "question regeneration exhausted after "
            f"{total_attempts} attempts; failed_dimensions="
            f"{[dimension.value for dimension in failed_dimensions]}; issue_codes={issue_codes}"
        )


class QuestionSetGenerationError(QuestionEvaluationError):
    """Raised when fail-closed QuestionSet generation stops before completion."""

    def __init__(
        self,
        *,
        failed_position: int,
        question_type: str,
        total_attempts: int,
        failed_dimensions: list[QualityDimension],
        issue_codes: list[str],
    ) -> None:
        self.failed_position = failed_position
        self.question_type = question_type
        self.total_attempts = total_attempts
        self.failed_dimensions = failed_dimensions
        self.issue_codes = issue_codes
        super().__init__(
            "question set generation failed closed at "
            f"position={failed_position}; question_type={question_type}; "
            f"total_attempts={total_attempts}; failed_dimensions="
            f"{[dimension.value for dimension in failed_dimensions]}; issue_codes={issue_codes}"
        )
