"""Centralized application settings."""

from __future__ import annotations

from functools import cached_property

from dotenv import load_dotenv
from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_GROQ_MODEL = "qwen/qwen3.6-27b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

load_dotenv()


class GroqSettings(BaseSettings):
    """Groq provider configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="GROQ_",
        extra="ignore",
        validate_assignment=True,
    )

    api_key: SecretStr | None = Field(default=None)
    base_url: str | None = Field(default=None)

    generation_model: str = Field(default=DEFAULT_GROQ_MODEL, min_length=1)
    generation_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    generation_max_completion_tokens: int = Field(default=2048, gt=0)
    generation_timeout_seconds: float = Field(default=30.0, gt=0)
    generation_reasoning_effort: str | None = Field(default="none")

    judge_model: str | None = Field(default=None)
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    judge_max_completion_tokens: int = Field(default=2048, gt=0)
    judge_timeout_seconds: float = Field(default=30.0, gt=0)
    judge_reasoning_effort: str | None = Field(default="none")

    sdk_max_retries: int = Field(default=2, ge=0, le=10)
    generation_max_attempts: int = Field(default=3, ge=1, le=10)
    enable_quality_judge: bool = True

    @field_validator("api_key")
    @classmethod
    def empty_api_key_as_missing(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            return None
        return value

    @field_validator(
        "base_url",
        "generation_reasoning_effort",
        "judge_model",
        "judge_reasoning_effort",
        mode="before",
    )
    @classmethod
    def empty_string_as_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def default_judge_model(self) -> GroqSettings:
        if self.judge_model is None:
            self.judge_model = self.generation_model
        return self

    def require_api_key(self) -> SecretStr:
        if self.api_key is None:
            raise ValueError("GROQ_API_KEY must be configured before constructing Groq services")
        return self.api_key


class GeminiSettings(BaseSettings):
    """Gemini configuration optimized for low-latency interactive practice."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        extra="ignore",
        validate_assignment=True,
    )

    api_key: SecretStr | None = Field(default=None)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    generation_model: str = Field(default=DEFAULT_GEMINI_MODEL, min_length=1)
    generation_temperature: float = Field(default=0.35, ge=0.0, le=2.0)
    generation_max_completion_tokens: int = Field(default=768, gt=0)
    generation_timeout_seconds: float = Field(default=20.0, gt=0)
    generation_reasoning_effort: str | None = None
    generation_max_attempts: int = Field(default=2, ge=1, le=3)

    judge_model: str | None = None
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    judge_max_completion_tokens: int = Field(default=512, gt=0)
    judge_timeout_seconds: float = Field(default=20.0, gt=0)
    judge_reasoning_effort: str | None = None
    enable_quality_judge: bool = False

    sdk_max_retries: int = Field(default=1, ge=0, le=2)
    retry_max_delay_seconds: float = Field(default=2.0, ge=0.0, le=5.0)
    thinking_budget: int = Field(default=0, ge=0, le=24_576)

    @field_validator("api_key")
    @classmethod
    def empty_api_key_as_missing(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            return None
        return value

    @model_validator(mode="after")
    def default_judge_model(self) -> GeminiSettings:
        if self.judge_model is None:
            self.judge_model = self.generation_model
        return self

    def require_api_key(self) -> SecretStr:
        if self.api_key is None:
            raise ValueError("GEMINI_API_KEY must be configured before constructing Gemini services")
        return self.api_key


class Settings(BaseSettings):
    """Root settings object.

    Compatibility properties preserve older code paths while new LLM modules use
    the grouped ``groq`` settings.
    """

    model_config = SettingsConfigDict(extra="ignore", validate_assignment=True)

    @cached_property
    def groq(self) -> GroqSettings:
        return GroqSettings()

    @cached_property
    def gemini(self) -> GeminiSettings:
        return GeminiSettings()

    @property
    def GROQ_API_KEY(self) -> SecretStr | None:
        return self.groq.api_key

    @property
    def MODEL_NAME(self) -> str:
        return self.groq.generation_model

    @property
    def TEMPERATURE(self) -> float:
        return self.groq.generation_temperature

    @property
    def MAX_RETRIES(self) -> int:
        return self.groq.sdk_max_retries


try:
    settings = Settings()
except ValidationError:
    raise
