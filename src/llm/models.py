"""Provider-neutral LLM request and response contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(min_length=1, strict=True)]


class ChatRole(StrEnum):
    """Supported provider-neutral chat roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Provider-neutral chat message without SDK or LangChain metadata."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    role: ChatRole
    content: NonEmptyText


class CompletionProfile(BaseModel):
    """Immutable provider-neutral completion configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    model: NonEmptyText
    temperature: float = Field(ge=0.0, le=2.0)
    max_completion_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    json_mode: bool = False
    reasoning_effort: str | None = None


class LLMCompletionResult(BaseModel):
    """Safe provider-neutral completion result."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    content: NonEmptyText
    model: NonEmptyText
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None


def generation_profile_from_settings(settings: object) -> CompletionProfile:
    """Build an immutable generation profile from provider settings."""

    return CompletionProfile(
        model=settings.generation_model,
        temperature=settings.generation_temperature,
        max_completion_tokens=settings.generation_max_completion_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
        json_mode=True,
        reasoning_effort=settings.generation_reasoning_effort,
    )


def judge_profile_from_settings(settings: object) -> CompletionProfile:
    """Build an immutable judge profile from provider settings."""

    return CompletionProfile(
        model=settings.judge_model or settings.generation_model,
        temperature=settings.judge_temperature,
        max_completion_tokens=settings.judge_max_completion_tokens,
        timeout_seconds=settings.judge_timeout_seconds,
        json_mode=True,
        reasoning_effort=settings.judge_reasoning_effort,
    )
