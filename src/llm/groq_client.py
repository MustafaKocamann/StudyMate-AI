"""Low-level asynchronous Groq chat-completions client."""

from __future__ import annotations

import inspect
import time
from collections.abc import Sequence
from typing import Any

from src.common.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMEmptyResponseError,
    LLMInvalidRequestError,
    LLMModelUnavailableError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnexpectedProviderError,
)
from src.common.logger import get_logger
from src.config.settings import GroqSettings
from src.llm.gateway import LLMGateway
from src.llm.models import ChatMessage, CompletionProfile, LLMCompletionResult


PROVIDER_NAME = "groq"
logger = get_logger(__name__)


class GroqClient(LLMGateway):
    """Provider-neutral gateway backed by the official Groq AsyncGroq SDK."""

    def __init__(
        self,
        settings: GroqSettings,
        *,
        sdk_client: object | None = None,
    ) -> None:
        self._settings = settings
        if sdk_client is not None:
            self._client = sdk_client
            return

        api_key = settings.require_api_key().get_secret_value()
        from groq import AsyncGroq

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": settings.sdk_max_retries,
        }
        if settings.base_url is not None:
            kwargs["base_url"] = settings.base_url
        self._client = AsyncGroq(**_supported_kwargs(AsyncGroq, kwargs))

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        profile: CompletionProfile,
    ) -> LLMCompletionResult:
        started_at = time.monotonic()
        kwargs = {
            "model": profile.model,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ],
            "temperature": profile.temperature,
            "max_completion_tokens": profile.max_completion_tokens,
            "timeout": profile.timeout_seconds,
        }
        if profile.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if profile.reasoning_effort is not None:
            kwargs["reasoning_effort"] = profile.reasoning_effort

        create = self._client.chat.completions.create
        kwargs = _supported_kwargs(create, kwargs)
        try:
            response = await create(**kwargs)
        except Exception as exc:
            raise _map_groq_exception(exc, model=profile.model) from exc

        try:
            result = _completion_result_from_response(response, fallback_model=profile.model)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMUnexpectedProviderError(
                "Groq response could not be interpreted",
                provider=PROVIDER_NAME,
                model=profile.model,
                error_category="response_shape",
                provider_request_id=_provider_request_id(response),
            ) from exc

        latency_ms = int((time.monotonic() - started_at) * 1000)
        logger.info(
            "llm_completion_success",
            extra={
                "event": "llm_completion_success",
                "provider": PROVIDER_NAME,
                "model": result.model,
                "operation": "chat.completions.create",
                "json_mode": profile.json_mode,
                "temperature": profile.temperature,
                "latency_ms": latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "finish_reason": result.finish_reason,
                "provider_request_id": result.provider_request_id,
            },
        )
        return result


def _supported_kwargs(callable_obj: object, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _completion_result_from_response(response: object, *, fallback_model: str) -> LLMCompletionResult:
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMEmptyResponseError(
            "Groq returned no completion choices",
            provider=PROVIDER_NAME,
            model=fallback_model,
            error_category="empty_response",
            provider_request_id=_provider_request_id(response),
        )
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise LLMEmptyResponseError(
            "Groq returned an empty completion message",
            provider=PROVIDER_NAME,
            model=fallback_model,
            error_category="empty_response",
            provider_request_id=_provider_request_id(response),
        )

    usage = getattr(response, "usage", None)
    return LLMCompletionResult(
        content=content,
        model=getattr(response, "model", None) or fallback_model,
        finish_reason=getattr(first_choice, "finish_reason", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        provider_request_id=_provider_request_id(response),
    )


def _provider_request_id(value: object) -> str | None:
    for attribute in ("request_id", "_request_id", "id"):
        request_id = getattr(value, attribute, None)
        if isinstance(request_id, str) and request_id:
            return request_id
    return None


def _map_groq_exception(exc: Exception, *, model: str) -> LLMProviderError:
    name = type(exc).__name__
    status_code = getattr(exc, "status_code", None)
    request_id = _provider_request_id(exc)
    safe_message = f"Groq provider request failed: {name}"
    context = {
        "provider": PROVIDER_NAME,
        "model": model,
        "provider_request_id": request_id,
    }

    body_text = _safe_error_text(exc).lower()
    if status_code in {401, 403} or name in {"AuthenticationError", "PermissionDeniedError"}:
        return LLMAuthenticationError(safe_message, error_category="authentication", **context)
    if status_code == 429 or name == "RateLimitError":
        return LLMRateLimitError(safe_message, error_category="rate_limit", **context)
    if name in {"APITimeoutError", "TimeoutException"}:
        return LLMTimeoutError(safe_message, error_category="timeout", **context)
    if name in {"APIConnectionError", "ConnectError", "ConnectionError"}:
        return LLMConnectionError(safe_message, error_category="connection", **context)
    if status_code == 400 and _looks_like_model_unavailable(body_text):
        return LLMModelUnavailableError(safe_message, error_category="model_unavailable", **context)
    if status_code in {404, 410} and (name == "NotFoundError" or _looks_like_model_unavailable(body_text)):
        return LLMModelUnavailableError(safe_message, error_category="model_unavailable", **context)
    if status_code is not None and 400 <= int(status_code) < 500:
        return LLMInvalidRequestError(safe_message, error_category="invalid_request", **context)
    return LLMUnexpectedProviderError(safe_message, error_category="unexpected_provider_error", **context)


def _looks_like_model_unavailable(text: str) -> bool:
    model_markers = ("model", "qwen", "does not exist", "not found", "unavailable", "decommissioned", "retired")
    access_markers = ("not have access", "unauthorized", "permission", "invalid")
    return any(marker in text for marker in model_markers) and (
        any(marker in text for marker in access_markers)
        or any(marker in text for marker in ("does not exist", "not found", "unavailable", "decommissioned", "retired"))
    )


def _safe_error_text(exc: Exception) -> str:
    for attribute in ("message", "body", "response"):
        value = getattr(exc, attribute, None)
        if isinstance(value, str):
            return value[:500]
        if isinstance(value, dict):
            return str(value)[:500]
    return str(exc)[:500]
