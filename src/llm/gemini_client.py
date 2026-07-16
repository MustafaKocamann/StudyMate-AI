"""Low-latency Gemini implementation of the provider-neutral LLM gateway."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

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
from src.config.settings import GeminiSettings
from src.llm.gateway import LLMGateway
from src.llm.models import ChatMessage, ChatRole, CompletionProfile, LLMCompletionResult


PROVIDER_NAME = "gemini"
logger = get_logger(__name__)


class GeminiClient(LLMGateway):
    """Call Gemini with structured JSON output and bounded retry latency."""

    def __init__(
        self,
        settings: GeminiSettings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._api_key = settings.require_api_key().get_secret_value()
        self._client = http_client or httpx.AsyncClient(base_url=settings.base_url.rstrip("/"))

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        profile: CompletionProfile,
    ) -> LLMCompletionResult:
        started_at = time.monotonic()
        payload = _request_payload(
            messages,
            profile,
            thinking_budget=self._settings.thinking_budget,
        )
        response = await self._post_with_bounded_retry(
            f"/models/{profile.model}:generateContent",
            payload=payload,
            profile=profile,
        )
        result = _completion_result(response, fallback_model=profile.model)
        logger.info(
            "llm_completion_success",
            extra={
                "event": "llm_completion_success",
                "provider": PROVIDER_NAME,
                "model": result.model,
                "operation": "models.generateContent",
                "json_mode": profile.json_mode,
                "temperature": profile.temperature,
                "latency_ms": int((time.monotonic() - started_at) * 1000),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "finish_reason": result.finish_reason,
                "provider_request_id": result.provider_request_id,
            },
        )
        return result

    async def _post_with_bounded_retry(
        self,
        path: str,
        *,
        payload: dict[str, Any],
        profile: CompletionProfile,
    ) -> httpx.Response:
        for attempt in range(self._settings.sdk_max_retries + 1):
            try:
                response = await self._client.post(
                    path,
                    headers={"x-goog-api-key": self._api_key},
                    json=payload,
                    timeout=profile.timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(
                    "Gemini provider request timed out",
                    provider=PROVIDER_NAME,
                    model=profile.model,
                    error_category="timeout",
                ) from exc
            except httpx.RequestError as exc:
                raise LLMConnectionError(
                    "Gemini provider could not be reached",
                    provider=PROVIDER_NAME,
                    model=profile.model,
                    error_category="connection",
                ) from exc

            if response.status_code != 429 or attempt >= self._settings.sdk_max_retries:
                break
            await asyncio.sleep(_bounded_retry_delay(response, self._settings.retry_max_delay_seconds))

        if response.is_error:
            raise _map_http_error(response, model=profile.model)
        return response


def _request_payload(
    messages: Sequence[ChatMessage],
    profile: CompletionProfile,
    *,
    thinking_budget: int,
) -> dict[str, Any]:
    system_text = "\n\n".join(
        message.content for message in messages if message.role is ChatRole.SYSTEM
    )
    contents = [
        {
            "role": "model" if message.role is ChatRole.ASSISTANT else "user",
            "parts": [{"text": message.content}],
        }
        for message in messages
        if message.role is not ChatRole.SYSTEM
    ]
    generation_config: dict[str, Any] = {
        "temperature": profile.temperature,
        "maxOutputTokens": profile.max_completion_tokens,
        "thinkingConfig": {"thinkingBudget": thinking_budget},
    }
    if profile.json_mode:
        generation_config["responseMimeType"] = "application/json"
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    return payload


def _completion_result(response: httpx.Response, *, fallback_model: str) -> LLMCompletionResult:
    try:
        payload = response.json()
        candidate = payload["candidates"][0]
        parts = candidate["content"]["parts"]
        content = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, Mapping)
        ).strip()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMUnexpectedProviderError(
            "Gemini response could not be interpreted",
            provider=PROVIDER_NAME,
            model=fallback_model,
            error_category="response_shape",
            provider_request_id=_request_id(response),
        ) from exc
    if not content:
        raise LLMEmptyResponseError(
            "Gemini returned an empty completion",
            provider=PROVIDER_NAME,
            model=fallback_model,
            error_category="empty_response",
            provider_request_id=_request_id(response),
        )
    usage = payload.get("usageMetadata", {})
    return LLMCompletionResult(
        content=content,
        model=payload.get("modelVersion") or fallback_model,
        finish_reason=candidate.get("finishReason"),
        prompt_tokens=usage.get("promptTokenCount"),
        completion_tokens=usage.get("candidatesTokenCount"),
        total_tokens=usage.get("totalTokenCount"),
        provider_request_id=_request_id(response),
    )


def _bounded_retry_delay(response: httpx.Response, maximum: float) -> float:
    retry_after = response.headers.get("retry-after")
    try:
        requested = float(retry_after) if retry_after is not None else 0.5
    except ValueError:
        requested = 0.5
    return min(maximum, max(0.0, requested))


def _request_id(response: httpx.Response) -> str | None:
    return response.headers.get("x-request-id") or response.headers.get("x-guploader-uploadid")


def _map_http_error(response: httpx.Response, *, model: str) -> LLMProviderError:
    context = {
        "provider": PROVIDER_NAME,
        "model": model,
        "provider_request_id": _request_id(response),
    }
    if response.status_code in {401, 403}:
        return LLMAuthenticationError(
            "Gemini authentication or permission failed",
            error_category="authentication",
            **context,
        )
    if response.status_code == 429:
        return LLMRateLimitError(
            "Gemini rate limit was reached",
            error_category="rate_limit",
            **context,
        )
    if response.status_code == 404:
        return LLMModelUnavailableError(
            "Gemini model is unavailable",
            error_category="model_unavailable",
            **context,
        )
    if 400 <= response.status_code < 500:
        return LLMInvalidRequestError(
            "Gemini rejected the request",
            error_category="invalid_request",
            **context,
        )
    return LLMUnexpectedProviderError(
        "Gemini provider request failed",
        error_category="unexpected_provider_error",
        **context,
    )
