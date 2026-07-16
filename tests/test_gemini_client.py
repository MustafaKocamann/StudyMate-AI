"""Tests for the low-latency Gemini gateway."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.common.exceptions import LLMRateLimitError
from src.config.settings import DEFAULT_GEMINI_MODEL, GeminiSettings
from src.llm.gemini_client import GeminiClient
from src.llm.models import ChatMessage, ChatRole, CompletionProfile


def profile() -> CompletionProfile:
    return CompletionProfile(
        model=DEFAULT_GEMINI_MODEL,
        temperature=0.35,
        max_completion_tokens=768,
        timeout_seconds=20,
        json_mode=True,
    )


def messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=ChatRole.SYSTEM, content="Return one question as JSON."),
        ChatMessage(role=ChatRole.USER, content="Topic: supervised learning"),
    ]


def test_gemini_settings_are_low_latency_by_default() -> None:
    settings = GeminiSettings(api_key="secret")

    assert settings.generation_model == "gemini-2.5-flash"
    assert settings.generation_max_completion_tokens == 768
    assert settings.thinking_budget == 0
    assert settings.sdk_max_retries == 1
    assert not settings.enable_quality_judge
    assert "secret" not in repr(settings.api_key)


def test_gemini_client_requests_structured_json_without_key_in_url() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["api_key"] = request.headers.get("x-goog-api-key")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": '{"type":"mcq"}'}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 28,
                },
                "modelVersion": "gemini-2.5-flash",
            },
        )

    async def run_test():
        http_client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            transport=httpx.MockTransport(handler),
        )
        client = GeminiClient(
            GeminiSettings(api_key="secret", sdk_max_retries=0),
            http_client=http_client,
        )
        try:
            return await client.complete(messages=messages(), profile=profile())
        finally:
            await http_client.aclose()

    result = asyncio.run(run_test())
    payload = captured["payload"]

    assert "secret" not in captured["url"]
    assert captured["api_key"] == "secret"
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert payload["generationConfig"]["maxOutputTokens"] == 768
    assert result.total_tokens == 28


def test_gemini_client_maps_rate_limit_without_long_retry() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "quota"}})

    async def run_test() -> None:
        http_client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            transport=httpx.MockTransport(handler),
        )
        client = GeminiClient(
            GeminiSettings(api_key="secret", sdk_max_retries=0),
            http_client=http_client,
        )
        try:
            await client.complete(messages=messages(), profile=profile())
        finally:
            await http_client.aclose()

    with pytest.raises(LLMRateLimitError):
        asyncio.run(run_test())
