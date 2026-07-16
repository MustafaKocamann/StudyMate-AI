"""Deterministic tests for provider-neutral LLM infrastructure."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.exceptions import (
    LLMEmptyResponseError,
    LLMMessageAdapterError,
    LLMModelUnavailableError,
    LLMResponseParsingError,
)
from src.config.settings import DEFAULT_GROQ_MODEL, GroqSettings
from src.llm.groq_client import GroqClient, _map_groq_exception
from src.llm.json_utils import decode_json_object
from src.llm.message_adapter import langchain_messages_to_chat_messages
from src.llm.models import ChatMessage, ChatRole, CompletionProfile, generation_profile_from_settings, judge_profile_from_settings


def test_groq_settings_defaults_judge_model_to_generation_model() -> None:
    settings = GroqSettings(api_key="secret")

    assert settings.generation_model == DEFAULT_GROQ_MODEL
    assert settings.judge_model == DEFAULT_GROQ_MODEL
    assert settings.api_key is not None
    assert "secret" not in repr(settings.api_key)


def test_completion_profiles_are_separate_immutable_objects() -> None:
    settings = GroqSettings(api_key="secret")

    generation = generation_profile_from_settings(settings)
    judge = judge_profile_from_settings(settings)

    assert generation is not judge
    assert generation.temperature == 0.7
    assert judge.temperature == 0.0
    with pytest.raises(Exception):
        generation.temperature = 1.0


def test_langchain_adapter_preserves_supported_text_messages() -> None:
    adapted = langchain_messages_to_chat_messages(
        [SystemMessage(content="system text"), HumanMessage(content="user text")]
    )

    assert adapted == [
        ChatMessage(role=ChatRole.SYSTEM, content="system text"),
        ChatMessage(role=ChatRole.USER, content="user text"),
    ]


def test_langchain_adapter_rejects_non_text_content() -> None:
    with pytest.raises(LLMMessageAdapterError):
        langchain_messages_to_chat_messages([HumanMessage(content=[{"type": "text", "text": "hi"}])])


def test_decode_json_object_accepts_only_object_root() -> None:
    assert decode_json_object('{"ok": true}') == {"ok": True}
    with pytest.raises(LLMResponseParsingError):
        decode_json_object("[1, 2]")
    with pytest.raises(LLMResponseParsingError):
        decode_json_object("not json")


class _FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


class _FakeChat:
    def __init__(self, response: object) -> None:
        self.completions = _FakeCompletions(response)


class _FakeSdkClient:
    def __init__(self, response: object) -> None:
        self.chat = _FakeChat(response)


class _Message:
    content = '{"type":"ok"}'


class _Choice:
    message = _Message()
    finish_reason = "stop"


class _Usage:
    prompt_tokens = 3
    completion_tokens = 4
    total_tokens = 7


class _Response:
    choices = [_Choice()]
    model = "returned-model"
    usage = _Usage()
    request_id = "req_123"


def test_groq_client_shapes_json_mode_request_and_extracts_safe_result() -> None:
    sdk = _FakeSdkClient(_Response())
    client = GroqClient(GroqSettings(api_key="secret"), sdk_client=sdk)
    profile = CompletionProfile(
        model="configured-model",
        temperature=0.7,
        max_completion_tokens=128,
        timeout_seconds=3.0,
        json_mode=True,
        reasoning_effort="none",
    )

    result = asyncio.run(async_complete(client, profile))

    kwargs = sdk.chat.completions.kwargs
    assert kwargs is not None
    assert kwargs["model"] == "configured-model"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["reasoning_effort"] == "none"
    assert result.content == '{"type":"ok"}'
    assert result.model == "returned-model"
    assert result.provider_request_id == "req_123"


async def async_complete(client: GroqClient, profile: CompletionProfile):
    return await client.complete(
        messages=[ChatMessage(role=ChatRole.USER, content="hello")],
        profile=profile,
    )


def test_groq_client_rejects_empty_successful_response() -> None:
    class EmptyResponse:
        choices: list[object] = []

    client = GroqClient(GroqSettings(api_key="secret"), sdk_client=_FakeSdkClient(EmptyResponse()))
    profile = CompletionProfile(
        model="configured-model",
        temperature=0.0,
        max_completion_tokens=1,
        timeout_seconds=1.0,
    )

    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(async_complete(client, profile))


def test_groq_model_rejection_maps_to_model_unavailable() -> None:
    class FakeBadRequest(Exception):
        status_code = 400
        body = {"error": {"message": "The model qwen/qwen3.6-27b does not exist or you do not have access."}}

    mapped = _map_groq_exception(FakeBadRequest(), model="qwen/qwen3.6-27b")

    assert isinstance(mapped, LLMModelUnavailableError)
    assert mapped.model == "qwen/qwen3.6-27b"
    assert mapped.provider == "groq"
