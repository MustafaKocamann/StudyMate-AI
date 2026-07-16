"""Provider-neutral LLM gateway protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.llm.models import ChatMessage, CompletionProfile, LLMCompletionResult


class LLMGateway(Protocol):
    """Small asynchronous completion gateway used by application services."""

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        profile: CompletionProfile,
    ) -> LLMCompletionResult: ...
