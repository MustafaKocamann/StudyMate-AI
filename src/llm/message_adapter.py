"""Adapters from LangChain prompt messages to provider-neutral messages."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.common.exceptions import LLMMessageAdapterError
from src.llm.models import ChatMessage, ChatRole


def langchain_messages_to_chat_messages(messages: Sequence[BaseMessage]) -> list[ChatMessage]:
    """Convert supported LangChain text messages into strict chat messages."""

    adapted: list[ChatMessage] = []
    for message in messages:
        role = _role_for_message(message)
        content = message.content
        if not isinstance(content, str):
            raise LLMMessageAdapterError(
                f"unsupported LangChain message content type: {type(content).__name__}"
            )
        if not content.strip():
            raise LLMMessageAdapterError("LangChain message content must be non-empty")
        adapted.append(ChatMessage(role=role, content=content))
    return adapted


def _role_for_message(message: BaseMessage) -> ChatRole:
    if isinstance(message, SystemMessage):
        return ChatRole.SYSTEM
    if isinstance(message, HumanMessage):
        return ChatRole.USER
    if isinstance(message, AIMessage):
        return ChatRole.ASSISTANT
    raise LLMMessageAdapterError(f"unsupported LangChain message type: {type(message).__name__}")
