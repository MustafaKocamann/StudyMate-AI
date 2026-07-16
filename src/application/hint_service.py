"""Provider-independent hint contracts for the learning experience."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.models.question_schemas import GeneratedQuestion


class HintRequest(BaseModel):
    """Validated request for a progressive learner hint."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    question: GeneratedQuestion
    level: int = Field(ge=1, le=3)


class HintProvider(Protocol):
    """Asynchronous hint source that must not intentionally reveal the answer."""

    async def get_hint(self, question: GeneratedQuestion, level: int) -> str: ...
