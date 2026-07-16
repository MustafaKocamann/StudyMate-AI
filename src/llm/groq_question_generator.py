"""Groq-backed question payload generator adapter."""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter, ValidationError

from src.common.exceptions import (
    QuestionGenerationPromptError,
    QuestionGenerationResponseValidationError,
)
from src.generator.regeneration import QuestionGenerationRequest
from src.generator.repair_prompts import (
    QuestionRepairFeedback,
    fill_blank_repair_prompt_template,
    mcq_repair_prompt_template,
)
from src.llm.gateway import LLMGateway
from src.llm.json_utils import decode_json_object
from src.llm.message_adapter import langchain_messages_to_chat_messages
from src.llm.models import CompletionProfile
from src.models.question_payloads import (
    GeneratedFillBlankPayload,
    GeneratedMCQPayload,
    GeneratedQuestionPayload,
)
from src.models.question_schemas import QuestionType
from src.prompts.question_prompts import fill_blank_prompt_template, mcq_prompt_template


class GroqQuestionGenerator:
    """Render versioned question prompts and validate Groq JSON payloads.

    This adapter owns prompt selection, LangChain-message adaptation, JSON
    decoding, and strict raw-payload validation. It deliberately does not map
    payloads into domain questions, evaluate quality, or retry. Controlled
    regeneration remains the orchestration boundary for educational failures.
    """

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        generation_profile: CompletionProfile,
    ) -> None:
        self.gateway = gateway
        self.generation_profile = generation_profile

    async def generate(
        self,
        request: QuestionGenerationRequest,
        *,
        previous_payload: GeneratedQuestionPayload | None = None,
        repair_feedback: QuestionRepairFeedback | None = None,
    ) -> GeneratedQuestionPayload:
        if previous_payload is None and repair_feedback is None:
            messages = self._render_generation_messages(request)
        else:
            messages = self._render_repair_messages(
                request,
                previous_payload=previous_payload,
                repair_feedback=repair_feedback,
            )

        result = await self.gateway.complete(messages=messages, profile=self.generation_profile)
        payload = decode_json_object(result.content)
        return _validate_payload_for_type(payload, request.question_type)

    def _render_generation_messages(self, request: QuestionGenerationRequest):
        template = (
            mcq_prompt_template
            if request.question_type is QuestionType.MCQ
            else fill_blank_prompt_template
        )
        try:
            rendered = template.format_messages(
                topic=request.topic,
                difficulty=request.difficulty.value,
                position=request.position,
                language=request.language,
            )
        except Exception as exc:
            raise QuestionGenerationPromptError("question generation prompt rendering failed") from exc
        return langchain_messages_to_chat_messages(rendered)

    def _render_repair_messages(
        self,
        request: QuestionGenerationRequest,
        *,
        previous_payload: GeneratedQuestionPayload | None,
        repair_feedback: QuestionRepairFeedback | None,
    ):
        if previous_payload is None or repair_feedback is None:
            raise QuestionGenerationPromptError(
                "repair generation requires previous_payload and repair_feedback"
            )
        template = (
            mcq_repair_prompt_template
            if request.question_type is QuestionType.MCQ
            else fill_blank_repair_prompt_template
        )
        target_schema = (
            GeneratedMCQPayload.model_json_schema()
            if request.question_type is QuestionType.MCQ
            else GeneratedFillBlankPayload.model_json_schema()
        )
        try:
            rendered = template.format_messages(
                original_generation_request=_safe_json(request.model_dump(mode="json")),
                previous_payload=_safe_json(previous_payload.model_dump(mode="json")),
                repair_feedback=_safe_json(repair_feedback.model_dump(mode="json")),
                target_output_schema=_safe_json(target_schema),
            )
        except Exception as exc:
            raise QuestionGenerationPromptError("question repair prompt rendering failed") from exc
        return langchain_messages_to_chat_messages(rendered)


def _validate_payload_for_type(
    payload: dict[str, object],
    question_type: QuestionType,
) -> GeneratedQuestionPayload:
    adapter: TypeAdapter[Any]
    if question_type is QuestionType.MCQ:
        adapter = TypeAdapter(GeneratedMCQPayload)
    else:
        adapter = TypeAdapter(GeneratedFillBlankPayload)
    try:
        return adapter.validate_python(payload)
    except ValidationError as exc:
        raise QuestionGenerationResponseValidationError(
            "generated question payload validation failed"
        ) from exc


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
