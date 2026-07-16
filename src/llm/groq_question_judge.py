"""Groq-backed LLM-as-a-judge adapter."""

from __future__ import annotations

import json

from src.common.exceptions import LLMResponseParsingError, QuestionJudgeResponseValidationError
from src.evaluation.judge import validate_judge_evaluation
from src.evaluation.judge_prompts import question_quality_judge_prompt
from src.evaluation.models import LLMJudgeReport, QualityDimensionResult, QuestionEvaluationContext
from src.llm.gateway import LLMGateway
from src.llm.json_utils import decode_json_object
from src.llm.message_adapter import langchain_messages_to_chat_messages
from src.llm.models import CompletionProfile


class GroqQuestionJudge:
    """Render the quality-judge prompt and validate strict judge JSON.

    The judge adapter treats malformed JSON and schema-invalid judge output as
    technical evaluation errors. It returns only the public judge report fields;
    hidden reasoning or chain-of-thought is neither requested nor stored.
    """

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        judge_profile: CompletionProfile,
    ) -> None:
        self.gateway = gateway
        self.judge_profile = judge_profile

    async def evaluate(
        self,
        context: QuestionEvaluationContext,
        deterministic_findings: list[QualityDimensionResult],
    ) -> LLMJudgeReport:
        try:
            rendered = question_quality_judge_prompt.format_messages(
                topic=context.topic,
                source_content=context.source_content or "",
                question_json=_safe_json(context.question.model_dump(mode="json")),
                deterministic_findings=_safe_json(
                    [finding.model_dump(mode="json") for finding in deterministic_findings]
                ),
            )
        except Exception as exc:
            raise QuestionJudgeResponseValidationError("judge prompt rendering failed") from exc

        messages = langchain_messages_to_chat_messages(rendered)
        result = await self.gateway.complete(messages=messages, profile=self.judge_profile)
        try:
            payload = decode_json_object(result.content)
        except LLMResponseParsingError as exc:
            raise QuestionJudgeResponseValidationError("judge response parsing failed") from exc
        return validate_judge_evaluation(payload)


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
