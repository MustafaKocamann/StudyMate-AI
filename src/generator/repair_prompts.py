"""Repair feedback generation for controlled question regeneration."""

from __future__ import annotations

from typing import Final

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from src.evaluation.models import EvaluationModel, QualityDimension, QualityDimensionResult, QualityIssue, QualityReport


QUESTION_REPAIR_PROMPT_VERSION: Final[str] = "question-repair-v1"


QUESTION_REPAIR_SYSTEM_PROMPT: Final[str] = """You are a controlled question-repair component for Study Buddy AI.

Produce a revised generated-question JSON payload that fixes all listed failed quality dimensions.
Return exactly one valid JSON object and nothing else.
Do not include Markdown, code fences, analysis, commentary, hidden reasoning, or a UUID.
Do not repeat the failed question unchanged.
Preserve the requested topic, language, difficulty, question type, and position.
Use the previous payload only as context for repair; do not copy defects forward.
Do not include raw chain-of-thought, hidden judge prompts, internal policy text, secret configuration, or unrelated passed-dimension details."""

MCQ_REPAIR_USER_PROMPT: Final[str] = """Repair the previous multiple-choice question.

<original_generation_request>
{original_generation_request}
</original_generation_request>

<previous_validated_payload>
{previous_payload}
</previous_validated_payload>

<concise_repair_feedback>
{repair_feedback}
</concise_repair_feedback>

<target_output_schema>
{target_output_schema}
</target_output_schema>

MCQ repair requirements:
- Return fields only for the MCQ payload schema.
- Use type "mcq".
- Keep exactly four option objects with IDs A, B, C, and D.
- Keep correct_option_id as an option ID, not answer text.
- Improve answer validity, distractor quality, explanation quality, difficulty alignment, context alignment, and answer leakage issues named in the feedback."""

FILL_BLANK_REPAIR_USER_PROMPT: Final[str] = """Repair the previous fill-in-the-blank question.

<original_generation_request>
{original_generation_request}
</original_generation_request>

<previous_validated_payload>
{previous_payload}
</previous_validated_payload>

<concise_repair_feedback>
{repair_feedback}
</concise_repair_feedback>

<target_output_schema>
{target_output_schema}
</target_output_schema>

Fill-blank repair requirements:
- Return fields only for the fill_blank payload schema.
- Use type "fill_blank".
- Include exactly one standalone ___ placeholder.
- Keep answer as the missing word or phrase only.
- Improve answer validity, explanation quality, difficulty alignment, context alignment, and answer leakage issues named in the feedback."""

mcq_repair_prompt_template: Final[ChatPromptTemplate] = ChatPromptTemplate.from_messages(
    [
        ("system", QUESTION_REPAIR_SYSTEM_PROMPT),
        ("human", MCQ_REPAIR_USER_PROMPT),
    ]
)
"""Chat prompt for controlled MCQ repair generation."""

fill_blank_repair_prompt_template: Final[ChatPromptTemplate] = ChatPromptTemplate.from_messages(
    [
        ("system", QUESTION_REPAIR_SYSTEM_PROMPT),
        ("human", FILL_BLANK_REPAIR_USER_PROMPT),
    ]
)
"""Chat prompt for controlled fill-blank repair generation."""


class QuestionRepairFeedback(EvaluationModel):
    """Concise structured feedback for controlled regeneration."""

    failed_dimensions: list[QualityDimension] = Field(default_factory=list)
    issues: list[QualityIssue] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)


def build_repair_feedback(checks: list[QualityDimensionResult]) -> list[str]:
    """Create concise regeneration feedback from failed or risky checks."""

    feedback = [
        f"{check.dimension.value}: {check.reason}"
        for check in checks
        if check.passed is False
    ]
    return feedback or ["Regenerate the question with stronger educational quality."]


def build_question_repair_feedback(report: QualityReport) -> QuestionRepairFeedback:
    """Build strict repair feedback from failed applicable dimensions."""

    failed_checks = [check for check in report.checks if check.passed is False]
    issues = _deduplicate_issues([issue for check in failed_checks for issue in check.issues])
    instructions = [
        _instruction_for_issue(issue)
        for issue in issues
    ]
    return QuestionRepairFeedback(
        failed_dimensions=[check.dimension for check in failed_checks],
        issues=issues,
        revision_instructions=instructions or ["Revise the question to address the failed quality dimensions."],
    )


def _deduplicate_issues(issues: list[QualityIssue]) -> list[QualityIssue]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduplicated: list[QualityIssue] = []
    for issue in issues:
        key = (issue.code, tuple(issue.affected_option_ids))
        if key not in seen:
            seen.add(key)
            deduplicated.append(issue)
    return deduplicated


def _instruction_for_issue(issue: QualityIssue) -> str:
    instructions = {
        "irrelevant_distractor": "Replace unrelated distractors with plausible misconceptions from the same concept family.",
        "obviously_wrong_distractor": "Replace trivially eliminable distractors with fair, relevant alternatives.",
        "duplicate_distractor_meaning": "Make each option semantically distinct.",
        "difficulty_too_low": "Increase conceptual depth to match the requested difficulty.",
        "difficulty_too_high": "Reduce reasoning complexity to match the requested difficulty.",
        "answer_revealed_in_question": "Rewrite the question so it does not reveal the answer.",
        "weak_explanation": "Explain why the answer is correct using the underlying concept.",
        "answer_only_explanation": "Expand the explanation beyond simply naming the answer.",
        "topic_mismatch": "Refocus the question on the requested topic.",
        "unsupported_by_source": "Align the question and explanation with the supplied source content.",
        "exact_duplicate_question": "Generate a question that assesses a different angle of the topic.",
        "high_similarity_question": "Make the question meaningfully distinct from earlier generated questions.",
    }
    return instructions.get(issue.code, issue.message)
