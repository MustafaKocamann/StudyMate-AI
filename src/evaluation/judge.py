"""Provider-independent LLM-as-a-judge contracts and reconciliation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import ValidationError

from src.common.exceptions import QuestionJudgeResponseValidationError
from src.evaluation.models import (
    LLMJudgeReport,
    QualityDimension,
    QualityDimensionResult,
    QualityStatus,
    QuestionEvaluationContext,
)


class QuestionJudge(Protocol):
    """Asynchronous provider-independent question judge."""

    async def evaluate(
        self,
        context: QuestionEvaluationContext,
        deterministic_findings: list[QualityDimensionResult],
    ) -> LLMJudgeReport: ...


def validate_judge_evaluation(payload: Mapping[str, Any]) -> LLMJudgeReport:
    """Validate parsed judge JSON against the judge evaluation contract."""

    try:
        return LLMJudgeReport.model_validate(_normalize_issue_strings(payload))
    except ValidationError as exc:
        raise QuestionJudgeResponseValidationError("judge response validation failed") from exc


def _normalize_issue_strings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a common provider deviation without weakening the contract."""

    normalized = dict(payload)
    issue_codes = {
        "answer_validity": "incorrect_declared_answer",
        "distractor_quality": "obviously_wrong_distractor",
        "explanation_quality": "weak_explanation",
        "difficulty_alignment": "difficulty_mismatch",
        "context_alignment": "topic_mismatch",
        "answer_leakage": "answer_revealed_in_question",
    }
    for field_name, issue_code in issue_codes.items():
        raw_result = normalized.get(field_name)
        if not isinstance(raw_result, Mapping):
            continue
        result = dict(raw_result)
        raw_issues = result.get("issues")
        if isinstance(raw_issues, list):
            result["issues"] = [
                {
                    "code": issue_code,
                    "message": issue[:1_000],
                    "affected_option_ids": [],
                }
                if isinstance(issue, str) and issue.strip()
                else issue
                for issue in raw_issues
            ]
        normalized[field_name] = result
    return normalized


def requires_secondary_judge(report: LLMJudgeReport) -> bool:
    """Return whether primary judge output contains critical uncertainty."""

    if report.requires_secondary_review:
        return True
    critical_codes = {
        "multiple_plausible_correct_answers",
        "version_context_missing",
        "incorrect_declared_answer",
        "insufficient_question_context",
        "unsupported_by_source",
    }
    return any(
        issue.code in critical_codes
        for result in report.checks
        for issue in result.issues
    )


def reconcile_judge_reports(
    primary_report: LLMJudgeReport,
    secondary_report: LLMJudgeReport | None,
) -> LLMJudgeReport:
    """Reconcile primary and optional secondary judge reports.

    When no secondary report is available for unresolved critical ambiguity,
    fail closed by returning the primary report with its answer-validity result
    marked failed. If judges disagree on answer validity, the returned report
    also fails closed without exposing hidden reasoning.
    """

    if not requires_secondary_judge(primary_report):
        return primary_report

    if secondary_report is None:
        return _with_failed_answer_validity(
            primary_report,
            reason="Primary judge requested secondary review but no secondary judge is configured.",
            issue_code="insufficient_question_context",
        )

    if primary_report.answer_validity.passed != secondary_report.answer_validity.passed:
        return _with_failed_answer_validity(
            primary_report,
            reason="Primary and secondary judges disagree on answer validity.",
            issue_code="incorrect_declared_answer",
        )

    return secondary_report if secondary_report.confidence >= primary_report.confidence else primary_report


def _with_failed_answer_validity(
    report: LLMJudgeReport,
    *,
    reason: str,
    issue_code: str,
) -> LLMJudgeReport:
    answer_validity = QualityDimensionResult(
        dimension=QualityDimension.ANSWER_VALIDITY,
        status=QualityStatus.FAILED,
        passed=False,
        score=0.0,
        reason=reason,
        issues=[
            {
                "code": issue_code,
                "message": reason,
                "affected_option_ids": [],
            }
        ],
    )
    return report.model_copy(
        update={
            "answer_validity": answer_validity,
            "overall_score": min(report.overall_score, 0.0),
            "requires_secondary_review": False,
            "feedback": reason,
        }
    )
