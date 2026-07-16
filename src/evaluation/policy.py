"""Quality policy for accepting, regenerating, or escalating questions."""

from __future__ import annotations

from src.evaluation.models import (
    LLMJudgeReport,
    QualityDimension,
    QualityDimensionResult,
    QualityDecision,
    QualityPolicyConfig,
)


CRITICAL_DIMENSIONS = {
    QualityDimension.ANSWER_VALIDITY,
    QualityDimension.EXPLANATION_QUALITY,
    QualityDimension.DIFFICULTY_ALIGNMENT,
    QualityDimension.CONTEXT_ALIGNMENT,
    QualityDimension.ANSWER_LEAKAGE,
    QualityDimension.DUPLICATE_RISK,
}


def decide_quality_outcome(
    checks: list[QualityDimensionResult],
    *,
    duplicate_risk_score: float,
    judge_evaluation: LLMJudgeReport | None = None,
    config: QualityPolicyConfig | None = None,
) -> QualityDecision:
    """Convert quality signals into one application decision."""

    policy_config = config or QualityPolicyConfig()
    if duplicate_risk_score >= policy_config.duplicate_fuzzy_threshold:
        return QualityDecision.REGENERATE

    if any(_is_applicable_critical_failure(check) for check in checks):
        return QualityDecision.REGENERATE

    if judge_evaluation is None:
        return QualityDecision.ACCEPT

    if any(_is_applicable_critical_failure(check) for check in judge_evaluation.checks):
        return QualityDecision.REGENERATE

    if judge_evaluation.confidence < policy_config.minimum_judge_confidence:
        return QualityDecision.ESCALATE_TO_SECONDARY_JUDGE

    if abs(judge_evaluation.overall_score - policy_config.minimum_judge_score) <= policy_config.secondary_judge_margin:
        return QualityDecision.ESCALATE_TO_SECONDARY_JUDGE

    if judge_evaluation.overall_score < policy_config.minimum_judge_score:
        return QualityDecision.REGENERATE

    if any(
        check.score is not None and check.score < policy_config.minimum_dimension_score
        for check in judge_evaluation.checks
        if check.passed is not None
    ):
        return QualityDecision.REGENERATE

    return QualityDecision.ACCEPT


def _is_applicable_critical_failure(check: QualityDimensionResult) -> bool:
    if check.passed is not False:
        return False
    if check.dimension in CRITICAL_DIMENSIONS:
        return True
    return check.dimension is QualityDimension.DISTRACTOR_QUALITY
