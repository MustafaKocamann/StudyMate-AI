"""Quality policy for accepting, regenerating, or escalating questions."""

from __future__ import annotations

from src.evaluation.models import (
    LLMJudgeReport,
    QualityDimension,
    QualityDimensionResult,
    QualityDecision,
    QualityPolicyConfig,
)


SEMANTIC_CRITICAL_DIMENSIONS = {
    QualityDimension.ANSWER_VALIDITY,
    QualityDimension.CONTEXT_ALIGNMENT,
    QualityDimension.ANSWER_LEAKAGE,
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

    # Deterministic failures are concrete defects and always regenerate. The
    # LLM judge is intentionally narrower: subjective weaknesses affect the
    # overall score, while correctness, context, and leakage remain fail-closed.
    if any(check.passed is False for check in checks):
        return QualityDecision.REGENERATE

    if judge_evaluation is None:
        return QualityDecision.ACCEPT

    if any(_is_semantic_critical_failure(check) for check in judge_evaluation.checks):
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
        if check.dimension in SEMANTIC_CRITICAL_DIMENSIONS and check.passed is not None
    ):
        return QualityDecision.REGENERATE

    return QualityDecision.ACCEPT


def _is_semantic_critical_failure(check: QualityDimensionResult) -> bool:
    if check.passed is not False:
        return False
    return check.dimension in SEMANTIC_CRITICAL_DIMENSIONS
