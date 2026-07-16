"""Quality report aggregation service without LLM invocation."""

from __future__ import annotations

from src.common.exceptions import QuestionJudgeError, StudyBuddyException
from src.evaluation.deterministic import run_deterministic_quality_checks
from src.evaluation.judge import QuestionJudge, reconcile_judge_reports, requires_secondary_judge
from src.evaluation.models import (
    JudgeEvaluation,
    LLMJudgeReport,
    QualityDecision,
    QualityDimension,
    QualityDimensionResult,
    QualityPolicyConfig,
    QuestionEvaluationContext,
    QuestionQualityReport,
)
from src.evaluation.policy import decide_quality_outcome
from src.generator.repair_prompts import build_repair_feedback
from src.models.question_schemas import GeneratedQuestion


def evaluate_question_quality(
    question: GeneratedQuestion,
    *,
    topic: str,
    source_content: str | None = None,
    previous_questions: list[GeneratedQuestion] | None = None,
    language: str | None = None,
    judge_evaluation: JudgeEvaluation | None = None,
    policy_config: QualityPolicyConfig | None = None,
) -> QuestionQualityReport:
    """Aggregate deterministic checks, optional judge output, and policy.

    The question must already be structurally valid and mapped into the domain
    model. This function does not invoke a judge LLM; a validated judge result
    may be supplied by a future service layer and is treated as one quality
    signal among deterministic checks.
    """

    checks, _, duplicate_score = run_deterministic_quality_checks(
        question,
        topic=topic,
        source_content=source_content,
        previous_questions=previous_questions,
        language=language,
        policy_config=policy_config,
    )
    return _build_quality_report(
        question=question,
        checks=checks,
        duplicate_score=duplicate_score,
        judge_evaluation=judge_evaluation,
        secondary_judge_evaluation=None,
        policy_config=policy_config,
    )


def evaluate_question_context(
    context: QuestionEvaluationContext,
    *,
    judge_evaluation: JudgeEvaluation | None = None,
    policy_config: QualityPolicyConfig | None = None,
) -> QuestionQualityReport:
    """Evaluate a question using a strict input context object."""

    checks, _, duplicate_score = run_deterministic_quality_checks(
        context.question,
        topic=context.topic,
        source_content=context.source_content,
        previous_questions=context.existing_questions,
        language=context.language,
        requested_difficulty=context.requested_difficulty,
        policy_config=policy_config,
    )
    return _build_quality_report(
        question=context.question,
        checks=checks,
        duplicate_score=duplicate_score,
        judge_evaluation=judge_evaluation,
        secondary_judge_evaluation=None,
        policy_config=policy_config,
    )


class QuestionQualityEvaluator:
    """Evaluate domain questions with deterministic checks and injected judges."""

    def __init__(
        self,
        *,
        primary_judge: QuestionJudge | None = None,
        secondary_judge: QuestionJudge | None = None,
        policy_config: QualityPolicyConfig | None = None,
    ) -> None:
        self.primary_judge = primary_judge
        self.secondary_judge = secondary_judge
        self.policy_config = policy_config

    async def evaluate(self, context: QuestionEvaluationContext) -> QuestionQualityReport:
        checks, _, duplicate_score = run_deterministic_quality_checks(
            context.question,
            topic=context.topic,
            source_content=context.source_content,
            previous_questions=context.existing_questions,
            language=context.language,
            requested_difficulty=context.requested_difficulty,
            policy_config=self.policy_config,
        )
        deterministic_decision = decide_quality_outcome(
            checks,
            duplicate_risk_score=duplicate_score,
            config=self.policy_config,
        )
        if deterministic_decision is QualityDecision.REGENERATE:
            return _build_quality_report(
                question=context.question,
                checks=checks,
                duplicate_score=duplicate_score,
                judge_evaluation=None,
                secondary_judge_evaluation=None,
                policy_config=self.policy_config,
            )
        if self.primary_judge is None:
            return _build_quality_report(
                question=context.question,
                checks=checks,
                duplicate_score=duplicate_score,
                judge_evaluation=None,
                secondary_judge_evaluation=None,
                policy_config=self.policy_config,
            )
        try:
            primary_report = await self.primary_judge.evaluate(context, checks)
            secondary_report = None
            if requires_secondary_judge(primary_report) and self.secondary_judge is not None:
                secondary_report = await self.secondary_judge.evaluate(context, checks)
            reconciled_report = reconcile_judge_reports(primary_report, secondary_report)
        except QuestionJudgeError:
            raise
        except StudyBuddyException:
            raise
        except Exception as exc:
            raise QuestionJudgeError("question judge failed for technical reasons") from exc

        merged_checks = _merge_deterministic_and_judge_results(checks, reconciled_report)
        return _build_quality_report(
            question=context.question,
            checks=merged_checks,
            policy_checks=checks,
            duplicate_score=duplicate_score,
            judge_evaluation=reconciled_report,
            secondary_judge_evaluation=secondary_report,
            policy_config=self.policy_config,
        )


def _build_quality_report(
    *,
    question: GeneratedQuestion,
    checks: list[QualityDimensionResult],
    policy_checks: list[QualityDimensionResult] | None = None,
    duplicate_score: float,
    judge_evaluation: LLMJudgeReport | None,
    secondary_judge_evaluation: LLMJudgeReport | None,
    policy_config: QualityPolicyConfig | None,
) -> QuestionQualityReport:
    decision = decide_quality_outcome(
        policy_checks or checks,
        duplicate_risk_score=duplicate_score,
        judge_evaluation=judge_evaluation,
        config=policy_config,
    )

    results_by_dimension = _results_by_dimension(checks)

    return QuestionQualityReport(
        question_id=question.id,
        question_type=question.type,
        overall_passed=decision is QualityDecision.ACCEPT,
        answer_validity=results_by_dimension[QualityDimension.ANSWER_VALIDITY],
        distractor_quality=results_by_dimension[QualityDimension.DISTRACTOR_QUALITY],
        explanation_quality=results_by_dimension[QualityDimension.EXPLANATION_QUALITY],
        difficulty_alignment=results_by_dimension[QualityDimension.DIFFICULTY_ALIGNMENT],
        context_alignment=results_by_dimension[QualityDimension.CONTEXT_ALIGNMENT],
        answer_leakage=results_by_dimension[QualityDimension.ANSWER_LEAKAGE],
        duplicate_risk=results_by_dimension[QualityDimension.DUPLICATE_RISK],
        judge_evaluation=judge_evaluation,
        secondary_judge_evaluation=secondary_judge_evaluation,
        decision=decision,
        repair_feedback=build_repair_feedback(checks),
    )


def _results_by_dimension(
    checks: list[QualityDimensionResult],
) -> dict[QualityDimension, QualityDimensionResult]:
    results = {check.dimension: check for check in checks}
    missing_dimensions = set(QualityDimension) - set(results)
    if missing_dimensions:
        missing = ", ".join(sorted(dimension.value for dimension in missing_dimensions))
        raise ValueError(f"missing quality dimension results: {missing}")
    return results


def _merge_deterministic_and_judge_results(
    deterministic_checks: list[QualityDimensionResult],
    judge_report: LLMJudgeReport,
) -> list[QualityDimensionResult]:
    deterministic_by_dimension = _results_by_dimension(deterministic_checks)
    merged = dict(deterministic_by_dimension)
    for judge_result in judge_report.checks:
        deterministic_result = deterministic_by_dimension[judge_result.dimension]
        if deterministic_result.passed is False:
            continue
        merged[judge_result.dimension] = judge_result
    return [merged[dimension] for dimension in QualityDimension]
