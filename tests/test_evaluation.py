from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from uuid import UUID

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.exceptions import LLMRateLimitError, QuestionJudgeError, QuestionJudgeResponseValidationError
from src.evaluation.deterministic import run_deterministic_quality_checks
from src.evaluation.duplicate_detection import duplicate_risk_score, token_jaccard_similarity
from src.evaluation.judge import validate_judge_evaluation
from src.evaluation.judge_prompts import (
    JUDGE_PROMPT_VERSION,
    QUALITY_JUDGE_PROMPT_VERSION,
    question_quality_judge_prompt,
    judge_prompt_template,
)
from src.evaluation.models import (
    ContextAlignmentMode,
    JudgeEvaluation,
    QUALITY_EVALUATION_VERSION,
    QuestionEvaluationContext,
    LLMJudgeReport,
    QualityDimensionResult,
    QualityDecision,
    QualityDimension,
    QualityIssue,
    QualityPolicyConfig,
    QualityStatus,
)
from src.evaluation.normalization import (
    contains_normalized_phrase,
    normalize_for_comparison,
    normalize_for_duplicate_detection,
    normalize_text,
)
from src.evaluation.policy import decide_quality_outcome
from src.evaluation.service import QuestionQualityEvaluator, evaluate_question_context, evaluate_question_quality
from src.generator.regeneration import build_regeneration_plan
from src.generator.repair_prompts import (
    QuestionRepairFeedback,
    build_question_repair_feedback,
    build_repair_feedback,
)
from src.models.question_schemas import FillBlankQuestion, MCQQuestion, QuestionSet


def mcq_question(**overrides: object) -> MCQQuestion:
    payload: dict[str, object] = {
        "type": "mcq",
        "position": 1,
        "question": "Which process allows plants to convert light energy into chemical energy?",
        "difficulty": "easy",
        "explanation": "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "options": [
            {"id": "A", "text": "Respiration"},
            {"id": "B", "text": "Photosynthesis"},
            {"id": "C", "text": "Transpiration"},
            {"id": "D", "text": "Fermentation"},
        ],
        "correct_option_id": "B",
    }
    payload.update(overrides)
    return MCQQuestion.model_validate(payload)


def fill_blank_question(**overrides: object) -> FillBlankQuestion:
    payload: dict[str, object] = {
        "type": "fill_blank",
        "position": 1,
        "question": "Plants convert light energy into chemical energy through ___.",
        "difficulty": "easy",
        "explanation": "Photosynthesis is the process that stores light energy in chemical form.",
        "answer": "photosynthesis",
    }
    payload.update(overrides)
    return FillBlankQuestion.model_validate(payload)


def dimension_result(
    dimension: QualityDimension,
    *,
    score: float = 0.9,
    status: QualityStatus = QualityStatus.PASSED,
    passed: bool | None = True,
    reason: str = "The dimension passes.",
) -> dict[str, object]:
    return {
        "dimension": dimension.value,
        "status": status.value,
        "passed": passed,
        "score": score,
        "reason": reason,
        "issues": []
        if passed is not False
        else [{"code": "weak_explanation", "message": "The dimension failed.", "affected_option_ids": []}],
    }


def valid_judge_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "answer_validity": dimension_result(QualityDimension.ANSWER_VALIDITY),
        "distractor_quality": dimension_result(QualityDimension.DISTRACTOR_QUALITY),
        "explanation_quality": dimension_result(QualityDimension.EXPLANATION_QUALITY),
        "difficulty_alignment": {
            **dimension_result(QualityDimension.DIFFICULTY_ALIGNMENT),
            "requested_difficulty": "easy",
            "estimated_difficulty": "easy",
        },
        "context_alignment": {
            **dimension_result(QualityDimension.CONTEXT_ALIGNMENT),
            "context_alignment_mode": "topic_relevance",
        },
        "answer_leakage": dimension_result(QualityDimension.ANSWER_LEAKAGE),
        "overall_score": 0.86,
        "confidence": 0.8,
        "requires_secondary_review": False,
        "feedback": "Acceptable question.",
    }
    payload.update(overrides)
    return payload


class FakeJudge:
    def __init__(self, report: LLMJudgeReport) -> None:
        self.report = report
        self.calls = 0

    async def evaluate(self, context, deterministic_findings):  # noqa: ANN001
        self.calls += 1
        return self.report


class RaisingJudge:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def evaluate(self, context, deterministic_findings):  # noqa: ANN001
        raise self.exc


def test_quality_dimension_enum_contains_required_values() -> None:
    assert {dimension.value for dimension in QualityDimension} == {
        "answer_validity",
        "distractor_quality",
        "explanation_quality",
        "difficulty_alignment",
        "context_alignment",
        "answer_leakage",
        "duplicate_risk",
    }


def test_quality_status_enum_contains_required_values() -> None:
    assert {status.value for status in QualityStatus} == {
        "passed",
        "failed",
        "not_applicable",
        "not_evaluated",
    }


def test_quality_issue_is_structured_and_strict() -> None:
    issue = QualityIssue(
        code="answer_revealed_in_question",
        message="The answer appears in the question.",
        affected_option_ids=["A"],
    )

    assert issue.code == "answer_revealed_in_question"
    assert issue.affected_option_ids == ["A"]

    with pytest.raises(ValidationError):
        QualityIssue(code="", message="Missing code.")


def test_quality_dimension_result_validates_status_consistency() -> None:
    with pytest.raises(ValidationError):
        QualityDimensionResult(
            dimension=QualityDimension.ANSWER_VALIDITY,
            status=QualityStatus.PASSED,
            passed=False,
            score=0.9,
            reason="Inconsistent state.",
        )

    with pytest.raises(ValidationError):
        QualityDimensionResult(
            dimension=QualityDimension.ANSWER_VALIDITY,
            status=QualityStatus.FAILED,
            passed=False,
            score=0.2,
            reason="Failed without issue.",
        )

    not_applicable = QualityDimensionResult(
        dimension=QualityDimension.DISTRACTOR_QUALITY,
        status=QualityStatus.NOT_APPLICABLE,
        passed=None,
        score=None,
        reason="Distractors do not apply.",
    )
    assert not_applicable.passed is None


def test_question_evaluation_context_is_strict_and_uses_default_factory() -> None:
    question = mcq_question()
    context = QuestionEvaluationContext(
        question=question,
        topic="photosynthesis",
        requested_difficulty=question.difficulty,
        language="Turkish",
    )
    second_context = QuestionEvaluationContext(
        question=question,
        topic="photosynthesis",
        requested_difficulty=question.difficulty,
        language="Turkish",
    )

    assert context.existing_questions == []
    assert context.existing_questions is not second_context.existing_questions

    with pytest.raises(ValidationError):
        QuestionEvaluationContext(
            question=question,
            topic=" ",
            requested_difficulty=question.difficulty,
            language="Turkish",
        )

    with pytest.raises(ValidationError):
        QuestionEvaluationContext(
            question=question,
            topic="photosynthesis",
            requested_difficulty=question.difficulty,
            language=" ",
        )

    with pytest.raises(ValidationError):
        QuestionEvaluationContext(
            question=question,
            topic="photosynthesis",
            requested_difficulty=question.difficulty,
            language="Turkish",
            source_content=" ",
        )

    with pytest.raises(ValidationError):
        QuestionEvaluationContext(
            question=question,
            topic="photosynthesis",
            requested_difficulty=question.difficulty,
            language="Turkish",
            existing_questions=[question],
        )


def test_normalization_utilities_are_case_and_whitespace_stable() -> None:
    assert normalize_text("  PyThOn   Lists ") == "python lists"
    assert contains_normalized_phrase("Python list comprehensions", " python LIST ")
    assert normalize_for_duplicate_detection("Python,  lists!") == "python lists"


def test_normalization_handles_turkish_dotted_and_dotless_i() -> None:
    assert normalize_for_comparison("IŞIK", language="tr") == "ışık"
    assert normalize_for_comparison("İSTANBUL", language="Türkçe") == "istanbul"


def test_duplicate_detection_scores_exact_and_partial_matches() -> None:
    question = mcq_question()
    exact_previous = mcq_question(position=2)
    different_previous = fill_blank_question(position=3)

    assert duplicate_risk_score(question, [exact_previous]) == 1.0
    assert 0.0 <= duplicate_risk_score(question, [different_previous]) < 1.0
    assert token_jaccard_similarity("python lists", "python dictionaries") == pytest.approx(1 / 3)


def test_duplicate_detection_includes_correct_answer_in_representation() -> None:
    first = mcq_question(question="Which concept is described?", position=1)
    second = mcq_question(
        question="Which concept is described?",
        options=[
            {"id": "A", "text": "Respiration"},
            {"id": "B", "text": "Photosynthesis"},
            {"id": "C", "text": "Fermentation"},
            {"id": "D", "text": "Transpiration"},
        ],
        correct_option_id="A",
        position=2,
    )

    assert duplicate_risk_score(first, [second]) < 1.0


def test_deterministic_checks_pass_for_reasonable_mcq() -> None:
    checks, mode, duplicate_score = run_deterministic_quality_checks(
        mcq_question(),
        topic="plants light energy photosynthesis",
    )

    dimensions = {check.dimension for check in checks}
    assert QualityDimension.ANSWER_VALIDITY in dimensions
    assert QualityDimension.DISTRACTOR_QUALITY in dimensions
    assert QualityDimension.EXPLANATION_QUALITY in dimensions
    assert QualityDimension.CONTEXT_ALIGNMENT in dimensions
    assert QualityDimension.ANSWER_LEAKAGE in dimensions
    assert QualityDimension.DUPLICATE_RISK in dimensions
    assert mode is ContextAlignmentMode.TOPIC_RELEVANCE
    assert duplicate_score == 0.0


def test_distractor_quality_is_not_applicable_for_fill_blank() -> None:
    checks, _, _ = run_deterministic_quality_checks(
        fill_blank_question(),
        topic="plants light energy photosynthesis",
    )
    distractor_quality = next(
        check for check in checks if check.dimension is QualityDimension.DISTRACTOR_QUALITY
    )

    assert distractor_quality.status is QualityStatus.NOT_APPLICABLE
    assert distractor_quality.passed is None


def test_deterministic_checks_detect_answer_leakage() -> None:
    question = fill_blank_question(question="Photosynthesis happens through ___.")
    checks, _, _ = run_deterministic_quality_checks(question, topic="photosynthesis")
    leakage = next(check for check in checks if check.dimension is QualityDimension.ANSWER_LEAKAGE)

    assert leakage.status is QualityStatus.FAILED
    assert leakage.passed is False
    assert leakage.issues[0].code == "answer_revealed_in_question"


def test_fill_blank_leakage_masks_placeholder_but_checks_remaining_question() -> None:
    safe_question = fill_blank_question(question="The process is called ___.")
    leaked_question = fill_blank_question(question="Photosynthesis is the process called ___.")

    safe_checks, _, _ = run_deterministic_quality_checks(safe_question, topic="photosynthesis")
    leaked_checks, _, _ = run_deterministic_quality_checks(leaked_question, topic="photosynthesis")

    safe_leakage = next(check for check in safe_checks if check.dimension is QualityDimension.ANSWER_LEAKAGE)
    leaked_leakage = next(
        check for check in leaked_checks if check.dimension is QualityDimension.ANSWER_LEAKAGE
    )
    assert safe_leakage.status is QualityStatus.PASSED
    assert leaked_leakage.status is QualityStatus.FAILED


def test_weak_explanation_patterns_are_failed_deterministically() -> None:
    question = mcq_question(explanation="B doğru cevaptır.")
    checks, _, _ = run_deterministic_quality_checks(question, topic="plants", language="tr")
    explanation = next(check for check in checks if check.dimension is QualityDimension.EXPLANATION_QUALITY)

    assert explanation.status is QualityStatus.FAILED
    assert explanation.issues[0].code == "answer_only_explanation"


def test_context_alignment_uses_source_groundedness_when_source_is_supplied() -> None:
    checks, mode, _ = run_deterministic_quality_checks(
        mcq_question(),
        topic="plants",
        source_content="Photosynthesis lets plants convert light energy.",
    )

    context_check = next(check for check in checks if check.dimension is QualityDimension.CONTEXT_ALIGNMENT)
    assert mode is ContextAlignmentMode.SOURCE_GROUNDEDNESS
    assert context_check.context_alignment_mode is ContextAlignmentMode.SOURCE_GROUNDEDNESS
    assert "source_groundedness" in context_check.reason


def test_judge_evaluation_contract_accepts_valid_payload() -> None:
    evaluation = validate_judge_evaluation(valid_judge_payload())

    assert isinstance(evaluation, LLMJudgeReport)
    assert evaluation.overall_score == 0.86
    assert not evaluation.requires_secondary_review


def test_judge_evaluation_rejects_wrong_dimension_field() -> None:
    with pytest.raises(QuestionJudgeResponseValidationError):
        validate_judge_evaluation(
            valid_judge_payload(
                answer_validity=dimension_result(QualityDimension.EXPLANATION_QUALITY)
            )
        )


def test_judge_prompt_is_chat_template_with_expected_variables() -> None:
    assert JUDGE_PROMPT_VERSION == "question-quality-judge-v1"
    assert QUALITY_JUDGE_PROMPT_VERSION == "question-quality-judge-v1"
    assert isinstance(judge_prompt_template, ChatPromptTemplate)
    assert isinstance(question_quality_judge_prompt, ChatPromptTemplate)
    assert set(judge_prompt_template.input_variables) == {
        "topic",
        "source_content",
        "question_json",
        "deterministic_findings",
    }


def test_judge_prompt_renders_contract_without_extra_placeholders() -> None:
    messages = judge_prompt_template.format_messages(
        topic="photosynthesis",
        source_content="",
        question_json='{"type": "mcq"}',
        deterministic_findings="No deterministic failures.",
    )

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    system_text = str(messages[0].content)
    human_text = str(messages[1].content)
    assert "Return exactly one valid JSON object" in system_text
    assert "not the original question generator" in system_text
    assert "declared correct answer may be wrong" in system_text
    assert "requires_secondary_review" in system_text
    assert "topic_relevance" in system_text
    assert "source_groundedness" in system_text
    assert "Do not call topic-only evaluation groundedness" in system_text
    assert "{topic}" not in human_text
    assert '"answer_validity"' in human_text
    assert '"duplicate_risk"' not in human_text


def test_policy_accepts_when_checks_and_judge_are_good() -> None:
    report = evaluate_question_quality(
        mcq_question(),
        topic="plants light energy photosynthesis",
        judge_evaluation=validate_judge_evaluation(valid_judge_payload()),
    )

    assert report.decision is QualityDecision.ACCEPT
    assert report.passed
    assert report.overall_passed
    assert report.evaluation_version == QUALITY_EVALUATION_VERSION
    assert isinstance(report.question_id, UUID)
    assert report.answer_validity.dimension is QualityDimension.ANSWER_VALIDITY
    assert "checks" not in report.model_dump()


def test_evaluate_question_context_uses_requested_difficulty_and_language() -> None:
    question = mcq_question()
    context = QuestionEvaluationContext(
        question=question,
        topic="plants light energy photosynthesis",
        requested_difficulty=question.difficulty,
        language="tr-TR",
    )

    report = evaluate_question_context(context)

    assert report.question_id == question.id
    assert report.difficulty_alignment.status is QualityStatus.PASSED


def test_context_detects_requested_difficulty_mismatch() -> None:
    question = mcq_question()
    context = QuestionEvaluationContext(
        question=question,
        topic="plants light energy photosynthesis",
        requested_difficulty="hard",
        language="English",
    )

    report = evaluate_question_context(context)

    assert report.difficulty_alignment.status is QualityStatus.FAILED
    assert report.difficulty_alignment.issues[0].code == "difficulty_too_low"


def test_policy_regenerates_on_failed_deterministic_check() -> None:
    report = evaluate_question_quality(
        fill_blank_question(question="Photosynthesis happens through ___."),
        topic="photosynthesis",
    )

    assert report.decision is QualityDecision.REGENERATE
    assert not report.passed
    assert any("answer_leakage" in feedback for feedback in report.repair_feedback)
    assert report.answer_leakage.issues[0].code == "answer_revealed_in_question"


def test_policy_escalates_low_confidence_judge() -> None:
    checks, _, duplicate_score = run_deterministic_quality_checks(
        mcq_question(),
        topic="plants light energy photosynthesis",
    )
    decision = decide_quality_outcome(
        checks,
        duplicate_risk_score=duplicate_score,
        judge_evaluation=validate_judge_evaluation(valid_judge_payload(confidence=0.4)),
    )

    assert decision is QualityDecision.ESCALATE_TO_SECONDARY_JUDGE


def test_policy_escalates_near_threshold_judge_score() -> None:
    checks, _, duplicate_score = run_deterministic_quality_checks(
        mcq_question(),
        topic="plants light energy photosynthesis",
    )
    decision = decide_quality_outcome(
        checks,
        duplicate_risk_score=duplicate_score,
        judge_evaluation=validate_judge_evaluation(valid_judge_payload(overall_score=0.75)),
        config=QualityPolicyConfig(secondary_judge_margin=0.03),
    )

    assert decision is QualityDecision.ESCALATE_TO_SECONDARY_JUDGE


def test_duplicate_question_causes_regeneration() -> None:
    previous = mcq_question(position=2)
    report = evaluate_question_quality(
        mcq_question(position=1),
        topic="plants light energy photosynthesis",
        previous_questions=[previous],
    )

    assert report.duplicate_risk_score == 1.0
    assert report.decision is QualityDecision.REGENERATE


def test_fuzzy_duplicate_threshold_can_be_configured() -> None:
    previous = mcq_question(position=2)
    candidate = mcq_question(
        position=1,
        question="Which process lets plants convert light energy into chemical energy?",
    )
    report = evaluate_question_quality(
        candidate,
        topic="plants light energy photosynthesis",
        previous_questions=[previous],
        policy_config=QualityPolicyConfig(duplicate_fuzzy_threshold=0.5),
    )

    assert report.duplicate_risk.status is QualityStatus.FAILED
    assert report.duplicate_risk.issues[0].code in {
        "exact_duplicate_question",
        "high_similarity_question",
    }


def test_repair_feedback_and_regeneration_plan_are_deterministic() -> None:
    report = evaluate_question_quality(
        fill_blank_question(question="Photosynthesis happens through ___."),
        topic="photosynthesis",
    )
    feedback = build_repair_feedback(report.checks)
    plan = build_regeneration_plan(report)

    assert feedback
    assert plan.should_regenerate
    assert not plan.should_escalate_to_secondary_judge
    assert plan.feedback == report.repair_feedback


def test_structured_repair_feedback_contains_only_failed_dimensions() -> None:
    report = evaluate_question_quality(
        fill_blank_question(question="Photosynthesis happens through ___."),
        topic="photosynthesis",
    )

    feedback = build_question_repair_feedback(report)

    assert isinstance(feedback, QuestionRepairFeedback)
    assert QualityDimension.ANSWER_LEAKAGE in feedback.failed_dimensions
    assert all(issue.code for issue in feedback.issues)
    assert feedback.revision_instructions


def test_question_quality_evaluator_uses_primary_judge_without_secondary_when_not_needed() -> None:
    primary_judge = FakeJudge(validate_judge_evaluation(valid_judge_payload()))
    secondary_judge = FakeJudge(validate_judge_evaluation(valid_judge_payload()))
    question = mcq_question()
    context = QuestionEvaluationContext(
        question=question,
        topic="plants light energy photosynthesis",
        requested_difficulty=question.difficulty,
        language="English",
    )

    report = asyncio.run(
        QuestionQualityEvaluator(
            primary_judge=primary_judge,
            secondary_judge=secondary_judge,
        ).evaluate(context)
    )

    assert report.judge_evaluation is not None
    assert primary_judge.calls == 1
    assert secondary_judge.calls == 0
    assert report.secondary_judge_evaluation is None


def test_question_quality_evaluator_calls_secondary_only_for_required_review() -> None:
    primary_judge = FakeJudge(
        validate_judge_evaluation(valid_judge_payload(requires_secondary_review=True, confidence=0.7))
    )
    secondary_judge = FakeJudge(validate_judge_evaluation(valid_judge_payload(confidence=0.9)))
    question = mcq_question()
    context = QuestionEvaluationContext(
        question=question,
        topic="plants light energy photosynthesis",
        requested_difficulty=question.difficulty,
        language="English",
    )

    report = asyncio.run(
        QuestionQualityEvaluator(
            primary_judge=primary_judge,
            secondary_judge=secondary_judge,
        ).evaluate(context)
    )

    assert primary_judge.calls == 1
    assert secondary_judge.calls == 1
    assert report.secondary_judge_evaluation is not None


def test_question_quality_evaluator_fails_closed_without_secondary_for_ambiguity() -> None:
    primary_judge = FakeJudge(
        validate_judge_evaluation(valid_judge_payload(requires_secondary_review=True, confidence=0.7))
    )
    question = mcq_question()
    context = QuestionEvaluationContext(
        question=question,
        topic="plants light energy photosynthesis",
        requested_difficulty=question.difficulty,
        language="English",
    )

    report = asyncio.run(QuestionQualityEvaluator(primary_judge=primary_judge).evaluate(context))

    assert report.decision is QualityDecision.REGENERATE
    assert report.answer_validity.status is QualityStatus.FAILED
    assert report.answer_validity.issues[0].code == "insufficient_question_context"


def test_question_quality_evaluator_fails_closed_when_judges_disagree_on_answer_validity() -> None:
    failed_answer = {
        **dimension_result(
            QualityDimension.ANSWER_VALIDITY,
            status=QualityStatus.FAILED,
            passed=False,
            score=0.2,
            reason="The declared answer is wrong.",
        ),
        "issues": [
            {
                "code": "incorrect_declared_answer",
                "message": "The declared answer is wrong.",
                "affected_option_ids": ["B"],
            }
        ],
    }
    primary_judge = FakeJudge(
        validate_judge_evaluation(valid_judge_payload(requires_secondary_review=True, confidence=0.7))
    )
    secondary_judge = FakeJudge(
        validate_judge_evaluation(valid_judge_payload(answer_validity=failed_answer, confidence=0.9))
    )
    question = mcq_question()
    context = QuestionEvaluationContext(
        question=question,
        topic="plants light energy photosynthesis",
        requested_difficulty=question.difficulty,
        language="English",
    )

    report = asyncio.run(
        QuestionQualityEvaluator(primary_judge=primary_judge, secondary_judge=secondary_judge).evaluate(
            context
        )
    )

    assert report.decision is QualityDecision.REGENERATE
    assert report.answer_validity.status is QualityStatus.FAILED
    assert report.answer_validity.issues[0].code == "incorrect_declared_answer"


def test_question_quality_evaluator_preserves_provider_error_categories() -> None:
    question = mcq_question()
    context = QuestionEvaluationContext(
        question=question,
        topic="plants light energy photosynthesis",
        requested_difficulty=question.difficulty,
        language="English",
    )
    provider_error = LLMRateLimitError(
        "provider rate limit",
        provider="groq",
        model="configured-model",
        error_category="rate_limit",
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        asyncio.run(QuestionQualityEvaluator(primary_judge=RaisingJudge(provider_error)).evaluate(context))

    assert exc_info.value.error_category == "rate_limit"
    assert exc_info.value.provider == "groq"


def test_question_quality_evaluator_wraps_unknown_judge_failures() -> None:
    question = mcq_question()
    context = QuestionEvaluationContext(
        question=question,
        topic="plants light energy photosynthesis",
        requested_difficulty=question.difficulty,
        language="English",
    )

    with pytest.raises(QuestionJudgeError):
        asyncio.run(
            QuestionQualityEvaluator(primary_judge=RaisingJudge(RuntimeError("boom"))).evaluate(context)
        )


def test_evaluated_questions_can_still_form_question_set() -> None:
    first = mcq_question(position=1)
    second = fill_blank_question(position=2)
    first_report = evaluate_question_quality(first, topic="plants light energy photosynthesis")
    second_report = evaluate_question_quality(second, topic="plants light energy photosynthesis")

    question_set = QuestionSet.model_validate({"questions": [first, second]})

    assert question_set.questions[0].id != question_set.questions[1].id
    assert first_report.decision in {QualityDecision.ACCEPT, QualityDecision.REGENERATE}
    assert second_report.decision in {QualityDecision.ACCEPT, QualityDecision.REGENERATE}
