"""Deterministic educational quality checks for structurally valid questions."""

from __future__ import annotations

from src.evaluation.duplicate_detection import duplicate_risk_score
from src.evaluation.models import (
    ContextAlignmentMode,
    QualityPolicyConfig,
    QualityDimension,
    QualityDimensionResult,
    QualityIssue,
    QualityStatus,
)
from src.evaluation.normalization import contains_normalized_phrase, normalize_for_comparison, tokenize
from src.models.question_schemas import DifficultyLevel, FillBlankQuestion, GeneratedQuestion, MCQQuestion


GENERIC_BAD_OPTIONS = {
    "all of the above",
    "none of the above",
    "all",
    "none",
}

WEAK_EXPLANATION_PHRASES = {
    "a is correct",
    "b is correct",
    "c is correct",
    "d is correct",
    "a is the answer",
    "b is the answer",
    "c is the answer",
    "d is the answer",
    "the correct answer is a",
    "the correct answer is b",
    "the correct answer is c",
    "the correct answer is d",
    "this is correct",
    "a doğrudur",
    "b doğrudur",
    "c doğrudur",
    "d doğrudur",
    "a doğru cevaptır",
    "b doğru cevaptır",
    "c doğru cevaptır",
    "d doğru cevaptır",
    "doğru cevap a",
    "doğru cevap b",
    "doğru cevap c",
    "doğru cevap d",
}

PROMPT_META_PHRASES = {
    "as an ai language model",
    "prompt instructions",
    "json format",
    "system prompt",
    "language model",
}


def run_deterministic_quality_checks(
    question: GeneratedQuestion,
    *,
    topic: str,
    source_content: str | None = None,
    previous_questions: list[GeneratedQuestion] | None = None,
    language: str | None = None,
    policy_config: QualityPolicyConfig | None = None,
    requested_difficulty: DifficultyLevel | None = None,
) -> tuple[list[QualityDimensionResult], ContextAlignmentMode, float]:
    """Run local quality checks that require no model call.

    These checks catch obvious quality failures after structural validation:
    answer leakage, weak explanations, suspicious MCQ distractors, topic/source
    mismatch signals, and duplicate risk. They are intentionally conservative;
    nuanced correctness and difficulty judgments are left to the judge contract.
    """

    config = policy_config or QualityPolicyConfig()
    duplicate_score = duplicate_risk_score(question, previous_questions or [], language=language)
    mode = (
        ContextAlignmentMode.SOURCE_GROUNDEDNESS
        if source_content
        else ContextAlignmentMode.TOPIC_RELEVANCE
    )
    checks = [
        _answer_validity_check(question),
        _distractor_quality_check(question),
        _explanation_quality_check(question, language=language),
        _difficulty_alignment_placeholder_check(question, requested_difficulty=requested_difficulty),
        _context_alignment_check(
            question,
            topic=topic,
            source_content=source_content,
            mode=mode,
            language=language,
        ),
        _answer_leakage_check(question, language=language),
        _duplicate_risk_check(duplicate_score, threshold=config.duplicate_fuzzy_threshold),
    ]
    return checks, mode, duplicate_score


def _answer_validity_check(question: GeneratedQuestion) -> QualityDimensionResult:
    if isinstance(question, MCQQuestion):
        reason = "The declared correct option references one supplied option."
    else:
        reason = "The fill-blank answer is present and structurally valid."

    return _passed_result(
        QualityDimension.ANSWER_VALIDITY,
        score=1.0,
        reason=reason,
    )


def _distractor_quality_check(question: GeneratedQuestion) -> QualityDimensionResult:
    if not isinstance(question, MCQQuestion):
        return QualityDimensionResult(
            dimension=QualityDimension.DISTRACTOR_QUALITY,
            status=QualityStatus.NOT_APPLICABLE,
            passed=None,
            score=None,
            reason="Distractor quality does not apply to fill-in-the-blank questions.",
        )

    option_texts = [option.text for option in question.options]
    has_generic_option = any(
        contains_normalized_phrase(option_text, generic_option)
        for option_text in option_texts
        for generic_option in GENERIC_BAD_OPTIONS
    )
    correct_option = next(option for option in question.options if option.id == question.correct_option_id)
    longest_option_length = max(len(option.text) for option in question.options)
    correct_is_much_longer = len(correct_option.text) > longest_option_length * 0.75 and len(correct_option.text) > 80

    if has_generic_option or correct_is_much_longer:
        return _failed_result(
            QualityDimension.DISTRACTOR_QUALITY,
            score=0.4,
            reason="Distractors contain deterministic red flags.",
            issue=QualityIssue(
                code="obviously_wrong_distractor",
                message="At least one distractor appears generic or structurally unfair.",
            ),
        )

    return _passed_result(
        QualityDimension.DISTRACTOR_QUALITY,
        score=0.85,
        reason="No deterministic distractor red flags detected.",
    )


def _explanation_quality_check(
    question: GeneratedQuestion,
    *,
    language: str | None,
) -> QualityDimensionResult:
    answer_text = _answer_text(question)
    explanation_mentions_answer = contains_normalized_phrase(
        question.explanation,
        answer_text,
        language=language,
    )
    explanation_tokens = tokenize(question.explanation, language=language)
    normalized_explanation = normalize_for_comparison(question.explanation, language=language)
    if any(phrase in normalized_explanation for phrase in PROMPT_META_PHRASES):
        return _failed_result(
            QualityDimension.EXPLANATION_QUALITY,
            score=0.2,
            reason="Explanation discusses prompt instructions, JSON format, or AI model behavior.",
            issue=QualityIssue(
                code="weak_explanation",
                message="The explanation contains prompt or model meta-language.",
            ),
        )
    if any(phrase in normalized_explanation for phrase in WEAK_EXPLANATION_PHRASES):
        return _failed_result(
            QualityDimension.EXPLANATION_QUALITY,
            score=0.35,
            reason="Explanation is answer-only or formulaic.",
            issue=QualityIssue(
                code="answer_only_explanation",
                message="The explanation appears to only state the answer.",
            ),
        )
    if normalize_for_comparison(question.explanation, language=language) == normalize_for_comparison(
        answer_text,
        language=language,
    ):
        return _failed_result(
            QualityDimension.EXPLANATION_QUALITY,
            score=0.2,
            reason="Explanation is only the correct answer.",
            issue=QualityIssue(
                code="answer_only_explanation",
                message="The explanation repeats only the answer with no conceptual context.",
            ),
        )
    if explanation_mentions_answer and len(explanation_tokens) >= 6:
        return _passed_result(
            QualityDimension.EXPLANATION_QUALITY,
            score=0.9,
            reason="Explanation names the answer and provides enough context.",
        )
    return _failed_result(
        QualityDimension.EXPLANATION_QUALITY,
        score=0.55,
        reason="Explanation may not sufficiently connect the answer to the concept.",
        issue=QualityIssue(
            code="weak_explanation",
            message="The explanation may be too thin or insufficiently connected to the answer.",
        ),
    )


def _difficulty_alignment_placeholder_check(
    question: GeneratedQuestion,
    *,
    requested_difficulty: DifficultyLevel | None,
) -> QualityDimensionResult:
    if requested_difficulty is not None and question.difficulty is not requested_difficulty:
        return _failed_result(
            QualityDimension.DIFFICULTY_ALIGNMENT,
            score=0.2,
            reason="Generated difficulty does not match the requested difficulty.",
            issue=QualityIssue(
                code="difficulty_too_low",
                message="The generated difficulty enum differs from the requested difficulty.",
            ),
            requested_difficulty=requested_difficulty,
            estimated_difficulty=question.difficulty,
        )

    question_tokens = tokenize(question.question)
    if question_tokens:
        return _passed_result(
            QualityDimension.DIFFICULTY_ALIGNMENT,
            score=0.75,
            reason="Difficulty alignment requires judge review; deterministic structure is present.",
            requested_difficulty=requested_difficulty,
            estimated_difficulty=question.difficulty,
        )
    return _failed_result(
        QualityDimension.DIFFICULTY_ALIGNMENT,
        score=0.0,
        reason="Question text is missing tokenizable content.",
        issue=QualityIssue(code="difficulty_too_low", message="The question has no assessable content."),
        requested_difficulty=requested_difficulty,
        estimated_difficulty=question.difficulty,
    )


def _context_alignment_check(
    question: GeneratedQuestion,
    *,
    topic: str,
    source_content: str | None,
    mode: ContextAlignmentMode,
    language: str | None,
) -> QualityDimensionResult:
    evidence = source_content if source_content else topic
    evidence_tokens = tokenize(evidence, language=language)
    question_tokens = tokenize(question.question, language=language)
    overlap = len(evidence_tokens & question_tokens)
    if overlap:
        return QualityDimensionResult(
            dimension=QualityDimension.CONTEXT_ALIGNMENT,
            status=QualityStatus.PASSED,
            passed=True,
            score=0.85,
            reason=f"Deterministic {mode.value} token overlap is present.",
            issues=[],
            context_alignment_mode=mode,
        )

    issue_code = "unsupported_by_source" if source_content else "topic_mismatch"
    return QualityDimensionResult(
        dimension=QualityDimension.CONTEXT_ALIGNMENT,
        status=QualityStatus.FAILED,
        passed=False,
        score=0.45,
        reason=f"Deterministic {mode.value} token overlap is missing.",
        issues=[
            QualityIssue(
                code=issue_code,
                message="The question text has no detectable token overlap with the supplied context.",
            )
        ],
        context_alignment_mode=mode,
    )


def _answer_leakage_check(
    question: GeneratedQuestion,
    *,
    language: str | None,
) -> QualityDimensionResult:
    answer_text = _answer_text(question)
    question_text = question.question
    if isinstance(question, FillBlankQuestion):
        question_text = question_text.replace("___", " ")
    leaks_answer = contains_normalized_phrase(
        question_text,
        answer_text,
        language=language,
        remove_punctuation=True,
    )
    if leaks_answer:
        return _failed_result(
            QualityDimension.ANSWER_LEAKAGE,
            score=0.2,
            reason=(
                "Question appears to reveal the answer text. Explanations are assumed to be shown "
                "after the learner answers and are not treated as pre-answer leakage."
            ),
            issue=QualityIssue(
                code="answer_revealed_in_question",
                message="The answer text appears directly in the question.",
            ),
        )

    return _passed_result(
        QualityDimension.ANSWER_LEAKAGE,
        score=1.0,
        reason="No direct answer phrase leakage detected.",
    )


def _duplicate_risk_check(duplicate_score: float, *, threshold: float) -> QualityDimensionResult:
    quality_score = 1.0 - duplicate_score
    if duplicate_score >= threshold:
        issue_code = "exact_duplicate_question" if duplicate_score == 1.0 else "high_similarity_question"
        return _failed_result(
            QualityDimension.DUPLICATE_RISK,
            score=quality_score,
            reason=f"Duplicate risk score is {duplicate_score:.2f}.",
            issue=QualityIssue(
                code=issue_code,
                message="The question appears to duplicate or closely match a previously generated question.",
            ),
        )

    return _passed_result(
        QualityDimension.DUPLICATE_RISK,
        score=quality_score,
        reason=f"Duplicate risk score is {duplicate_score:.2f}.",
    )


def _answer_text(question: GeneratedQuestion) -> str:
    if isinstance(question, FillBlankQuestion):
        return question.answer
    return next(option.text for option in question.options if option.id == question.correct_option_id)


def _passed_result(
    dimension: QualityDimension,
    *,
    score: float,
    reason: str,
    requested_difficulty: DifficultyLevel | None = None,
    estimated_difficulty: DifficultyLevel | None = None,
) -> QualityDimensionResult:
    return QualityDimensionResult(
        dimension=dimension,
        status=QualityStatus.PASSED,
        passed=True,
        score=score,
        reason=reason,
        issues=[],
        requested_difficulty=requested_difficulty,
        estimated_difficulty=estimated_difficulty,
    )


def _failed_result(
    dimension: QualityDimension,
    *,
    score: float,
    reason: str,
    issue: QualityIssue,
    requested_difficulty: DifficultyLevel | None = None,
    estimated_difficulty: DifficultyLevel | None = None,
) -> QualityDimensionResult:
    return QualityDimensionResult(
        dimension=dimension,
        status=QualityStatus.FAILED,
        passed=False,
        score=score,
        reason=reason,
        issues=[issue],
        requested_difficulty=requested_difficulty,
        estimated_difficulty=estimated_difficulty,
    )
