"""Learner answer evaluation for validated question domain models."""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.normalization import normalize_for_comparison
from src.models.question_schemas import FillBlankQuestion, GeneratedQuestion, MCQQuestion
from src.models.study_session import (
    AttemptOutcome,
    ConfidenceLevel,
    FillBlankLearnerAnswer,
    LearnerAnswer,
    MCQLearnerAnswer,
    MistakeCategory,
)


@dataclass(frozen=True)
class AnswerEvaluationResult:
    """Safe feedback payload returned after a learner submits an answer."""

    outcome: AttemptOutcome
    learner_answer_text: str
    correct_answer_text: str
    explanation: str
    mistake_category: MistakeCategory


class AnswerEvaluationService:
    """Evaluate learner answers without calling an LLM or mutating questions.

    MCQ correctness is exact option-ID equality. Fill-blank comparison is a
    conservative lexical normalization path using Unicode normalization,
    surrounding whitespace trim, internal whitespace collapse, case-insensitive
    comparison, and the existing Turkish-aware normalization utility. Semantic
    alternative answers are intentionally left for a future evaluator layer.
    """

    def evaluate(
        self,
        *,
        question: GeneratedQuestion,
        learner_answer: LearnerAnswer,
        confidence: ConfidenceLevel | None,
        language: str | None = None,
    ) -> AnswerEvaluationResult:
        if isinstance(question, MCQQuestion) and isinstance(learner_answer, MCQLearnerAnswer):
            return self._evaluate_mcq(question, learner_answer, confidence)
        if isinstance(question, FillBlankQuestion) and isinstance(
            learner_answer, FillBlankLearnerAnswer
        ):
            return self._evaluate_fill_blank(question, learner_answer, confidence, language)
        raise ValueError("learner answer type must match question type")

    def _evaluate_mcq(
        self,
        question: MCQQuestion,
        learner_answer: MCQLearnerAnswer,
        confidence: ConfidenceLevel | None,
    ) -> AnswerEvaluationResult:
        correct_option = next(
            option for option in question.options if option.id == question.correct_option_id
        )
        if learner_answer.unknown:
            outcome = AttemptOutcome.UNKNOWN
            learner_text = "I do not know"
        else:
            assert learner_answer.selected_option_id is not None
            outcome = (
                AttemptOutcome.CORRECT
                if learner_answer.selected_option_id == question.correct_option_id
                else AttemptOutcome.INCORRECT
            )
            selected_option = next(
                option for option in question.options if option.id == learner_answer.selected_option_id
            )
            learner_text = f"{selected_option.id}. {selected_option.text}"

        return AnswerEvaluationResult(
            outcome=outcome,
            learner_answer_text=learner_text,
            correct_answer_text=f"{correct_option.id}. {correct_option.text}",
            explanation=question.explanation,
            mistake_category=classify_mistake(outcome, confidence),
        )

    def _evaluate_fill_blank(
        self,
        question: FillBlankQuestion,
        learner_answer: FillBlankLearnerAnswer,
        confidence: ConfidenceLevel | None,
        language: str | None,
    ) -> AnswerEvaluationResult:
        if learner_answer.unknown:
            outcome = AttemptOutcome.UNKNOWN
            learner_text = "I do not know"
        else:
            assert learner_answer.submitted_answer is not None
            learner_text = learner_answer.submitted_answer
            outcome = (
                AttemptOutcome.CORRECT
                if normalize_for_comparison(learner_answer.submitted_answer, language=language)
                == normalize_for_comparison(question.answer, language=language)
                else AttemptOutcome.INCORRECT
            )

        return AnswerEvaluationResult(
            outcome=outcome,
            learner_answer_text=learner_text,
            correct_answer_text=question.answer,
            explanation=question.explanation,
            mistake_category=classify_mistake(outcome, confidence),
        )


def classify_mistake(
    outcome: AttemptOutcome,
    confidence: ConfidenceLevel | None,
    *,
    repeated_successes: int = 0,
) -> MistakeCategory:
    """Classify attempts for learner review without making psychological claims."""

    if repeated_successes >= 2 and outcome == AttemptOutcome.CORRECT:
        return MistakeCategory.RESOLVED
    if outcome == AttemptOutcome.UNKNOWN:
        return MistakeCategory.KNOWLEDGE_GAP
    if outcome == AttemptOutcome.INCORRECT and confidence in {
        ConfidenceLevel.HIGH,
        ConfidenceLevel.VERY_HIGH,
    }:
        return MistakeCategory.HIGH_CONFIDENCE_MISCONCEPTION
    if outcome == AttemptOutcome.INCORRECT and confidence in {
        ConfidenceLevel.GUESSED,
        ConfidenceLevel.LOW,
    }:
        return MistakeCategory.LOW_CONFIDENCE
    if outcome == AttemptOutcome.CORRECT and confidence in {
        ConfidenceLevel.GUESSED,
        ConfidenceLevel.LOW,
    }:
        return MistakeCategory.CARELESS_OR_UNCERTAIN
    return MistakeCategory.LOW_CONFIDENCE if outcome == AttemptOutcome.INCORRECT else MistakeCategory.RESOLVED
