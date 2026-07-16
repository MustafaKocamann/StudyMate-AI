"""Tests for the Streamlit learning-experience domain and services."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.application.answer_evaluation_service import AnswerEvaluationService
from src.application.hint_service import HintRequest
from src.application.progress_service import ProgressService
from src.application.review_service import ReviewService
from src.application.study_session_service import AnswerSubmissionConflictError, StudySessionService
from src.models.question_schemas import DifficultyLevel, FillBlankQuestion, MCQQuestion, QuestionOption, QuestionSet, QuestionType
from src.models.study_session import (
    AttemptOutcome,
    ConfidenceLevel,
    FillBlankLearnerAnswer,
    MCQLearnerAnswer,
    MistakeCategory,
    StudyQuestionMode,
    StudySession,
    StudySessionStatus,
)
from src.repositories.in_memory_study_repository import DuplicateAttemptError, InMemoryStudyRepository


def mcq_question(position: int = 1) -> MCQQuestion:
    return MCQQuestion(
        type=QuestionType.MCQ,
        position=position,
        question="What does len return in Python?",
        difficulty=DifficultyLevel.EASY,
        explanation="len returns the number of items in a supported collection.",
        options=[
            QuestionOption(id="A", text="The number of items"),
            QuestionOption(id="B", text="The last item"),
            QuestionOption(id="C", text="The first item"),
            QuestionOption(id="D", text="The collection type"),
        ],
        correct_option_id="A",
    )


def fill_question(position: int = 1) -> FillBlankQuestion:
    return FillBlankQuestion(
        type=QuestionType.FILL_BLANK,
        position=position,
        question="Python'da __init__ metodu nesneyi ___ eder.",
        difficulty=DifficultyLevel.MEDIUM,
        explanation="__init__ yeni nesnenin başlangıç durumunu kurar.",
        answer="başlatır",
    )


def question_set() -> QuestionSet:
    return QuestionSet(questions=[mcq_question(1), fill_question(2)])


def test_study_session_creation_and_invariants() -> None:
    session = StudySession(
        topic="Python",
        requested_difficulty=DifficultyLevel.EASY,
        requested_question_type=StudyQuestionMode.MIXED,
        language="Turkish",
        questions=question_set().questions,
    )

    assert session.current_question.position == 1
    with pytest.raises(ValidationError):
        session.current_position = 3
    with pytest.raises(ValidationError):
        StudySession(
            topic="Python",
            requested_difficulty=DifficultyLevel.EASY,
            requested_question_type=StudyQuestionMode.MCQ,
            language="English",
            questions=[],
        )


def test_study_session_service_appends_lazy_question_safely() -> None:
    service = StudySessionService()
    first = mcq_question(1)
    session = service.start_session(
        topic="Python",
        difficulty=DifficultyLevel.EASY,
        question_mode=StudyQuestionMode.MIXED,
        language="English",
        question_set=QuestionSet(questions=[first]),
    )

    updated = service.append_question(session, fill_question(2))

    assert len(session.questions) == 1
    assert len(updated.questions) == 2
    assert updated.questions[1].position == 2
    with pytest.raises(ValueError):
        service.append_question(updated, first)
    with pytest.raises(ValidationError):
        StudySession(
            topic="Python",
            requested_difficulty=DifficultyLevel.EASY,
            requested_question_type=StudyQuestionMode.MCQ,
            language="English",
            questions=[mcq_question()],
            started_at=datetime.now(),
        )


def test_answer_model_discrimination_and_extra_fields() -> None:
    assert MCQLearnerAnswer(selected_option_id="A").type == QuestionType.MCQ
    assert FillBlankLearnerAnswer(submitted_answer=" başlatır ").submitted_answer == "başlatır"
    with pytest.raises(ValidationError):
        MCQLearnerAnswer()
    with pytest.raises(ValidationError):
        FillBlankLearnerAnswer(submitted_answer="x", extra_field=True)  # type: ignore[call-arg]


def test_answer_evaluation_mcq_and_unknown() -> None:
    service = AnswerEvaluationService()
    question = mcq_question()

    correct = service.evaluate(
        question=question,
        learner_answer=MCQLearnerAnswer(selected_option_id="A"),
        confidence=ConfidenceLevel.HIGH,
    )
    incorrect = service.evaluate(
        question=question,
        learner_answer=MCQLearnerAnswer(selected_option_id="B"),
        confidence=ConfidenceLevel.VERY_HIGH,
    )
    unknown = service.evaluate(
        question=question,
        learner_answer=MCQLearnerAnswer(unknown=True),
        confidence=ConfidenceLevel.GUESSED,
    )

    assert correct.outcome == AttemptOutcome.CORRECT
    assert incorrect.outcome == AttemptOutcome.INCORRECT
    assert incorrect.mistake_category == MistakeCategory.HIGH_CONFIDENCE_MISCONCEPTION
    assert unknown.outcome == AttemptOutcome.UNKNOWN


def test_answer_evaluation_fill_blank_normalization_and_turkish() -> None:
    service = AnswerEvaluationService()
    question = fill_question()

    result = service.evaluate(
        question=question,
        learner_answer=FillBlankLearnerAnswer(submitted_answer="  BAŞLATIR  "),
        confidence=ConfidenceLevel.MEDIUM,
        language="Turkish",
    )

    assert result.outcome == AttemptOutcome.CORRECT
    assert question.answer == "başlatır"


def test_study_session_submission_and_duplicate_protection() -> None:
    session_service = StudySessionService()
    session = session_service.start_session(
        topic="Python",
        difficulty=DifficultyLevel.EASY,
        question_mode=StudyQuestionMode.MIXED,
        language="English",
        question_set=question_set(),
    )

    updated, attempt, feedback = session_service.submit_answer(
        session=session,
        learner_answer=MCQLearnerAnswer(selected_option_id="A"),
        confidence=ConfidenceLevel.HIGH,
        hints_used=1,
    )

    assert attempt.outcome == AttemptOutcome.CORRECT
    assert feedback.correct_answer_text.startswith("A.")
    with pytest.raises(AnswerSubmissionConflictError):
        session_service.submit_answer(
            session=updated,
            learner_answer=MCQLearnerAnswer(selected_option_id="A"),
            confidence=ConfidenceLevel.HIGH,
        )


def test_review_policy_scheduling_and_no_past_dates() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    service = ReviewService(clock=lambda: now)

    immediate = service.next_review_time(
        outcome=AttemptOutcome.INCORRECT,
        confidence=ConfidenceLevel.HIGH,
    )
    later = service.next_review_time(
        outcome=AttemptOutcome.CORRECT,
        confidence=ConfidenceLevel.HIGH,
        successful_review_count=1,
    )

    assert immediate == now
    assert later == now + timedelta(days=14)


def test_repository_and_progress_snapshot() -> None:
    repo = InMemoryStudyRepository()
    session_service = StudySessionService()
    session = session_service.start_session(
        topic="Python",
        difficulty=DifficultyLevel.EASY,
        question_mode=StudyQuestionMode.MCQ,
        language="English",
        question_set=QuestionSet(questions=[mcq_question()]),
    )
    updated, attempt, _ = session_service.submit_answer(
        session=session,
        learner_answer=MCQLearnerAnswer(selected_option_id="B"),
        confidence=ConfidenceLevel.VERY_HIGH,
    )

    repo.save_session(updated)
    repo.save_attempt(attempt)
    with pytest.raises(DuplicateAttemptError):
        repo.save_attempt(attempt)

    snapshot = repo.progress_snapshot(now=datetime.now(UTC))
    assert snapshot.total_questions_answered == 1
    assert snapshot.high_confidence_wrong_count == 1
    assert repo.list_mistakes() == [attempt]


def test_session_summary_recommendation_threshold() -> None:
    service = ProgressService()
    session_service = StudySessionService()
    session = session_service.start_session(
        topic="Python",
        difficulty=DifficultyLevel.EASY,
        question_mode=StudyQuestionMode.MCQ,
        language="English",
        question_set=QuestionSet(questions=[mcq_question()]),
    )
    summary = service.summarize_session(session)

    assert summary.total_questions == 0
    assert summary.accuracy == 0.0
    assert "Keep practicing" in summary.recommended_next_action


def test_hint_request_supports_three_levels_only() -> None:
    assert HintRequest(question=mcq_question(), level=3).level == 3
    with pytest.raises(ValidationError):
        HintRequest(question=mcq_question(), level=4)


def test_completed_session_requires_timestamp() -> None:
    with pytest.raises(ValidationError):
        StudySession(
            topic="Python",
            requested_difficulty=DifficultyLevel.EASY,
            requested_question_type=StudyQuestionMode.MCQ,
            language="English",
            questions=[mcq_question()],
            status=StudySessionStatus.COMPLETED,
        )
