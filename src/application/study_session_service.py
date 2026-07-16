"""Application service for study-session state transitions."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from src.application.answer_evaluation_service import AnswerEvaluationResult, AnswerEvaluationService
from src.models.question_schemas import GeneratedQuestion, QuestionSet
from src.models.study_session import (
    ConfidenceLevel,
    LearnerAnswer,
    QuestionAttempt,
    StudyQuestionMode,
    StudySession,
    StudySessionStatus,
)


class AnswerSubmissionConflictError(ValueError):
    """Raised when a rerun tries to record an already submitted answer."""


class StudySessionService:
    """Coordinate learner attempts without embedding rules in Streamlit pages."""

    def __init__(self, *, answer_evaluator: AnswerEvaluationService | None = None) -> None:
        self.answer_evaluator = answer_evaluator or AnswerEvaluationService()

    def start_session(
        self,
        *,
        topic: str,
        difficulty,
        question_mode: StudyQuestionMode,
        language: str,
        question_set: QuestionSet,
    ) -> StudySession:
        return StudySession(
            topic=topic,
            requested_difficulty=difficulty,
            requested_question_type=question_mode,
            language=language,
            questions=question_set.questions,
            status=StudySessionStatus.ACTIVE,
        )

    def submit_answer(
        self,
        *,
        session: StudySession,
        learner_answer: LearnerAnswer,
        confidence: ConfidenceLevel | None,
        hints_used: int = 0,
        response_time_seconds: float = 0.0,
    ) -> tuple[StudySession, QuestionAttempt, AnswerEvaluationResult]:
        """Record exactly one attempt for the current question.

        Streamlit reruns can replay button states; this method rejects a second
        attempt for the same question while the session remains on that
        position, preventing accidental double scoring and duplicate review
        items.
        """

        question = session.current_question
        if any(attempt.question_id == question.id for attempt in session.attempts):
            raise AnswerSubmissionConflictError("answer for current question was already recorded")

        result = self.answer_evaluator.evaluate(
            question=question,
            learner_answer=learner_answer,
            confidence=confidence,
            language=session.language,
        )
        attempt = QuestionAttempt(
            question_id=question.id,
            session_id=session.session_id,
            learner_answer=learner_answer,
            outcome=result.outcome,
            confidence=confidence,
            hints_used=hints_used,
            response_time_seconds=response_time_seconds,
            topic=session.topic,
            question_type=question.type,
            difficulty=question.difficulty,
        )
        updated_attempts = [*session.attempts, attempt]
        updated_session = session.model_copy(update={"attempts": updated_attempts})
        return updated_session, attempt, result

    def advance(self, session: StudySession) -> StudySession:
        if session.current_position == len(session.questions):
            return session.model_copy(
                update={
                    "status": StudySessionStatus.COMPLETED,
                    "completed_at": datetime.now(UTC),
                }
            )
        return session.model_copy(update={"current_position": session.current_position + 1})

    def question_by_id(self, session: StudySession, question_id: UUID) -> GeneratedQuestion:
        return next(question for question in session.questions if question.id == question_id)
