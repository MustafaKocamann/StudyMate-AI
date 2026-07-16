"""In-memory repository for local Streamlit state and deterministic tests."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from src.application.progress_service import ProgressService
from src.models.study_session import AttemptOutcome, ProgressSnapshot, QuestionAttempt, ReviewItem, StudySession


class DuplicateAttemptError(ValueError):
    """Raised when the same attempt is saved twice."""


class InMemoryStudyRepository:
    """Small non-durable store suitable for one Streamlit user session.

    Data is intentionally process-local and should not be treated as production
    persistence. The repository prevents duplicate attempt IDs and duplicate
    active review items for the same question so reruns do not inflate metrics.
    """

    def __init__(self, *, progress_service: ProgressService | None = None) -> None:
        self._sessions: dict[UUID, StudySession] = {}
        self._attempts: dict[UUID, QuestionAttempt] = {}
        self._review_items: dict[UUID, ReviewItem] = {}
        self.progress_service = progress_service or ProgressService()

    def save_session(self, session: StudySession) -> None:
        self._sessions[session.session_id] = session

    def get_session(self, session_id: UUID) -> StudySession | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[StudySession]:
        return list(self._sessions.values())

    def save_attempt(self, attempt: QuestionAttempt) -> None:
        if attempt.attempt_id in self._attempts:
            raise DuplicateAttemptError("attempt is already saved")
        self._attempts[attempt.attempt_id] = attempt

    def list_attempts(self) -> list[QuestionAttempt]:
        return list(self._attempts.values())

    def save_review_item(self, item: ReviewItem) -> None:
        existing = self.get_active_review_item_for_question(item.question.id)
        if existing and existing.review_item_id != item.review_item_id:
            raise ValueError("active review item already exists for this question")
        self._review_items[item.review_item_id] = item

    def get_active_review_item_for_question(self, question_id: UUID) -> ReviewItem | None:
        for item in self._review_items.values():
            if item.question.id == question_id and not item.resolved:
                return item
        return None

    def list_due_review_items(self, *, now: datetime) -> list[ReviewItem]:
        return [
            item
            for item in self._review_items.values()
            if item.next_review_at <= now and not item.resolved
        ]

    def list_mistakes(self) -> list[QuestionAttempt]:
        return [
            attempt
            for attempt in self._attempts.values()
            if attempt.outcome in {AttemptOutcome.INCORRECT, AttemptOutcome.UNKNOWN}
        ]

    def progress_snapshot(self, *, now: datetime) -> ProgressSnapshot:
        return self.progress_service.build_progress_snapshot(
            attempts=self.list_attempts(),
            review_items=list(self._review_items.values()),
            sessions=self.list_sessions(),
            now=now,
        )
