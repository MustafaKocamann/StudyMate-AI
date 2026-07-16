"""Repository protocol for learner sessions, attempts, and reviews."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from src.models.study_session import (
    ProgressSnapshot,
    QuestionAttempt,
    ReviewItem,
    StudySession,
)


class StudyRepository(Protocol):
    """Persistence boundary for the Streamlit MVP and future durable stores."""

    def save_session(self, session: StudySession) -> None: ...

    def get_session(self, session_id: UUID) -> StudySession | None: ...

    def list_sessions(self) -> list[StudySession]: ...

    def save_attempt(self, attempt: QuestionAttempt) -> None: ...

    def list_attempts(self) -> list[QuestionAttempt]: ...

    def save_review_item(self, item: ReviewItem) -> None: ...

    def get_active_review_item_for_question(self, question_id: UUID) -> ReviewItem | None: ...

    def list_due_review_items(self, *, now: datetime) -> list[ReviewItem]: ...

    def list_mistakes(self) -> list[QuestionAttempt]: ...

    def progress_snapshot(self, *, now: datetime) -> ProgressSnapshot: ...
