"""Domain models for learner study sessions and attempts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.models.question_constraints import OptionId
from src.models.question_schemas import DifficultyLevel, GeneratedQuestion, QuestionType


StrictText = Annotated[str, StringConstraints(min_length=1, strict=True)]


class StrictStudyModel(BaseModel):
    """Shared strict configuration for learner-experience models."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class StudySessionStatus(StrEnum):
    """Lifecycle states for a study session."""

    CONFIGURED = "configured"
    GENERATING = "generating"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class StudyQuestionMode(StrEnum):
    """Question selection mode requested by the learner."""

    MCQ = "mcq"
    FILL_BLANK = "fill_blank"
    MIXED = "mixed"


class ConfidenceLevel(IntEnum):
    """Stored confidence scale; labels are owned by the UI."""

    GUESSED = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


class AttemptOutcome(StrEnum):
    """Objective result of a learner answer."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"


class MistakeCategory(StrEnum):
    """Deterministic learning-risk category for review prioritization."""

    KNOWLEDGE_GAP = "knowledge_gap"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_CONFIDENCE_MISCONCEPTION = "high_confidence_misconception"
    CARELESS_OR_UNCERTAIN = "careless_or_uncertain"
    RESOLVED = "resolved"


class MCQLearnerAnswer(StrictStudyModel):
    """Learner response for one multiple-choice question."""

    type: Literal[QuestionType.MCQ] = QuestionType.MCQ
    selected_option_id: OptionId | None = None
    unknown: bool = False

    @model_validator(mode="after")
    def validate_answer_shape(self) -> MCQLearnerAnswer:
        if self.unknown == (self.selected_option_id is None):
            return self
        raise ValueError("MCQ answer must be either an option selection or explicit unknown")


class FillBlankLearnerAnswer(StrictStudyModel):
    """Learner response for one fill-in-the-blank question."""

    type: Literal[QuestionType.FILL_BLANK] = QuestionType.FILL_BLANK
    submitted_answer: StrictText | None = None
    unknown: bool = False

    @model_validator(mode="after")
    def validate_answer_shape(self) -> FillBlankLearnerAnswer:
        if self.unknown == (self.submitted_answer is None):
            return self
        raise ValueError("fill-blank answer must be either submitted text or explicit unknown")


LearnerAnswer = Annotated[
    MCQLearnerAnswer | FillBlankLearnerAnswer,
    Field(discriminator="type"),
]
"""Discriminated union of supported learner answer models."""


class QuestionAttempt(StrictStudyModel):
    """Recorded learner attempt for analytics, feedback, and review scheduling."""

    attempt_id: UUID = Field(default_factory=uuid4)
    question_id: UUID
    session_id: UUID
    learner_answer: LearnerAnswer
    outcome: AttemptOutcome
    confidence: ConfidenceLevel | None = None
    hints_used: int = Field(default=0, ge=0, le=3)
    response_time_seconds: float = Field(default=0.0, ge=0.0)
    attempted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    first_attempt: bool = True
    topic: StrictText
    question_type: QuestionType
    difficulty: DifficultyLevel

    @model_validator(mode="after")
    def validate_timestamp(self) -> QuestionAttempt:
        if self.attempted_at.tzinfo is None:
            raise ValueError("attempted_at must be timezone-aware")
        return self


class StudySession(StrictStudyModel):
    """A rerun-safe unit of learner practice.

    The model owns state invariants that UI and repository code rely on:
    one-based current position, unique question identities, attempts tied only
    to contained questions, and completed sessions carrying a completion time.
    """

    session_id: UUID = Field(default_factory=uuid4)
    topic: StrictText
    requested_difficulty: DifficultyLevel
    requested_question_type: StudyQuestionMode
    language: StrictText
    questions: list[GeneratedQuestion] = Field(min_length=1, max_length=20)
    current_position: int = Field(default=1, ge=1)
    status: StudySessionStatus = StudySessionStatus.ACTIVE
    attempts: list[QuestionAttempt] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_session_invariants(self) -> StudySession:
        """Reject session states that would break Streamlit rerun navigation.

        Keeping these checks in the model prevents page code from accidentally
        skipping questions, completing a session without a timestamp, or saving
        attempts for questions outside the active set after a rerun.
        """

        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.completed_at is not None and self.completed_at.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")
        question_ids = [question.id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("session questions must have unique IDs")
        if self.current_position > len(self.questions):
            raise ValueError("current_position cannot exceed the question count")
        question_id_set = set(question_ids)
        if any(attempt.question_id not in question_id_set for attempt in self.attempts):
            raise ValueError("attempts must reference questions in the session")
        if self.status == StudySessionStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed sessions require completed_at")
        return self

    @property
    def current_question(self) -> GeneratedQuestion:
        return self.questions[self.current_position - 1]


class SessionSummary(StrictStudyModel):
    """Aggregate learning metrics for a completed or active session."""

    session_id: UUID
    total_questions: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    incorrect_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    first_attempt_accuracy: float = Field(ge=0.0, le=1.0)
    average_confidence: float | None = Field(default=None, ge=1.0, le=5.0)
    hints_used: int = Field(ge=0)
    high_confidence_incorrect_count: int = Field(ge=0)
    weak_question_ids: list[UUID] = Field(default_factory=list)
    recommended_next_action: str


class ReviewItem(StrictStudyModel):
    """Scheduled question review item for the in-memory MVP repository."""

    review_item_id: UUID = Field(default_factory=uuid4)
    question: GeneratedQuestion
    topic: StrictText
    last_outcome: AttemptOutcome
    confidence: ConfidenceLevel | None = None
    repetition_count: int = Field(default=0, ge=0)
    successful_review_count: int = Field(default=0, ge=0)
    last_reviewed_at: datetime
    next_review_at: datetime
    resolved: bool = False

    @model_validator(mode="after")
    def validate_review_timestamps(self) -> ReviewItem:
        if self.last_reviewed_at.tzinfo is None or self.next_review_at.tzinfo is None:
            raise ValueError("review timestamps must be timezone-aware")
        if self.next_review_at < self.last_reviewed_at:
            raise ValueError("next_review_at cannot be before last_reviewed_at")
        return self


class ProgressSnapshot(StrictStudyModel):
    """Current aggregate progress metrics for dashboard rendering."""

    total_questions_answered: int = Field(ge=0)
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    first_attempt_accuracy: float = Field(ge=0.0, le=1.0)
    average_confidence: float | None = Field(default=None, ge=1.0, le=5.0)
    high_confidence_wrong_count: int = Field(ge=0)
    hints_used: int = Field(ge=0)
    review_due_count: int = Field(ge=0)
    accuracy_by_difficulty: dict[DifficultyLevel, float] = Field(default_factory=dict)
    accuracy_by_question_type: dict[QuestionType, float] = Field(default_factory=dict)
    recent_session_summaries: list[SessionSummary] = Field(default_factory=list)
