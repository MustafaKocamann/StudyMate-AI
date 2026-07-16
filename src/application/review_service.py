"""Review scheduling policy for learner attempts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from src.models.question_schemas import GeneratedQuestion
from src.models.study_session import AttemptOutcome, ConfidenceLevel, QuestionAttempt, ReviewItem


class ReviewPolicyConfig(BaseModel):
    """Configurable deterministic review intervals for the MVP."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    immediate_review_hours: int = Field(default=0, ge=0)
    incorrect_days: int = Field(default=1, ge=0)
    low_confidence_correct_days: int = Field(default=2, ge=1)
    medium_confidence_correct_days: int = Field(default=4, ge=1)
    high_confidence_correct_days: int = Field(default=7, ge=1)
    successful_review_intervals_days: list[int] = Field(default_factory=lambda: [14, 30])


class ReviewService:
    """Schedule reviews from objective outcomes and confidence.

    High-confidence incorrect and unknown answers are prioritized because they
    are useful review signals, but this is a transparent deterministic policy,
    not a claim of scientifically optimal spaced repetition.
    """

    def __init__(
        self,
        *,
        config: ReviewPolicyConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config or ReviewPolicyConfig()
        self.clock = clock or (lambda: datetime.now(UTC))

    def next_review_time(
        self,
        *,
        outcome: AttemptOutcome,
        confidence: ConfidenceLevel | None,
        successful_review_count: int = 0,
    ) -> datetime:
        now = self._now()
        if outcome == AttemptOutcome.UNKNOWN or (
            outcome == AttemptOutcome.INCORRECT
            and confidence in {ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH}
        ):
            return now + timedelta(hours=self.config.immediate_review_hours)
        if outcome == AttemptOutcome.INCORRECT:
            return now + timedelta(days=self.config.incorrect_days)
        if confidence in {ConfidenceLevel.GUESSED, ConfidenceLevel.LOW, None}:
            return now + timedelta(days=self.config.low_confidence_correct_days)
        if confidence == ConfidenceLevel.MEDIUM:
            return now + timedelta(days=self.config.medium_confidence_correct_days)
        if successful_review_count > 0:
            index = min(successful_review_count - 1, len(self.config.successful_review_intervals_days) - 1)
            return now + timedelta(days=self.config.successful_review_intervals_days[index])
        return now + timedelta(days=self.config.high_confidence_correct_days)

    def build_review_item(
        self,
        *,
        question: GeneratedQuestion,
        topic: str,
        attempt: QuestionAttempt,
        existing_item: ReviewItem | None = None,
    ) -> ReviewItem:
        now = self._now()
        successful_count = (
            (existing_item.successful_review_count + 1)
            if existing_item and attempt.outcome == AttemptOutcome.CORRECT
            else 0
        )
        repetition_count = (existing_item.repetition_count + 1) if existing_item else 0
        review_data = {
            "question": question,
            "topic": topic,
            "last_outcome": attempt.outcome,
            "confidence": attempt.confidence,
            "repetition_count": repetition_count,
            "successful_review_count": successful_count,
            "last_reviewed_at": now,
            "next_review_at": self.next_review_time(
                outcome=attempt.outcome,
                confidence=attempt.confidence,
                successful_review_count=successful_count,
            ),
            "resolved": successful_count >= 2,
        }
        if existing_item:
            review_data["review_item_id"] = existing_item.review_item_id
        return ReviewItem(**review_data)

    def _now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetimes")
        return now
