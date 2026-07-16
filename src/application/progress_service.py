"""Session summary and progress aggregation services."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from src.models.question_schemas import DifficultyLevel, QuestionType
from src.models.study_session import (
    AttemptOutcome,
    ConfidenceLevel,
    ProgressSnapshot,
    QuestionAttempt,
    ReviewItem,
    SessionSummary,
    StudySession,
)


class ProgressService:
    """Build reusable learning summaries for UI, tests, and future persistence."""

    def summarize_session(self, session: StudySession) -> SessionSummary:
        attempts = session.attempts
        total = len(attempts)
        correct = sum(attempt.outcome == AttemptOutcome.CORRECT for attempt in attempts)
        incorrect = sum(attempt.outcome == AttemptOutcome.INCORRECT for attempt in attempts)
        unknown = sum(attempt.outcome == AttemptOutcome.UNKNOWN for attempt in attempts)
        first_attempts = [attempt for attempt in attempts if attempt.first_attempt]
        confidence_values = [int(attempt.confidence) for attempt in attempts if attempt.confidence]
        high_conf_wrong = sum(
            attempt.outcome == AttemptOutcome.INCORRECT
            and attempt.confidence in {ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH}
            for attempt in attempts
        )
        return SessionSummary(
            session_id=session.session_id,
            total_questions=total,
            correct_count=correct,
            incorrect_count=incorrect,
            unknown_count=unknown,
            accuracy=_ratio(correct, total),
            first_attempt_accuracy=_ratio(
                sum(attempt.outcome == AttemptOutcome.CORRECT for attempt in first_attempts),
                len(first_attempts),
            ),
            average_confidence=(
                sum(confidence_values) / len(confidence_values) if confidence_values else None
            ),
            hints_used=sum(attempt.hints_used for attempt in attempts),
            high_confidence_incorrect_count=high_conf_wrong,
            weak_question_ids=[
                attempt.question_id
                for attempt in attempts
                if attempt.outcome in {AttemptOutcome.INCORRECT, AttemptOutcome.UNKNOWN}
            ],
            recommended_next_action=self.recommend_next_action(attempts),
        )

    def build_progress_snapshot(
        self,
        *,
        attempts: list[QuestionAttempt],
        review_items: list[ReviewItem],
        sessions: list[StudySession],
        now: datetime | None = None,
    ) -> ProgressSnapshot:
        now = now or datetime.now(UTC)
        summaries = [self.summarize_session(session) for session in sessions[-5:]]
        total = len(attempts)
        correct = sum(attempt.outcome == AttemptOutcome.CORRECT for attempt in attempts)
        first_attempts = [attempt for attempt in attempts if attempt.first_attempt]
        confidence_values = [int(attempt.confidence) for attempt in attempts if attempt.confidence]
        return ProgressSnapshot(
            total_questions_answered=total,
            overall_accuracy=_ratio(correct, total),
            first_attempt_accuracy=_ratio(
                sum(attempt.outcome == AttemptOutcome.CORRECT for attempt in first_attempts),
                len(first_attempts),
            ),
            average_confidence=(
                sum(confidence_values) / len(confidence_values) if confidence_values else None
            ),
            high_confidence_wrong_count=sum(
                attempt.outcome == AttemptOutcome.INCORRECT
                and attempt.confidence in {ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH}
                for attempt in attempts
            ),
            hints_used=sum(attempt.hints_used for attempt in attempts),
            review_due_count=sum(item.next_review_at <= now and not item.resolved for item in review_items),
            accuracy_by_difficulty=self._accuracy_by_difficulty(attempts),
            accuracy_by_question_type=self._accuracy_by_question_type(attempts),
            recent_session_summaries=summaries,
        )

    def recommend_next_action(self, attempts: list[QuestionAttempt]) -> str:
        if len(attempts) < 5:
            return "Keep practicing to build a clearer progress signal."
        confidence_values = [int(attempt.confidence) for attempt in attempts if attempt.confidence]
        accuracy = _ratio(sum(attempt.outcome == AttemptOutcome.CORRECT for attempt in attempts), len(attempts))
        average_confidence = (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        )
        if accuracy >= 0.80 and average_confidence >= 3.5:
            return "Consider increasing difficulty for the next session."
        if accuracy < 0.50:
            return "Review this concept before increasing difficulty."
        return "Continue at the same difficulty."

    def _accuracy_by_difficulty(self, attempts: list[QuestionAttempt]) -> dict[DifficultyLevel, float]:
        grouped: dict[DifficultyLevel, list[QuestionAttempt]] = defaultdict(list)
        for attempt in attempts:
            grouped[attempt.difficulty].append(attempt)
        return {
            difficulty: _ratio(
                sum(attempt.outcome == AttemptOutcome.CORRECT for attempt in grouped_attempts),
                len(grouped_attempts),
            )
            for difficulty, grouped_attempts in grouped.items()
        }

    def _accuracy_by_question_type(self, attempts: list[QuestionAttempt]) -> dict[QuestionType, float]:
        grouped: dict[QuestionType, list[QuestionAttempt]] = defaultdict(list)
        for attempt in attempts:
            grouped[attempt.question_type].append(attempt)
        return {
            question_type: _ratio(
                sum(attempt.outcome == AttemptOutcome.CORRECT for attempt in grouped_attempts),
                len(grouped_attempts),
            )
            for question_type, grouped_attempts in grouped.items()
        }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
