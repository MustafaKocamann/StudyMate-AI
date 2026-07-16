"""Download-ready export helpers for learner attempts."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable
from datetime import UTC, datetime

from src.models.study_session import (
    FillBlankLearnerAnswer,
    MCQLearnerAnswer,
    QuestionAttempt,
)


EXPORT_COLUMNS = (
    "attempted_at",
    "topic",
    "question_type",
    "difficulty",
    "outcome",
    "learner_answer",
    "confidence",
    "hints_used",
    "response_time_seconds",
    "first_attempt",
)


def attempts_to_export_rows(attempts: Iterable[QuestionAttempt]) -> list[dict[str, object]]:
    """Return safe learner-facing rows without internal IDs or judge metadata."""

    return [
        {
            "attempted_at": attempt.attempted_at.astimezone(UTC).isoformat(),
            "topic": attempt.topic,
            "question_type": attempt.question_type.value,
            "difficulty": attempt.difficulty.value,
            "outcome": attempt.outcome.value,
            "learner_answer": _learner_answer_text(attempt),
            "confidence": int(attempt.confidence) if attempt.confidence is not None else "",
            "hints_used": attempt.hints_used,
            "response_time_seconds": attempt.response_time_seconds,
            "first_attempt": attempt.first_attempt,
        }
        for attempt in attempts
    ]


def dataframe_to_csv_bytes(rows: Iterable[dict[str, object]]) -> bytes:
    """Serialize export rows as UTF-8 with BOM for Turkish Excel compatibility."""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in EXPORT_COLUMNS})
    return buffer.getvalue().encode("utf-8-sig")


def attempts_to_csv_bytes(attempts: Iterable[QuestionAttempt]) -> bytes:
    return dataframe_to_csv_bytes(attempts_to_export_rows(attempts))


def build_export_filename(*, prefix: str = "study-buddy-attempts", now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", prefix.strip()).strip(".-").lower()
    return f"{safe_prefix or 'study-buddy-attempts'}-{timestamp}.csv"


def _learner_answer_text(attempt: QuestionAttempt) -> str:
    answer = attempt.learner_answer
    if isinstance(answer, MCQLearnerAnswer):
        return "I do not know" if answer.unknown else str(answer.selected_option_id)
    if isinstance(answer, FillBlankLearnerAnswer):
        return "I do not know" if answer.unknown else str(answer.submitted_answer)
    return ""
