"""Tests for download-ready attempt export helpers."""

from __future__ import annotations

import builtins
import csv
import io
from datetime import UTC, datetime

from src.models.question_schemas import DifficultyLevel, QuestionType
from src.models.study_session import (
    AttemptOutcome,
    ConfidenceLevel,
    FillBlankLearnerAnswer,
    MCQLearnerAnswer,
    QuestionAttempt,
)
from src.ui.export_helpers import (
    EXPORT_COLUMNS,
    attempts_to_csv_bytes,
    attempts_to_export_rows,
    build_export_filename,
)


def _attempt(*, topic: str = "Python başlangıç", question_type: QuestionType = QuestionType.MCQ) -> QuestionAttempt:
    learner_answer = (
        MCQLearnerAnswer(selected_option_id="A")
        if question_type == QuestionType.MCQ
        else FillBlankLearnerAnswer(submitted_answer="başlatır")
    )
    return QuestionAttempt(
        question_id="00000000-0000-0000-0000-000000000001",
        session_id="00000000-0000-0000-0000-000000000002",
        learner_answer=learner_answer,
        outcome=AttemptOutcome.CORRECT,
        confidence=ConfidenceLevel.HIGH,
        hints_used=1,
        response_time_seconds=2.5,
        attempted_at=datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC),
        topic=topic,
        question_type=question_type,
        difficulty=DifficultyLevel.MEDIUM,
    )


def test_attempts_to_export_rows_has_deterministic_safe_shape() -> None:
    rows = attempts_to_export_rows([_attempt()])

    assert list(rows[0]) == list(EXPORT_COLUMNS)
    assert rows[0]["topic"] == "Python başlangıç"
    assert "question_id" not in rows[0]
    assert "session_id" not in rows[0]
    assert "prompt" not in rows[0]


def test_csv_bytes_include_bom_no_index_and_turkish_round_trip() -> None:
    data = attempts_to_csv_bytes([_attempt(question_type=QuestionType.FILL_BLANK)])

    assert data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    assert rows[0]["topic"] == "Python başlangıç"
    assert rows[0]["learner_answer"] == "başlatır"
    assert "" not in rows[0]


def test_empty_export_behavior_writes_only_header() -> None:
    data = attempts_to_csv_bytes([])

    assert data.decode("utf-8-sig") == ",".join(EXPORT_COLUMNS) + "\n"


def test_export_helpers_do_not_create_disk_files(monkeypatch) -> None:
    def fail_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("export helpers must not open files")

    monkeypatch.setattr(builtins, "open", fail_open)

    data = attempts_to_csv_bytes([_attempt()])

    assert data


def test_safe_timestamped_filename() -> None:
    filename = build_export_filename(
        prefix="Study Buddy: Türkçe / Session",
        now=datetime(2026, 7, 16, 12, 13, 14, tzinfo=UTC),
    )

    assert filename == "study-buddy-t-rk-e-session-20260716-121314.csv"
    assert "/" not in filename
    assert ":" not in filename
