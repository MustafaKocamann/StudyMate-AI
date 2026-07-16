"""Architecture checks for the Streamlit study flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.study_session_service import AnswerSubmissionConflictError, StudySessionService
from src.models.question_schemas import DifficultyLevel, MCQQuestion, QuestionOption, QuestionSet, QuestionType
from src.models.study_session import ConfidenceLevel, MCQLearnerAnswer, StudyQuestionMode


def _question() -> MCQQuestion:
    return MCQQuestion(
        type=QuestionType.MCQ,
        position=1,
        question="What does len return?",
        difficulty=DifficultyLevel.EASY,
        explanation="It returns the number of items.",
        options=[
            QuestionOption(id="A", text="The number of items"),
            QuestionOption(id="B", text="The final item"),
            QuestionOption(id="C", text="The first item"),
            QuestionOption(id="D", text="The type name"),
        ],
        correct_option_id="A",
    )


def test_duplicate_answer_submission_is_rejected_by_session_service() -> None:
    service = StudySessionService()
    session = service.start_session(
        topic="Python",
        difficulty=DifficultyLevel.EASY,
        question_mode=StudyQuestionMode.MCQ,
        language="English",
        question_set=QuestionSet(questions=[_question()]),
    )
    updated_session, _, _ = service.submit_answer(
        session=session,
        learner_answer=MCQLearnerAnswer(selected_option_id="A"),
        confidence=ConfidenceLevel.HIGH,
    )

    with pytest.raises(AnswerSubmissionConflictError):
        service.submit_answer(
            session=updated_session,
            learner_answer=MCQLearnerAnswer(selected_option_id="A"),
            confidence=ConfidenceLevel.HIGH,
        )


def test_active_ui_has_no_quiz_manager_dependency() -> None:
    source_files = [*Path("src").rglob("*.py"), *Path("streamlit_app.py").parent.glob("streamlit_app.py")]
    offenders = [
        path.as_posix()
        for path in source_files
        if "QuizManager" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
