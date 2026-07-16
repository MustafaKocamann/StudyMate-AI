"""Semantic AppTest coverage for the learner-facing Streamlit flows."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def _practice_source(*, question_type: str = "mcq", fail: bool = False, hints: bool = False) -> str:
    question_code = (
        """
        question = MCQQuestion(
            type=QuestionType.MCQ,
            position=position,
            question="Which structure stores key-value pairs?",
            difficulty=difficulty,
            explanation="A dictionary maps unique keys to values.",
            options=[
                QuestionOption(id="A", text="Dictionary"),
                QuestionOption(id="B", text="Tuple"),
                QuestionOption(id="C", text="Set"),
                QuestionOption(id="D", text="String"),
            ],
            correct_option_id="A",
        )
        """
        if question_type == "mcq"
        else """
        question = FillBlankQuestion(
            type=QuestionType.FILL_BLANK,
            position=position,
            question="A Python mapping is called a ___.",
            difficulty=difficulty,
            explanation="A dictionary stores mappings from keys to values.",
            answer="dictionary",
        )
        """
    )
    service_body = (
        'raise RuntimeError("private provider detail")'
        if fail else """
        self.calls += 1
        return QuestionSet(questions=[self._build_question(kwargs["difficulty"], 1)])
        """
    )
    next_service_body = (
        'raise RuntimeError("private provider detail")'
        if fail else """
        self.calls += 1
        return SimpleNamespace(
            question=self._build_question(request.difficulty, request.position)
        )
        """
    )
    hint_setup = (
        """
class FakeHintProvider:
    async def get_hint(self, question, level):
        return f"Hint {level}: think about Python collection types."
"""
        if hints
        else ""
    )
    hint_state = (
        "st.session_state[StateKey.HINT_PROVIDER.value] = FakeHintProvider()"
        if hints
        else ""
    )
    return f"""
import streamlit as st
from types import SimpleNamespace
from src.models.question_schemas import (
    DifficultyLevel,
    FillBlankQuestion,
    MCQQuestion,
    QuestionOption,
    QuestionSet,
    QuestionType,
)
from src.ui.pages.practice import render_practice_page
from src.ui.state import StateKey, initialize_state

class FakeQuestionService:
    def __init__(self):
        self.calls = 0

    def _build_question(self, difficulty, position):
{question_code}
        return question

    async def generate_questions(self, **kwargs):
        {service_body}

    async def generate_question(self, request, **kwargs):
        {next_service_body}

{hint_setup}
initialize_state()
if st.session_state.get(StateKey.QUESTION_SERVICE.value) is None:
    st.session_state[StateKey.QUESTION_SERVICE.value] = FakeQuestionService()
    {hint_state}
render_practice_page()
"""


def _start_practice(
    *,
    question_type: str = "mcq",
    hints: bool = False,
    question_count: int = 1,
) -> AppTest:
    app = AppTest.from_string(
        _practice_source(question_type=question_type, hints=hints)
    ).run(timeout=10)
    next(
        field for field in app.text_input if field.label == "What would you like to study?"
    ).set_value("Python dictionaries")
    next(
        field for field in app.number_input if field.label == "Number of questions"
    ).set_value(question_count)
    if hints:
        next(
            checkbox for checkbox in app.checkbox if checkbox.label == "Enable progressive hints"
        ).check()
    next(button for button in app.button if button.label == "Generate session").click().run(
        timeout=10
    )
    return app


def _visible_text(app: AppTest) -> str:
    elements = [
        *app.title,
        *app.subheader,
        *app.markdown,
        *app.caption,
        *app.info,
        *app.warning,
        *app.success,
        *app.error,
    ]
    return "\n".join(str(element.value) for element in elements)


def test_navigation_and_home_primary_action_open_practice() -> None:
    app = AppTest.from_file("streamlit_app.py").run(timeout=10)

    assert not app.exception
    start_button = next(button for button in app.button if button.label == "Start studying")

    start_button.click().run(timeout=10)

    assert not app.exception
    assert any(title.value == "Practice" for title in app.title)


def test_practice_requires_topic_before_generation() -> None:
    app = AppTest.from_string(_practice_source()).run(timeout=10)

    next(button for button in app.button if button.label == "Generate session").click().run(
        timeout=10
    )

    assert any("study topic" in error.value for error in app.error)
    assert app.session_state["study_question_service"].calls == 0


def test_valid_configuration_calls_service_once_and_hides_explanation() -> None:
    app = _start_practice()

    assert not app.exception
    assert app.session_state["study_question_service"].calls == 1
    assert len(app.radio) == 1
    assert "A dictionary maps unique keys to values." not in _visible_text(app)

    app.run(timeout=10)

    assert app.session_state["study_question_service"].calls == 1
    assert len(app.radio) == 1


def test_generation_failure_closes_status_and_offers_safe_retry() -> None:
    app = AppTest.from_string(_practice_source(fail=True)).run(timeout=10)
    next(
        field for field in app.text_input if field.label == "What would you like to study?"
    ).set_value("Python dictionaries")

    next(button for button in app.button if button.label == "Generate session").click().run(
        timeout=10
    )

    assert not app.exception
    assert len(app.status) == 0
    assert any(button.label == "Try again" for button in app.button)
    assert "private provider detail" not in _visible_text(app)


def test_mcq_feedback_confidence_and_duplicate_rerun_safety() -> None:
    app = _start_practice()
    app.radio[0].set_value(app.radio[0].options[0])
    confidence = next(
        field for field in app.selectbox if field.label == "How confident are you?"
    )
    confidence.set_value(confidence.options[4])

    next(button for button in app.button if button.label == "Check answer").click().run(
        timeout=10
    )

    assert [success.value for success in app.success] == ["Correct"]
    assert "A dictionary maps unique keys to values." in _visible_text(app)
    assert len(app.session_state["study_repository"].list_attempts()) == 1


def test_practice_generates_follow_up_question_lazily() -> None:
    app = _start_practice(question_count=2)
    assert app.session_state["study_question_service"].calls == 1
    assert len(app.session_state["active_session"].questions) == 1

    app.radio[0].set_value(app.radio[0].options[0])
    confidence = next(
        field for field in app.selectbox if field.label == "How confident are you?"
    )
    confidence.set_value(confidence.options[3])
    next(button for button in app.button if button.label == "Check answer").click().run(
        timeout=10
    )
    next(button for button in app.button if button.label == "Next question").click().run(
        timeout=10
    )

    assert not app.exception
    assert app.session_state["study_question_service"].calls == 2
    assert len(app.session_state["active_session"].questions) == 2
    assert app.session_state["active_session"].current_position == 2

    app.run(timeout=10)

    assert len(app.session_state["study_repository"].list_attempts()) == 1


def test_fill_blank_and_i_do_not_know_feedback() -> None:
    fill_app = _start_practice(question_type="fill_blank")
    next(field for field in fill_app.text_input if field.label == "Your answer").set_value(
        "dictionary"
    )
    fill_confidence = next(
        field for field in fill_app.selectbox if field.label == "How confident are you?"
    )
    fill_confidence.set_value(fill_confidence.options[3])
    next(
        button for button in fill_app.button if button.label == "Check answer"
    ).click().run(timeout=10)

    assert [success.value for success in fill_app.success] == ["Correct"]

    unknown_app = _start_practice()
    next(
        checkbox for checkbox in unknown_app.checkbox if checkbox.label == "I do not know"
    ).check()
    unknown_confidence = next(
        field for field in unknown_app.selectbox if field.label == "How confident are you?"
    )
    unknown_confidence.set_value(unknown_confidence.options[1])
    next(
        button for button in unknown_app.button if button.label == "Check answer"
    ).click().run(timeout=10)

    assert any(info.value == "I do not know" for info in unknown_app.info)


def test_confidence_is_required_and_hint_panel_needs_content() -> None:
    confidence_app = _start_practice()
    confidence_app.radio[0].set_value(confidence_app.radio[0].options[0])
    next(
        button for button in confidence_app.button if button.label == "Check answer"
    ).click().run(timeout=10)

    assert any("confidence level" in error.value for error in confidence_app.error)

    hint_app = _start_practice(hints=True)
    assert not any("Hint 1:" in info.value for info in hint_app.info)
    assert any(button.label == "Get a hint" for button in hint_app.button)

    next(button for button in hint_app.button if button.label == "Get a hint").click().run(
        timeout=10
    )

    assert any("Hint 1:" in info.value for info in hint_app.info)
    assert any("Hint 1 of 3" in caption.value for caption in hint_app.caption)


def test_session_completion_summary_and_download_render() -> None:
    app = _start_practice()
    app.radio[0].set_value(app.radio[0].options[0])
    confidence = next(
        field for field in app.selectbox if field.label == "How confident are you?"
    )
    confidence.set_value(confidence.options[4])
    next(button for button in app.button if button.label == "Check answer").click().run(
        timeout=10
    )
    next(button for button in app.button if button.label == "Finish session").click().run(
        timeout=10
    )

    labels = {metric.label for metric in app.metric}
    assert {
        "Total questions",
        "Correct",
        "Incorrect",
        "I do not know",
        "Accuracy",
        "First-attempt accuracy",
        "Average confidence",
        "Hints used",
        "High-confidence mistakes",
    } <= labels
    assert [download.label for download in app.download_button] == [
        "Download session results"
    ]


def test_empty_review_mistakes_and_progress_states_render() -> None:
    for module, renderer, expected in [
        ("src.ui.pages.review", "render_review_page", "caught up"),
        ("src.ui.pages.mistakes", "render_mistakes_page", "No mistakes"),
        ("src.ui.pages.progress", "render_progress_page", "No progress"),
    ]:
        app = AppTest.from_string(
            f"""
from src.ui.state import initialize_state
from {module} import {renderer}
initialize_state()
{renderer}()
"""
        ).run(timeout=10)

        assert not app.exception
        assert any(expected in info.value for info in app.info)


def test_review_start_progress_feedback_and_completion() -> None:
    app = AppTest.from_string(
        """
from datetime import UTC, datetime
import streamlit as st
from src.application.review_service import ReviewService
from src.application.study_session_service import StudySessionService
from src.models.question_schemas import (
    DifficultyLevel,
    MCQQuestion,
    QuestionOption,
    QuestionSet,
    QuestionType,
)
from src.models.study_session import ConfidenceLevel, MCQLearnerAnswer, StudyQuestionMode
from src.ui.pages.review import render_review_page
from src.ui.state import initialize_state, repository

initialize_state()
if not st.session_state.get("review_seeded"):
    question = MCQQuestion(
        type=QuestionType.MCQ,
        position=1,
        question="Which structure stores key-value pairs?",
        difficulty=DifficultyLevel.EASY,
        explanation="A dictionary maps unique keys to values.",
        options=[
            QuestionOption(id="A", text="Dictionary"),
            QuestionOption(id="B", text="Tuple"),
            QuestionOption(id="C", text="Set"),
            QuestionOption(id="D", text="String"),
        ],
        correct_option_id="A",
    )
    service = StudySessionService()
    session = service.start_session(
        topic="Python",
        difficulty=DifficultyLevel.EASY,
        question_mode=StudyQuestionMode.MCQ,
        language="English",
        question_set=QuestionSet(questions=[question]),
    )
    answered, attempt, _ = service.submit_answer(
        session=session,
        learner_answer=MCQLearnerAnswer(selected_option_id="B"),
        confidence=ConfidenceLevel.HIGH,
    )
    repo = repository()
    repo.save_session(answered)
    repo.save_attempt(attempt)
    repo.save_review_item(
        ReviewService(clock=lambda: datetime.now(UTC)).build_review_item(
            question=question,
            topic="Python",
            attempt=attempt,
        )
    )
    st.session_state["review_seeded"] = True
render_review_page()
"""
    ).run(timeout=10)

    assert any(metric.label == "Due questions" for metric in app.metric)
    next(button for button in app.button if button.label == "Start review").click().run(
        timeout=10
    )

    assert len(app.get("progress")) == 1
    assert len(app.radio) == 1
    app.radio[0].set_value(app.radio[0].options[0])
    confidence = next(
        field for field in app.selectbox if field.label == "How confident are you?"
    )
    confidence.set_value(confidence.options[4])
    next(button for button in app.button if button.label == "Check answer").click().run(
        timeout=10
    )

    assert [success.value for success in app.success] == ["Correct"]
    next(button for button in app.button if button.label == "Continue review").click().run(
        timeout=10
    )

    assert any(metric.label == "Questions reviewed" for metric in app.metric)
    assert len(app.session_state["study_repository"].list_attempts()) == 2


def test_settings_has_one_save_action_and_visible_confirmation() -> None:
    app = AppTest.from_string(
        """
from src.ui.pages.settings import render_settings_page
from src.ui.state import initialize_state
initialize_state()
render_settings_page()
"""
    ).run(timeout=10)

    save_buttons = [button for button in app.button if button.label == "Save preferences"]
    assert len(save_buttons) == 1

    save_buttons[0].click().run(timeout=10)

    assert [success.value for success in app.success] == ["Preferences saved."]


def test_ui_source_has_no_unsafe_cards_or_provider_imports() -> None:
    ui_paths = [Path("streamlit_app.py"), *Path("src/ui").rglob("*.py")]
    source = "\n".join(path.read_text(encoding="utf-8") for path in ui_paths)

    assert "unsafe_allow_html" not in source
    assert "<div" not in source
    assert "st.empty(" not in source
    assert ".model_dump(" not in source
    assert "from groq" not in source.lower()
    assert "AsyncGroq" not in source
