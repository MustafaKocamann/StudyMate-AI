"""Practice page with configuration and one-question answering flow."""

from __future__ import annotations

import asyncio

from pydantic import ValidationError
import streamlit as st

from src.application.review_service import ReviewService
from src.application.study_session_service import AnswerSubmissionConflictError, StudySessionService
from src.common.exceptions import StudyBuddyException
from src.models.question_schemas import DifficultyLevel
from src.models.study_session import StudyQuestionMode
from src.ui.components.answer_form import render_answer_form
from src.ui.components.feedback_panel import render_feedback_panel
from src.ui.components.question_card import render_question_card
from src.ui.helpers import format_difficulty_label, request_rerun, safe_error_message
from src.ui.state import StateKey, UIPhase, phase, repository, set_phase


def render_practice_page() -> None:
    st.title("Practice")
    active_session = st.session_state.get(StateKey.ACTIVE_SESSION.value)
    if active_session is None or phase() == UIPhase.CONFIGURING:
        _render_configuration()
        return

    session = active_session
    difficulty_label = format_difficulty_label(
        session.requested_difficulty,
        language=session.language,
    )
    st.caption(f"{session.topic} · {difficulty_label} · {session.language}")
    st.progress(session.current_position / len(session.questions))
    st.write(f"Question {session.current_position} of {len(session.questions)}")

    render_question_card(session.current_question)

    if phase() == UIPhase.FEEDBACK:
        feedback = st.session_state.get(StateKey.LAST_FEEDBACK.value)
        if feedback:
            render_feedback_panel(feedback)
        if st.button("Next question", type="primary"):
            updated = StudySessionService().advance(session)
            st.session_state[StateKey.ACTIVE_SESSION.value] = updated
            set_phase(UIPhase.COMPLETED if updated.status.value == "completed" else UIPhase.ANSWERING)
            request_rerun()
        return

    preferences = st.session_state[StateKey.USER_PREFERENCES.value]
    if preferences["enable_hints"]:
        if st.session_state.get(StateKey.HINT_PROVIDER.value) is None:
            st.caption("Hints are unavailable in this environment.")
        else:
            st.button("Get hint")
    submitted, learner_answer, confidence = render_answer_form(
        session.current_question,
        session_id=session.session_id,
        confidence_required=preferences["enable_confidence_capture"],
    )
    if not submitted:
        return

    try:
        updated_session, attempt, feedback = StudySessionService().submit_answer(
            session=session,
            learner_answer=learner_answer,
            confidence=confidence,
            hints_used=st.session_state.get(StateKey.HINT_LEVEL.value, 0),
        )
    except AnswerSubmissionConflictError:
        st.warning("Your previous answer was already recorded.")
        return

    repo = repository()
    repo.save_attempt(attempt)
    repo.save_session(updated_session)
    existing_review = repo.get_active_review_item_for_question(session.current_question.id)
    review_item = ReviewService().build_review_item(
        question=session.current_question,
        topic=session.topic,
        attempt=attempt,
        existing_item=existing_review,
    )
    repo.save_review_item(review_item)

    st.session_state[StateKey.ACTIVE_SESSION.value] = updated_session
    st.session_state[StateKey.LAST_FEEDBACK.value] = feedback
    st.session_state[StateKey.HINT_LEVEL.value] = 0
    set_phase(UIPhase.FEEDBACK)
    request_rerun()


def _render_configuration() -> None:
    preferences = st.session_state[StateKey.USER_PREFERENCES.value]
    with st.form("practice-config"):
        topic = st.text_input("Study topic")
        difficulty = st.selectbox(
            "Difficulty",
            [level.value for level in DifficultyLevel],
            index=[level.value for level in DifficultyLevel].index(preferences["default_difficulty"]),
        )
        question_mode = st.selectbox(
            "Question type",
            [mode.value for mode in StudyQuestionMode],
            index=[mode.value for mode in StudyQuestionMode].index(preferences["default_question_mode"]),
            format_func=lambda value: "MCQ" if value == "mcq" else ("Fill blank" if value == "fill_blank" else "Mixed"),
        )
        count = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=20,
            value=int(preferences["default_question_count"]),
        )
        language = st.text_input("Learner-facing language", value=preferences["default_language"])
        submitted = st.form_submit_button("Start practice", type="primary")

    if not submitted:
        return
    if not topic.strip():
        st.error("Please enter a study topic.")
        return
    question_service = st.session_state.get(StateKey.QUESTION_SERVICE.value)
    if question_service is None:
        st.error("Question generation is not configured yet. Please connect a generator service.")
        return

    set_phase(UIPhase.GENERATING)
    with st.spinner("Generating questions..."):
        try:
            question_set = asyncio.run(
                question_service.generate_questions(
                    topic=topic,
                    difficulty=DifficultyLevel(difficulty),
                    question_mode=StudyQuestionMode(question_mode),
                    count=int(count),
                    language=language,
                )
            )
        except (ValidationError, ValueError, StudyBuddyException) as exc:
            _handle_generation_failure(exc)
            return

    session = StudySessionService().start_session(
        topic=topic,
        difficulty=DifficultyLevel(difficulty),
        question_mode=StudyQuestionMode(question_mode),
        language=language,
        question_set=question_set,
    )
    repo = repository()
    repo.save_session(session)
    st.session_state[StateKey.ACTIVE_SESSION.value] = session
    set_phase(UIPhase.ANSWERING)
    request_rerun()


def _handle_generation_failure(exc: Exception) -> None:
    set_phase(UIPhase.CONFIGURING)
    st.error(safe_error_message(exc))
