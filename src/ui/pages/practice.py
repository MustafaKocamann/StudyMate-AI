"""Practice page with configuration and a rerun-safe one-question flow."""

from __future__ import annotations

import asyncio

from pydantic import ValidationError
import streamlit as st

from src.application.question_service import question_type_blueprint
from src.application.progress_service import ProgressService
from src.application.review_service import ReviewService
from src.application.study_session_service import AnswerSubmissionConflictError, StudySessionService
from src.common.exceptions import StudyBuddyException
from src.common.logger import get_logger
from src.generator.regeneration import QuestionGenerationRequest
from src.models.question_schemas import DifficultyLevel, QuestionSet
from src.models.study_session import StudyQuestionMode
from src.ui.components.answer_form import render_answer_form
from src.ui.components.feedback_panel import render_feedback_panel
from src.ui.components.question_card import render_question_card
from src.ui.export_helpers import attempts_to_csv_bytes, build_export_filename
from src.ui.helpers import build_widget_key, format_difficulty_label, request_rerun, safe_error_message
from src.ui.layout import metric_row, page_header, section_header
from src.ui.navigation import AppRoute, page_for
from src.ui.state import StateKey, UIPhase, phase, repository, reset_practice_flow, set_phase


logger = get_logger(__name__)


def render_practice_page() -> None:
    page_header(
        "Practice",
        "Set a clear goal, then work through one question at a time.",
        eyebrow="FOCUSED STUDY",
    )

    active_session = st.session_state.get(StateKey.ACTIVE_SESSION.value)
    if active_session is None or phase() == UIPhase.CONFIGURING:
        _render_configuration()
        return
    if phase() == UIPhase.COMPLETED or active_session.status.value == "completed":
        _render_completed_session(active_session)
        return

    session = active_session
    difficulty_label = format_difficulty_label(
        session.requested_difficulty,
        language=session.language,
    )
    st.caption(f"{session.topic} · {difficulty_label} · {session.language}")

    completed_count = len(session.attempts)
    total_count = max(
        len(session.questions),
        int(st.session_state.get(StateKey.PRACTICE_TARGET_COUNT.value, len(session.questions))),
    )
    st.progress(
        completed_count / total_count,
        text=f"Question {session.current_position} of {total_count}",
    )
    render_question_card(session.current_question)

    preferences = st.session_state[StateKey.USER_PREFERENCES.value]
    if phase() == UIPhase.FEEDBACK:
        generation_error = st.session_state.get(StateKey.GENERATION_ERROR.value)
        if generation_error:
            st.error(generation_error)
        feedback = st.session_state.get(StateKey.LAST_FEEDBACK.value)
        if feedback is not None:
            latest_attempt = session.attempts[-1]
            render_feedback_panel(
                feedback,
                show_explanation_automatically=preferences["show_explanations_automatically"],
                confidence=latest_attempt.confidence,
                hints_used=latest_attempt.hints_used,
            )
        next_label = "Finish session" if session.current_position == total_count else (
            "Try next question" if generation_error else "Next question"
        )
        if st.button(
            next_label,
            type="primary",
            icon=":material/arrow_forward:",
            width="stretch",
            key=build_widget_key(
                "practice-next",
                session_id=session.session_id,
                question_id=session.current_question.id,
                position=session.current_position,
            ),
        ):
            working_session = session
            if session.current_position < total_count and session.current_position == len(session.questions):
                next_position = len(session.questions) + 1
                question_service = st.session_state.get(StateKey.QUESTION_SERVICE.value)
                if question_service is None:
                    st.session_state[StateKey.GENERATION_ERROR.value] = (
                        "The next question could not be prepared. Please try again."
                    )
                    request_rerun()
                    return
                with st.status("Preparing the next question", expanded=True) as next_status:
                    try:
                        generation_result = asyncio.run(
                            question_service.generate_question(
                                QuestionGenerationRequest(
                                    topic=session.topic,
                                    difficulty=session.requested_difficulty,
                                    question_type=question_type_blueprint(
                                        session.requested_question_type,
                                        total_count,
                                    )[next_position - 1],
                                    position=next_position,
                                    language=session.language,
                                ),
                                existing_questions=session.questions,
                            )
                        )
                        working_session = StudySessionService().append_question(
                            session,
                            generation_result.question,
                        )
                    except (ValidationError, ValueError, StudyBuddyException) as exc:
                        next_status.update(
                            label="Next question could not be prepared",
                            state="error",
                            expanded=False,
                        )
                        st.session_state[StateKey.GENERATION_ERROR.value] = safe_error_message(exc)
                        request_rerun()
                        return
                    except Exception as exc:
                        logger.exception(
                            "practice_next_generation_unexpected_failure",
                            extra={
                                "event": "practice_next_generation_unexpected_failure",
                                "error_type": type(exc).__name__,
                            },
                        )
                        next_status.update(
                            label="Next question could not be prepared",
                            state="error",
                            expanded=False,
                        )
                        st.session_state[StateKey.GENERATION_ERROR.value] = safe_error_message(exc)
                        request_rerun()
                        return
                    next_status.update(
                        label="Next question ready",
                        state="complete",
                        expanded=False,
                    )
            updated = StudySessionService().advance(working_session)
            repository().save_session(updated)
            st.session_state[StateKey.ACTIVE_SESSION.value] = updated
            st.session_state[StateKey.LAST_FEEDBACK.value] = None
            st.session_state[StateKey.HINT_TEXT.value] = None
            st.session_state[StateKey.GENERATION_ERROR.value] = None
            set_phase(UIPhase.COMPLETED if updated.status.value == "completed" else UIPhase.ANSWERING)
            request_rerun()
        return

    _render_hint_controls(session)
    submitted, learner_answer, confidence = render_answer_form(
        session.current_question,
        session_id=session.session_id,
        confidence_required=preferences["enable_confidence_capture"],
        language=session.language,
    )
    if not submitted or learner_answer is None:
        return

    try:
        updated_session, attempt, feedback = StudySessionService().submit_answer(
            session=session,
            learner_answer=learner_answer,
            confidence=confidence,
            hints_used=st.session_state.get(StateKey.HINT_LEVEL.value, 0),
        )
    except AnswerSubmissionConflictError:
        st.warning("This answer was already recorded. Continue to the next question.")
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
    st.session_state[StateKey.HINT_TEXT.value] = None
    set_phase(UIPhase.FEEDBACK)
    request_rerun()


def _render_configuration() -> None:
    preferences = st.session_state[StateKey.USER_PREFERENCES.value]
    generation_error = st.session_state.get(StateKey.GENERATION_ERROR.value)
    section_header("Plan this session", "A smaller, specific topic usually produces better practice.")
    if generation_error:
        st.error(generation_error)
    with st.form("practice-config", border=True):
        topic = st.text_input(
            "What would you like to study?",
            placeholder="For example: Python list comprehensions",
        )
        left, right = st.columns(2)
        with left:
            difficulty = st.selectbox(
                "Difficulty",
                [level.value for level in DifficultyLevel],
                index=[level.value for level in DifficultyLevel].index(preferences["default_difficulty"]),
                format_func=lambda value: value.replace("_", " ").title(),
            )
            count = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=20,
                value=int(preferences["default_question_count"]),
                step=1,
            )
        with right:
            question_mode = st.selectbox(
                "Question type",
                [mode.value for mode in StudyQuestionMode],
                index=[mode.value for mode in StudyQuestionMode].index(preferences["default_question_mode"]),
                format_func=_format_question_mode,
            )
            language = st.text_input(
                "Question language",
                value=preferences["default_language"],
                help="Questions, answers, and explanations will use this language.",
            )

        with st.expander("Practice preferences"):
            show_explanations = st.checkbox(
                "Show explanations automatically after each answer",
                value=preferences["show_explanations_automatically"],
                key="practice-config-show-explanations",
            )
            confidence_capture = st.checkbox(
                "Ask for confidence with each answer",
                value=preferences["enable_confidence_capture"],
                help="Confidence supports review planning and never changes correctness.",
                key="practice-config-confidence",
            )
            hints_available = st.session_state.get(StateKey.HINT_PROVIDER.value) is not None
            hints_requested = st.checkbox(
                "Enable progressive hints",
                value=preferences["enable_hints"] and hints_available,
                disabled=not hints_available,
                key="practice-config-hints",
            )
            if not hints_available:
                st.caption("Hints are not available in this environment.")
        submitted = st.form_submit_button(
            "Try again" if generation_error else "Generate session",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        )

    if not submitted:
        return
    st.session_state[StateKey.GENERATION_ERROR.value] = None
    if not topic.strip():
        st.error("Enter a study topic to begin.")
        return
    if not language.strip():
        st.error("Enter a question language.")
        return
    question_service = st.session_state.get(StateKey.QUESTION_SERVICE.value)
    if question_service is None:
        st.error("Question generation is not ready yet. Please try again later.")
        return

    set_phase(UIPhase.GENERATING)
    with st.status("Preparing your study session", expanded=True) as generation_status:
        generation_status.write("Generating questions and checking educational quality…")
        try:
            requested_mode = StudyQuestionMode(question_mode)
            requested_count = int(count)
            generation_result = asyncio.run(
                question_service.generate_question(
                    QuestionGenerationRequest(
                        topic=topic,
                        difficulty=DifficultyLevel(difficulty),
                        question_type=question_type_blueprint(
                            requested_mode,
                            requested_count,
                        )[0],
                        position=1,
                        language=language,
                    )
                )
            )
            question_set = QuestionSet(questions=[generation_result.question])
        except (ValidationError, ValueError, StudyBuddyException) as exc:
            generation_status.update(
                label="Study session could not be prepared",
                state="error",
                expanded=False,
            )
            _handle_generation_failure(exc)
            return
        except Exception as exc:
            logger.exception(
                "practice_generation_unexpected_failure",
                extra={
                    "event": "practice_generation_unexpected_failure",
                    "error_type": type(exc).__name__,
                },
            )
            generation_status.update(
                label="Study session could not be prepared",
                state="error",
                expanded=False,
            )
            _handle_generation_failure(exc)
            return
        generation_status.write("Building your session…")
        session = StudySessionService().start_session(
            topic=topic,
            difficulty=DifficultyLevel(difficulty),
            question_mode=StudyQuestionMode(question_mode),
            language=language,
            question_set=question_set,
        )
        generation_status.update(
            label="Study session ready",
            state="complete",
            expanded=False,
        )

    st.session_state[StateKey.USER_PREFERENCES.value] = {
        **preferences,
        "show_explanations_automatically": show_explanations,
        "enable_confidence_capture": confidence_capture,
        "enable_hints": hints_requested and hints_available,
    }
    repository().save_session(session)
    st.session_state[StateKey.ACTIVE_SESSION.value] = session
    st.session_state[StateKey.PRACTICE_TARGET_COUNT.value] = int(count)
    set_phase(UIPhase.ANSWERING)
    request_rerun()


def _render_hint_controls(session) -> None:  # noqa: ANN001
    preferences = st.session_state[StateKey.USER_PREFERENCES.value]
    if not preferences["enable_hints"]:
        return

    hint_provider = st.session_state.get(StateKey.HINT_PROVIDER.value)
    if hint_provider is None:
        st.caption("Hints are not available for this session.")
        return

    hint_text = st.session_state.get(StateKey.HINT_TEXT.value)
    hint_level = st.session_state.get(StateKey.HINT_LEVEL.value, 0)
    if hint_text:
        st.info(hint_text, icon=":material/lightbulb:")
        st.caption(f"Hint {hint_level} of 3 · {hint_level} used")

    if hint_level >= 3:
        return
    if st.button(
        "Get next hint" if hint_level else "Get a hint",
        icon=":material/lightbulb:",
        key=build_widget_key(
            "practice-hint",
            session_id=session.session_id,
            question_id=session.current_question.id,
            position=session.current_position,
            suffix=hint_level + 1,
        ),
    ):
        try:
            hint_text = asyncio.run(hint_provider.get_hint(session.current_question, hint_level + 1))
        except StudyBuddyException as exc:
            st.error(safe_error_message(exc))
            return
        except Exception as exc:
            logger.exception(
                "practice_hint_unexpected_failure",
                extra={
                    "event": "practice_hint_unexpected_failure",
                    "error_type": type(exc).__name__,
                },
            )
            st.error(safe_error_message(exc))
            return
        st.session_state[StateKey.HINT_LEVEL.value] = hint_level + 1
        st.session_state[StateKey.HINT_TEXT.value] = hint_text
        request_rerun()


def _render_completed_session(session) -> None:  # noqa: ANN001
    summary = ProgressService().summarize_session(session)
    section_header("Session complete", "Take a moment to notice what is clear and what deserves review.")
    metric_row(
        [
            ("Total questions", summary.total_questions, None),
            ("Correct", summary.correct_count, None),
            ("Incorrect", summary.incorrect_count, None),
            ("I do not know", summary.unknown_count, None),
        ]
    )
    learning_metrics = [
            ("Accuracy", f"{summary.accuracy:.0%}", None),
            ("First-attempt accuracy", f"{summary.first_attempt_accuracy:.0%}", None),
            ("Hints used", summary.hints_used, None),
            (
                "High-confidence mistakes",
                summary.high_confidence_incorrect_count,
                "Incorrect answers submitted with high confidence.",
            ),
    ]
    if summary.average_confidence is not None:
        learning_metrics.insert(
            2,
            ("Average confidence", f"{summary.average_confidence:.1f} / 5", None),
        )
    metric_row(learning_metrics)
    st.info(summary.recommended_next_action, icon=":material/route:")
    primary, secondary, tertiary = st.columns(3)
    with primary:
        if st.button(
            "Start another session",
            type="primary",
            icon=":material/refresh:",
            width="stretch",
            key="practice-start-another",
        ):
            reset_practice_flow()
            request_rerun()
    with secondary:
        st.page_link(
            page_for(AppRoute.MISTAKES),
            label="Review mistakes",
            icon=":material/edit_note:",
            width="stretch",
        )
    with tertiary:
        st.page_link(
            page_for(AppRoute.HOME),
            label="Return home",
            icon=":material/home:",
            width="stretch",
        )

    st.download_button(
        "Download session results",
        data=attempts_to_csv_bytes(session.attempts),
        file_name=build_export_filename(prefix="studymate-session"),
        mime="text/csv",
        icon=":material/download:",
        key=f"practice-download:{session.session_id}",
    )


def _format_question_mode(value: str) -> str:
    return {
        StudyQuestionMode.MCQ.value: "Multiple choice",
        StudyQuestionMode.FILL_BLANK.value: "Fill in the blank",
        StudyQuestionMode.MIXED.value: "Mixed",
    }[value]


def _handle_generation_failure(exc: Exception) -> None:
    set_phase(UIPhase.CONFIGURING)
    st.session_state[StateKey.GENERATION_ERROR.value] = safe_error_message(exc)
    request_rerun()
