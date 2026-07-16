"""Scheduled review page with explicit batch progress and feedback states."""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from src.application.review_service import ReviewService
from src.application.study_session_service import AnswerSubmissionConflictError, StudySessionService
from src.models.question_schemas import QuestionSet, QuestionType
from src.models.study_session import ReviewItem, StudyQuestionMode, StudySession
from src.ui.components.answer_form import render_answer_form
from src.ui.components.empty_state import render_empty_state
from src.ui.components.feedback_panel import render_feedback_panel
from src.ui.components.question_card import render_question_card
from src.ui.helpers import format_difficulty_label, request_rerun
from src.ui.layout import metric_row, page_header, section_header
from src.ui.navigation import AppRoute, page_for
from src.ui.state import (
    StateKey,
    UIPhase,
    repository,
    reset_current_review_item,
    reset_review_flow,
)


def render_review_page() -> None:
    page_header(
        "Review",
        "Revisit questions at useful moments and strengthen recall with another attempt.",
        eyebrow="SCHEDULED REVIEW",
    )

    repo = repository()
    due_items = sorted(
        repo.list_due_review_items(now=datetime.now(UTC)),
        key=lambda item: item.next_review_at,
    )
    review_phase = UIPhase(
        st.session_state.get(StateKey.REVIEW_PHASE.value, UIPhase.CONFIGURING.value)
    )

    if review_phase == UIPhase.CONFIGURING:
        _render_review_start(due_items)
        return
    if review_phase == UIPhase.COMPLETED:
        _render_review_completion(due_items)
        return

    active_item = st.session_state.get(StateKey.ACTIVE_REVIEW_ITEM.value)
    if active_item is None:
        reset_review_flow()
        request_rerun()
        return

    _render_review_progress(review_phase)
    _render_review_item(active_item, review_phase)


def _render_review_start(due_items: list[ReviewItem]) -> None:
    if not due_items:
        render_empty_state(
            "You're caught up",
            "No review questions are due right now. A short practice session is a good next step.",
            icon=":material/check_circle:",
        )
        st.page_link(
            page_for(AppRoute.PRACTICE),
            label="Start practice",
            icon=":material/edit:",
        )
        return

    metric_row(
        [("Due questions", len(due_items), "Questions whose scheduled review time has arrived.")]
    )
    st.write("Review one question at a time. Each answer updates its next review date.")
    if st.button(
        "Start review",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        key="review-start",
    ):
        _start_review_batch(due_items)
        request_rerun()


def _start_review_batch(due_items: list[ReviewItem]) -> None:
    reset_review_flow()
    st.session_state[StateKey.REVIEW_QUEUE.value] = list(due_items)
    st.session_state[StateKey.REVIEW_TOTAL_COUNT.value] = len(due_items)
    st.session_state[StateKey.ACTIVE_REVIEW_ITEM.value] = due_items[0]
    st.session_state[StateKey.REVIEW_PHASE.value] = UIPhase.ANSWERING.value


def _render_review_progress(review_phase: UIPhase) -> None:
    completed = st.session_state[StateKey.REVIEW_COMPLETED_COUNT.value]
    total = st.session_state[StateKey.REVIEW_TOTAL_COUNT.value]
    answered_current = 1 if review_phase == UIPhase.FEEDBACK else 0
    visible_completed = min(completed + answered_current, total)
    current_position = min(completed + 1, total)
    st.progress(
        visible_completed / total,
        text=f"Review {current_position} of {total}",
    )


def _render_review_item(item: ReviewItem, review_phase: UIPhase) -> None:
    preferences = st.session_state[StateKey.USER_PREFERENCES.value]
    session = st.session_state.get(StateKey.REVIEW_SESSION.value)
    if session is None:
        session = _build_review_session(item, preferences["default_language"])
        st.session_state[StateKey.REVIEW_SESSION.value] = session

    difficulty = format_difficulty_label(item.question.difficulty, language=session.language)
    st.caption(f"Review mode · {item.topic} · {difficulty}")
    render_question_card(session.current_question)

    if review_phase == UIPhase.FEEDBACK:
        feedback = st.session_state.get(StateKey.REVIEW_FEEDBACK.value)
        if feedback is not None:
            latest_attempt = session.attempts[-1]
            render_feedback_panel(
                feedback,
                show_explanation_automatically=preferences["show_explanations_automatically"],
                confidence=latest_attempt.confidence,
                hints_used=latest_attempt.hints_used,
            )
        if st.button(
            "Continue review",
            type="primary",
            icon=":material/arrow_forward:",
            width="stretch",
            key=f"review-continue:{item.review_item_id}",
        ):
            _advance_review_batch()
            request_rerun()
        return

    submitted, learner_answer, confidence = render_answer_form(
        session.current_question,
        session_id=session.session_id,
        confidence_required=preferences["enable_confidence_capture"],
        language=session.language,
    )
    if not submitted or learner_answer is None:
        return

    try:
        session_service = StudySessionService()
        updated_session, attempt, feedback = session_service.submit_answer(
            session=session,
            learner_answer=learner_answer,
            confidence=confidence,
        )
    except AnswerSubmissionConflictError:
        st.warning("This review answer was already recorded.")
        return

    repo = repository()
    repo.save_attempt(attempt)
    completed_session = session_service.advance(updated_session)
    repo.save_session(completed_session)
    updated_item = ReviewService().build_review_item(
        question=item.question,
        topic=item.topic,
        attempt=attempt,
        existing_item=item,
    )
    repo.save_review_item(updated_item)

    st.session_state[StateKey.ACTIVE_REVIEW_ITEM.value] = updated_item
    st.session_state[StateKey.REVIEW_SESSION.value] = completed_session
    st.session_state[StateKey.REVIEW_FEEDBACK.value] = feedback
    st.session_state[StateKey.REVIEW_PHASE.value] = UIPhase.FEEDBACK.value
    request_rerun()


def _build_review_session(item: ReviewItem, language: str) -> StudySession:
    review_question = item.question.model_copy(update={"position": 1})
    question_set = QuestionSet(questions=[review_question])
    mode = (
        StudyQuestionMode.MCQ
        if item.question.type == QuestionType.MCQ
        else StudyQuestionMode.FILL_BLANK
    )
    return StudySessionService().start_session(
        topic=item.topic,
        difficulty=item.question.difficulty,
        question_mode=mode,
        language=language,
        question_set=question_set,
    )


def _advance_review_batch() -> None:
    queue = st.session_state[StateKey.REVIEW_QUEUE.value]
    remaining = queue[1:]
    st.session_state[StateKey.REVIEW_QUEUE.value] = remaining
    st.session_state[StateKey.REVIEW_COMPLETED_COUNT.value] += 1
    reset_current_review_item()
    if remaining:
        st.session_state[StateKey.ACTIVE_REVIEW_ITEM.value] = remaining[0]
        st.session_state[StateKey.REVIEW_PHASE.value] = UIPhase.ANSWERING.value
    else:
        st.session_state[StateKey.REVIEW_PHASE.value] = UIPhase.COMPLETED.value


def _render_review_completion(due_items: list[ReviewItem]) -> None:
    completed = st.session_state[StateKey.REVIEW_COMPLETED_COUNT.value]
    section_header("Review complete", "You worked through every question in this review set.")
    metric_row([("Questions reviewed", completed, None)])
    if due_items:
        st.info(
            f"{len(due_items)} question(s) are still due based on the answers you just submitted.",
            icon=":material/replay:",
        )
        if st.button(
            "Start another review",
            type="primary",
            icon=":material/refresh:",
            width="stretch",
            key="review-start-another",
        ):
            _start_review_batch(due_items)
            request_rerun()
        return

    actions = st.columns(2)
    with actions[0]:
        st.page_link(
            page_for(AppRoute.PRACTICE),
            label="Start practice",
            icon=":material/edit:",
            width="stretch",
        )
    with actions[1]:
        st.page_link(
            page_for(AppRoute.HOME),
            label="Return home",
            icon=":material/home:",
            width="stretch",
        )
