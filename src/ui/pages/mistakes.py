"""Learner-friendly mistake notebook with useful filters and details."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import streamlit as st

from src.application.answer_evaluation_service import AnswerEvaluationService
from src.application.study_session_service import StudySessionService
from src.models.question_schemas import GeneratedQuestion, QuestionType
from src.models.study_session import (
    AttemptOutcome,
    FillBlankLearnerAnswer,
    MCQLearnerAnswer,
    QuestionAttempt,
    ReviewItem,
)
from src.ui.components.empty_state import render_empty_state
from src.ui.export_helpers import attempts_to_csv_bytes, build_export_filename
from src.ui.layout import page_header, section_header
from src.ui.navigation import AppRoute, page_for
from src.ui.state import repository


_OUTCOME_LABELS = {
    AttemptOutcome.INCORRECT: "Needs another look",
    AttemptOutcome.UNKNOWN: "I do not know",
}

_QUESTION_TYPE_LABELS = {
    QuestionType.MCQ: "Multiple choice",
    QuestionType.FILL_BLANK: "Fill in the blank",
}


@dataclass(frozen=True)
class MistakeView:
    attempt: QuestionAttempt
    question: GeneratedQuestion | None
    review_item: ReviewItem | None
    language: str | None

    @property
    def review_status(self) -> str:
        return "unresolved" if self.review_item is not None else "resolved"


def render_mistakes_page() -> None:
    page_header(
        "Mistake notebook",
        "Use earlier attempts as study material—not as a scorecard.",
        eyebrow="LEARN FROM ATTEMPTS",
    )

    repo = repository()
    attempts = repo.list_mistakes()
    if not attempts:
        render_empty_state(
            "No mistakes recorded yet",
            "Incorrect and unanswered questions will appear here with their explanations.",
            icon=":material/edit_note:",
        )
        st.page_link(
            page_for(AppRoute.PRACTICE),
            label="Start practice",
            icon=":material/edit:",
        )
        return

    views = [_build_mistake_view(attempt) for attempt in attempts]
    filtered = _render_filters(views)
    if not filtered:
        render_empty_state(
            "No matching attempts",
            "Adjust the filters to see another part of your notebook.",
            icon=":material/filter_alt_off:",
        )
        return

    section_header(
        "Attempts to revisit",
        f"Showing {len(filtered)} of {len(views)} saved mistakes.",
    )
    for view in reversed(filtered):
        _render_mistake(view)

    st.download_button(
        "Download notebook data",
        data=attempts_to_csv_bytes([view.attempt for view in filtered]),
        file_name=build_export_filename(prefix="studymate-mistakes"),
        mime="text/csv",
        icon=":material/download:",
        key="mistakes-download",
    )


def _build_mistake_view(attempt: QuestionAttempt) -> MistakeView:
    repo = repository()
    session = repo.get_session(attempt.session_id)
    question = None
    if session is not None:
        try:
            question = StudySessionService().question_by_id(session, attempt.question_id)
        except StopIteration:
            question = None
    return MistakeView(
        attempt=attempt,
        question=question,
        review_item=repo.get_active_review_item_for_question(attempt.question_id),
        language=session.language if session is not None else None,
    )


def _render_filters(views: list[MistakeView]) -> list[MistakeView]:
    topics = sorted({view.attempt.topic for view in views})
    first_row = st.columns(2)
    selected_topic = first_row[0].selectbox(
        "Topic",
        ["All topics", *topics],
        key="mistakes-topic-filter",
    )
    selected_type = first_row[1].selectbox(
        "Question type",
        ["all", *[question_type.value for question_type in QuestionType]],
        format_func=lambda value: (
            "All question types"
            if value == "all"
            else _QUESTION_TYPE_LABELS[QuestionType(value)]
        ),
        key="mistakes-type-filter",
    )

    second_row = st.columns(2)
    selected_difficulty = second_row[0].selectbox(
        "Difficulty",
        ["all", *sorted({view.attempt.difficulty.value for view in views})],
        format_func=lambda value: "All difficulties" if value == "all" else value.title(),
        key="mistakes-difficulty-filter",
    )
    selected_review_status = second_row[1].selectbox(
        "Review status",
        ["all", "unresolved", "resolved"],
        format_func=lambda value: "All statuses" if value == "all" else value.title(),
        key="mistakes-review-status-filter",
    )
    selected_outcome = st.selectbox(
        "Answer status",
        ["all", *[outcome.value for outcome in _OUTCOME_LABELS]],
        format_func=lambda value: (
            "All attempts" if value == "all" else _OUTCOME_LABELS[AttemptOutcome(value)]
        ),
        key="mistakes-outcome-filter",
    )

    return [
        view
        for view in views
        if (selected_topic == "All topics" or view.attempt.topic == selected_topic)
        and (selected_type == "all" or view.attempt.question_type.value == selected_type)
        and (
            selected_difficulty == "all"
            or view.attempt.difficulty.value == selected_difficulty
        )
        and (
            selected_review_status == "all"
            or view.review_status == selected_review_status
        )
        and (selected_outcome == "all" or view.attempt.outcome.value == selected_outcome)
    ]


def _render_mistake(view: MistakeView) -> None:
    attempt = view.attempt
    label = (
        f"{attempt.topic} · {_OUTCOME_LABELS[attempt.outcome]} · "
        f"{attempt.attempted_at:%d %b %Y}"
    )
    with st.expander(label):
        status_label = "Unresolved" if view.review_item is not None else "Resolved"
        st.caption(
            f"{_QUESTION_TYPE_LABELS[attempt.question_type]} · "
            f"{attempt.difficulty.value.title()} · {status_label}"
        )
        if view.question is not None:
            st.markdown("**Question**")
            st.write(view.question.question)
            feedback = AnswerEvaluationService().evaluate(
                question=view.question,
                learner_answer=attempt.learner_answer,
                confidence=attempt.confidence,
                language=view.language,
            )
            st.markdown("**Your answer**")
            st.write(feedback.learner_answer_text)
            st.markdown("**Correct answer**")
            st.write(feedback.correct_answer_text)
            st.markdown("**Why this works**")
            st.write(feedback.explanation)
        else:
            st.markdown("**Your answer**")
            st.write(_learner_answer_fallback(attempt.learner_answer))

        confidence = (
            attempt.confidence.name.replace("_", " ").title()
            if attempt.confidence
            else "Not captured"
        )
        st.caption(f"Confidence · {confidence}   |   Hints used · {attempt.hints_used}")
        if view.review_item is not None:
            st.caption(f"Next review · {view.review_item.next_review_at:%d %b %Y, %H:%M UTC}")
            if view.review_item.next_review_at <= datetime.now(UTC):
                st.page_link(
                    page_for(AppRoute.REVIEW),
                    label="Open due reviews",
                    icon=":material/replay:",
                )
        else:
            st.caption("This item no longer has an active review schedule.")


def _learner_answer_fallback(answer) -> str:  # noqa: ANN001
    if isinstance(answer, MCQLearnerAnswer):
        return "I do not know" if answer.unknown else str(answer.selected_option_id)
    if isinstance(answer, FillBlankLearnerAnswer):
        return "I do not know" if answer.unknown else str(answer.submitted_answer)
    return "Answer unavailable"
