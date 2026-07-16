"""Answer form components for supported question types."""

from __future__ import annotations

from uuid import UUID

import streamlit as st

from src.models.question_schemas import FillBlankQuestion, GeneratedQuestion, MCQQuestion
from src.models.study_session import ConfidenceLevel, FillBlankLearnerAnswer, LearnerAnswer, MCQLearnerAnswer
from src.ui.helpers import build_widget_key


CONFIDENCE_LABELS = {
    ConfidenceLevel.GUESSED: "1 - Guessed",
    ConfidenceLevel.LOW: "2 - Low",
    ConfidenceLevel.MEDIUM: "3 - Medium",
    ConfidenceLevel.HIGH: "4 - High",
    ConfidenceLevel.VERY_HIGH: "5 - Very high",
}


def render_answer_form(
    question: GeneratedQuestion,
    *,
    session_id: UUID,
    confidence_required: bool,
) -> tuple[bool, LearnerAnswer | None, ConfidenceLevel | None]:
    form_key = build_widget_key(
        "answer-form",
        session_id=session_id,
        question_id=question.id,
        position=question.position,
    )
    with st.form(key=form_key, clear_on_submit=False):
        unknown = st.checkbox(
            "I do not know",
            key=build_widget_key(
                "answer-unknown",
                session_id=session_id,
                question_id=question.id,
                position=question.position,
            ),
        )
        learner_answer: LearnerAnswer | None = None
        if isinstance(question, MCQQuestion):
            selected = st.radio(
                "Choose one option",
                [option.id for option in question.options],
                index=None,
                disabled=unknown,
                key=build_widget_key(
                    "answer-mcq-option",
                    session_id=session_id,
                    question_id=question.id,
                    position=question.position,
                ),
            )
            if unknown:
                learner_answer = MCQLearnerAnswer(unknown=True)
            elif selected:
                learner_answer = MCQLearnerAnswer(selected_option_id=selected)
        elif isinstance(question, FillBlankQuestion):
            submitted = st.text_input(
                "Your answer",
                disabled=unknown,
                key=build_widget_key(
                    "answer-fill-blank",
                    session_id=session_id,
                    question_id=question.id,
                    position=question.position,
                ),
            )
            if unknown:
                learner_answer = FillBlankLearnerAnswer(unknown=True)
            elif submitted.strip():
                learner_answer = FillBlankLearnerAnswer(submitted_answer=submitted)

        confidence_value = None
        if confidence_required:
            confidence_value = st.select_slider(
                "Confidence",
                options=list(CONFIDENCE_LABELS),
                format_func=lambda value: CONFIDENCE_LABELS[value],
                help="Use confidence for review priority; it does not change correctness.",
                key=build_widget_key(
                    "answer-confidence",
                    session_id=session_id,
                    question_id=question.id,
                    position=question.position,
                ),
            )
        submitted_form = st.form_submit_button(
            "Submit answer",
            type="primary",
        )

    if submitted_form and learner_answer is None:
        st.error("Please answer the question or choose I do not know.")
        return False, None, None
    if submitted_form and confidence_required and confidence_value is None:
        st.error("Please choose your confidence before submitting.")
        return False, None, None
    return submitted_form, learner_answer, confidence_value
