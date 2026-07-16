"""Post-submission feedback panel."""

from __future__ import annotations

import streamlit as st

from src.application.answer_evaluation_service import AnswerEvaluationResult
from src.models.study_session import AttemptOutcome, ConfidenceLevel


def render_feedback_panel(
    result: AnswerEvaluationResult,
    *,
    show_explanation_automatically: bool = True,
    confidence: ConfidenceLevel | None = None,
    hints_used: int = 0,
) -> None:
    """Render safe learner feedback without interpolating content into HTML."""

    if result.outcome == AttemptOutcome.CORRECT:
        st.success("Correct", icon=":material/check_circle:")
        next_action = "Move on or review why this answer works."
    elif result.outcome == AttemptOutcome.UNKNOWN:
        st.info("I do not know", icon=":material/lightbulb:")
        next_action = "Study the explanation, then review this item soon."
    else:
        st.warning("Incorrect", icon=":material/cancel:")
        next_action = "Compare your answer with the explanation and retry during review."

    with st.container(border=True):
        st.markdown("**Your answer**")
        st.write(result.learner_answer_text)
        st.markdown("**Correct answer**")
        st.write(result.correct_answer_text)
        if show_explanation_automatically:
            st.markdown("**Why this works**")
            st.write(result.explanation)
        else:
            with st.expander("View explanation"):
                st.write(result.explanation)
        if hints_used:
            st.caption(f"Hints used · {hints_used}")
        confidence_insight = _confidence_insight(result.outcome, confidence)
        if confidence_insight:
            st.caption(confidence_insight)
        st.caption(next_action)


def _confidence_insight(
    outcome: AttemptOutcome,
    confidence: ConfidenceLevel | None,
) -> str | None:
    if outcome == AttemptOutcome.INCORRECT and confidence in {
        ConfidenceLevel.HIGH,
        ConfidenceLevel.VERY_HIGH,
    }:
        return "Because you felt confident, prioritize this concept during review."
    if outcome == AttemptOutcome.CORRECT and confidence in {
        ConfidenceLevel.GUESSED,
        ConfidenceLevel.LOW,
    }:
        return "You were correct with low confidence; one review can help make it stick."
    return None
