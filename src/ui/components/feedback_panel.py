"""Post-submission feedback panel."""

from __future__ import annotations

import streamlit as st

from src.application.answer_evaluation_service import AnswerEvaluationResult
from src.models.study_session import AttemptOutcome


def render_feedback_panel(result: AnswerEvaluationResult) -> None:
    if result.outcome == AttemptOutcome.CORRECT:
        st.success("✅ Correct")
        next_action = "Move on or review why this answer works."
    elif result.outcome == AttemptOutcome.UNKNOWN:
        st.info("➖ I do not know")
        next_action = "Study the explanation, then review this item soon."
    else:
        st.error("❌ Incorrect")
        next_action = "Compare your answer with the explanation and retry during review."

    st.write(f"**Your answer:** {result.learner_answer_text}")
    st.write(f"**Correct answer:** {result.correct_answer_text}")
    st.write(f"**Explanation:** {result.explanation}")
    st.caption(next_action)
