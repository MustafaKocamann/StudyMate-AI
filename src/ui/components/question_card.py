"""Question rendering helpers."""

from __future__ import annotations

import streamlit as st

from src.models.question_schemas import FillBlankQuestion, GeneratedQuestion, MCQQuestion


def render_question_card(question: GeneratedQuestion) -> None:
    """Keep the question prompt visually primary and all model text escaped."""

    question_type = "Multiple choice" if isinstance(question, MCQQuestion) else "Fill in the blank"
    with st.container(border=True):
        st.caption(question_type)
        st.subheader(f"Question {question.position}")
        st.write(question.question)
        if isinstance(question, FillBlankQuestion):
            st.caption("Replace the ___ marker with the missing word or phrase.")
