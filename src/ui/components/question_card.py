"""Question rendering helpers."""

from __future__ import annotations

import streamlit as st

from src.models.question_schemas import FillBlankQuestion, GeneratedQuestion, MCQQuestion


def render_question_card(question: GeneratedQuestion) -> None:
    st.subheader(f"Question {question.position}")
    st.write(question.question)
    if isinstance(question, MCQQuestion):
        for option in question.options:
            st.write(f"**{option.id}.** {option.text}")
    elif isinstance(question, FillBlankQuestion):
        st.caption("Fill the single blank marked with ___.")
