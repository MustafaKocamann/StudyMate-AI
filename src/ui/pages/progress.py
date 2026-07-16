"""Progress dashboard page."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src.ui.components.empty_state import render_empty_state
from src.ui.components.progress_cards import render_progress_cards
from src.ui.state import repository


def render_progress_page() -> None:
    st.title("Progress")
    snapshot = repository().progress_snapshot(now=datetime.now(UTC))
    render_progress_cards(snapshot)

    if snapshot.total_questions_answered == 0:
        render_empty_state("No progress data yet.", "Answer a question to populate the dashboard.")
        return

    st.subheader("Accuracy by difficulty")
    st.caption("Shows objective correctness grouped by requested difficulty.")
    if snapshot.accuracy_by_difficulty:
        st.bar_chart(
            pd.DataFrame(
                {
                    "difficulty": [key.value for key in snapshot.accuracy_by_difficulty],
                    "accuracy": list(snapshot.accuracy_by_difficulty.values()),
                }
            ).set_index("difficulty")
        )

    st.subheader("Accuracy by question type")
    st.caption("Shows objective correctness grouped by schema question type.")
    if snapshot.accuracy_by_question_type:
        st.bar_chart(
            pd.DataFrame(
                {
                    "question_type": [key.value for key in snapshot.accuracy_by_question_type],
                    "accuracy": list(snapshot.accuracy_by_question_type.values()),
                }
            ).set_index("question_type")
        )
