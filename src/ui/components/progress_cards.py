"""Progress metric rendering helpers."""

from __future__ import annotations

import streamlit as st

from src.models.study_session import ProgressSnapshot


def render_progress_cards(snapshot: ProgressSnapshot) -> None:
    columns = st.columns(4)
    columns[0].metric("Answered", snapshot.total_questions_answered)
    columns[1].metric("Accuracy", f"{snapshot.overall_accuracy:.0%}")
    columns[2].metric("Avg confidence", "-" if snapshot.average_confidence is None else f"{snapshot.average_confidence:.1f}")
    columns[3].metric("Due reviews", snapshot.review_due_count)
