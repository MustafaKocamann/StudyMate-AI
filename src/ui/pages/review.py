"""Review page for due review items."""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from src.ui.components.empty_state import render_empty_state
from src.ui.state import repository


def render_review_page() -> None:
    st.title("Review")
    due_items = repository().list_due_review_items(now=datetime.now(UTC))
    st.metric("Due questions", len(due_items))
    if not due_items:
        render_empty_state("No review questions are due today.", "Keep practicing or check your mistake notebook.")
        return

    st.write("Due review items")
    for item in due_items:
        with st.expander(item.question.question):
            st.write(f"Topic: {item.topic}")
            st.write(f"Last outcome: {item.last_outcome.value}")
            st.write(f"Next review: {item.next_review_at:%Y-%m-%d %H:%M}")
