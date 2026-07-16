"""Home page for Study Buddy AI."""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from src.application.progress_service import ProgressService
from src.ui.components.empty_state import render_empty_state
from src.ui.components.progress_cards import render_progress_cards
from src.ui.state import repository


def render_home_page() -> None:
    st.title("Study Buddy AI")
    st.write("Practice one focused question at a time, review mistakes, and track progress.")

    repo = repository()
    snapshot = repo.progress_snapshot(now=datetime.now(UTC))
    render_progress_cards(snapshot)

    if snapshot.recent_session_summaries:
        latest = snapshot.recent_session_summaries[-1]
        st.subheader("Most recent session")
        st.write(
            f"{latest.correct_count}/{latest.total_questions} correct. "
            f"{latest.recommended_next_action}"
        )
    else:
        render_empty_state("No sessions yet", "Start a practice session when you are ready.")

    st.subheader("Quick actions")
    col1, col2, col3 = st.columns(3)
    col1.button("Start practice", type="primary")
    col2.button(f"Review due ({snapshot.review_due_count})")
    col3.button("Open mistake notebook")
    st.caption("Use the sidebar navigation to open each workspace.")
