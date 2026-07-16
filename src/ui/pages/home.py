"""Focused StudyMate home page."""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from src.ui.components.empty_state import render_empty_state
from src.ui.components.progress_cards import render_progress_cards
from src.ui.layout import page_header, section_header
from src.ui.navigation import AppRoute, page_for
from src.ui.state import repository


def render_home_page() -> None:
    page_header(
        "StudyMate — your Study Buddy AI",
        "Build understanding through focused practice, useful feedback, and well-timed review.",
        eyebrow="YOUR LEARNING SPACE",
    )

    repo = repository()
    snapshot = repo.progress_snapshot(now=datetime.now(UTC))
    has_learning_data = bool(snapshot.total_questions_answered or snapshot.recent_session_summaries)

    if st.button(
        "Start studying",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        key="home-start-studying",
    ):
        st.switch_page(page_for(AppRoute.PRACTICE))

    if not has_learning_data:
        render_empty_state(
            "No sessions yet",
            "Choose a topic and StudyMate will prepare a short, focused practice session.",
            icon=":material/menu_book:",
        )
        return

    section_header("Your progress", "A quick view of the learning signals available so far.")
    render_progress_cards(snapshot)

    if snapshot.recent_session_summaries:
        latest = snapshot.recent_session_summaries[-1]
        with st.container(border=True):
            st.subheader("Most recent session")
            st.write(f"{latest.correct_count} of {latest.total_questions} answered correctly")
            if latest.total_questions:
                st.progress(latest.accuracy, text=f"Accuracy · {latest.accuracy:.0%}")
            st.caption(latest.recommended_next_action)

    section_header("Continue learning")
    practice, review, mistakes = st.columns(3)
    with practice:
        st.page_link(
            page_for(AppRoute.PRACTICE),
            label="Practice",
            icon=":material/edit:",
            help="Start a new focused session.",
            width="stretch",
        )
    with review:
        review_label = f"Review · {snapshot.review_due_count} due" if snapshot.review_due_count else "Review"
        st.page_link(
            page_for(AppRoute.REVIEW),
            label=review_label,
            icon=":material/replay:",
            help="Complete scheduled reviews.",
            width="stretch",
        )
    with mistakes:
        st.page_link(
            page_for(AppRoute.MISTAKES),
            label="Mistake notebook",
            icon=":material/edit_note:",
            help="Revisit questions that need more attention.",
            width="stretch",
        )

    st.page_link(
        page_for(AppRoute.PROGRESS),
        label="Open detailed progress",
        icon=":material/trending_up:",
    )
