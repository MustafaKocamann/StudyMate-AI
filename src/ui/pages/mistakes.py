"""Mistake notebook page."""

from __future__ import annotations

import streamlit as st

from src.models.study_session import AttemptOutcome
from src.ui.components.empty_state import render_empty_state
from src.ui.state import repository


def render_mistakes_page() -> None:
    st.title("Mistake notebook")
    mistakes = repository().list_mistakes()
    if not mistakes:
        render_empty_state("No mistakes recorded yet.", "Incorrect and I do not know attempts will appear here.")
        return

    topics = sorted({attempt.topic for attempt in mistakes})
    selected_topic = st.selectbox("Topic filter", ["All", *topics])
    outcome_filter = st.selectbox("Status", ["All", AttemptOutcome.INCORRECT.value, AttemptOutcome.UNKNOWN.value])
    filtered = [
        attempt
        for attempt in mistakes
        if (selected_topic == "All" or attempt.topic == selected_topic)
        and (outcome_filter == "All" or attempt.outcome.value == outcome_filter)
    ]
    for attempt in filtered:
        with st.expander(f"{attempt.topic} · {attempt.outcome.value}"):
            st.write(f"Learner answer: {attempt.learner_answer.model_dump(mode='json')}")
            st.write(f"Confidence: {attempt.confidence.name if attempt.confidence else 'Not captured'}")
            st.write(f"Hints used: {attempt.hints_used}")
            st.write(f"Attempted: {attempt.attempted_at:%Y-%m-%d %H:%M}")
