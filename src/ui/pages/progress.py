"""Calm, learner-focused progress dashboard."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src.application.progress_service import ProgressService
from src.ui.components.empty_state import render_empty_state
from src.ui.components.progress_cards import render_progress_cards
from src.ui.layout import metric_row, page_header, section_header
from src.ui.navigation import AppRoute, page_for
from src.ui.state import repository


def render_progress_page() -> None:
    page_header(
        "Progress",
        "Look for patterns in your practice and choose a useful next step.",
        eyebrow="LEARNING SIGNALS",
    )

    repo = repository()
    snapshot = repo.progress_snapshot(now=datetime.now(UTC))
    if snapshot.total_questions_answered == 0:
        render_empty_state(
            "No progress data yet",
            "Answer your first question to begin building a progress picture.",
            icon=":material/trending_up:",
        )
        st.page_link(
            page_for(AppRoute.PRACTICE),
            label="Start practice",
            icon=":material/edit:",
        )
        return

    render_progress_cards(snapshot)
    supporting_metrics = [
        (
            "First-attempt accuracy",
            f"{snapshot.first_attempt_accuracy:.0%}",
            "Correct answers on the first recorded attempt.",
        )
    ]
    if snapshot.high_confidence_wrong_count > 0:
        supporting_metrics.append(
            (
                "High-confidence mistakes",
                snapshot.high_confidence_wrong_count,
                "Incorrect answers submitted with high confidence.",
            )
        )
    metric_row(supporting_metrics)

    if snapshot.high_confidence_wrong_count > 0:
        st.warning(
            f"{snapshot.high_confidence_wrong_count} high-confidence answer needs another look.",
            icon=":material/priority_high:",
        )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        section_header("By difficulty", "Accuracy for each difficulty you have practiced.")
        difficulty_data = pd.DataFrame(
            {
                "Difficulty": [key.value.title() for key in snapshot.accuracy_by_difficulty],
                "Accuracy": list(snapshot.accuracy_by_difficulty.values()),
            }
        ).set_index("Difficulty")
        if not difficulty_data.empty:
            st.bar_chart(difficulty_data, y="Accuracy", y_label="Accuracy")
            strongest_difficulty = max(
                snapshot.accuracy_by_difficulty,
                key=snapshot.accuracy_by_difficulty.get,
            )
            st.caption(
                f"Strongest recorded difficulty: {strongest_difficulty.value.title()} · "
                f"{snapshot.accuracy_by_difficulty[strongest_difficulty]:.0%} accuracy."
            )

    with chart_right:
        section_header("By question type", "Accuracy across the formats you have practiced.")
        question_type_data = pd.DataFrame(
            {
                "Question type": [
                    "Multiple choice" if key.value == "mcq" else "Fill in the blank"
                    for key in snapshot.accuracy_by_question_type
                ],
                "Accuracy": list(snapshot.accuracy_by_question_type.values()),
            }
        ).set_index("Question type")
        if not question_type_data.empty:
            st.bar_chart(question_type_data, y="Accuracy", y_label="Accuracy")
            strongest_type = max(
                snapshot.accuracy_by_question_type,
                key=snapshot.accuracy_by_question_type.get,
            )
            strongest_label = (
                "Multiple choice" if strongest_type.value == "mcq" else "Fill in the blank"
            )
            st.caption(
                f"Strongest recorded format: {strongest_label} · "
                f"{snapshot.accuracy_by_question_type[strongest_type]:.0%} accuracy."
            )

    sessions = repo.list_sessions()
    if sessions:
        section_header("Recent sessions", "Your five most recent study sessions.")
        progress_service = ProgressService()
        for session in reversed(sessions[-5:]):
            summary = progress_service.summarize_session(session)
            with st.container(border=True):
                st.markdown(f"**{session.topic}**")
                st.caption(
                    f"{summary.total_questions} answered · {summary.accuracy:.0%} accuracy"
                )
                st.write(summary.recommended_next_action)
