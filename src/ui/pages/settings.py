"""Learner preference page."""

from __future__ import annotations

import streamlit as st

from src.models.question_schemas import DifficultyLevel
from src.models.study_session import StudyQuestionMode
from src.ui.state import StateKey


def render_settings_page() -> None:
    st.title("Settings")
    preferences = st.session_state[StateKey.USER_PREFERENCES.value]

    with st.form("settings-form"):
        language = st.text_input("Default language", value=preferences["default_language"])
        difficulty = st.selectbox(
            "Default difficulty",
            [level.value for level in DifficultyLevel],
            index=[level.value for level in DifficultyLevel].index(preferences["default_difficulty"]),
        )
        question_count = st.number_input(
            "Default question count",
            min_value=1,
            max_value=20,
            value=int(preferences["default_question_count"]),
        )
        question_mode = st.selectbox(
            "Default question type",
            [mode.value for mode in StudyQuestionMode],
            index=[mode.value for mode in StudyQuestionMode].index(preferences["default_question_mode"]),
        )
        show_explanations = st.checkbox(
            "Show explanations automatically after submission",
            value=preferences["show_explanations_automatically"],
        )
        confidence = st.checkbox(
            "Enable confidence capture",
            value=preferences["enable_confidence_capture"],
        )
        hints = st.checkbox("Enable hints", value=preferences["enable_hints"])
        submitted = st.form_submit_button("Save settings")

    if submitted:
        preferences.update(
            {
                "default_language": language,
                "default_difficulty": difficulty,
                "default_question_count": int(question_count),
                "default_question_mode": question_mode,
                "show_explanations_automatically": show_explanations,
                "enable_confidence_capture": confidence,
                "enable_hints": hints,
            }
        )
        st.success("Settings saved.")
