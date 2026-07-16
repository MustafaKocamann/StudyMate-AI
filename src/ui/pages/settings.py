"""Learner preference page."""

from __future__ import annotations

import streamlit as st

from src.models.question_schemas import DifficultyLevel
from src.models.study_session import StudyQuestionMode
from src.ui.layout import page_header, section_header
from src.ui.state import StateKey


def render_settings_page() -> None:
    page_header(
        "Settings",
        "Choose defaults that make each new study session easier to begin.",
        eyebrow="LEARNING PREFERENCES",
    )
    preferences = st.session_state[StateKey.USER_PREFERENCES.value]

    with st.form("settings-form", border=True):
        section_header("Session defaults")
        language = st.text_input(
            "Default question language",
            value=preferences["default_language"],
        )
        left, right = st.columns(2)
        with left:
            difficulty = st.selectbox(
                "Default difficulty",
                [level.value for level in DifficultyLevel],
                index=[level.value for level in DifficultyLevel].index(preferences["default_difficulty"]),
                format_func=lambda value: value.title(),
            )
            question_count = st.number_input(
                "Default question count",
                min_value=1,
                max_value=20,
                value=int(preferences["default_question_count"]),
                step=1,
            )
        with right:
            question_mode = st.selectbox(
                "Default question type",
                [mode.value for mode in StudyQuestionMode],
                index=[mode.value for mode in StudyQuestionMode].index(preferences["default_question_mode"]),
                format_func=lambda value: {
                    "mcq": "Multiple choice",
                    "fill_blank": "Fill in the blank",
                    "mixed": "Mixed",
                }[value],
            )

        section_header("During practice")
        show_explanations = st.checkbox(
            "Show explanations automatically after each answer",
            value=preferences["show_explanations_automatically"],
        )
        confidence = st.checkbox(
            "Ask for confidence with each answer",
            value=preferences["enable_confidence_capture"],
            help="Confidence helps prioritize review and never changes correctness.",
        )
        hints = st.checkbox(
            "Enable hints when available",
            value=preferences["enable_hints"],
        )
        submitted = st.form_submit_button(
            "Save preferences",
            type="primary",
            icon=":material/save:",
            width="stretch",
        )

    if not submitted:
        return
    if not language.strip():
        st.error("Enter a default question language.")
        return

    st.session_state[StateKey.USER_PREFERENCES.value] = {
        "default_language": language,
        "default_difficulty": difficulty,
        "default_question_count": int(question_count),
        "default_question_mode": question_mode,
        "show_explanations_automatically": show_explanations,
        "enable_confidence_capture": confidence,
        "enable_hints": hints,
    }
    st.success("Preferences saved.", icon=":material/check_circle:")
