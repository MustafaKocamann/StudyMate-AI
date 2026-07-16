"""Centralized Streamlit session-state access."""

from __future__ import annotations

from enum import StrEnum

import streamlit as st

from src.models.question_schemas import DifficultyLevel
from src.models.study_session import StudyQuestionMode
from src.repositories.in_memory_study_repository import InMemoryStudyRepository


class UIPhase(StrEnum):
    """Explicit learner flow states used across reruns."""

    CONFIGURING = "configuring"
    GENERATING = "generating"
    ANSWERING = "answering"
    FEEDBACK = "feedback"
    NEXT_QUESTION = "next_question"
    COMPLETED = "completed"


class StateKey(StrEnum):
    """Shared session-state keys to avoid raw string scattering."""

    REPOSITORY = "study_repository"
    QUESTION_SERVICE = "study_question_service"
    HINT_PROVIDER = "hint_provider"
    ACTIVE_SESSION = "active_session"
    UI_PHASE = "ui_phase"
    LAST_FEEDBACK = "last_feedback"
    USER_PREFERENCES = "user_preferences"
    HINT_LEVEL = "hint_level"


DEFAULT_PREFERENCES = {
    "default_language": "English",
    "default_difficulty": DifficultyLevel.MEDIUM.value,
    "default_question_count": 5,
    "show_explanations_automatically": True,
    "enable_confidence_capture": True,
    "enable_hints": False,
    "default_question_mode": StudyQuestionMode.MIXED.value,
}


def initialize_state() -> None:
    """Prepare rerun-safe UI state without storing provider secrets or prompts."""

    st.session_state.setdefault(StateKey.REPOSITORY.value, InMemoryStudyRepository())
    st.session_state.setdefault(StateKey.ACTIVE_SESSION.value, None)
    st.session_state.setdefault(StateKey.UI_PHASE.value, UIPhase.CONFIGURING.value)
    st.session_state.setdefault(StateKey.LAST_FEEDBACK.value, None)
    st.session_state.setdefault(StateKey.HINT_LEVEL.value, 0)
    st.session_state.setdefault(StateKey.USER_PREFERENCES.value, DEFAULT_PREFERENCES.copy())


def repository() -> InMemoryStudyRepository:
    initialize_state()
    return st.session_state[StateKey.REPOSITORY.value]


def set_phase(phase: UIPhase) -> None:
    st.session_state[StateKey.UI_PHASE.value] = phase.value


def phase() -> UIPhase:
    return UIPhase(st.session_state.get(StateKey.UI_PHASE.value, UIPhase.CONFIGURING.value))
