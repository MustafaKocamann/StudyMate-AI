"""Streamlit entrypoint for the StudyMate learning experience."""

from __future__ import annotations

import streamlit as st

from src.common.exceptions import ApplicationConfigurationError, LLMModelUnavailableError, LLMProviderError
from src.ui.dependencies import build_streamlit_application_container
from src.ui.navigation import build_pages
from src.ui.state import StateKey, initialize_state
from src.ui.styles import apply_global_styles


def main() -> None:
    st.set_page_config(
        page_title="StudyMate",
        page_icon=":material/school:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    _initialize_application_services()
    apply_global_styles()

    st.sidebar.title("StudyMate")
    st.sidebar.caption("Focused practice. Useful feedback. Steady progress.")

    st.navigation(build_pages(), position="sidebar", expanded=True).run()


def _initialize_application_services() -> None:
    if st.session_state.get(StateKey.QUESTION_SERVICE.value) is not None:
        return
    try:
        container = build_streamlit_application_container()
    except ApplicationConfigurationError:
        st.sidebar.warning("Question generation is not ready yet.")
        return
    except LLMModelUnavailableError:
        st.sidebar.warning("Question generation is temporarily unavailable.")
        return
    except LLMProviderError:
        st.sidebar.warning("Question generation is temporarily unavailable.")
        return
    st.session_state[StateKey.QUESTION_SERVICE.value] = container.study_question_service


if __name__ == "__main__":
    main()
