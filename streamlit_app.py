"""Streamlit entrypoint for the Study Buddy AI learning experience."""

from __future__ import annotations

import streamlit as st

from src.common.exceptions import ApplicationConfigurationError, LLMModelUnavailableError, LLMProviderError
from src.ui.dependencies import build_streamlit_application_container
from src.ui.pages.home import render_home_page
from src.ui.pages.mistakes import render_mistakes_page
from src.ui.pages.practice import render_practice_page
from src.ui.pages.progress import render_progress_page
from src.ui.pages.review import render_review_page
from src.ui.pages.settings import render_settings_page
from src.ui.state import StateKey, initialize_state


def main() -> None:
    st.set_page_config(page_title="Study Buddy AI", page_icon="book", layout="wide")
    initialize_state()
    _initialize_application_services()

    st.sidebar.title("Study Buddy AI")
    st.sidebar.caption("Focused practice, review, and progress.")

    pages = [
        st.Page(render_home_page, title="Home", icon=":material/home:"),
        st.Page(render_practice_page, title="Practice", icon=":material/edit:"),
        st.Page(render_review_page, title="Review", icon=":material/replay:"),
        st.Page(render_mistakes_page, title="Mistakes", icon=":material/edit_note:"),
        st.Page(render_progress_page, title="Progress", icon=":material/trending_up:"),
        st.Page(render_settings_page, title="Settings", icon=":material/settings:"),
    ]
    st.navigation(pages).run()


def _initialize_application_services() -> None:
    if st.session_state.get(StateKey.QUESTION_SERVICE.value) is not None:
        return
    try:
        container = build_streamlit_application_container()
    except ApplicationConfigurationError:
        st.sidebar.warning("The AI question service is not configured.")
        return
    except LLMModelUnavailableError:
        st.sidebar.warning(
            "The selected AI model is currently unavailable. Check the application model configuration."
        )
        return
    except LLMProviderError:
        st.sidebar.warning("The AI question service is temporarily unavailable.")
        return
    st.session_state[StateKey.QUESTION_SERVICE.value] = container.study_question_service


if __name__ == "__main__":
    main()
