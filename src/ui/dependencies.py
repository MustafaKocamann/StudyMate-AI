"""Streamlit dependency boundary for cached application services."""

from __future__ import annotations

import streamlit as st

from src.application.dependencies import ApplicationContainer, build_application_container


@st.cache_resource
def build_streamlit_application_container() -> ApplicationContainer:
    """Build and cache provider-backed application services across reruns.

    The cached object is application infrastructure, not learner session data.
    API keys remain in settings and SDK internals and are never copied into
    ``st.session_state``.
    """

    return build_application_container()
