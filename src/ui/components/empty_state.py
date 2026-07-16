"""Small empty-state helper."""

from __future__ import annotations

import streamlit as st


def render_empty_state(title: str, body: str) -> None:
    st.info(f"**{title}**\n\n{body}")
