"""Small empty-state helper."""

from __future__ import annotations

import streamlit as st


def render_empty_state(title: str, body: str, *, icon: str = ":material/inbox:") -> None:
    """Render a compact empty state only when meaningful copy is available."""

    st.info(f"**{title}**\n\n{body}", icon=icon)
