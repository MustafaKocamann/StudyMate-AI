"""Small, content-aware layout helpers shared by StudyMate pages."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st


Metric = tuple[str, str | int, str | None]


def page_header(title: str, description: str, *, eyebrow: str | None = None) -> None:
    """Render a consistent page introduction using native Streamlit text."""

    if eyebrow:
        st.caption(eyebrow)
    st.title(title)
    st.write(description)


def section_header(title: str, description: str | None = None) -> None:
    st.subheader(title)
    if description:
        st.caption(description)


def metric_row(metrics: Sequence[Metric]) -> None:
    """Render meaningful metrics in responsive rows of at most three columns."""

    if not metrics:
        return
    chunk_size = 2 if len(metrics) == 4 else 3
    for start in range(0, len(metrics), chunk_size):
        chunk = metrics[start : start + chunk_size]
        columns = st.columns(len(chunk))
        for column, (label, value, help_text) in zip(columns, chunk, strict=True):
            column.metric(label, value, help=help_text)
