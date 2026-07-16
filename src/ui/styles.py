"""Centralized visual refinements for the Streamlit interface."""

from __future__ import annotations

import streamlit as st


_GLOBAL_STYLES = """
<style>
    :root {
        --studymate-content-width: 72rem;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: var(--studymate-content-width);
        padding-top: 2.25rem;
        padding-bottom: 4rem;
    }

    [data-testid="stMainBlockContainer"] h1 {
        letter-spacing: -0.035em;
        line-height: 1.12;
    }

    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3 {
        letter-spacing: -0.02em;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0.85rem;
        box-shadow: 0 0.25rem 1.25rem rgba(31, 42, 39, 0.045);
    }

    [data-testid="stMetric"] {
        padding: 0.9rem 1rem;
        border: 1px solid color-mix(in srgb, currentColor 12%, transparent);
        border-radius: 0.75rem;
    }

    [data-testid="stForm"] {
        border-radius: 0.85rem;
    }

    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebar"] button {
        border-radius: 0.55rem;
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stPageLink"] a {
        font-weight: 650;
    }

    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    [data-testid="stPageLink"] a:focus-visible {
        outline: 3px solid color-mix(in srgb, var(--primary-color) 35%, transparent);
        outline-offset: 2px;
    }

    @media (max-width: 48rem) {
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.35rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
        }
    }
</style>
"""


def apply_global_styles() -> None:
    """Apply trusted, content-free CSS once per Streamlit rerun."""

    st.html(_GLOBAL_STYLES)
