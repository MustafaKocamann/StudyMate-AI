"""Single source of truth for StudyMate's Streamlit navigation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import streamlit as st


class AppRoute(StrEnum):
    HOME = "home"
    PRACTICE = "practice"
    REVIEW = "review"
    MISTAKES = "mistakes"
    PROGRESS = "progress"
    SETTINGS = "settings"


_PAGE_DETAILS = {
    AppRoute.HOME: ("Home", ":material/home:"),
    AppRoute.PRACTICE: ("Practice", ":material/edit:"),
    AppRoute.REVIEW: ("Review", ":material/replay:"),
    AppRoute.MISTAKES: ("Mistakes", ":material/edit_note:"),
    AppRoute.PROGRESS: ("Progress", ":material/trending_up:"),
    AppRoute.SETTINGS: ("Settings", ":material/settings:"),
}


def build_pages() -> list[Any]:
    return [page_for(route) for route in AppRoute]


def page_for(route: AppRoute) -> Any:
    """Return a page target whose source and URL match the registered page."""

    title, icon = _PAGE_DETAILS[route]
    return st.Page(
        _renderer_for(route),
        title=title,
        icon=icon,
        url_path=route.value,
        default=route is AppRoute.HOME,
    )


def _renderer_for(route: AppRoute):  # noqa: ANN202
    # Local imports keep page modules free to create navigation links without
    # introducing an import cycle at module import time.
    from src.ui.pages.home import render_home_page
    from src.ui.pages.mistakes import render_mistakes_page
    from src.ui.pages.practice import render_practice_page
    from src.ui.pages.progress import render_progress_page
    from src.ui.pages.review import render_review_page
    from src.ui.pages.settings import render_settings_page

    return {
        AppRoute.HOME: render_home_page,
        AppRoute.PRACTICE: render_practice_page,
        AppRoute.REVIEW: render_review_page,
        AppRoute.MISTAKES: render_mistakes_page,
        AppRoute.PROGRESS: render_progress_page,
        AppRoute.SETTINGS: render_settings_page,
    }[route]
