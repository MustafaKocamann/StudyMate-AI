"""Smoke tests for the Streamlit entrypoint."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_home_page() -> None:
    app = AppTest.from_file("streamlit_app.py")
    app.run(timeout=10)

    assert not app.exception
    assert any("Study Buddy AI" in item.value for item in app.title)
    assert any("No sessions yet" in item.value for item in app.info)
