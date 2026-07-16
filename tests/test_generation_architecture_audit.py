from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_FILES = [
    path
    for path in (PROJECT_ROOT / "src").rglob("*.py")
    if "__pycache__" not in path.parts
]


def production_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION_FILES)


def test_active_generation_path_does_not_use_legacy_llm_patterns() -> None:
    source = production_text()

    assert "PydanticOutputParser" not in source
    assert "get_groq_llm" not in source
    assert ".invoke(" not in source
    assert "src.prompts.templates" not in source


def test_streamlit_pages_do_not_import_provider_or_generation_internals() -> None:
    ui_files = list((PROJECT_ROOT / "src" / "ui").rglob("*.py")) + [PROJECT_ROOT / "streamlit_app.py"]
    ui_source = "\n".join(path.read_text(encoding="utf-8") for path in ui_files)

    forbidden = [
        "AsyncGroq",
        "GroqClient",
        "GroqQuestionGenerator",
        "GeneratedMCQPayload",
        "GeneratedFillBlankPayload",
        "decode_json_object",
        "QuestionQualityEvaluator",
        "QualityGatedQuestionGenerator",
        "mcq_prompt_template",
        "fill_blank_prompt_template",
    ]
    for symbol in forbidden:
        assert symbol not in ui_source


def test_direct_groq_sdk_usage_is_isolated_to_low_level_client() -> None:
    offenders = []
    for path in PRODUCTION_FILES:
        text = path.read_text(encoding="utf-8")
        if "AsyncGroq" in text or "chat.completions.create" in text:
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

    assert offenders == ["src/llm/groq_client.py"]
