"""Focused JSON response decoding helpers."""

from __future__ import annotations

import json
from typing import Any

from src.common.exceptions import LLMResponseParsingError


def decode_json_object(content: str) -> dict[str, object]:
    """Decode provider text as a JSON object root."""

    try:
        parsed: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseParsingError(
            "LLM response was not valid JSON",
            error_category="invalid_json",
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMResponseParsingError(
            "LLM response JSON root must be an object",
            error_category="invalid_json_root",
        )
    return parsed
