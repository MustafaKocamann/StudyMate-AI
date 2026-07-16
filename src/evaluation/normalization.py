"""Text normalization utilities for deterministic quality checks."""

from __future__ import annotations

import re
import string
import unicodedata


WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)
TURKIC_LANGUAGE_HINTS = {"tr", "tr-tr", "turkish", "türkçe", "turkce", "az", "azerbaijani"}


def normalize_for_comparison(text: str, *, language: str | None = None) -> str:
    """Normalize text for conservative lexical comparison.

    The normalization is lexical: it applies Unicode NFKC normalization,
    trims/collapses whitespace, and uses case-insensitive comparison. For
    Turkish and Azerbaijani language hints, dotted and dotless I are handled
    before case normalization. This is not stemming, lemmatization, or a
    morphology system, so it cannot detect every valid language variation.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.strip().split())
    if _is_turkic_language(language):
        normalized = normalized.replace("I", "ı").replace("İ", "i")
    return normalized.casefold()


def normalize_for_duplicate_detection(
    text: str,
    *,
    language: str | None = None,
    remove_punctuation: bool = True,
) -> str:
    """Normalize text for exact and fuzzy duplicate checks."""

    normalized = normalize_for_comparison(text, language=language)
    if remove_punctuation:
        normalized = normalized.translate(PUNCTUATION_TRANSLATION)
        normalized = " ".join(normalized.split())
    return normalized


def normalize_text(text: str) -> str:
    """Backward-compatible comparison normalization."""

    return normalize_for_comparison(text)


def tokenize(text: str, *, language: str | None = None) -> set[str]:
    """Tokenize normalized text into Unicode word-like tokens."""

    return set(WORD_PATTERN.findall(normalize_for_duplicate_detection(text, language=language)))


def contains_normalized_phrase(
    text: str,
    phrase: str,
    *,
    language: str | None = None,
    remove_punctuation: bool = False,
) -> bool:
    """Check whether a normalized phrase appears in normalized text."""

    normalizer = normalize_for_duplicate_detection if remove_punctuation else normalize_for_comparison
    normalized_phrase = normalizer(phrase, language=language)
    if not normalized_phrase:
        return False
    return normalized_phrase in normalizer(text, language=language)


def _is_turkic_language(language: str | None) -> bool:
    if not language:
        return False
    return normalize_for_comparison(language) in TURKIC_LANGUAGE_HINTS
