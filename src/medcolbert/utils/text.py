"""Lightweight text normalization utilities.

Kept simple: lowercase, Unicode normalize, whitespace collapse, trim punctuation.
Medical-specific normalization (UMLS strings) lives in medcolbert.data.umls.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces and strip."""
    return " ".join(text.split())


def normalize_unicode(text: str) -> str:
    """Apply NFKC Unicode normalization."""
    return unicodedata.normalize("NFKC", text)


def trim_punctuation(text: str) -> str:
    """Strip leading and trailing punctuation but preserve internal punctuation."""
    return re.sub(r"^\p{P}+|\p{P}+$", "", text, flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Lightweight normalization: lowercase, NFKC, collapse whitespace, trim edges.

    This does NOT remove internal punctuation — token boundaries are preserved.
    """
    text = normalize_unicode(text.lower())
    text = normalize_whitespace(text)
    text = text.strip()
    return text


def is_abbreviation_candidate(text: str) -> bool:
    """Heuristic: short uppercase string is likely an abbreviation."""
    text = text.strip()
    if len(text) > 10:
        return False
    if len(text) < 2:
        return False
    return text.isupper() or (len(text) <= 5 and text.isalpha())


def token_overlap_ratio(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Jaccard overlap between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def tokenize_simple(text: str) -> set[str]:
    """Simple whitespace tokenization for overlap checks."""
    return set(text.split())
