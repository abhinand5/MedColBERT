"""Deterministic hashing utilities for generated artifacts.

Uses SHA-256 and MinHash for deduplication and fingerprinting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash(*parts: str) -> str:
    """Create a deterministic SHA-256 hash from one or more string parts.

    Parts are joined with a delimiter so hash("a", "bc") != hash("ab", "c").
    Returns a hex string.
    """
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def file_content_hash(path: Path | str) -> str:
    """SHA-256 hash of a file's contents."""
    path = Path(path)
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def dict_hash(d: dict[str, Any]) -> str:
    """Deterministic hash of a dictionary (sorted keys, canonical JSON)."""
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_text_for_fingerprint(text: str) -> str:
    """Normalize text for fingerprinting: lowercase, strip, collapse whitespace.

    This is intentionally less aggressive than full medical normalization.
    It preserves word boundaries for MinHash tokenization.
    """
    return " ".join(text.lower().split())


def shingle(text: str, k: int = 3) -> list[str]:
    """Generate character k-shingles from normalized text."""
    return [text[i : i + k] for i in range(len(text) - k + 1)]
