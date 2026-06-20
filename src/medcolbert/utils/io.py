"""Path helpers for private/public data boundaries.

Enforces the privacy rules defined in DS_GOAL.md:
- Private data under data/processed/private/
- Public-safe data under data/processed/public/
- Public outputs must not contain restricted vocabulary strings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Resolve from environment or default to repo root.
# In production, MEDCOLBERT_ROOT should be set to the repo root.
_REPO_ROOT = Path(os.environ.get("MEDCOLBERT_ROOT", Path(__file__).resolve().parents[3]))


def repo_root() -> Path:
    """Return the resolved repository root."""
    return _REPO_ROOT


def private_dir() -> Path:
    """Return the private data output directory."""
    return _REPO_ROOT / "data" / "processed" / "private"


def public_dir() -> Path:
    """Return the public-safe data output directory."""
    return _REPO_ROOT / "data" / "processed" / "public"


def private_output_path(*parts: str) -> Path:
    """Build a path under data/processed/private/."""
    p = private_dir()
    for part in parts:
        p = p / part
    return p


def public_output_path(*parts: str) -> Path:
    """Build a path under data/processed/public/."""
    p = public_dir()
    for part in parts:
        p = p / part
    return p


def assert_private_path(path: Path) -> None:
    """Assert that a path lives under the private data directory.

    Raises ValueError if the path is outside the private boundary.
    """
    resolved = path.resolve()
    priv = private_dir().resolve()
    try:
        resolved.relative_to(priv)
    except ValueError:
        raise ValueError(
            f"Path {path} is outside the private data boundary ({priv}). "
            "Restricted data must be written under data/processed/private/."
        )


# Fields that are safe for public manifests (no restricted vocabulary strings).
ALLOWED_PUBLIC_FIELDS = frozenset(
    {
        "cui",
        "count",
        "hash",
        "source",
        "source_name",
        "semantic_group",
        "vocab_shift_type",
        "generation_mode",
        "task_family",
        "role",
        "quality_label",
        "judge_model",
        "prompt_hash",
        "validator_version",
        "acceptance_rate",
        "total",
        "passed",
        "failed",
        "rejected",
        "batches",
        "completed",
        "per_family_acceptance",
        "per_dimension_pass_rates",
        "quality_trend",
        "mode_stats",
        "timestamp",
        "version",
        "git_commit",
        "config_hash",
        "dependency_lockfile_hash",
        "id_fields",
        "description",
        "note",
        "schema_version",
        "num_perm",
        "minhash_threshold",
        "target",
        "threshold",
    }
)

# Fields that MUST NOT appear in public outputs.
RESTRICTED_FIELDS = frozenset(
    {
        "string",
        "normalized_string",
        "source_string",
        "target_string",
        "query",
        "passage",
        "text",
        "positive_passage",
        "negative_passage",
        "content",
    }
)


def assert_no_restricted_public_fields(record: dict[str, Any]) -> None:
    """Raise ValueError if any restricted field names appear in the record.

    This checks field NAMES, not values. It is a blunt safety check.
    """
    restricted = set(record.keys()) & RESTRICTED_FIELDS
    if restricted:
        raise ValueError(
            f"Public record contains restricted field(s): {sorted(restricted)}. "
            "These fields may contain UMLS or corpus string content. "
            "Remove them or route this artifact to the private directory."
        )


def is_public_safe(record: dict[str, Any]) -> bool:
    """Return True if the record contains no restricted field names."""
    return not (set(record.keys()) & RESTRICTED_FIELDS)
