"""Structured logging with counts and rejection reasons.

Never silently drop examples — always return counts and rejection reasons.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Counts:
    """Track counts of processed, accepted, and rejected items with reasons."""

    total: int = 0
    accepted: int = 0
    rejected: Counter[str] = field(default_factory=Counter)

    def add_accepted(self, n: int = 1) -> None:
        self.total += n
        self.accepted += n

    def add_rejected(self, n: int = 1, *, reason: str = "unspecified") -> None:
        self.total += n
        self.rejected[reason] += n

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected_total,
            "acceptance_rate": (
                self.accepted / self.total if self.total > 0 else 0.0
            ),
            "rejection_reasons": dict(self.rejected),
        }

    def report(self) -> str:
        s = self.summary()
        lines = [
            f"Total: {s['total']}",
            f"Accepted: {s['accepted']} ({s['acceptance_rate']:.2%})",
            f"Rejected: {s['rejected']}",
        ]
        if s["rejection_reasons"]:
            lines.append("Rejection reasons:")
            for reason, count in s["rejection_reasons"].items():
                lines.append(f"  {reason}: {count}")
        return "\n".join(lines)


def write_json_report(
    path: Path,
    data: dict[str, Any],
    *,
    assert_public_safe: bool = False,
) -> None:
    """Write a JSON report to disk, optionally asserting public safety."""
    if assert_public_safe:
        from medcolbert.utils.io import RESTRICTED_FIELDS as _RESTRICTED

        # Check every top-level key
        restricted = set(data.keys()) & _RESTRICTED
        if restricted:
            raise ValueError(
                f"Cannot write public report: restricted fields present: {restricted}"
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def write_parquet(df: Any, path: Path) -> None:
    """Write a pandas/polars DataFrame to parquet, creating parent dirs.

    Args:
        df: A pandas or polars DataFrame.
        path: Output path (will be created if it doesn't exist).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Detect DataFrame type
    module_name = type(df).__module__
    if "polars" in module_name:
        df.write_parquet(str(path))
    else:
        df.to_parquet(path, index=False)
