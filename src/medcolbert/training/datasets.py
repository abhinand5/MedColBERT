"""Triplet dataset loading and transformation for PyLate contrastive training.

The canonical training data is the private Hugging Face dataset
``fierysurf/medcolbert-training-v1``. This module loads the ``long`` config
(one row per kept negative, with the positive passage joined) and transforms
it into the column shape PyLate's ``ColBERTCollator`` expects
(``query`` / ``positive`` / ``negative``).

Audit fields on the negatives (``audit_relevant_looks``, ``audit_grounding``,
``weak`` ...) are preserved so callers can filter or weight per-negative. No
examples are silently dropped: every filter returns rejection counts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, load_dataset

DEFAULT_REPO = "fierysurf/medcolbert-training-v1"
DEFAULT_CONFIG = "long"
DEFAULT_SPLIT = "train"

# Columns PyLate's ColBERTCollator routes to sentence features.
PYLATE_QUERY_COL = "query"
PYLATE_POSITIVE_COL = "positive"
PYLATE_NEGATIVE_COL = "negative"

# sentence-transformers losses (MultipleNegativesRankingLoss and the sparse
# variants) consume ``anchor`` / ``positive`` (+ optional ``negative``). The
# triplet dataset ships ``query`` / ``positive`` / ``negative``; only the
# query column is renamed.
SBERT_QUERY_COL = "anchor"
SBERT_POSITIVE_COL = "positive"
SBERT_NEGATIVE_COL = "negative"

# Source column names on the HF ``long`` config.
SOURCE_QUERY_COL = "query"
SOURCE_POSITIVE_COL = "positive_passage"
SOURCE_NEGATIVE_COL = "negative_passage"

# Audit + metadata columns worth keeping alongside the triplet.
AUDIT_COLUMNS = (
    "negative_source",
    "failure_mode",
    "audit_relevant_looks",
    "audit_answers_query",
    "audit_grounding",
    "audit_in_failure_mode",
    "weak",
    "query_style",
    "task_family",
    "semantic_group",
    "vocab_shift_type",
    "cui_label",
    "positive_passage_id",
    "query_id",
)


@dataclass
class FilterReport:
    """Counts of kept/dropped examples across filter stages.

    ``kept`` is the post-filter row count; ``rejections`` maps a reason label
    to the number of rows dropped for that reason. Rejection reasons are
    mutually exclusive per row (a row is dropped by the first failing rule).
    """

    input: int = 0
    kept: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, n: int) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + n

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "kept": self.kept,
            "dropped": self.input - self.kept,
            "rejections": dict(self.rejections),
        }


def load_triplet_dataset(
    repo: str = DEFAULT_REPO,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    hf_token: str | None = None,
) -> Dataset:
    """Load a triplet config from the Hugging Face Hub.

    Args:
        repo: HF dataset repository id.
        config: config name (``long``, ``triplets``, or ``dev``).
        split: split name.
        hf_token: optional HF token; defaults to ``$HF_TOKEN``.

    Returns:
        The raw ``datasets.Dataset`` with source column names.
    """
    token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
    return load_dataset(repo, name=config, split=split, token=token)


def to_pylate_columns(
    ds: Dataset,
    keep_audit: bool = True,
) -> Dataset:
    """Rename source columns to the PyLate triplet shape.

    Renames ``positive_passage`` → ``positive`` and ``negative_passage`` →
    ``negative`` (``query`` is already correct). Drops empty/null rows and
    returns a report via the returned dataset's ``.features`` is not possible,
    so callers should run :func:`drop_null_triplets` separately for counts.

    Args:
        ds: dataset with source column names.
        keep_audit: if True, retain audit/metadata columns; if False, keep
            only ``query``/``positive``/``negative``.

    Returns:
        Dataset with PyLate column names.
    """
    rename_map = {
        SOURCE_POSITIVE_COL: PYLATE_POSITIVE_COL,
        SOURCE_NEGATIVE_COL: PYLATE_NEGATIVE_COL,
    }
    rename_map = {k: v for k, v in rename_map.items() if k in ds.column_names}
    ds = ds.rename_columns(rename_map)

    if not keep_audit:
        keep = [PYLATE_QUERY_COL, PYLATE_POSITIVE_COL, PYLATE_NEGATIVE_COL]
        ds = ds.select_columns([c for c in keep if c in ds.column_names])
    return ds


def to_sbert_columns(ds: Dataset) -> Dataset:
    """Rename the triplet ``query`` column to ``anchor`` for sentence-transformers.

    sentence-transformers ``MultipleNegativesRankingLoss`` (and the sparse
    ``SparseMultipleNegativesRankingLoss``) consume ``anchor`` / ``positive``
    with an optional ``negative`` hard-negative column. Call this after
    :func:`to_pylate_columns` / :func:`prepare_training_dataset`, which leaves
    the dataset in the ``query`` / ``positive`` / ``negative`` shape; only the
    ``query`` → ``anchor`` rename is applied.
    """
    rename_map = {PYLATE_QUERY_COL: SBERT_QUERY_COL} if PYLATE_QUERY_COL in ds.column_names else {}
    if rename_map:
        ds = ds.rename_columns(rename_map)
    return ds


def drop_null_triplets(ds: Dataset) -> tuple[Dataset, FilterReport]:
    """Drop rows where query/positive/negative is null or empty string.

    Returns the cleaned dataset and a :class:`FilterReport` with counts.
    """
    report = FilterReport(input=len(ds))

    def _bad(row: dict[str, Any]) -> str | None:
        for col in (PYLATE_QUERY_COL, PYLATE_POSITIVE_COL, PYLATE_NEGATIVE_COL):
            v = row.get(col)
            if v is None:
                return f"null_{col}"
            if isinstance(v, str) and v.strip() == "":
                return f"empty_{col}"
        return None

    keep_mask = []
    rejections: dict[str, int] = {}
    for row in ds:
        reason = _bad(row)
        keep_mask.append(reason is None)
        if reason:
            rejections[reason] = rejections.get(reason, 0) + 1
    for reason, n in rejections.items():
        report.add(reason, n)
    ds_clean = ds.select([i for i, k in enumerate(keep_mask) if k])
    report.kept = len(ds_clean)
    return ds_clean, report


def filter_by_audit(
    ds: Dataset,
    min_relevant_looks: int | None = None,
    max_weak: int | None = None,
    exclude_weak: bool = False,
    failure_modes: list[str] | None = None,
) -> tuple[Dataset, FilterReport]:
    """Filter negatives by audit quality fields.

    All filters are inclusive thresholds: a row is kept only if it passes
    every enabled rule. Rejection counts are reported per reason.

    Args:
        ds: dataset with audit columns (call after :func:`to_pylate_columns`
            with ``keep_audit=True``).
        min_relevant_looks: drop rows with ``audit_relevant_looks`` below this.
        max_weak: drop rows with ``weak`` above this (``weak`` is 0/1 int).
        exclude_weak: if True, drop any row where ``weak`` is truthy.
        failure_modes: if set, keep only rows whose ``failure_mode`` is in
            this allow-list.

    Returns:
        (filtered_dataset, FilterReport).
    """
    report = FilterReport(input=len(ds))

    def _reason(row: dict[str, Any]) -> str | None:
        if min_relevant_looks is not None:
            v = row.get("audit_relevant_looks")
            if v is None or v < min_relevant_looks:
                return "below_min_relevant_looks"
        if exclude_weak and row.get("weak"):
            return "weak_excluded"
        if max_weak is not None and (row.get("weak") or 0) > max_weak:
            return "above_max_weak"
        if failure_modes is not None:
            fm = row.get("failure_mode")
            if fm not in failure_modes:
                return "failure_mode_not_allowed"
        return None

    keep_idx: list[int] = []
    rejections: dict[str, int] = {}
    for i, row in enumerate(ds):
        reason = _reason(row)
        if reason is None:
            keep_idx.append(i)
        else:
            rejections[reason] = rejections.get(reason, 0) + 1
    for reason, n in rejections.items():
        report.add(reason, n)
    ds_out = ds.select(keep_idx)
    report.kept = len(ds_out)
    return ds_out, report


def subset_dataset(
    ds: Dataset,
    max_examples: int | None,
    seed: int = 42,
) -> Dataset:
    """Return a deterministic shuffled subset of ``ds``.

    Uses ``Dataset.shuffle`` with the given seed then selects the first
    ``max_examples`` rows. If ``max_examples`` is None or >= len(ds), returns
    ``ds`` unchanged (no copy).
    """
    if max_examples is None or max_examples >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(max_examples))


def prepare_training_dataset(
    repo: str = DEFAULT_REPO,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    hf_token: str | None = None,
    keep_audit: bool = True,
    min_relevant_looks: int | None = None,
    exclude_weak: bool = False,
    failure_modes: list[str] | None = None,
    max_examples: int | None = None,
    seed: int = 42,
) -> tuple[Dataset, list[FilterReport]]:
    """End-to-end: load → rename → drop nulls → audit filter → subset.

    Returns the final dataset and the list of :class:`FilterReport` from each
    stage (null-drop, audit-filter) for logging.
    """
    ds = load_triplet_dataset(repo=repo, config=config, split=split, hf_token=hf_token)
    ds = to_pylate_columns(ds, keep_audit=keep_audit)
    reports: list[FilterReport] = []
    ds, r_null = drop_null_triplets(ds)
    reports.append(r_null)
    if (
        min_relevant_looks is not None
        or exclude_weak
        or failure_modes is not None
    ):
        ds, r_audit = filter_by_audit(
            ds,
            min_relevant_looks=min_relevant_looks,
            exclude_weak=exclude_weak,
            failure_modes=failure_modes,
        )
        reports.append(r_audit)
    ds = subset_dataset(ds, max_examples=max_examples, seed=seed)
    return ds, reports
