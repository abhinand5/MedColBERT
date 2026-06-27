"""Tests for the training dataset loader and filters.

Uses small in-memory ``Dataset`` objects so no Hugging Face network access is
required. The end-to-end :func:`load_triplet_dataset` is exercised only by
smoke runs against the real Hub.
"""

from __future__ import annotations

from datasets import Dataset

from medcolbert.training.datasets import (
    AUDIT_COLUMNS,
    PYLATE_NEGATIVE_COL,
    PYLATE_POSITIVE_COL,
    PYLATE_QUERY_COL,
    drop_null_triplets,
    filter_by_audit,
    prepare_training_dataset,
    subset_dataset,
    to_pylate_columns,
)
from medcolbert.training.datasets import FilterReport


def _sample_rows() -> list[dict]:
    return [
        {
            "query": "q1",
            "positive_passage": "p1",
            "negative_passage": "n1",
            "audit_relevant_looks": 2,
            "audit_grounding": 5,
            "weak": 0,
            "failure_mode": "lexical_trap_real",
            "negative_source": "bm25_real",
            "query_style": "keyword",
            "task_family": "symptom_to_diagnosis",
        },
        {
            "query": "q2",
            "positive_passage": "p2",
            "negative_passage": "n2",
            "audit_relevant_looks": 1,
            "audit_grounding": 3,
            "weak": 1,
            "failure_mode": "off_topic",
            "negative_source": "synthetic",
            "query_style": "layperson",
            "task_family": "symptom_to_diagnosis",
        },
        {
            "query": "q3",
            "positive_passage": "p3",
            "negative_passage": "n3",
            "audit_relevant_looks": 3,
            "audit_grounding": 4,
            "weak": 0,
            "failure_mode": "lexical_trap_real",
            "negative_source": "bm25_real",
            "query_style": "technical",
            "task_family": "brand_to_generic",
        },
    ]


def _ds(rows: list[dict]) -> Dataset:
    return Dataset.from_list(rows)


def test_to_pylate_columns_renames_and_keeps_audit():
    ds = to_pylate_columns(_ds(_sample_rows()), keep_audit=True)
    assert PYLATE_QUERY_COL in ds.column_names
    assert PYLATE_POSITIVE_COL in ds.column_names
    assert PYLATE_NEGATIVE_COL in ds.column_names
    assert "positive_passage" not in ds.column_names
    assert "negative_passage" not in ds.column_names
    assert ds[0][PYLATE_POSITIVE_COL] == "p1"
    assert ds[0][PYLATE_NEGATIVE_COL] == "n1"
    assert "audit_relevant_looks" in ds.column_names


def test_to_pylate_columns_drops_audit_when_requested():
    ds = to_pylate_columns(_ds(_sample_rows()), keep_audit=False)
    assert ds.column_names == [PYLATE_QUERY_COL, PYLATE_POSITIVE_COL, PYLATE_NEGATIVE_COL]


def test_drop_null_triplets_reports_counts():
    rows = _sample_rows()
    rows[1]["negative_passage"] = None
    rows[2]["positive_passage"] = "   "
    ds = to_pylate_columns(_ds(rows))
    ds_clean, report = drop_null_triplets(ds)
    assert report.input == 3
    assert report.kept == 1
    assert report.rejections.get("empty_positive") == 1
    assert report.rejections.get("null_negative") == 1
    assert ds_clean[0][PYLATE_QUERY_COL] == "q1"


def test_filter_by_audit_min_relevant_looks():
    ds = to_pylate_columns(_ds(_sample_rows()))
    ds_f, report = filter_by_audit(ds, min_relevant_looks=2)
    assert report.input == 3
    assert report.kept == 2  # rows with relevant_looks 2 and 3
    assert report.rejections.get("below_min_relevant_looks") == 1
    assert ds_f[0]["audit_relevant_looks"] == 2


def test_filter_by_audit_exclude_weak():
    ds = to_pylate_columns(_ds(_sample_rows()))
    ds_f, report = filter_by_audit(ds, exclude_weak=True)
    assert report.kept == 2
    assert report.rejections.get("weak_excluded") == 1


def test_filter_by_audit_failure_modes_allowlist():
    ds = to_pylate_columns(_ds(_sample_rows()))
    ds_f, report = filter_by_audit(ds, failure_modes=["lexical_trap_real"])
    assert report.kept == 2
    assert report.rejections.get("failure_mode_not_allowed") == 1


def test_filter_by_audit_no_filters_returns_all():
    ds = to_pylate_columns(_ds(_sample_rows()))
    ds_f, report = filter_by_audit(ds)
    assert report.kept == 3
    assert report.rejections == {}


def test_subset_dataset_deterministic_and_capped():
    ds = _ds(_sample_rows())
    sub = subset_dataset(ds, max_examples=2, seed=42)
    assert len(sub) == 2
    sub_again = subset_dataset(ds, max_examples=2, seed=42)
    assert sub["query"] == sub_again["query"]


def test_subset_dataset_none_returns_full():
    ds = _ds(_sample_rows())
    sub = subset_dataset(ds, max_examples=None)
    assert len(sub) == 3


def test_filter_report_as_dict_roundtrip():
    r = FilterReport(input=10, kept=7)
    r.add("x", 2)
    r.add("y", 1)
    d = r.as_dict()
    assert d["input"] == 10 and d["kept"] == 7 and d["dropped"] == 3
    assert d["rejections"] == {"x": 2, "y": 1}


def test_prepare_training_dataset_audit_columns_present():
    """Smoke the local pipeline stages (no network): build a fake dataset
    by monkeypatching load_triplet_dataset."""
    rows = _sample_rows()
    rows[0]["negative_passage"] = ""
    ds_in = _ds(rows)

    import medcolbert.training.datasets as mod

    orig = mod.load_triplet_dataset

    def fake_load(**kwargs):
        return ds_in

    mod.load_triplet_dataset = fake_load
    try:
        ds, reports = prepare_training_dataset(
            min_relevant_looks=2, max_examples=10, seed=42
        )
    finally:
        mod.load_triplet_dataset = orig

    assert PYLATE_POSITIVE_COL in ds.column_names
    # null drop removed the empty-negative row -> 2 rows
    assert reports[0].kept == 2
    # audit filter kept only relevant_looks>=2 -> could be 1 or 2
    assert reports[1].input == 2
    assert all(r["audit_relevant_looks"] >= 2 for r in ds)


def test_audit_columns_constant_complete():
    # Sanity: the audit column tuple covers the fields used downstream.
    for needed in (
        "audit_relevant_looks",
        "weak",
        "failure_mode",
        "negative_source",
    ):
        assert needed in AUDIT_COLUMNS
