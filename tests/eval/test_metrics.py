"""Tests for the pure retrieval metric functions."""

from __future__ import annotations

import math

import pytest

from medcolbert.eval.real_corpus import (
    aggregate_metrics,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_mrr_at_k_first_relevant():
    ranked = ["d1", "d2", "d3", "d4"]
    assert mrr_at_k(ranked, ["d2"], k=10) == 0.5
    assert mrr_at_k(ranked, ["d1"], k=10) == 1.0
    assert mrr_at_k(ranked, ["d4"], k=10) == 0.25


def test_mrr_at_k_truncated():
    ranked = ["d1", "d2", "d3", "d4"]
    assert mrr_at_k(ranked, ["d4"], k=3) == 0.0
    assert mrr_at_k(ranked, ["d4"], k=4) == 0.25


def test_mrr_at_k_no_relevant():
    assert mrr_at_k(["d1", "d2"], ["dX"], k=10) == 0.0


def test_recall_at_k():
    ranked = ["d1", "d2", "d3", "d4"]
    assert recall_at_k(ranked, ["d4"], k=4) == 1.0
    assert recall_at_k(ranked, ["d4"], k=3) == 0.0
    assert recall_at_k(ranked, ["d2"], k=100) == 1.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k(["d1"], [], k=10) == 0.0


def test_ndcg_at_k_single_relevant():
    # rank 1 -> 1/log2(2) = 1.0
    assert ndcg_at_k(["d1", "d2"], ["d1"], k=10) == pytest.approx(1.0)
    # rank 2 -> 1/log2(3)
    assert ndcg_at_k(["d1", "d2"], ["d2"], k=10) == pytest.approx(1.0 / math.log2(3))
    # rank 3 -> 1/log2(4) = 0.5
    assert ndcg_at_k(["d1", "d2", "d3"], ["d3"], k=10) == pytest.approx(0.5)


def test_ndcg_at_k_outside_cutoff():
    assert ndcg_at_k(["d1", "d2", "d3"], ["d3"], k=2) == 0.0


def test_aggregate_metrics_means():
    ranked = [["d1", "d2"], ["d3", "d1"], ["d9", "d9"]]
    relevant = [["d1"], ["d1"], ["d1"]]
    m = aggregate_metrics(ranked, relevant)
    assert m.n_queries == 3
    # query0: rank1 -> mrr=1, r10=1, r100=1, ndcg=1
    # query1: rank2 -> mrr=0.5, r10=1, r100=1, ndcg=1/log2(3)
    # query2: not found -> all 0
    assert m.mrr_at_10 == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert m.recall_at_10 == pytest.approx(2 / 3)
    assert m.recall_at_100 == pytest.approx(2 / 3)
    assert m.ndcg_at_10 == pytest.approx((1.0 + 1.0 / math.log2(3) + 0.0) / 3)
    assert len(m.per_query) == 3


def test_aggregate_metrics_empty():
    m = aggregate_metrics([], [])
    assert m.n_queries == 0
    assert m.mrr_at_10 == 0.0


def test_aggregate_metrics_length_mismatch():
    with pytest.raises(ValueError):
        aggregate_metrics([["d1"]], [["d1"], ["d2"]])


def test_aggregate_metrics_as_dict():
    m = aggregate_metrics([["d1"]], [["d1"]])
    d = m.as_dict()
    assert d["n_queries"] == 1
    assert d["mrr_at_10"] == 1.0
    assert d["recall_at_100"] == 1.0
