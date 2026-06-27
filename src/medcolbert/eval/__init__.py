"""Evaluation pipeline: real-corpus retrieval and metric computation."""

from medcolbert.eval.real_corpus import (
    RetrievalMetrics,
    aggregate_metrics,
    build_index,
    load_dev_queries,
    load_real_corpus,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    run_real_corpus_eval,
)

__all__ = [
    "RetrievalMetrics",
    "aggregate_metrics",
    "build_index",
    "load_dev_queries",
    "load_real_corpus",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "run_real_corpus_eval",
]
