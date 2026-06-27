"""Real-corpus retrieval evaluation — the train-on-synthetic quality gate.

Builds a ColBERT PLAID index over the real PubMed corpus, retrieves the held-out
dev queries, and measures MRR@10, Recall@100, nDCG@10 against the known
positive passage ids. The BM25 baseline (Recall@100 ≈ 0.63) lives in the dev
dataset; the trained model must beat it.

Metric functions are pure and unit-tested; the retrieval runner requires a
loaded ColBERT model and the corpus on disk.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pylate import indexes, models, retrieve

DEV_REPO = "fierysurf/medcolbert-training-v1"
DEV_CONFIG = "dev"
DEV_SPLIT = "test"
DEFAULT_KS = (10, 100)


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics over a query set."""

    n_queries: int
    mrr_at_10: float
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    # Per-query detail kept for diagnostics.
    per_query: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "mrr_at_10": self.mrr_at_10,
            "recall_at_10": self.recall_at_10,
            "recall_at_100": self.recall_at_100,
            "ndcg_at_10": self.ndcg_at_10,
        }


# ─── Pure metric functions (unit-tested) ────────────────────────────────────


def _first_relevant_rank(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> int | None:
    """1-indexed rank of the first relevant id within the top-k, or None."""
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_ids:
            return i + 1
    return None


def mrr_at_k(
    ranked_ids: list[str],
    relevant_ids: Iterable[str],
    k: int = 10,
) -> float:
    """MRR@k for a single query: 1/rank of first relevant in top-k, else 0."""
    rel = set(relevant_ids)
    rank = _first_relevant_rank(ranked_ids, rel, k)
    return 1.0 / rank if rank is not None else 0.0


def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: Iterable[str],
    k: int = 100,
) -> float:
    """Recall@k for a single query (binary: any relevant retrieved in top-k)."""
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    top = set(ranked_ids[:k])
    return 1.0 if top & rel else 0.0


def ndcg_at_k(
    ranked_ids: list[str],
    relevant_ids: Iterable[str],
    k: int = 10,
) -> float:
    """nDCG@k with binary relevance and a single relevant doc.

    DCG = sum_i rel_i / log2(i+2) for 0-indexed positions; with one relevant
    doc at rank r (1-indexed) inside top-k, DCG = 1/log2(r+1), IDCG = 1.
    """
    rel = set(relevant_ids)
    rank = _first_relevant_rank(ranked_ids, rel, k)
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def aggregate_metrics(
    per_query_ranked: list[list[str]],
    per_query_relevant: list[list[str]],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> RetrievalMetrics:
    """Compute aggregate MRR@10, Recall@10/100, nDCG@10 over a query set.

    Args:
        per_query_ranked: list of ranked doc-id lists (one per query).
        per_query_relevant: list of relevant doc-id lists (one per query).
        ks: k values for recall (must include 10 and 100 for the metrics).
    """
    if len(per_query_ranked) != len(per_query_relevant):
        raise ValueError(
            f"length mismatch: {len(per_query_ranked)} ranked vs "
            f"{len(per_query_relevant)} relevant"
        )
    n = len(per_query_ranked)
    if n == 0:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0, 0.0, [])

    details: list[dict[str, Any]] = []
    mrr10 = r10 = r100 = ndcg10 = 0.0
    for ranked, relevant in zip(per_query_ranked, per_query_relevant):
        m = mrr_at_k(ranked, relevant, 10)
        r10v = recall_at_k(ranked, relevant, 10)
        r100v = recall_at_k(ranked, relevant, 100)
        nd = ndcg_at_k(ranked, relevant, 10)
        mrr10 += m
        r10 += r10v
        r100 += r100v
        ndcg10 += nd
        details.append({
            "mrr_at_10": m,
            "recall_at_10": r10v,
            "recall_at_100": r100v,
            "ndcg_at_10": nd,
        })
    return RetrievalMetrics(
        n_queries=n,
        mrr_at_10=mrr10 / n,
        recall_at_10=r10 / n,
        recall_at_100=r100 / n,
        ndcg_at_10=ndcg10 / n,
        per_query=details,
    )


# ─── Data loading ────────────────────────────────────────────────────────────


def load_real_corpus(path: Path | str) -> list[dict[str, str]]:
    """Load the real PubMed corpus JSON (list of {passage_id, text, ...})."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of passages, got {type(data).__name__}")
    needed = ("passage_id", "text")
    for row in data:
        for k in needed:
            if k not in row:
                raise ValueError(f"Corpus entry missing '{k}': {list(row.keys())}")
    return data


def load_dev_queries(
    repo: str = DEV_REPO,
    config: str = DEV_CONFIG,
    split: str = DEV_SPLIT,
    hf_token: str | None = None,
) -> list[dict[str, Any]]:
    """Load the dev queries with their positive passage ids.

    Returns rows with at least ``query`` and ``positive_passage_id``.
    """
    token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
    from datasets import load_dataset

    ds = load_dataset(repo, name=config, split=split, token=token)
    return ds.to_list()


# ─── Retrieval runner (requires a loaded ColBERT model) ───────────────────────


def build_index(
    model: models.ColBERT,
    corpus: list[dict[str, str]],
    index_dir: Path | str,
    index_name: str = "real_corpus",
    batch_size: int = 64,
    override: bool = True,
) -> retrieve.ColBERT:
    """Encode the corpus and build a PLAID index; return a retriever."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index = indexes.PLAID(
        index_folder=str(index_dir),
        index_name=index_name,
        override=override,
    )
    doc_ids = [str(r["passage_id"]) for r in corpus]
    doc_texts = [r["text"] for r in corpus]
    doc_embs = model.encode(
        doc_texts,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=True,
    )
    index.add_documents(
        documents_ids=doc_ids,
        documents_embeddings=doc_embs,
    )
    return retrieve.ColBERT(index=index)


def run_real_corpus_eval(
    model_path: str,
    corpus_path: Path | str,
    index_dir: Path | str,
    dev_rows: list[dict[str, Any]] | None = None,
    k: int = 100,
    batch_size: int = 64,
    hf_token: str | None = None,
    index_name: str = "real_corpus",
    override_index: bool = True,
) -> RetrievalMetrics:
    """End-to-end real-corpus retrieval eval.

    Args:
        model_path: path/Hub id of the trained ColBERT model.
        corpus_path: path to real_annotated_500.json.
        index_dir: where to build the PLAID index.
        dev_rows: preloaded dev rows; if None, load from HF.
        k: top-k to retrieve (>= 100 to compute Recall@100).
    """
    corpus = load_real_corpus(corpus_path)
    if dev_rows is None:
        dev_rows = load_dev_queries(hf_token=hf_token)

    model = models.ColBERT(model_name_or_path=model_path)
    retriever = build_index(
        model=model,
        corpus=corpus,
        index_dir=index_dir,
        index_name=index_name,
        batch_size=batch_size,
        override=override_index,
    )

    queries = [r["query"] for r in dev_rows]
    q_embs = model.encode(
        queries,
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=True,
    )
    results = retriever.retrieve(queries_embeddings=q_embs, k=k)

    per_query_ranked = [[hit["id"] for hit in hits] for hits in results]
    per_query_relevant = [[r["positive_passage_id"]] for r in dev_rows]
    return aggregate_metrics(per_query_ranked, per_query_relevant)
