"""Public corpus ingestion and passage chunking.

Loads biomedical corpora (PubMed, BeIR, ClinicalTrials.gov) and normalizes
them into the standard passage schema with deterministic fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from medcolbert.utils.hashing import stable_hash, normalize_text_for_fingerprint
from medcolbert.utils.logging import Counts


@dataclass
class PassageRecord:
    """Standardized passage record."""

    passage_id: str
    source: str
    source_doc_id: str
    split: str | None = None
    title: str | None = None
    text: str = ""
    url: str | None = None
    license: str | None = None
    metadata: dict = field(default_factory=dict)
    text_hash: str = ""
    normalized_text_hash: str = ""
    minhash: str = ""


def chunk_text(
    text: str,
    max_tokens: int = 384,
    overlap_tokens: int = 64,
    tokenizer_fn: Any = None,
) -> list[str]:
    """Chunk long text into overlapping passages.

    Uses whitespace tokenization by default (no heavy model required).
    For medical text, this is usually sufficient for retrieval passages.
    """
    if not text or not text.strip():
        return []

    # Simple whitespace tokenization (fast, no model needed)
    tokens = text.split()

    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    step = max_tokens - overlap_tokens
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start += step

        if end >= len(tokens):
            break

    return chunks


def build_passage_id(source: str, doc_id: str, chunk_idx: int = 0) -> str:
    """Deterministic passage ID from source, doc ID, and chunk index."""
    return stable_hash(source, doc_id, str(chunk_idx))


def fingerprint_passage(text: str) -> dict[str, str]:
    """Compute deterministic fingerprints for a passage.

    Returns:
        Dict with text_hash (exact), normalized_text_hash, and minhash placeholder.
    """
    normalized = normalize_text_for_fingerprint(text)
    return {
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_text_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        # MinHash placeholder — computed lazily with datasketch if needed
        "minhash": "",
    }


# ── PubMed / MedRAG ingestion ────────────────────────────────────────────────


def ingest_pubmed_jsonl(
    path: Path,
    source: str = "pubmed",
    max_tokens: int = 384,
    overlap_tokens: int = 64,
    min_chars: int = 80,
) -> tuple[list[PassageRecord], Counts]:
    """Ingest PubMed articles from MedRAG-style JSONL files.

    Each JSON line contains: {id, title, content, ...}
    """
    records: list[PassageRecord] = []
    counts = Counts()

    if not path.exists():
        counts.add_rejected(reason=f"path_not_found:{path}")
        return records, counts

    # Handle both single file and directory of JSONL
    files = list(path.glob("*.jsonl")) if path.is_dir() else [path] if path.suffix == ".jsonl" else []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    counts.add_rejected(reason="invalid_json")
                    continue

                doc_id = str(doc.get("id", doc.get("pmid", "")))
                if not doc_id:
                    counts.add_rejected(reason="missing_id")
                    continue

                title = doc.get("title", "")
                content = doc.get("content", doc.get("abstract", ""))
                text = f"{title}\n{content}".strip() if title else content

                if not text or len(text) < min_chars:
                    counts.add_rejected(reason=f"too_short:{len(text)}")
                    continue

                chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
                for idx, chunk in enumerate(chunks):
                    pid = build_passage_id(source, doc_id, idx)
                    fps = fingerprint_passage(chunk)
                    records.append(PassageRecord(
                        passage_id=pid,
                        source=source,
                        source_doc_id=doc_id,
                        title=title if idx == 0 else f"{title} (chunk {idx})",
                        text=chunk,
                        url=doc.get("url", ""),
                        metadata={"chunk_index": idx, "num_chunks": len(chunks)},
                        text_hash=fps["text_hash"],
                        normalized_text_hash=fps["normalized_text_hash"],
                        minhash=fps["minhash"],
                    ))
                    counts.add_accepted()

    return records, counts


def ingest_beir_corpus(
    path: Path,
    source: str,
    max_tokens: int = 384,
    overlap_tokens: int = 64,
    min_chars: int = 80,
) -> tuple[list[PassageRecord], Counts]:
    """Ingest a BeIR-format corpus (JSONL with _id, title, text fields)."""
    records: list[PassageRecord] = []
    counts = Counts()

    if not path.exists():
        counts.add_rejected(reason=f"path_not_found:{path}")
        return records, counts

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                counts.add_rejected(reason="invalid_json")
                continue

            doc_id = str(doc.get("_id", ""))
            if not doc_id:
                counts.add_rejected(reason="missing_id")
                continue

            text = doc.get("text", "")
            title = doc.get("title", "")
            if title:
                text = f"{title}\n{text}"

            if not text or len(text) < min_chars:
                counts.add_rejected(reason=f"too_short:{len(text)}")
                continue

            chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            for idx, chunk in enumerate(chunks):
                pid = build_passage_id(source, doc_id, idx)
                fps = fingerprint_passage(chunk)
                records.append(PassageRecord(
                    passage_id=pid,
                    source=source,
                    source_doc_id=doc_id,
                    title=title if idx == 0 else f"{title} (chunk {idx})",
                    text=chunk,
                    metadata={"chunk_index": idx, "num_chunks": len(chunks)},
                    text_hash=fps["text_hash"],
                    normalized_text_hash=fps["normalized_text_hash"],
                    minhash=fps["minhash"],
                ))
                counts.add_accepted()

    return records, counts


def passages_to_dataframe(records: list[PassageRecord]) -> pd.DataFrame:
    """Convert a list of PassageRecords to a pandas DataFrame."""
    return pd.DataFrame([
        {
            "passage_id": r.passage_id,
            "source": r.source,
            "source_doc_id": r.source_doc_id,
            "split": r.split,
            "title": r.title,
            "text": r.text,
            "url": r.url,
            "license": r.license,
            "metadata": json.dumps(r.metadata),
            "text_hash": r.text_hash,
            "normalized_text_hash": r.normalized_text_hash,
            "minhash": r.minhash,
        }
        for r in records
    ])


def log_corpus_coverage(
    df_passages: pd.DataFrame,
    source: str,
) -> dict[str, Any]:
    """Report corpus statistics: passage count, doc count, avg length."""
    return {
        "source": source,
        "num_passages": int(len(df_passages)),
        "num_source_docs": int(df_passages["source_doc_id"].nunique()),
        "avg_text_length": float(df_passages["text"].str.len().mean()) if len(df_passages) > 0 else 0.0,
        "min_text_length": int(df_passages["text"].str.len().min()) if len(df_passages) > 0 else 0,
        "max_text_length": int(df_passages["text"].str.len().max()) if len(df_passages) > 0 else 0,
    }
