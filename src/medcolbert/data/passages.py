"""Passage store builder: ingest, chunk, fingerprint, and annotate passages.

Orchestrates the pipeline: ingest corpora → chunk → fingerprint → annotate with CUIs.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from medcolbert.data.corpora import (
    ingest_pubmed_jsonl,
    ingest_beir_corpus,
    passages_to_dataframe,
    log_corpus_coverage,
    PassageRecord,
    chunk_text,
    build_passage_id,
    fingerprint_passage,
)
from medcolbert.utils.hashing import stable_hash, normalize_text_for_fingerprint
from medcolbert.utils.io import private_output_path, public_output_path
from medcolbert.utils.logging import Counts


def build_passage_store(
    corpus_config: dict | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Build the full passage store from configured corpora.

    Args:
        corpus_config: Corpus-specific configuration.
        config: Full data configuration.

    Returns:
        Summary dict with counts, coverage stats, and output paths.
    """
    cfg = config or {}
    passage_cfg = cfg.get("passages", {})
    max_tokens = passage_cfg.get("max_tokens", 384)
    overlap_tokens = passage_cfg.get("overlap_tokens", 64)
    min_chars = passage_cfg.get("min_chars", 80)
    sources_cfg = passage_cfg.get("sources", {})

    all_records: list[PassageRecord] = []
    all_counts = Counts()
    coverage_reports: list[dict] = []

    print("=== Phase 3: Public Passage Store ===\n")

    # ── PubMed / MedRAG ───────────────────────────────────────────────────
    pubmed_path = Path(sources_cfg.get("pubmed", "data/raw/pubmed"))
    if pubmed_path.exists():
        print(f"[1] Ingesting PubMed from {pubmed_path}...")
        records, counts = ingest_pubmed_jsonl(
            pubmed_path,
            source="pubmed",
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_chars=min_chars,
        )
        all_records.extend(records)
        all_counts.total += counts.total
        all_counts.accepted += counts.accepted
        for reason, n in counts.rejected.items():
            all_counts.rejected[reason] += n
        print(f"    {len(records):,} passages from PubMed ({counts.accepted:,} accepted, {counts.rejected_total:,} rejected)")
    else:
        print(f"[1] PubMed path not found: {pubmed_path} — skipping")

    # ── Download BeIR/PubMed from HuggingFace if not on disk ──────────────
    # Try to download MedRAG/pubmed from HuggingFace
    if not pubmed_path.exists():
        print("[1b] Attempting to download MedRAG/pubmed from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset("MedRAG/pubmed", split="train", streaming=True)
            # We'll process in streaming mode to avoid memory issues
            record_count = 0
            batch: list[dict] = []
            for i, example in enumerate(ds):
                doc_id = str(example.get("id", example.get("pmid", str(i))))
                title = example.get("title", "")
                content = example.get("content", example.get("abstract", ""))
                text = f"{title}\n{content}".strip() if title else content

                if not text or len(text) < min_chars:
                    all_counts.add_rejected(reason="too_short")
                    continue

                chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
                for idx, chunk in enumerate(chunks):
                    pid = build_passage_id("pubmed", doc_id, idx)
                    fps = fingerprint_passage(chunk)
                    all_records.append(PassageRecord(
                        passage_id=pid,
                        source="pubmed",
                        source_doc_id=doc_id,
                        title=title if idx == 0 else f"{title} (chunk {idx})",
                        text=chunk,
                        metadata={"chunk_index": idx, "num_chunks": len(chunks)},
                        text_hash=fps["text_hash"],
                        normalized_text_hash=fps["normalized_text_hash"],
                        minhash=fps["minhash"],
                    ))
                    all_counts.add_accepted()
                    record_count += 1

            print(f"    {record_count:,} passages from HuggingFace MedRAG/pubmed")
        except Exception as e:
            print(f"    Could not download from HuggingFace: {e}")

    # ── BeIR datasets ─────────────────────────────────────────────────────
    beir_sources = [
        ("trec_covid", "data/raw/trec"),
        ("nfcorpus", "data/raw/nfcorpus"),
        ("bioasq", "data/raw/bioasq"),
    ]
    for source_name, raw_path in beir_sources:
        raw_dir = Path(raw_path)
        if raw_dir.exists():
            print(f"\n[2] Ingesting {source_name} from {raw_dir}...")
            corpus_file = raw_dir / "corpus.jsonl"
            if corpus_file.exists():
                records, counts = ingest_beir_corpus(
                    corpus_file,
                    source=source_name,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    min_chars=min_chars,
                )
                all_records.extend(records)
                for reason, n in counts.rejected.items():
                    all_counts.rejected[reason] += n
                all_counts.total += counts.total
                all_counts.accepted += counts.accepted
                print(f"    {len(records):,} passages from {source_name}")
            else:
                print(f"    corpus.jsonl not found in {raw_dir}")

    # ── ClinicalTrials.gov ────────────────────────────────────────────────
    ct_path = Path(sources_cfg.get("clinical_trials", "data/raw/clinical_trials"))
    if ct_path.exists():
        print(f"\n[3] Ingesting ClinicalTrials.gov from {ct_path}...")
        # Try ir_datasets format
        try:
            import ir_datasets
            dataset = ir_datasets.load("clinicaltrials/2024")
            ct_count = 0
            for doc in dataset.docs_iter():
                doc_id = doc.doc_id
                text = f"{doc.title}\n{getattr(doc, 'summary', '')}\n{getattr(doc, 'detailed_description', '')}"
                if len(text) < min_chars:
                    continue
                chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
                for idx, chunk in enumerate(chunks):
                    pid = build_passage_id("clinical_trials", doc_id, idx)
                    fps = fingerprint_passage(chunk)
                    all_records.append(PassageRecord(
                        passage_id=pid,
                        source="clinical_trials",
                        source_doc_id=doc_id,
                        title=doc.title,
                        text=chunk,
                        metadata={"chunk_index": idx, "num_chunks": len(chunks)},
                        text_hash=fps["text_hash"],
                        normalized_text_hash=fps["normalized_text_hash"],
                        minhash=fps["minhash"],
                    ))
                    all_counts.add_accepted()
                    ct_count += 1
            print(f"    {ct_count:,} passages from ClinicalTrials.gov")
        except Exception as e:
            print(f"    Could not load ClinicalTrials.gov via ir_datasets: {e}")

    # ── Build DataFrame ───────────────────────────────────────────────────
    print(f"\n[4] Building passage DataFrame ({len(all_records):,} records)...")
    df = passages_to_dataframe(all_records)

    # ── Deduplicate by text hash ─────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["text_hash"])
    after = len(df)
    if before > after:
        print(f"    Removed {before - after:,} duplicate passages (by text hash)")

    # ── Write private outputs ────────────────────────────────────────────
    out_dir = private_output_path("passages")
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_dir / "passages.parquet", index=False)
    print(f"\n    Passages written to {out_dir / 'passages.parquet'}")

    # Fingerprints as separate file (for decontamination)
    df_fp = df[["passage_id", "source", "source_doc_id", "text_hash", "normalized_text_hash", "minhash"]].copy()
    df_fp.to_parquet(out_dir / "fingerprints.parquet", index=False)

    # ── Build coverage report ─────────────────────────────────────────────
    coverage = {}
    for source_name in df["source"].unique():
        source_df = df[df["source"] == source_name]
        coverage[source_name] = log_corpus_coverage(source_df, source_name)

    # ── Write public manifest ────────────────────────────────────────────
    manifest_dir = public_output_path("manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.0",
        "total_passages": int(len(df)),
        "total_documents": int(df["source_doc_id"].nunique()),
        "sources": list(df["source"].unique()),
        "coverage": coverage,
        "rejection_summary": all_counts.summary(),
        "output_hashes": {
            "passages": stable_hash(str(len(df))),
            "fingerprints": stable_hash(str(len(df_fp))),
        },
    }
    with open(manifest_dir / "corpora.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"    Public manifest written to {manifest_dir / 'corpora.json'}")

    summary = {
        "total_passages": len(df),
        "total_documents": int(df["source_doc_id"].nunique()),
        "coverage": coverage,
        "rejection_summary": all_counts.summary(),
        "output_files": {
            "passages": str(out_dir / "passages.parquet"),
            "fingerprints": str(out_dir / "fingerprints.parquet"),
            "manifest": str(manifest_dir / "corpora.json"),
        },
    }

    return summary


def annotate_passages_with_cuis(
    passages_path: Path,
    concept_strings_df: pd.DataFrame,
    max_concepts_per_passage: int = 50,
) -> pd.DataFrame:
    """Annotate passages with UMLS CUI matches (private).

    Uses exact lowercase matching against UMLS concept strings.
    This is fast but imprecise — it's designed for concept coverage analysis,
    not for final relevance annotation.

    Args:
        passages_path: Path to passages.parquet.
        concept_strings_df: DataFrame with columns [cui, normalized_string, sab, semantic_group].
        max_concepts_per_passage: Max CUIs to store per passage.

    Returns:
        DataFrame with [passage_id, cui, matched_span, sab, semantic_group].
    """
    df = pd.read_parquet(passages_path)

    # Build a lookup: normalized_string → list of (cui, sab, semantic_group, orig_string)
    from collections import defaultdict
    string_lookup: dict[str, list[dict]] = defaultdict(list)
    for _, row in concept_strings_df.iterrows():
        key = row["normalized_string"]
        string_lookup[key].append({
            "cui": row["cui"],
            "sab": row.get("sab", ""),
            "semantic_group": row.get("semantic_group", "other"),
            "string": row.get("string", key),
        })

    annotations: list[dict] = []

    for _, passage_row in df.iterrows():
        passage_text = passage_row["text"]
        passage_id = passage_row["passage_id"]
        passage_lower = passage_text.lower()

        matched_cuis: set[str] = set()
        for norm_str, entries in string_lookup.items():
            if len(norm_str) < 4:  # Skip very short strings (too many false positives)
                continue
            if norm_str in passage_lower:
                for entry in entries:
                    if entry["cui"] not in matched_cuis:
                        matched_cuis.add(entry["cui"])
                        annotations.append({
                            "passage_id": passage_id,
                            "cui": entry["cui"],
                            "matched_span": norm_str,
                            "sab": entry["sab"],
                            "semantic_group": entry["semantic_group"],
                            "match_method": "exact_lowercase",
                            "confidence": "low",
                        })
                        if len(matched_cuis) >= max_concepts_per_passage:
                            break
                if len(matched_cuis) >= max_concepts_per_passage:
                    break

    df_annotations = pd.DataFrame(annotations)
    return df_annotations
