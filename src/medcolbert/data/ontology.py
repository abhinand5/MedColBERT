"""Ontology control layer: build concept pairs, edges, and vocab-shift pools.

Writes private parquet files and public-safe manifests.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from medcolbert.data.umls import (
    parse_mrconso,
    parse_mrsty,
    parse_mrrel,
    SOURCE_GROUPS,
    map_source_group,
)
from medcolbert.utils.hashing import stable_hash, dict_hash, file_content_hash
from medcolbert.utils.io import (
    private_output_path,
    public_output_path,
    assert_no_restricted_public_fields,
)
from medcolbert.utils.logging import Counts


def build_ontology(
    mrconso_path: Path,
    mrsty_path: Path,
    mrrel_path: Path,
    config: dict | None = None,
) -> dict[str, Any]:
    """Build the full ontology layer: concepts, strings, pairs, edges.

    Returns a summary dict suitable for gate verification.
    """
    cfg = config or {}
    counts = Counts()

    print("=== Phase 2: Ontology Control Layer ===\n")

    # ── Step 1: Parse MRCONSO → concept strings ──────────────────────────
    print("[1/4] Parsing MRCONSO.RRF...")
    concept_strings: list[dict] = []
    cui_to_strings: dict[str, list[dict]] = defaultdict(list)
    norm_string_to_cuis: dict[str, set[str]] = defaultdict(set)

    for record in parse_mrconso(mrconso_path, cfg):
        concept_strings.append(record)
        cui_to_strings[record["cui"]].append(record)
        norm_string_to_cuis[record["normalized_string"]].add(record["cui"])
        counts.add_accepted()

    print(f"  Parsed {len(concept_strings):,} concept strings "
          f"from {len(cui_to_strings):,} unique CUIs")

    df_strings = pd.DataFrame(concept_strings)

    # ── Step 2: Parse MRSTY → attach semantic types ───────────────────────
    print("\n[2/4] Parsing MRSTY.RRF...")
    cui_semantic = parse_mrsty(mrsty_path)
    print(f"  Semantic types for {len(cui_semantic):,} CUIs")

    # Attach semantic types to concept strings
    for record in concept_strings:
        stypes = cui_semantic.get(record["cui"], [])
        record["semantic_types"] = [s["tui"] for s in stypes]
        if stypes:
            # Use the first semantic group (most CUIs have one primary group)
            groups = {s["semantic_group"] for s in stypes}
            record["semantic_group"] = groups.pop() if len(groups) == 1 else (
                next(iter(groups)) if groups else "other"
            )

    # Rebuild dataframe with semantic info
    df_strings = pd.DataFrame(concept_strings)

    # ── Step 3: Parse MRREL → concept edges ──────────────────────────────
    print("\n[3/4] Parsing MRREL.RRF...")
    edges: list[dict] = []
    edge_count = Counts()
    for edge in parse_mrrel(mrrel_path, cfg):
        edges.append(edge)
        if edge["negative_use"] == "concept_near_negative":
            edge_count.add_accepted()
        else:
            edge_count.add_accepted()

    df_edges = pd.DataFrame(edges)
    print(f"  Parsed {len(edges):,} edges")
    print(f"  Hierarchical negatives: {edge_count.accepted:,}")

    # ── Step 4: Build vocabulary-shift pairs ──────────────────────────────
    print("\n[4/4] Building vocabulary-shift pair pools...")
    pairs = _build_vocab_shift_pairs(
        concept_strings, cui_to_strings, norm_string_to_cuis
    )
    df_pairs = pd.DataFrame(pairs)
    print(f"  Built {len(pairs):,} concept pairs across vocab shift types")

    # ── Write private parquet files ───────────────────────────────────────
    _write_private_outputs(df_strings, df_pairs, df_edges)

    # ── Write public manifests ────────────────────────────────────────────
    _write_public_manifests(df_strings, df_pairs, df_edges, concept_strings)

    # ── Build summary ─────────────────────────────────────────────────────
    semantic_group_counts = df_strings["semantic_group"].value_counts().to_dict()
    vocab_shift_counts = df_pairs["vocab_shift_type"].value_counts().to_dict()

    summary = {
        "total_concept_strings": len(concept_strings),
        "unique_cuis": len(cui_to_strings),
        "total_edges": len(edges),
        "total_pairs": len(pairs),
        "semantic_group_counts": {
            k: int(v) for k, v in semantic_group_counts.items()
        },
        "vocab_shift_type_counts": {
            k: int(v) for k, v in vocab_shift_counts.items()
        },
        "source_vocabularies": _source_vocab_counts(concept_strings),
        "files_written": {
            "concepts": str(private_output_path("ontology", "concepts.parquet")),
            "concept_pairs": str(private_output_path("ontology", "concept_pairs.parquet")),
            "concept_edges": str(private_output_path("ontology", "concept_edges.parquet")),
            "semantic_types": str(private_output_path("ontology", "semantic_types.parquet")),
        },
    }

    return summary


def _build_vocab_shift_pairs(
    concept_strings: list[dict],
    cui_to_strings: dict[str, list[dict]],
    norm_string_to_cuis: dict[str, set[str]],
) -> list[dict]:
    """Build vocabulary-shift concept pair pools.

    Shift types:
      - consumer_to_clinical: CHV → SNOMEDCT_US/ICD10CM/etc.
      - clinical_to_consumer: clinical → CHV
      - abbreviation_to_expanded: short abbrev → longer form (same CUI)
      - brand_to_generic: brand SAB → generic SAB
      - clinical_variant: clinical → clinical (different SAB)
    """
    pairs: list[dict] = []
    seen_pair_ids: set[str] = set()

    # Index strings by CUI and source group
    cui_sg_strings: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in concept_strings:
        cui_sg_strings[rec["cui"]][rec["source_group"]].append(rec)

    for cui, sg_map in cui_sg_strings.items():
        # Consumer → Clinical
        consumer_strings = sg_map.get("consumer", [])
        clinical_strings = sg_map.get("clinical", [])
        drug_strings = sg_map.get("drug", [])

        for left in consumer_strings:
            for right in clinical_strings:
                pair_id = stable_hash(left["string_hash"], right["string_hash"], "consumer_to_clinical")
                if pair_id not in seen_pair_ids:
                    seen_pair_ids.add(pair_id)
                    pairs.append({
                        "pair_id": pair_id,
                        "cui": cui,
                        "left_string_id": left["string_hash"],
                        "right_string_id": right["string_hash"],
                        "left_source_group": left["source_group"],
                        "right_source_group": right["source_group"],
                        "vocab_shift_type": "consumer_to_clinical",
                        "semantic_group": left.get("semantic_group", "other"),
                        "quality_flags": [],
                    })

        # Abbreviation → Expanded (same CUI, same SAB, different lengths)
        all_strings = []
        for group_strings in sg_map.values():
            all_strings.extend(group_strings)
        for i, left in enumerate(all_strings):
            if not left.get("is_abbreviation"):
                continue
            for right in all_strings[i:]:
                if right.get("is_abbreviation"):
                    continue
                if len(left["string"]) >= len(right["string"]):
                    continue
                pair_id = stable_hash(left["string_hash"], right["string_hash"], "abbreviation_to_expanded")
                if pair_id not in seen_pair_ids:
                    seen_pair_ids.add(pair_id)
                    pairs.append({
                        "pair_id": pair_id,
                        "cui": cui,
                        "left_string_id": left["string_hash"],
                        "right_string_id": right["string_hash"],
                        "left_source_group": left["source_group"],
                        "right_source_group": right["source_group"],
                        "vocab_shift_type": "abbreviation_to_expanded",
                        "semantic_group": left.get("semantic_group", "other"),
                        "quality_flags": [],
                    })

        # Drug: brand ↔ generic (within drug group)
        if len(drug_strings) >= 2:
            for i, left in enumerate(drug_strings):
                for right in drug_strings[i + 1:]:
                    if left["sab"] == right["sab"]:
                        continue
                    pair_id = stable_hash(left["string_hash"], right["string_hash"], "brand_to_generic")
                    if pair_id not in seen_pair_ids:
                        seen_pair_ids.add(pair_id)
                        pairs.append({
                            "pair_id": pair_id,
                            "cui": cui,
                            "left_string_id": left["string_hash"],
                            "right_string_id": right["string_hash"],
                            "left_source_group": left["source_group"],
                            "right_source_group": right["source_group"],
                            "vocab_shift_type": "brand_to_generic",
                            "semantic_group": left.get("semantic_group", "other"),
                            "quality_flags": [],
                        })

        # Clinical variant (different SAB within clinical)
        if len(clinical_strings) >= 2:
            for i, left in enumerate(clinical_strings):
                for right in clinical_strings[i + 1:]:
                    if left["sab"] == right["sab"]:
                        continue
                    pair_id = stable_hash(left["string_hash"], right["string_hash"], "clinical_variant")
                    if pair_id not in seen_pair_ids:
                        seen_pair_ids.add(pair_id)
                        pairs.append({
                            "pair_id": pair_id,
                            "cui": cui,
                            "left_string_id": left["string_hash"],
                            "right_string_id": right["string_hash"],
                            "left_source_group": left["source_group"],
                            "right_source_group": right["source_group"],
                            "vocab_shift_type": "clinical_variant",
                            "semantic_group": left.get("semantic_group", "other"),
                            "quality_flags": [],
                        })

    return pairs


def _source_vocab_counts(concept_strings: list[dict]) -> dict[str, int]:
    """Count strings by source vocabulary."""
    sab_counts: Counter[str] = Counter()
    for rec in concept_strings:
        sab_counts[rec["sab"]] += 1
    return dict(sab_counts.most_common(50))


def _write_private_outputs(
    df_strings: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_edges: pd.DataFrame,
) -> None:
    """Write all private parquet outputs."""
    out_dir = private_output_path("ontology")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_strings.to_parquet(out_dir / "concept_strings.parquet", index=False)
    df_strings.to_parquet(out_dir / "concepts.parquet", index=False)

    # Semantic types deduplicated
    semantic_records: list[dict] = []
    for _, row in df_strings.iterrows():
        for stype in row.get("semantic_types", []):
            semantic_records.append({
                "cui": row["cui"],
                "tui": stype,
                "semantic_group": row.get("semantic_group", "other"),
            })
    df_semantic = pd.DataFrame(semantic_records).drop_duplicates()
    df_semantic.to_parquet(out_dir / "semantic_types.parquet", index=False)

    df_pairs.to_parquet(out_dir / "concept_pairs.parquet", index=False)
    df_edges.to_parquet(out_dir / "concept_edges.parquet", index=False)

    print(f"\n  Private outputs written to {out_dir}/")


def _write_public_manifests(
    df_strings: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_edges: pd.DataFrame,
    concept_strings: list[dict],
) -> None:
    """Write public-safe manifests (no restricted strings)."""
    out_dir = public_output_path("manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count manifest
    counts = {
        "schema_version": "1.0",
        "total_concept_strings": len(df_strings),
        "unique_cuis": int(df_strings["cui"].nunique()),
        "total_concept_pairs": len(df_pairs),
        "total_edges": len(df_edges),
        "semantic_group_counts": {
            str(k): int(v)
            for k, v in df_strings["semantic_group"].value_counts().to_dict().items()
        },
        "vocab_shift_type_counts": {
            str(k): int(v)
            for k, v in df_pairs["vocab_shift_type"].value_counts().to_dict().items()
        },
        "source_vocabulary_counts": _source_vocab_counts(concept_strings),
        "edge_relation_counts": {
            str(k): int(v)
            for k, v in df_edges["relation"].value_counts().to_dict().items()
        },
    }

    # Verify no restricted fields
    assert_no_restricted_public_fields(counts)

    with open(out_dir / "ontology_counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2, ensure_ascii=False)

    # Hash manifest
    hashes = {
        "schema_version": "1.0",
        "concept_strings_hash": stable_hash(str(len(df_strings))),
        "concept_pairs_hash": stable_hash(str(len(df_pairs))),
        "concept_edges_hash": stable_hash(str(len(df_edges))),
    }

    with open(out_dir / "ontology_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)

    print(f"  Public manifests written to {out_dir}/")


def run_ontology_gate(summary: dict) -> bool:
    """Verify ontology gate checks.

    Returns True if the phase gate passes.
    """
    print("\n=== Ontology Gate Checks ===\n")

    checks: list[tuple[str, bool, str]] = []

    # CUI count check
    cuis = summary.get("unique_cuis", 0)
    checks.append((
        "CUIs load",
        cuis > 100_000,
        f"{cuis:,} CUIs loaded (threshold: 100,000)",
    ))

    # Concept strings check
    strings = summary.get("total_concept_strings", 0)
    checks.append((
        "Concept strings > 1M",
        strings > 1_000_000,
        f"{strings:,} strings (threshold: 1,000,000)",
    ))

    # Semantic groups check
    sg_counts = summary.get("semantic_group_counts", {})
    required_groups = {
        "disorders", "procedures", "drugs_chemicals",
        "anatomy", "findings_signs_symptoms",
    }
    present_groups = set(sg_counts.keys())
    groups_ok = required_groups.issubset(present_groups)
    checks.append((
        "Semantic groups present",
        groups_ok,
        f"Found: {sorted(present_groups)}, Required: {sorted(required_groups)}",
    ))

    # Pairs check
    pairs = summary.get("total_pairs", 0)
    checks.append((
        "Concept pairs > 100K",
        pairs > 100_000,
        f"{pairs:,} pairs (threshold: 100,000)",
    ))

    # Vocab shift types check
    vs_counts = summary.get("vocab_shift_type_counts", {})
    expected_types = {"consumer_to_clinical", "abbreviation_to_expanded", "clinical_variant"}
    shift_ok = expected_types.issubset(set(vs_counts.keys()))
    checks.append((
        "Vocab shift types present",
        shift_ok,
        f"Found: {sorted(vs_counts.keys())}, Required: {sorted(expected_types)}",
    ))

    # Edge check
    edges = summary.get("total_edges", 0)
    checks.append((
        "Edges > 100K",
        edges > 100_000,
        f"{edges:,} edges (threshold: 100,000)",
    ))

    all_passed = True
    for name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {detail}")
        if not passed:
            all_passed = False

    print(f"\n  Gate {'PASSED' if all_passed else 'FAILED'}")
    return all_passed
