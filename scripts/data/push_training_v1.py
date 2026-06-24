"""Build + push the standalone MedColBERT training dataset to the HF Hub.

Produces fierysurf/medcolbert-training-v1 with two configs + a dev split:

  config 'triplets'  (split 'train', ~53,703 rows): one row per positive, with its
    hard negatives joined as a list column. Directly trainable (ColBERT triplet form).
  config 'long'      (split 'train', ~711k rows): one row per kept negative
    (negatives_v1_clean.parquet long format) — flexible for re-collation.
  config 'dev'       (split 'test', 5,967 rows): the held-out 10% real-eval split,
    UNTOUCHED by negative generation (query + positive only; for post-training
    real-corpus retrieval eval).

Sources:
  positives : mode4_teacher_filtered_multistyle_cleaned_v3g.parquet
  negatives : data/processed/private/synthetic/negatives/negatives_v1_clean.parquet
  dev       : data/processed/private/synthetic/negatives/triplets_dev_candidates.parquet

Usage:
  HF_TOKEN=... uv run python scripts/data/push_training_v1.py --repo fierysurf/medcolbert-training-v1
  # --no-push to build locally only (write parquet to /tmp) for inspection
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, get_dataset_config_info
from huggingface_hub import HfApi

ROOT = Path("/workspace/MedColBERT")
V3G = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v3g.parquet"
NEG_DIR = ROOT / "data/processed/private/synthetic/negatives"
NEG_CLEAN = NEG_DIR / "negatives_v1_clean.parquet"
DEV = NEG_DIR / "triplets_dev_candidates.parquet"

# which positive-side columns to carry into the triplet config (the rest are audit/generation
# provenance not needed for training)
POS_COLS = ["example_id", "passage_id", "passage_text", "query", "query_style", "task_family",
            "cui_label", "alt_term", "semantic_group", "vocab_shift_type"]
# which negative-side columns to keep in the negatives list (drop the per-negative audit ids
# already present at the positive level; keep audit scores as they're useful per-negative signal)
NEG_COLS = ["negative_passage", "negative_source", "failure_mode", "why_wrong",
            "audit_relevant_looks", "audit_grounding", "weak"]


def build_triplets(positives: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    """One row per positive, negatives aggregated as a list of dicts."""
    pos = positives[POS_COLS].rename(columns={"example_id": "query_id",
                                              "passage_id": "positive_passage_id",
                                              "passage_text": "positive_passage"})
    # group negatives by query_id into list-of-dicts
    neg = negatives[NEG_COLS + ["query_id"]].copy()
    # ensure plain python types for the list column (parquet/arrow round-trips lists of structs)
    neg_lists = (neg.groupby("query_id", sort=False)
                   .apply(lambda g: g[NEG_COLS].to_dict("records"), include_groups=False))
    trip = pos.copy()
    trip["negatives"] = trip["query_id"].map(neg_lists)
    trip["negatives"] = trip["negatives"].apply(lambda v: v if isinstance(v, list) else [])
    trip["n_negatives"] = trip["negatives"].apply(len)
    return trip


def build_long(negatives: pd.DataFrame, positives: pd.DataFrame) -> pd.DataFrame:
    """One row per kept negative, with the positive's text joined in. The negatives table
    already carries query_id/query/query_style/task_family/positive_passage_id/semantic_group,
    so we only add positive_passage (the positive's text, not stored per-negative)."""
    pos = positives[["example_id", "passage_text"]].rename(
        columns={"example_id": "query_id", "passage_text": "positive_passage"})
    return negatives.merge(pos, on="query_id", how="left")


def build_dev(dev_df: pd.DataFrame) -> pd.DataFrame:
    """Dev real-eval split: query + positive only (no negatives, untouched)."""
    return dev_df[["query", "query_style", "task_family", "positive_passage_id",
                   "positive_passage_text", "positive_in_corpus",
                   "bm25_recall_at_10", "bm25_recall_at_100"]].rename(
        columns={"positive_passage_text": "positive_passage"})


def write_local(build_dir: Path, triplets, long, dev):
    build_dir.mkdir(parents=True, exist_ok=True)
    triplets.to_parquet(build_dir / "triplets_train.parquet", index=False)
    long.to_parquet(build_dir / "long_train.parquet", index=False)
    dev.to_parquet(build_dir / "dev_test.parquet", index=False)
    # a README manifest
    (build_dir / "manifest.json").write_text(json.dumps({
        "triplets_train": {"rows": len(triplets), "cols": list(triplets.columns)},
        "long_train": {"rows": len(long), "cols": list(long.columns)},
        "dev_test": {"rows": len(dev), "cols": list(dev.columns)},
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="fierysurf/medcolbert-training-v1")
    ap.add_argument("--no-push", action="store_true", help="build locally only, don't upload")
    ap.add_argument("--build-dir", default="/tmp/medcolbert_training_v1")
    args = ap.parse_args()

    print("loading positives + clean negatives + dev...")
    positives_all = pd.read_parquet(V3G)
    positives_all = positives_all[positives_all["drop_reason"].isna()].copy()
    negatives = pd.read_parquet(NEG_CLEAN)
    dev_df = pd.read_parquet(DEV)
    # The triplet config is TRAIN positives only: those that actually have negatives
    # (negatives cover the 53,703 train split; the 5,967 dev positives have none).
    train_ids = set(negatives["query_id"].unique())
    positives = positives_all[positives_all["example_id"].isin(train_ids)].copy()
    print(f"  positives(all)={len(positives_all):,}  train(with negatives)={len(positives):,}  "
          f"negatives(clean)={len(negatives):,}  dev={len(dev_df):,}")

    print("building configs...")
    triplets = build_triplets(positives, negatives)
    long = build_long(negatives, positives)
    dev = build_dev(dev_df)
    print(f"  triplets: {len(triplets):,} rows (neg/pos mean {triplets.n_negatives.mean():.2f}, "
          f"min {triplets.n_negatives.min()}, max {triplets.n_negatives.max()})")
    print(f"  long:     {len(long):,} rows")
    print(f"  dev:      {len(dev):,} rows")

    build_dir = Path(args.build_dir)
    write_local(build_dir, triplets, long, dev)
    print(f"wrote local build -> {build_dir}")

    if args.no_push:
        print("--no-push: stopping after local build."); return

    import os
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    who = api.whoami().get("name")
    print(f"HF whoami: {who}")

    # build three DatasetDicts, one per config, push each under its config name
    print(f"pushing {args.repo} ...")

    print("  config 'triplets' ...")
    dd_trip = DatasetDict({"train": Dataset.from_pandas(triplets, preserve_index=False)})
    dd_trip.push_to_hub(args.repo, config_name="triplets", token=token, private=True)

    print("  config 'long' ...")
    dd_long = DatasetDict({"train": Dataset.from_pandas(long, preserve_index=False)})
    dd_long.push_to_hub(args.repo, config_name="long", token=token, private=True)

    print("  config 'dev' ...")
    dd_dev = DatasetDict({"test": Dataset.from_pandas(dev, preserve_index=False)})
    dd_dev.push_to_hub(args.repo, config_name="dev", token=token, private=True)

    # README with config descriptions
    readme = f"""---
configs:
- config_name: triplets
  data_files:
  - split: train
    path: triplets/train-*
- config_name: long
  data_files:
  - split: train
    path: long/train-*
- config_name: dev
  data_files:
  - split: test
    path: dev/test-*
language:
- en
tags:
- medical
- retrieval
- colbert
- hard-negatives
size_categories:
- 100K<n<1M
---

# MedColBERT Training v1

Hard negatives for contrastive ColBERT training, mined from {len(positives):,} real-PubMed
(query, positive) positives. See the project repo `docs/audit_negatives_2026-06-24.md` for the
full QA report.

## Configs

- **triplets** (train, {len(triplets):,} rows): one row per positive with its hard negatives
  as a list column `negatives`. Negatives are mined by Gemma-4-31B (synthetic, taxonomy-routed)
  + BM25 over a 66k real-PubMed corpus (real anchors). Audit-gate verified: 0 false negatives
  in a 299-sample independent QA (Wilson 95% upper CI 1.27%). Use this for training.
- **long** (train, {len(long):,} rows): one row per kept negative (the `negatives_v1_clean`
  long format), with positive columns joined. Use for custom re-collation.
- **dev** (test, {len(dev):,} rows): held-out 10% real-eval split, UNTOUCHED by negative
  generation (query + positive only). For post-training real-corpus retrieval eval (MRR/Recall).

## Negative fields (in `triplets.negatives[]` / `long` rows)

- `negative_passage`: the hard-negative passage text
- `failure_mode`: one of confusable_disease, wrong_drug_same_class, wrong_stage_severity,
  wrong_dose_route, lay_clinical_mismatch, abbreviation_trap, biomedical_clinical_mismatch,
  wrong_context, treatment_diagnosis_confound, domain_boundary, same_group_wrong_concept_synth,
  lexical_trap(_real)
- `negative_source`: synthetic | bm25_real
- `audit_relevant_looks` (1-5), `audit_grounding` (1-5), `weak` (bool): per-negative audit scores
"""
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset", token=token)
    print(f"\nDONE -> https://huggingface.co/datasets/{args.repo}")
    print("verify configs:")
    for cfg in ["triplets", "long", "dev"]:
        try:
            info = get_dataset_config_info(args.repo, config_name=cfg, token=token)
            print(f"  {cfg}: {info.splits}")
        except Exception as e:
            print(f"  {cfg}: verify-failed {e}")


if __name__ == "__main__":
    main()
