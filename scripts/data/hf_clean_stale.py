#!/usr/bin/env python
"""Delete stale parquet/data files from the HF dataset repo so the fresh
consolidation is a true REPLACE (not a merge with the old 109K temp=0.0 corpus).

Run AFTER uploading the fresh mode4_*.parquet files. Only deletes files that are
NOT part of the fresh set. Non-parquet metadata (README, configs, .gitattributes)
is left untouched.
"""
import os
from huggingface_hub import HfApi

REPO = "fierysurf/medcolbert-synthetic-pilot"
TOKEN = os.environ.get("HF_TOKEN", "").strip()

# Fresh files that the current consolidation produces + keeps.
KEEP = {
    ".gitattributes",
    "README.md",
    "data/mode4_teacher_filtered_multistyle.parquet",
    "data/mode4_full_question.parquet",
    "data/mode4_keyword_clinical.parquet",
    "data/mode4_keyword_layperson.parquet",
    "data/mode4_keyword_technical.parquet",
    "data/consolidation_stats.json",
    "data/checkpoint.json",
    "stats/consolidation_stats.json",
    "checkpoint.json",
}

api = HfApi(token=TOKEN)
files = api.list_repo_files(REPO, repo_type="dataset")
print(f"Repo has {len(files)} files")

# Delete any parquet or stale json not in KEEP.
stale = [
    f for f in files
    if f not in KEEP and (f.endswith(".parquet") or f in {
        "synthetic_pilot.json", "dataset_stats.json", "ontology_counts.json",
        "go_decision.json", "corpora.json", "data/generation.yaml", "data/data.yaml",
        "data/mode3_ontology_grounded.parquet",
    })
]
# Also catch any other data/*.parquet not in KEEP.
stale += [f for f in files if f.startswith("data/") and f.endswith(".parquet") and f not in KEEP]
stale = sorted(set(stale))
print(f"Deleting {len(stale)} stale files:")
for f in stale:
    print(f"  - {f}")
    api.delete_file(f, REPO, repo_type="dataset", token=TOKEN)
print("Done.")
