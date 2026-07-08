"""Training pipeline: data loading, loss, and trainer construction.

Three retrieval architectures share the same triplet dataset and the same
HF-style trainer surface: ``sbert_train`` (dense single-vector),
``sparse_train`` (SPLADE sparse), and ``pylate_train`` (ColBERT late
interaction). Each module assembles its model/loss/trainer from config dicts
so launch scripts stay thin.
"""

from medcolbert.training.datasets import (
    AUDIT_COLUMNS,
    FilterReport,
    drop_null_triplets,
    filter_by_audit,
    load_triplet_dataset,
    prepare_training_dataset,
    subset_dataset,
    to_pylate_columns,
    to_sbert_columns,
)

__all__ = [
    "AUDIT_COLUMNS",
    "FilterReport",
    "drop_null_triplets",
    "filter_by_audit",
    "load_triplet_dataset",
    "prepare_training_dataset",
    "subset_dataset",
    "to_pylate_columns",
    "to_sbert_columns",
]
