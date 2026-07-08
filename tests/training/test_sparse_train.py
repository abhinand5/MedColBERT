"""Tests for the sparse trainer config resolution (pure, no GPU/network)."""

from __future__ import annotations


from sentence_transformers.training_args import BatchSamplers

from medcolbert.training.sparse_train import (
    build_training_args,
    enable_wandb_reporting,
    resolve_stage,
    wandb_enabled,
)


def _merged_cfg() -> dict:
    return {
        "model": "thomas-sounack/BioClinical-ModernBERT-base",
        "query_maxlen": 64,
        "doc_maxlen": 256,
        "outputs": {"save_steps": 10000, "eval_steps": 5000},
        "stage2_sparse": {
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 64,
            "gradient_accumulation_steps": 4,
            "bf16": True,
            "warmup_ratio": 0.1,
            "max_steps": 14000,
            "loss_type": "splade_mnr",
            "query_regularizer_weight": 5e-5,
            "document_regularizer_weight": 3e-5,
            "batch_sampler": "no_duplicates",
        },
    }


def test_resolve_sparse_defaults_splade():
    cfg = _merged_cfg()
    s = resolve_stage(cfg, "stage2_sparse")
    assert s.learning_rate == 2e-5
    assert s.per_device_train_batch_size == 64
    assert s.gradient_accumulation_steps == 4  # NOT forced (no gradcache)
    assert s.loss_type == "splade_mnr"
    assert s.query_regularizer_weight == 5e-5
    assert s.document_regularizer_weight == 3e-5
    assert s.batch_sampler == "no_duplicates"
    assert s.max_steps == 14000
    assert s.query_length == 64
    assert s.document_length == 256
    assert s.effective_batch == 64 * 4


def test_resolve_sparse_plain_mnr():
    cfg = _merged_cfg()
    cfg["stage2_sparse"]["loss_type"] = "sparse_mnrl"
    s = resolve_stage(cfg, "stage2_sparse")
    assert s.loss_type == "sparse_mnrl"


def test_resolve_sparse_defaults():
    s = resolve_stage({}, "stage_does_not_exist")
    assert s.per_device_train_batch_size == 16
    assert s.loss_type == "splade_mnr"  # default
    assert s.batch_sampler == "no_duplicates"
    assert s.num_train_epochs == 1


def test_build_training_args_sparse(tmp_path):
    cfg = _merged_cfg()
    s = resolve_stage(cfg, "stage2_sparse")
    args = build_training_args(s, tmp_path / "out", "run-sparse")
    assert args.max_steps == 14000
    assert args.per_device_train_batch_size == 64
    assert args.gradient_accumulation_steps == 4
    assert args.bf16 is True
    assert args.learning_rate == 2e-5
    assert args.dataloader_drop_last is True
    assert args.batch_sampler == BatchSamplers.NO_DUPLICATES


def test_build_training_args_sparse_batch_sampler_override(tmp_path):
    cfg = _merged_cfg()
    cfg["stage2_sparse"]["batch_sampler"] = "batch_sampler"
    s = resolve_stage(cfg, "stage2_sparse")
    args = build_training_args(s, tmp_path / "out", "run-x")
    assert args.batch_sampler == BatchSamplers.BATCH_SAMPLER


def test_build_training_args_sparse_uses_epochs_when_no_max_steps(tmp_path):
    cfg = _merged_cfg()
    cfg["stage2_sparse"]["max_steps"] = None
    cfg["stage2_sparse"]["num_train_epochs"] = 2
    s = resolve_stage(cfg, "stage2_sparse")
    args = build_training_args(s, tmp_path / "out", "run-smoke")
    assert args.max_steps == -1  # HF sentinel for "not set"
    assert args.num_train_epochs == 2


def test_save_strategy_epoch_sparse():
    cfg = {"stage2_sparse": {"save_strategy": "epoch", "save_total_limit": 5}}
    s = resolve_stage(cfg, "stage2_sparse")
    args = build_training_args(s, "/tmp/out", "run-x")
    assert args.save_strategy == "epoch"
    assert args.save_total_limit == 5


def test_default_report_to_is_none_sparse():
    s = resolve_stage({}, "stage_x")
    assert s.report_to == ["none"]


def test_enable_wandb_reporting_sparse_flips_when_key_set(monkeypatch):
    s = resolve_stage({}, "stage_x")
    assert s.report_to == ["none"]
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    assert wandb_enabled() is True
    enable_wandb_reporting(s)
    assert s.report_to == ["wandb"]


def test_enable_wandb_reporting_sparse_noop_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    s = resolve_stage({}, "stage_x")
    assert wandb_enabled() is False
    enable_wandb_reporting(s)
    assert s.report_to == ["none"]
