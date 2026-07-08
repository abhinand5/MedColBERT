"""Tests for the dense trainer config resolution (pure, no GPU/network)."""

from __future__ import annotations


from medcolbert.training.sbert_train import (
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
        "pooling": "mean",
        "outputs": {"save_steps": 10000, "eval_steps": 5000},
        "stage2_dense": {
            "learning_rate": 1e-5,
            "per_device_train_batch_size": 256,
            "gradient_accumulation_steps": 4,
            "bf16": True,
            "warmup_ratio": 0.03,
            "max_steps": 14000,
            "loss_type": "cached_mnrl",
            "mini_batch_size": 160,
            "pooling": "mean",
        },
    }


def test_resolve_dense_cached_forces_grad_accum_to_1():
    cfg = _merged_cfg()
    s = resolve_stage(cfg, "stage2_dense")
    assert s.learning_rate == 1e-5
    assert s.per_device_train_batch_size == 256
    # cached_mnrl + mini_batch_size => grad_accum forced to 1
    assert s.gradient_accumulation_steps == 1
    assert s.loss_type == "cached_mnrl"
    assert s.mini_batch_size == 160
    assert s.pooling == "mean"
    assert s.max_steps == 14000
    assert s.query_length == 64
    assert s.document_length == 256
    assert s.effective_batch == 256  # per_device * grad_accum(1)


def test_resolve_dense_plain_mnrl_keeps_grad_accum():
    cfg = _merged_cfg()
    cfg["stage2_dense"]["loss_type"] = "mnrl"
    cfg["stage2_dense"]["mini_batch_size"] = None
    s = resolve_stage(cfg, "stage2_dense")
    # plain mnrl keeps the configured accumulation
    assert s.gradient_accumulation_steps == 4
    assert s.effective_batch == 256 * 4


def test_resolve_dense_defaults():
    s = resolve_stage({}, "stage_does_not_exist")
    assert s.per_device_train_batch_size == 16
    assert s.loss_type == "cached_mnrl"  # default
    assert s.pooling == "mean"
    assert s.num_train_epochs == 1


def test_resolve_dense_pooling_falls_back_to_model_config():
    cfg = {"stage2_dense": {"per_device_train_batch_size": 8}, "pooling": "cls"}
    s = resolve_stage(cfg, "stage2_dense")
    assert s.pooling == "cls"


def test_build_training_args_dense_has_max_steps(tmp_path):
    cfg = _merged_cfg()
    s = resolve_stage(cfg, "stage2_dense")
    args = build_training_args(s, tmp_path / "out", "run-dense")
    assert args.max_steps == 14000
    assert args.per_device_train_batch_size == 256
    assert args.gradient_accumulation_steps == 1
    assert args.bf16 is True
    assert args.learning_rate == 1e-5
    assert args.dataloader_drop_last is True
    assert args.save_strategy == "steps"
    assert args.save_total_limit == 3


def test_build_training_args_dense_uses_epochs_when_no_max_steps(tmp_path):
    cfg = _merged_cfg()
    cfg["stage2_dense"]["max_steps"] = None
    cfg["stage2_dense"]["num_train_epochs"] = 2
    s = resolve_stage(cfg, "stage2_dense")
    args = build_training_args(s, tmp_path / "out", "run-smoke")
    assert args.max_steps == -1  # HF sentinel for "not set"
    assert args.num_train_epochs == 2


def test_save_strategy_epoch_dense():
    cfg = {"stage2_dense": {"save_strategy": "epoch", "save_total_limit": 5}}
    s = resolve_stage(cfg, "stage2_dense")
    assert s.save_strategy == "epoch"
    assert s.save_total_limit == 5
    args = build_training_args(s, "/tmp/out", "run-x")
    assert args.save_strategy == "epoch"
    assert args.save_total_limit == 5


def test_save_strategy_steps_dense_passes_save_steps():
    cfg = {"stage2_dense": {"save_strategy": "steps", "save_steps": 5000}}
    s = resolve_stage(cfg, "stage2_dense")
    args = build_training_args(s, "/tmp/out", "run-x")
    assert args.save_strategy == "steps"
    assert args.save_steps == 5000


def test_default_report_to_is_none_dense():
    s = resolve_stage({}, "stage_x")
    assert s.report_to == ["none"]


def test_enable_wandb_reporting_dense_flips_when_key_set(monkeypatch):
    s = resolve_stage({}, "stage_x")
    assert s.report_to == ["none"]
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    assert wandb_enabled() is True
    enable_wandb_reporting(s)
    assert s.report_to == ["wandb"]


def test_enable_wandb_reporting_dense_noop_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    s = resolve_stage({}, "stage_x")
    assert wandb_enabled() is False
    enable_wandb_reporting(s)
    assert s.report_to == ["none"]
