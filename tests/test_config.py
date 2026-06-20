"""Tests for config loading."""

import tempfile
from pathlib import Path

import pytest

from medcolbert.utils.config import load_yaml, load_configs


def test_load_yaml_simple():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("key: value\nnested:\n  sub: 42\n")
        tmp = Path(f.name)

    try:
        result = load_yaml(tmp)
        assert result == {"key": "value", "nested": {"sub": 42}}
    finally:
        tmp.unlink()


def test_load_yaml_not_found():
    with pytest.raises(FileNotFoundError):
        load_yaml("/nonexistent/path/config.yaml")


def test_load_configs_merge():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("a: 1\nb: {x: 10}\n")
        tmp1 = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("b: {y: 20}\nc: 3\n")
        tmp2 = Path(f.name)

    try:
        result = load_configs(tmp1, tmp2)
        assert result == {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
    finally:
        tmp1.unlink()
        tmp2.unlink()


def test_load_configs_empty():
    result = load_configs()
    assert result == {}
