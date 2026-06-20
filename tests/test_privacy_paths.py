"""Tests for privacy path enforcement."""

import tempfile
from pathlib import Path

import pytest

from medcolbert.utils.io import (
    private_dir,
    public_dir,
    private_output_path,
    public_output_path,
    assert_private_path,
    assert_no_restricted_public_fields,
    is_public_safe,
    ALLOWED_PUBLIC_FIELDS,
    RESTRICTED_FIELDS,
)


def test_private_dir_is_under_data_processed():
    d = private_dir()
    assert "data" in d.parts
    assert "private" in d.parts


def test_public_dir_is_under_data_processed():
    d = public_dir()
    assert "data" in d.parts
    assert "public" in d.parts


def test_private_output_path_nesting():
    p = private_output_path("ontology", "concepts.parquet")
    assert p.name == "concepts.parquet"
    assert "private" in str(p)
    assert "ontology" in str(p)


def test_public_output_path_nesting():
    p = public_output_path("manifests", "counts.json")
    assert p.name == "counts.json"
    assert "public" in str(p)
    assert "manifests" in str(p)


def test_assert_private_path_passes():
    p = private_output_path("test.parquet")
    assert_private_path(p)  # should not raise


def test_assert_private_path_fails_outside():
    with pytest.raises(ValueError):
        assert_private_path(Path("/tmp/some_file.txt"))


def test_assert_no_restricted_public_fields_allows_safe():
    record = {"cui": "C123", "count": 5, "hash": "abc123"}
    assert_no_restricted_public_fields(record)  # should not raise


def test_assert_no_restricted_public_fields_blocks_text():
    with pytest.raises(ValueError):
        assert_no_restricted_public_fields({"cui": "C123", "string": "Aspirin"})


def test_assert_no_restricted_public_fields_blocks_query():
    with pytest.raises(ValueError):
        assert_no_restricted_public_fields({"query": "heart attack treatment", "cui": "C123"})


def test_is_public_safe():
    assert is_public_safe({"cui": "C123", "hash": "abc"})
    assert not is_public_safe({"cui": "C123", "text": "patient has fever"})


def test_allowed_fields_excludes_restricted():
    common = ALLOWED_PUBLIC_FIELDS & RESTRICTED_FIELDS
    assert common == set(), f"Overlap between allowed and restricted: {common}"
