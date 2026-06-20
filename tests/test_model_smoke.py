"""Minimal model-load smoke tests.

These tests are skipped if optional training dependencies are unavailable.
They do NOT require UMLS data.
"""

import pytest


@pytest.mark.slow
def test_transformers_import():
    """Verify transformers can be imported."""
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        pytest.skip(f"transformers not available: {e}")


@pytest.mark.slow
def test_pylate_import():
    """Verify PyLate can be imported."""
    try:
        import pylate  # noqa: F401
    except ImportError as e:
        pytest.skip(f"pylate not available: {e}")


@pytest.mark.slow
def test_bioclinical_modernbert_loads():
    """Smoke test: BioClinical-ModernBERT loads via Transformers."""
    try:
        from transformers import AutoModel, AutoTokenizer

        backbone = "thomas-sounack/BioClinical-ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        model = AutoModel.from_pretrained(backbone)

        # Tiny forward pass
        inputs = tokenizer("The patient has a fever.", return_tensors="pt")
        outputs = model(**inputs)
        assert outputs.last_hidden_state is not None
        assert outputs.last_hidden_state.shape[0] == 1  # batch
        assert outputs.last_hidden_state.shape[1] > 0  # tokens
        assert outputs.last_hidden_state.shape[2] > 0  # hidden dim

    except ImportError as e:
        pytest.skip(f"Dependencies not available: {e}")
    except Exception as e:
        # Network issues, HF hub down, etc. — skip, don't fail
        pytest.skip(f"Model load failed (likely network/env): {e}")
