#!/usr/bin/env python3
"""Extract UMLS concept pairs, edges, and semantic types.

Thin CLI wrapper — core logic lives in src/medcolbert/data/ontology.py.

Usage:
    uv run python scripts/data/01_extract_umls_pairs.py --config configs/data.yaml
"""

from __future__ import annotations

from medcolbert.cli import app
from medcolbert.cli import extract_pairs  # noqa: F401 — registers command

if __name__ == "__main__":
    import sys

    # If called directly, simulate: medcolbert umls extract-pairs
    sys.argv = ["medcolbert", "umls", "extract-pairs"] + sys.argv[1:]
    app()
