# Phase 1: Environment and Privacy

## Goal

Make the project runnable, reproducible, and safe for private biomedical
artifacts before any large-scale data generation.

## Inputs

- Python 3.11 or 3.12.
- `uv` dependency manager.
- Hugging Face token if private repos are used.
- Local or private-HF UMLS path.
- Optional GPU for model smoke tests.

## Outputs

- CLI skeleton.
- Config loader.
- Private/public path validator.
- Dependency smoke tests.
- Minimal model load smoke test.
- Data safety checks.

## Tasks

1. Add package modules under `src/medcolbert/`.
2. Add a Typer CLI entrypoint named `medcolbert`.
3. Implement config loading for YAML files.
4. Implement path helpers:
   - `private_output_path`
   - `public_output_path`
   - `assert_private_path`
   - `assert_no_restricted_public_fields`
5. Add structured logging with counts and rejection reasons.
6. Verify package install with all optional extras.
7. Verify BioClinical-ModernBERT loads with Transformers.
8. Verify PyLate can instantiate a ColBERT-style model.
9. Add a tiny forward-pass test that does not require private data.

## Suggested File Targets

```text
src/medcolbert/cli.py
src/medcolbert/utils/config.py
src/medcolbert/utils/io.py
src/medcolbert/utils/hashing.py
src/medcolbert/utils/logging.py
tests/test_config.py
tests/test_privacy_paths.py
tests/test_model_smoke.py
```

## Safety Requirements

- Nothing under `data/processed/private/` should be committed.
- No generated audit files containing UMLS strings should be written to public
  paths.
- Public manifests may include counts, hashes, source names, and opaque IDs, but
  not restricted vocabulary strings.

## Smoke Commands

```bash
uv sync --extra data --extra eval --extra generation --extra training --extra dev
uv run pytest
uv run ruff check .
```

Add a lightweight command:

```bash
medcolbert doctor
```

It should report:

- Python version
- installed optional groups if detectable
- CUDA availability
- configured private/public data paths
- whether UMLS files are discoverable
- whether model load smoke checks pass

## Acceptance Criteria

- `medcolbert doctor` runs.
- Tests pass without private UMLS data.
- Private path checks prevent accidental public writes of restricted fields.
- Model smoke test can be skipped cleanly if optional training dependencies are
  unavailable.

## Common Mistakes

- Do not import GPU-heavy libraries at module import time.
- Do not require UMLS for generic tests.
- Do not make configs mutate global state.
- Do not hide failed optional dependency imports. Return actionable messages.

