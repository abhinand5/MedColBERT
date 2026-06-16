# Phase 8: Release and Paper

## Goal

Prepare a license-safe public release and a paper evidence package that makes
the research claim reproducible and reviewable.

## Inputs

- Final checkpoints.
- Training manifests.
- Evaluation outputs.
- Decontamination logs.
- Data provenance manifests.
- License notes for all sources.

## Public Release Artifacts

Release:

- source code
- configs
- evaluation harness
- public-safe manifests
- scripts to regenerate private UMLS-derived artifacts for licensed users
- toy data with no restricted vocabulary strings
- model card
- benchmark run files where licenses allow
- model weights only after review

Do not release:

- raw UMLS files
- UMLS-derived synonym tables
- CUI to restricted string mappings
- SNOMED hierarchy dumps
- private matched spans
- generated datasets that copy restricted vocabulary strings at scale

## Model Card Requirements

The model card must include:

- intended use
- out-of-scope use
- training data summary
- private data policy
- synthetic data generation summary
- benchmark results
- known limitations
- safety and clinical-use disclaimer
- license and provenance notes
- memorization and restricted-string audit summary

## Paper Evidence Package

Prepare:

- main benchmark table
- vocabulary-shift table
- ablation table
- efficiency table
- data quality audit table
- qualitative MaxSim alignment examples
- decontamination summary
- baseline settings appendix
- synthetic generation appendix
- release and licensing appendix

## Reviewer Questions To Preempt

Answer these explicitly:

- Why late interaction instead of dense pooling?
- Why synthetic supervision is not just prompt artifact learning?
- Why UMLS/SNOMED/RxNorm controls improve over generic synthetic data?
- How was train/test contamination prevented?
- Are restricted ontology strings redistributed?
- Does the model memorize restricted vocabulary content?
- Does performance improve on real benchmarks, not only synthetic slices?
- How expensive is indexing and querying?

## Release Audit

Before release:

1. Scan public files for private path leaks.
2. Scan public files for UMLS-derived string tables.
3. Verify `.gitignore` blocks private data and checkpoints if needed.
4. Verify model card provenance is complete.
5. Verify eval run files do not contain restricted strings.
6. Verify generated examples intended for release are public-safe.
7. Record review outcome in a release checklist.

## Acceptance Criteria

- Public release can be built from a clean checkout.
- Licensed users can reproduce private artifacts from scripts and configs.
- Non-licensed users can run tests and toy examples.
- Paper claims match available evidence.
- No restricted UMLS/SNOMED-derived string tables are public.

## Common Mistakes

- Do not release private audit files.
- Do not omit negative results or weak ablations from internal records.
- Do not claim clinical safety or diagnostic reliability.
- Do not hide synthetic data provenance.
- Do not release model weights before checking license and memorization risk.

