# MedColBERT

Late-interaction retrieval models for medical and clinical information retrieval.

Trained with ontology-guided supervision — using UMLS cross-vocabulary synonyms and SNOMED CT hard negatives — on top of BioClinical-ModernBERT.

## Status

Under development.

## Quick start

```bash
uv sync
```

## Repo structure

| Directory | Purpose |
|---|---|
| `src/medcolbert/` | Core library (model, losses, tokenizer, indexer) |
| `scripts/` | Data pipeline, training, and evaluation scripts |
| `configs/` | Model and training configurations |
| `data/` | Raw and processed datasets (gitignored) |

## License

Apache 2.0
