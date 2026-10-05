# MedColBERT v1 — public benchmark evaluation (2026-10-05/06)

**Verdict:** the v1 models (trained July 2026) are **below BM25 on every public medical
retrieval task measured**. The in-house real-corpus gate (ColBERT-base nDCG@10 0.754,
dense-large 0.764) measured fit to our own generator, not retrieval quality. v1 is not
released.

## Harness

`scripts/eval/run_mteb_local.py` (commit `82d54d5`), mteb 2.12.30, run on the L40S
pod. Validated by reproducing `lightonai/GTE-ModernColBERT-v1` on NFCorpus: **0.3804**
vs **0.378** published on the MTEB leaderboard.

## Results (nDCG@10, test split)

| task | MedColBERT-base (150M, ColBERT) | MedEmbed-dense-large (396M, dense) | BM25 (`mteb/baseline-bm25s`) | GTE-ModernColBERT-v1 (public) |
|---|---|---|---|---|
| NFCorpus | 0.151 | 0.210 | 0.321 | 0.378 |
| SciFact | 0.398 | 0.432 | 0.687 | 0.763 |
| TREC-COVID | 0.272 | 0.551 | 0.623 | 0.848 |
| MedicalQARetrieval | 0.187 | 0.185 | 0.458 | — |

The remaining tasks (CUREv1, PublicHealthQA, R2MED ×8) did not complete: the
ColBERT-base process died silently while loading CUREv1 (244,600 documents, ~23 GB
RSS, exit code 0, no result file). ColBERT-large and dense-base were not reached.
Public reference numbers are from the `embeddings-benchmark/results` repository
(sparse checkout, 2026-10-05).

Raw result JSONs were on the L40S pod at
`/workspace/MedColBERT/runs/eval/mteb_local/<model>/` and are lost if that pod is
terminated without a backup; the numbers above are the record.

## Root causes (diagnosed, not separately ablated)

1. **No retrieval pretraining.** Every model was fine-tuned directly from the
   BioClinical-ModernBERT MLM checkpoint; the planned stage-1 warmup never ran. Strong
   small retrievers start from backbones contrastively pretrained on hundreds of
   millions of pairs.
2. **Small, narrow, dated training set.** ~54k synthetic queries over 66k passages from
   two PubMed baseline shards (`pubmed23n0003`/`0004`, mostly 1970s abstracts).
3. **Synthetic negatives.** ~88% of hard negatives are LLM-written passages, which
   permits a "LLM style = irrelevant" shortcut.
4. **In-distribution gate.** The real-corpus dev queries were produced by the same
   generator over the same corpus, so the gate could not detect 1–3.

## What changes

See [plans/v2-retrieval-plan.md](plans/v2-retrieval-plan.md).
