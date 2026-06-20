# DS_GOAL: MedColBERT End-to-End Autonomous Build

Complete goal specification for Claude Code. ONE autonomous run from the current
scaffolded state to a production-ready synthetic dataset. Claude controls
everything — infrastructure build, generation loop, quality gating.

Gemma 4 31B (via vLLM at localhost:8000/v1) is the executor/LLM.
MedColBERT is the next iteration of MedEmbed — a late-interaction biomedical
retrieval model trained with ontology-guided supervision on top of
BioClinical-ModernBERT.

## End-to-End Scope

This goal encompasses ALL phases from current state through dataset delivery:

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 (generation loop) ──→ DONE
  │            │            │              │
  env setup    ontology     passage      calibration → batch loop
  privacy      layer        store +      → quality gates →
  paths        built        CUI annots   exit conditions
```

NOT in scope: Phase 5-8 (triplets, training, eval, release).
Stop after Phase 4 produces accepted dataset.

The generation loop is the final stage, not a separate run. Claude builds what's
missing, verifies it, then enters the loop. The Prerequisites section below
describes what gets BUILT first, not what must exist before starting.

MedColBERT is the next iteration of MedEmbed — a late-interaction biomedical
retrieval model trained with ontology-guided supervision on top of
BioClinical-ModernBERT.

## Session Rules

These rules govern ALL work in this session. Claude and all subagents must follow
them. Violations invalidate results.

### Python and Dependencies

1. **Always use `uv` for everything.** Never use bare `pip`, `python`, or `pytest`.
   - Install: `uv sync` (or `uv sync --extra data --extra eval --extra generation --extra training --extra dev`)
   - Run scripts: `uv run python scripts/...`
   - Run tests: `uv run pytest`
   - Run CLI: `uv run medcolbert ...`
   - Add deps: `uv add <package>`
2. **Never install anything outside the project venv.** The venv is at `.venv/`
   and managed by uv. No global pip installs, no conda, no system packages.
3. **Python version is 3.11 or 3.12** (pinned in `.python-version`). Do not use
   3.13+ features.

### Data Boundaries

4. **Private data lives in `data/processed/private/`.** This includes:
   - UMLS-derived strings, CUI mappings, synonym tables, concept annotations
   - Matched spans containing UMLS source strings
   - SNOMED hierarchy dumps
   - Any file containing restricted vocabulary content at scale
   - Generated synthetic queries that embed UMLS terms (until de-identified)
   - Audit files containing restricted terms
5. **Public-safe data lives in `data/processed/public/`.** This includes:
   - Code, configs, manifests with opaque IDs and hashes
   - Counts, source names, and provenance summaries
   - Reproduction scripts (requiring user-provided UMLS)
   - Model cards
6. **When unsure if an artifact is public-safe, keep it private.**
7. **Never commit private data to git.** Verify `.gitignore` covers private paths.

### Code Quality

8. **Logic lives in `src/medcolbert/`, scripts are thin wrappers.** Scripts in
   `scripts/` call library functions; they don't contain core logic.
9. **Use typed dataclasses or Pydantic schemas** for records crossing module
   boundaries.
10. **Use `pathlib.Path`** for all filesystem paths — never hardcoded strings.
11. **Use deterministic IDs and stable hashes** for generated artifacts.
12. **Write small pure functions** for parsing, normalization, filtering, hashing.
13. **Never silently drop examples.** Return counts and rejection reasons.
14. **Avoid custom retrieval math** until PyLate/ColBERT tools are proven inadequate.

### Provenance and Reproducibility

15. **Every training example must carry metadata:**
    `example_id, source, source_doc_id, positive_passage_id, generation_mode,
    target_cuis, semantic_groups, vocab_shift_type, negative_type,
    generator_model, prompt_hash, validator_version, teacher_model,
    teacher_margin, data_hash`
16. **Every experiment must record:**
    git commit, config files + hashes, model backbone + revision, training data
    manifest + hashes, dependency lockfile, random seed, hardware summary,
    wall-clock runtime.
17. **Config files are the source of truth for parameters.** Hardcoded values in
    scripts must be extractable to config.

### Research Integrity

18. **Never change the research claim.** The thesis is: ontology-controlled
    synthetic supervision improves retrieval under medical vocabulary shift.
19. **Never add or remove required baselines** without explicit approval.
    Mandatory: BM25, MedCPT or equivalent, generic ColBERT/PyLate,
    BioClinical-ColBERT without ontology supervision, MedColBERT.
20. **Never treat synthetic evals as primary evidence.**
21. **Never skip decontamination.** All training data must be fingerprinted and
    checked against eval splits.
22. **Do not let a single corpus dominate** all generated examples.
23. **Do not filter only by lexical overlap.** Check answerability and concept intent.

### Agent Roles

24. **Strong LLMs (Claude, Gemma 4 31B) are co-researchers.** They may critique
    hypotheses, compare data sources, propose ablations, judge examples, find
    false negatives, review evaluation tables, red-team claims.
25. **Small LLMs are executors.** They implement bounded tasks, write tests,
    refactor, run commands, summarize logs. They must not change research claims,
    add/remove baselines, decide noisy data is acceptable, weaken privacy rules,
    or skip decontamination.
26. **When a decision affects the research claim, data mixture, filtering rubric,
    baseline set, or release policy:** create an evidence packet and use the
    LLM Co-Researcher Protocol before proceeding.

### This Session

27. **We are building MedColBERT end-to-end with AI agents.** Claude is the lead
    co-researcher driving the loop. Subagents execute bounded tasks.
28. **Phase order is fixed.** Execute phases 0-8 in order per GOALS.md. Each
    phase has a go/no-go gate — do not proceed until the gate passes.
29. **The `docs/goals/` folder contains the step-by-step plan.** Refer to it
    before implementing any phase task.
30. **`.claude/custom/INITIAL_PLAN.md` and `EXECUTION_ROADMAP.md`** contain
    high-level architectural plans — reference them for context, not as
    step-by-step instructions.
31. **All MCP tools are available** — Context7 for library docs, Grep for GitHub
    code search, HuggingFace CLI for datasets/models.

## Environment

- **Generator + Judge LLM:** Gemma 4 31B via vLLM
- **vLLM endpoint:** `http://localhost:8000/v1`
- **API format:** OpenAI-compatible `/v1/chat/completions`
- **Controller:** Claude Code (this session) — does NOT generate, only judges samples and makes go/no-go decisions
- **Config:** `configs/generation.yaml`
- **Training mix config:** `configs/data.yaml`

## Goal

Produce a high-quality synthetic dataset of **ontology-grounded, teacher-filtered**
query-passage pairs that exhibit controlled medical vocabulary shift. The dataset
must pass a stratified audit at ≥80% acceptance before scaling to production.

### Scale Targets

| Stage | Count | Purpose |
|---|---|---|
| Pilot calibration | 50-100 examples | Align judge prompts, verify pipeline |
| Batch iteration | 500-1000 per batch | Quality loop until thresholds met |
| Pilot acceptance | 1,000 accepted examples | Demonstrate quality across all task families |
| Production shard | 100,000 per shard | Scale generation |
| v1 target | 2,900,000 accepted examples | Full training dataset |

## Build Order (Infrastructure — Phases 1-3)

Claude builds these in order. Each item is a task to complete, not a precondition.
Verify each before proceeding to the next. The vLLM endpoint is already running.

### Phase 1: Environment and Privacy

1. Confirm `uv sync` passes (base dependencies install cleanly)
2. Confirm `uv sync --extra data --extra eval --extra generation --extra training --extra dev` passes
3. Create private/public directory structure under `data/processed/`
4. Verify `.gitignore` covers private paths
5. Confirm `uv run pytest` runs (even with no tests yet)
6. Set `PYTHONPATH` or install package in editable mode so `from medcolbert import ...` works

### Phase 2: Ontology Control Layer

1. Implement `src/medcolbert/data/umls.py` — parse MRCONSO.RRF, MRREL.RRF, MRSTY.RRF
2. Build concept strings parquet, concept pairs parquet, concept edges parquet, semantic types parquet
3. Build vocabulary-shift pair pools (consumer→clinical, abbreviation→expanded, brand→generic, etc.)
4. Write public-safe count manifests (no restricted strings)
5. Run ontology gate checks: CUIs load, semantic groups countable, no restricted strings in public outputs
6. Output: `data/processed/private/ontology/*.parquet`, `data/processed/public/manifests/ontology_*.json`

### Phase 3: Public Passage Store

1. Ingest `MedRAG/pubmed` from HuggingFace (23.9M abstracts, 1166 JSONL chunk files)
2. Ingest BeIR medical datasets (trec-covid, nfcorpus, bioasq) for qrel-backed sources
3. Ingest ClinicalTrials.gov passages via `irds/` datasets
4. Chunk long documents (max 384 tokens, 64 token overlap)
5. Build deterministic fingerprints (exact hash, normalized hash, MinHash)
6. Annotate passages with UMLS CUI matches (private)
7. Run corpus gate checks: stable IDs, source metadata, decontamination fingerprints
8. Log concept coverage by corpus and semantic group
9. Output: `data/processed/private/passages/passages.parquet`, `passage_cui_matches.parquet`, `fingerprints.parquet`

### Phase 4 Prerequisites: Generation Scripts

Before the generation loop can start, build these:

1. `src/medcolbert/generation/dspy_programs.py` — all DSPy signatures (generator + 5 judges), pipeline module
2. `src/medcolbert/generation/prompts.py` — prompt templates per task family and role
3. `src/medcolbert/generation/validators.py` — deterministic validators (Stage 3 checks)
4. `src/medcolbert/generation/teacher_filter.py` — teacher margin scoring using judge LM
5. `scripts/data/04_generate_synthetic_queries.py` — thin CLI: loads config, calls dspy_programs, writes results
6. Script must accept: `--config`, `--mode`, `--limit`, `--batch-id`, `--output-dir`
7. Script must output: candidates.parquet + stats.json per batch
8. `scripts/data/04_judge_candidates.py` — thin CLI: loads candidates, calls 5 judges, writes judgments
9. `scripts/data/04_audit_sample.py` — thin CLI: stratified sample from batch, formats for Claude review
10. Verify end-to-end: generate 5 test queries → judge them → audit sample → all files on disk

## Generation Modes (Ordered)

Generate and audit each mode separately before moving to the next:

| # | Mode | Purpose |
|---|---|---|
| 1 | `generic_synthetic` | Lower bound — LLM query from passage, no ontology |
| 2 | `passage_grounded` | Sanity check — query must be answerable |
| 3 | `ontology_grounded` | Core method — targets CUIs with vocab shift |
| 4 | `ontology_grounded_teacher_filtered` | Best method — adds LLM judge margin |

Start with mode 1 (easiest, establishes baseline), progress to mode 4.

## Task Families

Every generated example must be tagged with exactly one task family:

| Family | Query Style | Positive Document |
|---|---|---|
| `lay_to_clinical` | Patient wording | Clinical or biomedical passage |
| `abbreviation_to_expanded` | Clinical abbreviation | Passage with expanded term |
| `brand_generic` | Brand/generic drug mention | Drug evidence passage |
| `symptom_to_diagnosis` | Symptom phrase | Diagnosis or treatment passage |
| `trial_matching` | Patient vignette | ClinicalTrials.gov trial |
| `biomedical_to_clinical` | Research wording | Clinical passage |
| `patient_similarity` | Patient summary | Case report or patient passage |
| `no_direct_lexical_overlap` | Paraphrased concept query | Passage with target concept |

## Roles

Weighted sampling from config:

| Role | Weight |
|---|---|
| patient | 0.30 |
| physician | 0.25 |
| researcher | 0.20 |
| nurse | 0.15 |
| student | 0.10 |

## Quality Dimensions

Five binary/categorical questions. The judge (Gemma) answers every question for
every candidate. Claude spot-checks samples to detect judge calibration drift.

### Q1: Answerability

```
Is this query answerable from ONLY the provided passage?

A) ANSWERABLE — the passage contains the specific information needed
   to answer this query
B) NOT_ANSWERABLE — the passage does not contain sufficient information
   to answer this query
```

**Gate:** ≥90% ANSWERABLE

### Q2: Vocabulary Shift

```
Does the query use different vocabulary from the passage to express
the same medical concept?

A) STRONG_SHIFT — entirely different terms (e.g., "heart attack prevention pill"
   vs "myocardial infarction prophylaxis medication")
B) MODERATE_SHIFT — some overlap but meaningfully different terms
C) MINIMAL_SHIFT — mostly same words, rearranged
D) NO_SHIFT — direct copy or near-copy of passage language
```

**Gate:** ≥70% STRONG_SHIFT or MODERATE_SHIFT. ≤10% NO_SHIFT.

### Q3: Passage Specificity

```
Is this query grounded in the specific content of this passage, or is it
a generic question that could apply to many passages about this topic?

A) SPECIFIC — the query references details, findings, or context unique
   to this passage
B) GENERIC — this query could be asked without ever reading this passage
```

**Gate:** ≥85% SPECIFIC

### Q4: Factual Grounding

```
Does the query assert any factual claim that is NOT supported by the passage?

A) CLEAN — all factual claims in the query are supported by the passage
B) HALLUCINATION — the query states or implies something not in the passage
C) CONTRADICTION — the query asserts something directly contradicted
   by the passage
```

**Gate:** ≥95% CLEAN

### Q5: Task Family Fit

```
Does this query match its assigned task family and role?

A) MATCH — the query's language, perspective, and vocabulary level
   correctly reflect the assigned task family and role
B) MISMATCH — the query reads as a different task family or role
   than assigned
```

**Gate:** ≥90% MATCH

## Judge Protocol

### Prompt Design Rules

- Binary/categorical output only — never numeric scales
- Answer letter must be the FIRST token after the reasoning line
- Each question is a separate API call (independent, no cross-contamination)
- Temperature = 0 for judging (deterministic, reproducible)
- Judge uses a DIFFERENT system prompt than generator
- Judge prompt includes: the passage, the query, the target CUI label, the task family, the role
- Judge must NOT see the generator's chain-of-thought (CoT stays private)

### Per-Example Judge Output Schema

```json
{
  "example_id": "str",
  "judge_model": "gemma-4-31b",
  "judge_prompt_hash": "str",
  "judgments": {
    "answerability": "ANSWERABLE | NOT_ANSWERABLE",
    "vocab_shift": "STRONG_SHIFT | MODERATE_SHIFT | MINIMAL_SHIFT | NO_SHIFT",
    "passage_specificity": "SPECIFIC | GENERIC",
    "factual_grounding": "CLEAN | HALLUCINATION | CONTRADICTION",
    "task_family_fit": "MATCH | MISMATCH"
  }
}
```

### Aggregate Decision Per Example

```
ACCEPT if:
  answerability == ANSWERABLE
  AND vocab_shift in (STRONG_SHIFT, MODERATE_SHIFT)
  AND passage_specificity == SPECIFIC
  AND factual_grounding == CLEAN
  AND task_family_fit == MATCH

REJECT otherwise (with specific rejection reasons)
```

## Claude's Role: Controller + Sanity Checker

Claude does NOT generate text. Claude does NOT bulk-judge examples. Claude:

### Per Batch

1. Receives aggregate stats:
   - Pass rates per quality dimension
   - Pass rates per task family
   - Pass rates per role
   - Pass rates per semantic group
   - Overall acceptance rate

2. Receives a stratified sample of 30 examples:
   - 10 random ACCEPT (by judge)
   - 10 random REJECT (by judge)
   - 5 from the most marginal dimension
   - 5 from the lowest-performing task family

3. Spot-checks the sample:
   - Verifies judge labels are reasonable
   - Identifies systematic failure patterns
   - Checks for judge leniency or harshness drift
   - Flags examples where judge is clearly wrong

4. Makes a decision:
   - **ADVANCE** — quality meets all gates, move to next batch or mode
   - **REGENERATE** — quality below threshold, adjust parameters and retry
   - **CALIBRATE** — judge labels appear systematically wrong, recalibrate judge prompt
   - **PAUSE** — systemic issue requiring human intervention

### Across Batches

- Tracks quality trends per dimension, family, and mode
- Detects drift (e.g., vocab_shift rate declining over batches)
- Ensures semantic group coverage is balanced
- Compares generation modes to confirm ontology_grounded > generic_synthetic

## Exit Conditions

### Per Mode

A generation mode is complete when:
- ≥1,000 examples accepted
- All five quality gates met in aggregate
- All task families have ≥50 accepted examples
- All quality gates met per task family (no family below 70% acceptance)

### Calibration Exit

Calibration is complete when:
- Claude and Gemma-judge agree on ≥85% of a 30-example sample
- No systematic leniency or harshness detected

### Overall Goal Exit

Goal is achieved when mode `ontology_grounded_teacher_filtered`:
- Has ≥1,000 accepted examples
- All quality gates met
- Audit acceptance rate ≥80%
- Outperforms `generic_synthetic` on at least 3 of 5 quality dimensions
- Audit sample across all 8 task families shows acceptable quality
- → Signal: **GO for production scale (2.9M)**

### Failure Conditions

Stop and reassess if:
- 5 consecutive batches fail the same quality dimension
- Judge calibration drifts (≥3 Claude-overruled labels in a single batch)
- Regeneration with adjusted parameters fails to improve a dimension after 3 attempts
- vLLM endpoint returns persistent errors

## Build Phase Protocol (Phases 1-3 + Scripts)

Before the generation loop, Claude must build the infrastructure. This is
deterministic engineering work, not a loop.

### Build Flow

```
For each phase in Build Order (1 → 2 → 3 → scripts):
  1. Read the detailed phase doc in docs/goals/0X-*.md
  2. Implement the code under src/medcolbert/
  3. Write thin CLI scripts under scripts/
  4. Run the phase gate check (verify outputs exist, counts are reasonable)
  5. If gate passes → advance to next phase
  6. If gate fails → fix and retry (max 3 attempts per gate)
```

### Build Decisions Claude Must Make

- Package structure: follow AGENTS.md planned layout
- Config parameter values: use configs/*.yaml as source of truth
- DSPy adapter: `ChatAdapter` (default) for all modules. Do NOT use
  `ChainOfThought` — Gemma 4 31B reasons natively. Use `dspy.Predict`.
- LM instances: separate instances for generator (temp=0.7) and judge (temp=0.0)
- Ingestion strategy: process one corpus at a time, verify fingerprints, log counts
- Private/public boundary: paranoid. If unsure, keep it private.

### Phase Gate Verification

At each gate, Claude reports:
- What was built (files, schemas, counts)
- Gate evidence (e.g., "3.2M concepts loaded, 847K pairs after filtering,
  0 restricted strings in public outputs")
- Any anomalies or concerns

Human can override a gate decision, but Claude must present the evidence first.

### Build Complete Signal

When Phase 4 prerequisites are met:
- `uv run python scripts/data/04_generate_synthetic_queries.py --config configs/generation.yaml --mode generic_synthetic --limit 5` produces output
- All 5 test queries are judged
- Audit sample parquet exists
- → Signal: build complete, entering generation loop

## Agent Loop Protocol

```
For each generation mode (1 → 4):

  # --- CALIBRATION (only for first mode, verify for subsequent) ---
  Generate 50 pilot examples
  Judge all 50 with Gemma
  Claude reviews 30 (10 accept + 10 reject + 5 marginal + 5 worst family)
  Compare Claude vs Gemma labels
  If disagreement > 15%:
    → adjust judge prompt, recalibrate, repeat
  If calibrated:
    → lock judge prompt, proceed to batch loop

  # --- BATCH LOOP ---
  batch_number = 0
  total_accepted = 0

  while total_accepted < 1000 and batch_number < 20:
    batch_number += 1

    # 1. Generate
    Generate batch (500-1000 candidates, Gemma)
    Run deterministic validators → split into pass / fail
    Log rejection reasons and counts

    # 2. Judge
    All deterministically-passed candidates → Gemma judge (5 questions each)
    Compute aggregate stats

    # 3. Claude reviews sample
    Receive: stats + stratified sample of 30
    Claude evaluates: quality, calibration, patterns

    # 4. Decision
    if Claude decides ADVANCE:
      write accepted to accepted.parquet
      total_accepted += count_accepted
      report: batch {n} accepted, {count} examples, running total {total_accepted}
      optional: adjust sampling params for next batch

    elif Claude decides REGENERATE:
      diagnose failure from stats + sample
      adjust: prompt template, role mix, forbidden terms list,
              target CUI selection, or passage filtering
      report: regenerating batch {n} with adjustment: {reason}
      retry same batch (max 3 retries per batch)

    elif Claude decides CALIBRATE:
      generate fresh 30-example calibration set
      compare Claude vs Gemma labels
      rewrite judge prompt if needed
      re-lock and continue

    elif Claude decides PAUSE:
      report: blockage reason, evidence, recommendation
      EXIT loop

  # --- MODE COMPLETE ---
  Report: mode {name} complete
    - total accepted: {count}
    - acceptance rate: {percent}
    - per-dimension pass rates
    - per-family pass rates
    - example audit sample (30 examples)

  Confirm: proceed to next mode? (y/n)

# --- GOAL COMPLETE ---
All 4 modes complete, mode 4 has ≥1000 accepted at ≥80% audit acceptance.
Signal GO for production scale.
```

## Context Compaction Survival

Claude Code auto-compacts conversation history when context exceeds the window.
During a multi-batch goal run, compaction WILL happen. The loop must survive it.

### What Gets Lost

- Specific examples reviewed in earlier batches
- Exact numerical stats from batch N (unless summarized in the compaction)
- Nuance of past diagnostic reasoning
- The "feel" for drift over time

### What Survives

- Files on disk (all parquet outputs, stats.json, decision logs)
- The DS_GOAL.md specification itself
- Recent conversation context (last ~10-20 messages typically survive)
- The compaction summary (high-level: mode, batch number, running totals, last decision)

### Checkpoint Protocol

At every decision point, Claude MUST write a checkpoint file BEFORE announcing the
decision. The checkpoint contains all state needed to resume the loop from this
exact point after compaction or session restart.

**Checkpoint schema** (`data/processed/private/synthetic/checkpoint.json`):

```json
{
  "timestamp": "ISO-8601",
  "current_mode": "ontology_grounded",
  "current_batch": 7,
  "total_accepted": 423,
  "total_generated": 3500,
  "mode_stats": {
    "generic_synthetic": {"completed": true, "accepted": 1100, "batches": 3},
    "passage_grounded": {"completed": true, "accepted": 1050, "batches": 3},
    "ontology_grounded": {"completed": false, "accepted": 423, "batches": 7},
    "ontology_grounded_teacher_filtered": {"completed": false, "accepted": 0, "batches": 0}
  },
  "last_decision": {
    "type": "advance",
    "batch": 7,
    "reasoning": "All dimensions met gates. Minor vocab_shift weakness in brand_generic family (68% MODERATE+), monitoring next batch.",
    "adjustments": null
  },
  "quality_trend": {
    "answerability": [0.94, 0.92, 0.93, 0.95, 0.93, 0.94, 0.95],
    "vocab_shift_moderate_plus": [0.72, 0.75, 0.71, 0.78, 0.76, 0.74, 0.77],
    "passage_specificity": [0.88, 0.90, 0.89, 0.92, 0.91, 0.93, 0.92],
    "factual_grounding": [0.97, 0.96, 0.98, 0.97, 0.98, 0.97, 0.98],
    "task_family_fit": [0.91, 0.92, 0.90, 0.93, 0.92, 0.94, 0.93]
  },
  "per_family_acceptance": {
    "lay_to_clinical": 0.82,
    "abbreviation_to_expanded": 0.88,
    "brand_generic": 0.68,
    "symptom_to_diagnosis": 0.84,
    "trial_matching": 0.79,
    "biomedical_to_clinical": 0.85,
    "patient_similarity": 0.81,
    "no_direct_lexical_overlap": 0.76
  },
  "locked_prompts": {
    "generator_prompt_hash": "abc123...",
    "judge_prompt_hash": "def456..."
  },
  "judge_calibration": {
    "last_checked_batch": 5,
    "agreement_rate": 0.90,
    "drift_detected": false
  },
  "resume_instructions": "Continue ontology_grounded mode, batch 8. Brand_generic family needs monitoring for vocab_shift weakness. Judge calibration verified at batch 5 (90% agreement). All prompts locked."
}
```

### Resume Protocol

When resuming after compaction or session restart:

1. Read `checkpoint.json`
2. Read `stats.json` from last completed batch
3. Compare checkpoint state with DS_GOAL.md exit conditions
4. Determine next action: continue current mode, advance to next mode, or re-audit last batch
5. If checkpoint is missing or stale (last batch not on disk): re-run last batch's audit from `audit_sample.parquet`
6. Continue the loop from the checkpoint state

### Anti-Drift Safeguards

After compaction, before making any new decisions:

1. Re-read the last batch's `audit_sample.parquet`
2. Spot-check 5 examples from it to re-establish calibration baseline
3. Compare current quality trends against historical trends from checkpoint
4. Only then make a decision on the next batch

This prevents "decision drift" where post-compaction Claude applies different
standards than pre-compaction Claude.

## Output Artifacts

Per mode, per batch:

```
data/processed/private/synthetic/{mode}/batch_{n}/
  candidates.parquet         # All generated (with judge labels)
  accepted.parquet           # Filtered by judge decision
  rejected.parquet           # Filtered by judge decision
  audit_sample.parquet       # Stratified sample for Claude review
  stats.json                 # Aggregate counts and rates
  claude_decision.json       # {decision, reasoning, adjustments, timestamp}
```

Final pilot output:

```
data/processed/private/synthetic/
  pilot_accepted.parquet     # All accepted from all modes (≥1000 from mode 4)
  pilot_rejected.parquet     # All rejected with reasons
  pilot_audit.parquet        # Final stratified audit sample
  pilot_stats.json           # Full aggregate statistics
  judge_prompts.json         # Locked judge prompts with hashes
  generation_prompts.json    # Locked generation prompts with hashes
  calibration_report.json    # Calibration results per mode
  go_decision.json           # Final GO/NO-GO with evidence
```

Public-safe manifest:

```
data/processed/public/manifests/synthetic_pilot.json
  # Counts, hashes, per-family stats, per-mode stats
  # NO restricted vocabulary strings
```

## Config Reference

From `configs/generation.yaml`:

- Generator temperature: 0.7 (creative generation)
- Judge temperature: 0 (deterministic judgment)
- Batch size: 64 (generation), 1 (judging — one call per question per example)
- Max query chars: 180
- Max passage chars: 1800
- Min alternative term count: 1
- Max forbidden surface overlap: 0.25
- Near-duplicate threshold: minhash 0.86
- Audit samples: 100 per role, 100 per semantic group
- Minimum acceptance rate: 0.80

From `configs/data.yaml`:

- Teacher margin minimum: 0.15
- Decontamination enabled, minhash threshold 0.8
- Preferred synthetic mode: `umls_grounded_teacher_filtered`

## Common Failure Patterns to Watch For

- **Weak vocabulary shift:** Generator copies passage language, Q2 mostly MINIMAL_SHIFT
  → Fix: add stricter forbidden terms, strengthen prompt instruction
- **Generic queries:** Q3 mostly GENERIC
  → Fix: add passage-specific details requirement to prompt, sample passages with more unique content
- **Hallucinated claims:** Q4 shows HALLUCINATION
  → Fix: add grounding instruction, reduce temperature, filter passages with clear facts
- **Role collapse:** Generator defaults to "researcher" voice regardless of assigned role
  → Fix: strengthen role instructions, add role-specific example queries to prompt
- **Family imbalance:** Some task families produce much lower quality than others
  → Fix: per-family prompt tuning, adjust family weights
- **Judge leniency:** Judge accepts queries Claude would reject
  → Fix: recalibrate judge prompt, add stricter criteria language
- **Semantic group gaps:** Certain UMLS semantic groups underrepresented
  → Fix: adjust concept sampling to oversample underrepresented groups
