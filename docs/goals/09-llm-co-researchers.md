# LLM Co-Researcher Protocol

## Goal

Use strong open LLMs as research collaborators and use small LLMs as execution
engines. The markdown plans are the source of truth; LLMs operate by reading
evidence packets and producing structured decisions, critiques, labels, or task
cards.

## Core Rule

Do not ask an LLM vague research questions. Give it a packet.

A good packet contains:

- phase goal
- exact decision needed
- relevant constraints
- current artifacts
- small samples
- allowed outputs
- JSON or Markdown schema
- explicit failure modes to look for

## Model Roles

### Strong Open LLMs: Co-Researchers

Use the strongest available open model, such as Gemma 4 31B or another current
frontier-class open model, for research judgment.

Best uses:

- hypothesis critique
- data-source ranking
- synthetic query quality review
- relevance adjudication
- hard-negative analysis
- ablation design
- eval-table interpretation
- reviewer-style paper criticism
- release and privacy audit

The co-researcher LLM should be asked to disagree with the plan when warranted.
Its job is not to be encouraging; its job is to find better experiments and
failure modes.

### Medium Specialist Models

Use specialist models when they are empirically useful:

- biomedical NER or entity-linking models for auxiliary annotation
- clinical de-identification models for privacy checks
- cross-encoders or rerankers distilled from LLM labels
- coding models for medium-complexity implementation

Specialist models are tools. Validate them against manual or strong-LLM audit
sets before trusting them.

### Small LLMs: Executors

Use small LLMs only for bounded execution:

- implement a named function
- write a CLI wrapper
- add unit tests
- convert schemas to dataclasses
- summarize logs
- produce audit tables
- run deterministic transformations

Small LLMs should receive task cards. They should not make research-policy
decisions.

## Required Artifacts

Create these artifacts as the project matures:

```text
docs/research/decision_log.md
docs/research/hypotheses.md
docs/research/data_source_reviews.md
docs/research/llm_judge_rubrics.md
runs/research/packets/
runs/research/llm_judgments/
runs/research/disagreements/
```

Do not store restricted UMLS/SNOMED strings in public research notes. If a
packet contains restricted strings, keep it under a private path.

## Research Loops

### Loop 1: Hypothesis Workshop

Use before major implementation.

Packet contents:

- current hypothesis
- intended mechanism
- benchmark list
- baseline list
- planned ablations
- known constraints

Ask the LLM to return:

```json
{
  "strongest_claim": "...",
  "weakest_assumption": "...",
  "likely_reviewer_objections": ["..."],
  "must_run_ablations": ["..."],
  "kill_criteria": ["..."],
  "recommended_reframe": "..."
}
```

Human action:

- accept, reject, or modify claims in `docs/research/decision_log.md`
- update phase docs only after a human decision

### Loop 2: Data Source Council

Use when choosing corpora or data mixtures.

Packet contents:

- dataset card or source summary
- license
- fields
- example rows
- expected supervision type
- contamination risk
- privacy risk
- how the data maps to vocabulary-shift buckets

Ask the LLM to return:

```json
{
  "use_recommendation": "use|maybe|avoid",
  "best_use": "training|eval|style_seed|negative_mining|audit_only",
  "risks": ["..."],
  "required_filters": ["..."],
  "expected_failure_modes": ["..."],
  "priority": 1
}
```

Human action:

- only add the dataset to configs after the decision is logged

### Loop 3: Synthetic Program Design

Use before bulk generation.

Packet contents:

- target task family
- passage samples
- target CUI metadata without exposing restricted public strings unless private
- allowed query roles
- forbidden copying rules
- validation rules
- rejected examples

Ask the LLM to return:

```json
{
  "prompt_changes": ["..."],
  "validator_changes": ["..."],
  "new_failure_modes": ["..."],
  "examples_to_reject": ["example_id"],
  "examples_to_keep": ["example_id"],
  "needs_human_review": ["example_id"]
}
```

Human action:

- update prompt/program only after comparing against audit samples

### Loop 4: Relevance Judge

Use for pilot filtering and ambiguous examples.

The judge must see query, positive passage, and candidate negatives in shuffled
order. Do not reveal which passage is supposed to be positive.

Recommended label scale:

| Label | Meaning |
|---|---|
| 0 | not relevant |
| 1 | weakly related but not sufficient |
| 2 | relevant, partial or indirect |
| 3 | clearly relevant |

Ask the LLM to return:

```json
{
  "query_id": "...",
  "passage_scores": [
    {
      "passage_id": "...",
      "score": 0,
      "evidence": "brief non-chain-of-thought justification",
      "failure_modes": ["wrong_concept"]
    }
  ],
  "best_passage_id": "...",
  "margin": 0,
  "ambiguous": false,
  "needs_human_review": false
}
```

Rules:

- Keep explanations brief.
- Store model name, prompt hash, temperature, and seed.
- Use deterministic settings when possible.
- Human-review low-margin examples.
- Never treat LLM labels as gold before calibration.

### Loop 5: Hard-Negative Adversary

Use when building negatives.

Packet contents:

- query
- positive passage
- negative candidates
- target concepts
- negative source type

Ask the LLM to identify:

```json
{
  "false_negatives": ["passage_id"],
  "too_easy_negatives": ["passage_id"],
  "good_hard_negatives": ["passage_id"],
  "missing_negative_types": ["..."],
  "notes": "..."
}
```

Human action:

- remove false negatives
- mine more of the suggested missing negative types

### Loop 6: Ablation Critic

Use before expensive training.

Packet contents:

- planned run matrix
- changed variables
- compute budget
- expected claims
- current pilot results

Ask the LLM to return:

```json
{
  "confounded_comparisons": ["..."],
  "missing_controls": ["..."],
  "redundant_runs": ["..."],
  "minimal_run_matrix": ["..."],
  "must_not_claim": ["..."]
}
```

Human action:

- freeze the run matrix before training

### Loop 7: Paper Reviewer

Use after evaluation tables exist.

Packet contents:

- abstract draft
- main claim
- result tables
- ablations
- decontamination summary
- release policy

Ask the LLM to write:

```json
{
  "likely_accept_reasons": ["..."],
  "likely_reject_reasons": ["..."],
  "overclaims": ["..."],
  "missing_experiments": ["..."],
  "best_framing": "...",
  "worst_table": "...",
  "action_items": ["..."]
}
```

Human action:

- fix the paper or narrow the claim

### Loop 8: Release Auditor

Use before publishing code, model, or data artifacts.

Packet contents:

- release file list
- model card
- data card
- public manifests
- sample records
- license notes

Ask the LLM to return:

```json
{
  "release_safe": false,
  "blocking_issues": ["..."],
  "risky_files": ["..."],
  "missing_license_notes": ["..."],
  "model_card_gaps": ["..."],
  "recommended_actions": ["..."]
}
```

Human action:

- do not release until blocking issues are resolved

## Calibration Protocol

Before trusting any LLM judge:

1. Create 200 to 500 human-reviewed examples.
2. Run the LLM judge blind.
3. Measure agreement with human labels.
4. Inspect disagreements.
5. Update rubric or prompt.
6. Repeat until quality is acceptable.

Track:

- exact-match agreement
- within-one-label agreement
- false-positive rate
- false-negative rate
- agreement by task family
- agreement by semantic group
- agreement by vocabulary-shift type

If the LLM judge fails a bucket, do not use it for that bucket without human
review or a better rubric.

## Multi-LLM Review

For high-impact decisions, use at least two strong open LLMs independently.

Use this when:

- changing the main research claim
- approving a new data source
- accepting a synthetic generation strategy
- selecting the final ablation matrix
- interpreting surprising results
- approving public release

Procedure:

1. Build one evidence packet.
2. Send it independently to each model.
3. Store structured outputs.
4. Compare disagreements.
5. Ask a final model or human to summarize unresolved disagreement.
6. Log the human decision.

Do not let the models debate interactively before they give independent first
answers. Independent disagreement is more useful.

## Task Cards For Small LLMs

Small LLMs should receive task cards like this:

```markdown
# Task Card

Goal:
Implement `load_mrconso` in `src/medcolbert/data/umls.py`.

Inputs:
- path to `MRCONSO.RRF`
- filter config

Outputs:
- pandas DataFrame with documented columns
- counts of accepted/rejected rows

Constraints:
- no public writes
- keep original strings private
- deterministic output
- no network calls

Tests:
- unit test with tiny synthetic MRCONSO fixture
- test suppression and language filters

Do not:
- change research docs
- add dependencies
- alter global config format
```

Small LLMs should return:

- files changed
- tests added
- commands run
- remaining blockers

## Anti-Patterns

- Asking an LLM: "What should we do next?"
- Letting an LLM invent labels without real passages.
- Letting an LLM see which candidate is the intended positive during judging.
- Using one model's answer as ground truth without calibration.
- Using a small executor model for research strategy.
- Letting generated explanations leak chain-of-thought into datasets.
- Publishing private packets that contain restricted ontology strings.

## Acceptance Criteria

This protocol is active when:

- every major research decision has a packet or decision-log entry
- every LLM-labeled dataset has model/prompt/config provenance
- every LLM judge has a calibration report
- small LLM tasks are issued as bounded task cards
- humans resolve high-impact disagreements

