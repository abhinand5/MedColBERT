# MedColBERT: Medical Late-Interaction Retrieval Model
## Comprehensive Build Plan

> **Author context:** This plan is written for Abhinand Balachandran, author of MedEmbed,
> targeting a top IR venue (SIGIR 2026 primary / EMNLP 2026 secondary).
>
> **Reading guide:**
> - If you are an AI agent executing tasks: follow sections sequentially, use `DECISION` blocks
>   to resolve ambiguities, use `OPEN QUESTION` blocks to flag items needing human input.
> - If you are a human engineer: skim headers first, read `RATIONALE` blocks before writing code,
>   treat `OPEN QUESTION` blocks as your action items.
> - If you are a senior researcher reviewing: focus on Sections 3, 5, and 8. Everything else is
>   implementation detail.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Theoretical Grounding](#2-theoretical-grounding)
3. [Core Hypothesis and Paper Argument](#3-core-hypothesis-and-paper-argument)
4. [Repository Structure](#4-repository-structure)
5. [Backbone Selection](#5-backbone-selection)
6. [Data Pipeline](#6-data-pipeline)
7. [Training Pipeline](#7-training-pipeline)
8. [Evaluation Suite](#8-evaluation-suite)
9. [Ablation Studies](#9-ablation-studies)
10. [Compute Budget](#10-compute-budget)
11. [Open Questions and Risks](#11-open-questions-and-risks)
12. [Timeline](#12-timeline)
13. [References and Prior Art](#13-references-and-prior-art)

---

## 1. Project Overview

### What we are building

**MedColBERT**: a late-interaction (ColBERT-style) retrieval model specifically trained for
medical and clinical information retrieval, using medical ontologies (UMLS, SNOMED CT) as
structured training supervision.

### What makes this novel

Nothing in the existing literature combines:
1. Late interaction (MaxSim scoring) with
2. A clinical-domain backbone encoder with
3. Ontology-guided training supervision (UMLS cross-vocabulary pairs + SNOMED hard negatives) with
4. Large-scale synthetic triplet generation constrained by medical concept grounding

Each of (1)–(4) exists independently in prior work. The combination does not.

### What we expect to prove

That ontology-guided late-interaction training produces token embeddings that align
cross-vocabulary medical synonyms at retrieval time in a way that:
- Dense bi-encoders structurally cannot (single-vector bottleneck)
- General-domain ColBERT cannot (no synonym supervision signal)
- BM25 cannot (lexical, not semantic)

### Deliverables

| Deliverable | Purpose |
|---|---|
| `MedColBERT-base` | 150M param model, fast inference |
| `MedColBERT-large` | 396M param model, headline numbers |
| Medical retrieval triplet dataset (~8–11M) | Released publicly, contribution in itself |
| Evaluation harness across 6+ benchmarks | Reproducible, released |
| Paper | SIGIR 2026 |

---

## 2. Theoretical Grounding

> This section exists so any agent or engineer can reconstruct the "why" behind every
> implementation decision. Do not skip it.

### 2.1 Why late interaction beats dense for medical IR

A dense bi-encoder maps an entire document to a single vector. This creates a
**representation bottleneck**: all semantic content — entities, relationships,
dosages, findings — must compress into one ~768-dimensional point. For general
text this is acceptable. For medical text it fails on two specific patterns:

**Pattern 1: Cross-vocabulary synonym variance**
- Query: `"heart attack treatment options"`
- Document: `"Management of acute myocardial infarction: reperfusion strategies"`
- Dense model sees low cosine similarity because the overlap in embedding space
  between lay-vocabulary queries and clinical-vocabulary documents is low unless
  explicitly trained on that pairing.

**Pattern 2: Fine-grained clinical distinction**
- `"Type 1 diabetes management"` vs `"Type 2 diabetes management"` have nearly
  identical dense embeddings. The single "diabetes" concept dominates the vector.
- A cross-encoder resolves this via full attention but cannot be indexed.
- ColBERT resolves this because "Type 1" and "Type 2" have distinct token
  embeddings that MaxSim routes independently.

Late interaction keeps per-token embeddings. MaxSim lets each query token independently
find its best matching document token. "heart" finds "cardiac"/"myocardial",
"attack" finds "infarction"/"MI"/"event", independently and simultaneously.
No bottleneck.

### 2.2 Why UMLS/SNOMED provide uniquely valuable training signal

**UMLS (Unified Medical Language System):**
- 3.5M+ biomedical concepts
- Each concept (identified by a CUI — Concept Unique Identifier) has multiple
  string representations drawn from ~200 source vocabularies
- Key vocabularies: SNOMEDCT_US, ICD10CM, ICD9CM, RXNORM (drugs),
  CHV (Consumer Health Vocabulary), NCI (National Cancer Institute), MSH (MeSH)
- CHV specifically maps clinical terms → patient-friendly lay language
  (e.g., "myocardial infarction" → "heart attack", "hypertension" → "high blood pressure")
- Same CUI = same concept = perfect positive pair signal

**SNOMED CT:**
- Hierarchical ontology (~370K concepts)
- Every concept has: parents, children, siblings
- Sibling concepts share a parent but are distinct (e.g., Type 1 DM and Type 2 DM
  are siblings under "Diabetes mellitus")
- Sibling pairs = structurally grounded **hard negatives** — medically meaningful
  distinctions that cannot be learned from surface-level text statistics

No general-purpose IR training set provides this level of structured semantic
organization. MS MARCO, Natural Questions, etc. use BM25 or annotator judgments
for negatives. SNOMED gives you the medical knowledge graph directly.

### 2.3 How ColBERT training works (for newcomers)

ColBERT uses **identical triplet format** to dense retrieval models:
```
(query, positive_passage, hard_negative_passage)
```

The only difference is the scoring function inside the loss:

**Dense:**
```
score(q, d) = dot(pool(q_tokens), pool(d_tokens))
```

**ColBERT (MaxSim):**
```
score(q, d) = sum over query tokens of [max over doc tokens of cosine_sim(q_i, d_j)]
```

This means ALL existing medical retrieval triplet datasets (BioASQ, MedEmbed v1,
TREC Clinical Trials qrels, etc.) can be used directly with ColBERT training code
with zero reformatting.

### 2.4 Matryoshka Representation Learning (MRL)

MRL trains the model such that the first k dimensions of each token embedding are
themselves a meaningful representation. This enables:
- `dim=32` for fast coarse retrieval over all documents
- `dim=128` for exact reranking of candidates
- No separate models needed

For medical ColBERT this enables a practical two-pass system:
```
Pass 1 (dim=32): score all N docs cheaply → top-1000 candidates
Pass 2 (dim=128): rescore top-1000 exactly → final ranking
```

MRL is implemented as an additional loss term weighted across truncated dimensions.
See Section 7.4 for implementation.

---

## 3. Core Hypothesis and Paper Argument

> This is the intellectual core. Every implementation decision should serve this argument.

### 3.1 The claim

> **Ontology-guided late interaction retrieval, trained with cross-vocabulary synonym pairs
> from UMLS and hierarchical hard negatives from SNOMED CT, achieves state-of-the-art
> medical information retrieval by learning token-level alignments that bridge terminology
> heterogeneity — a failure mode that dense retrieval and BM25 are architecturally
> unable to address.**

### 3.2 The argument structure (for the paper)

```
1. Medical IR fails because of terminology heterogeneity
   Evidence: show BM25 and dense recall@10 on synonym-split test sets

2. Late interaction can bridge terminology gaps IF trained on the right signal
   Evidence: general ColBERT (untrained on medical) vs our model

3. UMLS/SNOMED provide exactly that signal, at scale, for free
   Evidence: ablation — same model, random negatives vs SNOMED negatives

4. The resulting model is SOTA across all standard benchmarks
   Evidence: Table 2 (main results)

5. The gains are specifically attributable to cross-vocabulary alignment
   Evidence: Table 3 — performance broken down by query-document vocabulary overlap
             (same vocab vs cross-vocab queries)
```

Step 5 is the most important and most novel part. No prior work has reported
medical IR metrics broken down by query-document vocabulary divergence.
This framing of the problem is itself a contribution.

### 3.3 Falsification conditions

If any of the following are true, the hypothesis is weakened and the paper needs
reframing:

- BioClinical-ModernBERT as a dense bi-encoder (no late interaction) matches
  MedColBERT on cross-vocabulary test sets → late interaction is not necessary,
  domain pretraining alone suffices
- MedColBERT without UMLS supervision matches with UMLS → ontology signal is
  redundant given domain pretraining
- General ColBERT v2 with MedEmbed fine-tuning matches MedColBERT → synthetic
  scale matters more than ontology structure

**If falsified:** the paper becomes "scaling synthetic medical retrieval data works"
which is still publishable but less novel. Design ablations to detect this early.

---

## 4. Repository Structure

```
medcolbert/
│
├── PLAN.md                          ← this file
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── raw/                         ← never modify, gitignored if large
│   │   ├── umls/                    ← MRCONSO.RRF, MRREL.RRF, MRDEF.RRF
│   │   ├── snomed/                  ← RF2 release files
│   │   ├── pubmed/                  ← abstracts (bulk download)
│   │   ├── pmc/                     ← full text articles
│   │   └── mimic/                   ← requires PhysioNet credentialing
│   │
│   ├── processed/
│   │   ├── umls_pairs/              ← (str1, str2, cui) parquet files
│   │   ├── snomed_triplets/         ← (query, pos_cui, neg_cui) parquet files
│   │   ├── synthetic_queries/       ← generated query-passage pairs
│   │   └── final_triplets/          ← merged, deduplicated training data
│   │
│   └── eval/
│       ├── nfcorpus/
│       ├── trec_covid/
│       ├── bioasq/
│       ├── trec_clinical_trials/
│       ├── r2med/
│       └── mirage/
│
├── scripts/
│   ├── data/
│   │   ├── 01_extract_umls_pairs.py
│   │   ├── 02_extract_snomed_negatives.py
│   │   ├── 03_build_passage_index.py
│   │   ├── 04_generate_synthetic_queries.py
│   │   ├── 05_mine_bm25_negatives.py
│   │   ├── 06_mine_dense_negatives.py
│   │   └── 07_merge_and_deduplicate.py
│   │
│   ├── training/
│   │   ├── stage1_contrastive_warmup.py
│   │   ├── stage2_colbert_finetune.py
│   │   └── stage3_kl_distillation.py
│   │
│   └── eval/
│       ├── run_beir.py
│       ├── run_r2med.py
│       ├── run_mirage.py
│       └── run_trec_ct.py
│
├── medcolbert/
│   ├── __init__.py
│   ├── model.py                     ← ColBERT model class
│   ├── tokenizer.py
│   ├── losses.py                    ← MaxSim loss + MRL loss
│   ├── indexer.py                   ← document encoding + storage
│   ├── searcher.py                  ← query-time retrieval
│   └── utils.py
│
└── configs/
    ├── base.yaml
    └── large.yaml
```

---

## 5. Backbone Selection

### 5.1 Decision: Use BioClinical-ModernBERT

**Model:** `thomas-sounack/BioClinical-ModernBERT-base` and `-large`
**License:** MIT (commercially usable, no restrictions for research or release)
**Paper:** arxiv 2506.10896 (June 2025)

**Why not alternatives:**

| Model | Problem |
|---|---|
| `answerdotai/ModernBERT-base` | No medical domain knowledge. We'd need to replicate what BioClinical already did. |
| `microsoft/BiomedNLP-PubMedBERT-base` | 512 token limit — fatal for clinical notes. 2018 architecture. |
| `allenai/biomed_roberta_base` | RoBERTa, 512 tokens, no clinical note training |
| `answerdotai/ModernBERT-large` | Better than BioBERT but no domain data — suboptimal starting point |
| Train our own backbone | 3 months of compute for marginal gain over BioClinical. Not our contribution. |

**What BioClinical gives us:**
- 53.5B tokens of training: 50.7B biomedical (PubMed + PMC) + 2.8B clinical
- Clinical data from 20 institutions across 6 countries
- MIMIC-III (1,047M tokens) + MIMIC-IV (1,765M tokens) already incorporated
- 8192 token context window — handles full clinical notes, not just passages
- Flash Attention 2 — fast training and inference
- Beats every prior clinical encoder on every benchmark it was evaluated on

**OPEN QUESTION:** BioClinical-ModernBERT was published June 2025.
Check if any newer clinical MLM has been released. Search HF for
`clinical modernbert` or `biomedical encoder 2025`. If something substantially
better exists, update this decision.

### 5.2 Model sizes

Train both:
- **base (150M):** For ablations, fast iteration, efficiency claims in the paper
- **large (396M):** For headline benchmark numbers

The paper needs both because reviewers will ask about compute-quality tradeoff.
Base model also serves practitioners who cannot run large models.

### 5.3 Configuration for ColBERT use

```python
# Key modifications from standard BioClinical usage:
# 1. Add a linear projection layer: hidden_dim → colbert_dim (128)
# 2. Do NOT pool — keep all token embeddings
# 3. Normalize token embeddings to unit sphere (cosine similarity = dot product)

from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F

class MedColBERT(nn.Module):
    def __init__(self, backbone_name, colbert_dim=128, attend_to_mask=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden, colbert_dim, bias=False)
        self.attend_to_mask = attend_to_mask
        self.colbert_dim = colbert_dim

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state                     # [B, seq, hidden]
        tok = self.projection(tok)                      # [B, seq, colbert_dim]
        tok = F.normalize(tok, dim=-1)                  # unit sphere

        if not self.attend_to_mask:
            # zero out [MASK] and [PAD] token embeddings
            tok = tok * attention_mask.unsqueeze(-1)

        return tok                                      # [B, seq, colbert_dim]

    def maxsim(self, Q, D):
        # Q: [B_q, Q_len, dim], D: [B_d, D_len, dim]
        # returns: [B_q, B_d] scores
        scores = torch.einsum('qid,pjd->qipj', Q, D)   # [B_q, Q_len, B_d, D_len]
        scores = scores.max(dim=-1).values              # [B_q, Q_len, B_d]
        return scores.sum(dim=1)                        # [B_q, B_d]
```

**DECISION:** ColBERT dimension = 128. This matches ColBERTv2's default and is
sufficient for the full MRL range of 32 → 64 → 128. Increasing to 256 gives
marginal quality gain at 4× storage cost. Not worth it.

---

## 6. Data Pipeline

### 6.1 Overview

Total target: **8–11M training triplets**

```
Real data (ready to use):          ~850K triplets
Semi-synthetic (UMLS/SNOMED):      ~3M  triplets
Synthetic (LLM-generated):         ~5–7M triplets
─────────────────────────────────────────────────
Total:                             ~9–11M triplets
```

The ~850K real triplets are the quality anchor — they define what "relevant"
actually means. The synthetic triplets provide scale and vocabulary coverage.

### 6.2 Real data sources

#### 6.2.1 MedEmbed v1 training triplets
- Location: `abhinand/MedEmbed-training-triplets-v1` on HuggingFace
- Size: ~400K triplets
- Format: (query, positive, negative) — already clean, use directly
- This is your own prior work; using it is both legitimate and establishes
  continuity with the MedEmbed line

#### 6.2.2 BioASQ training data
- Source: bioasq.org (requires registration, free for research)
- Task B training: question → relevant PubMed article IDs
- Convert: for each question, positive = relevant snippets, negative = BM25 top-k
  that are not in the relevant set
- Expected yield: ~200–300K triplets

#### 6.2.3 TREC Clinical Trials 2021 and 2022
- Source: trec.nist.gov
- Topics: patient clinical notes as queries, clinical trial documents as corpus
- qrels provide 3-level relevance (0/1/2) — use 2 as positive, 0 as negative
- Expected yield: ~30–50K triplets (small but high quality, real clinical use case)

#### 6.2.4 TREC-COVID
- Source: BEIR benchmark / ir-datasets
- 50 topics, CORD-19 corpus
- Expected yield: ~20K triplets (small, but good for COVID-specific clinical language)

#### 6.2.5 NFCorpus
- Source: BEIR benchmark
- Nutrition/health queries → NutritionFacts.org documents
- Keep as eval only (it's one of your primary benchmarks — do not contaminate)

**IMPORTANT:** Do not train on any dataset you use for evaluation. NFCorpus, TREC-COVID
(if used for eval), R2MED, MIRAGE — these must be held out completely.

### 6.3 UMLS-based semi-synthetic pairs

#### 6.3.1 Extracting cross-vocabulary synonym pairs

```python
# UMLS MRCONSO.RRF format:
# CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRLV|SUPPRESS|CVF
# Field indices: CUI=0, LAT=1, SAB=11, STR=14, SUPPRESS=16

import pandas as pd
from itertools import combinations

def extract_umls_pairs(mrconso_path, output_path):
    df = pd.read_csv(mrconso_path, sep='|', header=None,
                     names=['CUI','LAT','TS','LUI','STT','SUI','ISPREF',
                            'AUI','SAUI','SCUI','SDUI','SAB','TTY','CODE',
                            'STR','SRLV','SUPPRESS','CVF'],
                     dtype=str, low_memory=False)

    # English only, not suppressed
    df = df[(df['LAT'] == 'ENG') & (df['SUPPRESS'] == 'N')]

    # Define vocabulary groups
    clinical_sabs = {'SNOMEDCT_US', 'ICD10CM', 'ICD9CM', 'NCI', 'MSH', 'MEDLINEPLUS'}
    lay_sabs      = {'CHV'}        # Consumer Health Vocabulary — gold lay language
    drug_sabs     = {'RXNORM', 'VANDF'}
    abbrev_types  = {'AB', 'ACR'}  # UMLS term type codes for abbreviations

    pairs = []
    for cui, group in df.groupby('CUI'):
        clinical = group[group['SAB'].isin(clinical_sabs)]['STR'].tolist()
        lay      = group[group['SAB'].isin(lay_sabs)]['STR'].tolist()
        abbrevs  = group[group['TTY'].isin(abbrev_types)]['STR'].tolist()

        # Cross-type pairs (most valuable — different surface forms, same concept)
        for c in clinical:
            for l in lay:
                if c.lower() != l.lower():  # skip exact duplicates
                    pairs.append({'cui': cui, 'str1': c, 'str2': l,
                                  'type': 'clinical_lay'})
            for a in abbrevs:
                pairs.append({'cui': cui, 'str1': c, 'str2': a,
                              'type': 'clinical_abbrev'})

        # Within-clinical pairs (same concept, different clinical vocab)
        if len(clinical) >= 2:
            for s1, s2 in combinations(clinical[:5], 2):  # cap at 5 to avoid explosion
                if s1.lower() != s2.lower():
                    pairs.append({'cui': cui, 'str1': s1, 'str2': s2,
                                  'type': 'clinical_clinical'})

    pd.DataFrame(pairs).to_parquet(output_path)
    print(f"Extracted {len(pairs)} UMLS synonym pairs")
    # Expected: 3–5M pairs total, ~800K clinical_lay (highest value)
```

#### 6.3.2 Converting pairs to retrieval triplets

UMLS pairs are `(str1, str2)` synonym pairs. To use them for ColBERT training
we need `(query, positive_passage, negative_passage)`.

**Approach A: Direct string-as-query → passage retrieval**

```python
# Use UMLS string as query, retrieve passages from PubMed/PMC
# that contain the paired string (different vocabulary)

# Build a passage index with CUI annotations first
# (use QuickUMLS or scispaCy to annotate PubMed abstracts with CUIs)

def string_pair_to_triplet(str1, str2, cui, passage_index, bm25_index):
    # str1 = lay term (query side), str2 = clinical term (document side)
    query = str1  # e.g., "heart attack"

    # Positive: passage that contains str2 (or its CUI)
    positives = passage_index.get_passages_by_cui(cui)
    if not positives:
        return None
    positive = random.choice(positives[:10])  # top-10 to add variety

    # Negative: BM25 top result for query that is NOT about this CUI
    candidates = bm25_index.search(query, k=20)
    negatives = [c for c in candidates
                 if cui not in passage_index.get_cuis(c['id'])]
    if not negatives:
        return None
    negative = negatives[0]

    return {'query': query, 'positive': positive['text'],
            'negative': negative['text'], 'source': 'umls_pair'}
```

**Approach B: Synthetic query generation grounded on UMLS (preferred)**

See Section 6.5 below.

### 6.4 SNOMED-based hard negative triplets

#### 6.4.1 Why SNOMED siblings are uniquely valuable hard negatives

A hard negative is a passage that looks relevant but isn't. The hardest medical
negatives are passages about a *similar but distinct* concept — exactly what
SNOMED sibling relationships encode.

```
Diabetes mellitus [parent]
├── Type 1 diabetes mellitus  [sibling A] ← concept of interest
├── Type 2 diabetes mellitus  [sibling B] ← structurally hard negative
├── Gestational diabetes      [sibling C] ← structurally hard negative
└── MODY                      [sibling D] ← structurally hard negative
```

A model trained on BM25 negatives will learn "avoid passages about non-diabetes."
A model trained on SNOMED negatives must learn "distinguish Type 1 from Type 2
at the token level" — a much harder and more clinically meaningful task.

#### 6.4.2 Extraction code

```python
# SNOMED RF2 format: Concept file + Relationship file
# Relationship file has: sourceId, typeId, destinationId
# typeId 116680003 = "is a" (parent relationship)

import pandas as pd
from collections import defaultdict

def build_snomed_hierarchy(relationship_file):
    df = pd.read_csv(relationship_file, sep='\t')
    # Filter to active "is a" relationships
    isa = df[(df['active'] == 1) & (df['typeId'] == 116680003)]

    # Build parent → children map
    children = defaultdict(set)
    for _, row in isa.iterrows():
        children[row['destinationId']].add(row['sourceId'])

    return children

def get_sibling_pairs(children, max_siblings=10):
    """For each parent, return sibling pairs (same parent, different concept)"""
    sibling_pairs = []
    for parent, kids in children.items():
        kids = list(kids)
        if len(kids) < 2:
            continue
        # Limit to avoid combinatorial explosion for large families
        kids = kids[:max_siblings]
        for i in range(len(kids)):
            for j in range(i+1, len(kids)):
                sibling_pairs.append((kids[i], kids[j], parent))
    return sibling_pairs

def build_snomed_triplets(sibling_pairs, passage_index, umls_strings):
    """
    For each sibling pair (A, B):
    - query: string representation of concept A
    - positive: passage containing concept A
    - negative: passage containing concept B (hard negative)
    """
    triplets = []
    for cui_a, cui_b, parent_cui in sibling_pairs:
        queries = umls_strings.get(str(cui_a), [])
        pos_passages = passage_index.get_by_cui(str(cui_a))
        neg_passages = passage_index.get_by_cui(str(cui_b))

        if not queries or not pos_passages or not neg_passages:
            continue

        for query in queries[:3]:   # 3 query variants per concept pair
            triplets.append({
                'query': query,
                'positive': random.choice(pos_passages)['text'],
                'negative': random.choice(neg_passages)['text'],
                'neg_type': 'snomed_sibling',
                'parent_cui': str(parent_cui)
            })

    return triplets
```

**Expected yield:** ~1–2M triplets from SNOMED sibling pairs
(SNOMED has ~370K concepts, ~200K have siblings, each pair generates ~3 triplets)

### 6.5 Synthetic query generation (main scale-up)

#### 6.5.1 Why constrained generation, not free generation

Unconstrained LLM query generation produces queries that are lexically similar
to the passage (the LLM copies terms from the passage). This creates easy training
problems and a **data artifact** — the model learns to match query-document surface
overlap rather than semantic equivalence.

**The fix:** explicitly instruct the LLM to use DIFFERENT surface forms,
constrained by UMLS synonyms extracted from the passage.

#### 6.5.2 Passage annotation pipeline

Before generating queries, annotate every passage with its UMLS concepts:

```python
# Option 1: QuickUMLS (fast, approximate)
# pip install quickumls
from quickumls import QuickUMLS
matcher = QuickUMLS('/path/to/quickumls_install')

def annotate_passage(text):
    matches = matcher.get_matches(text, best_match=True, ignore_syntax=False)
    cuis = set()
    for match_set in matches:
        for match in match_set:
            if match['similarity'] >= 0.9:
                cuis.add(match['cui'])
    return list(cuis)

# Option 2: scispaCy + UMLS linker (slower, more accurate)
# pip install scispacy
# python -m spacy download en_core_sci_lg
import spacy
nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True,
                                         "linker_name": "umls"})

def annotate_passage_spacy(text):
    doc = nlp(text)
    cuis = set()
    for ent in doc.ents:
        for linker_result in ent._.kb_ents:
            if linker_result[1] >= 0.85:  # confidence threshold
                cuis.add(linker_result[0])
    return list(cuis)
```

**DECISION:** Use QuickUMLS for bulk annotation (speed), scispaCy for the
highest-quality subset (final training data). QuickUMLS processes ~10K passages/sec
on CPU, adequate for the full PubMed corpus.

#### 6.5.3 Query generation prompt

```python
QUERY_GEN_PROMPT = """You are generating medical information retrieval queries for training a search model.

Given a medical passage, generate a realistic search query that:
1. Seeks the information in this passage
2. Uses DIFFERENT surface forms than those in the passage where possible
3. Matches the clinical role specified

Available alternative terms for concepts in this passage:
{synonym_map}

Passage:
{passage}

Clinical role: {role}
Role descriptions:
- physician: uses abbreviations, clinical shorthand, terse (e.g. "STEMI mgmt options")
- patient: lay language, may be informal (e.g. "what to do after a heart attack")
- researcher: formal full terminology (e.g. "pharmacological interventions in STEMI")
- nurse: clinical but accessible (e.g. "nursing care for acute myocardial infarction")
- student: educational framing (e.g. "how is myocardial infarction treated")

Return ONLY the query string. No explanation, no quotes, no formatting.
Query:"""

def generate_query(passage, cuis, umls_lookup, model_client, role):
    # Build synonym map: original_term → [alternative_terms]
    synonym_map = {}
    for cui in cuis:
        strings = umls_lookup.get_all_strings(cui)
        if len(strings) > 1:
            # Find which strings appear in passage
            in_passage = [s for s in strings if s.lower() in passage.lower()]
            not_in_passage = [s for s in strings if s.lower() not in passage.lower()]
            if in_passage and not_in_passage:
                synonym_map[in_passage[0]] = not_in_passage[:3]

    if not synonym_map:
        return None  # skip passages where we can't enforce cross-vocab generation

    prompt = QUERY_GEN_PROMPT.format(
        synonym_map=str(synonym_map),
        passage=passage[:800],  # truncate for context efficiency
        role=role
    )

    return model_client.complete(prompt, max_tokens=100, temperature=0.7)
```

**LLM choice for generation:**
- `meta-llama/Llama-3.1-70B-Instruct` — best quality, use for 20% of data
- `meta-llama/Llama-3.1-8B-Instruct` — good quality, use for 80% of data
- `mistralai/Mistral-7B-Instruct-v0.3` — backup if Llama unavailable

**OPEN QUESTION:** Budget for LLM API calls vs self-hosted inference.
At ~300 tokens per generation call (prompt + query output):
- 5M queries × 300 tokens = 1.5B tokens
- At Llama 3.1 70B via Fireworks/Together: ~$0.0009/1K tokens ≈ $1,350 total
- Self-hosted on 4×A100: ~3 days of generation at ~2K queries/min

Recommendation: self-host for cost at scale. Use API for prototyping first 50K.

#### 6.5.4 Source passages

**PubMed abstracts (primary — 35M abstracts)**
```bash
# Download PubMed baseline
wget -r ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
# Process into passages (split abstracts at ~200 word boundaries with overlap)
```

**PMC full text (secondary — 6M articles)**
```bash
# Download PMC Open Access subset
wget ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
# Split into paragraphs, filter to body text sections
```

**MIMIC-IV clinical notes (tertiary — requires PhysioNet credentialing)**
- URL: physionet.org/content/mimic-iv-note/
- Requires: completing CITI training + data use agreement (~2 hours)
- Data: 2.3M clinical notes (discharge summaries, radiology, nursing, etc.)
- Process: de-identify (already done in MIMIC-IV), split into passages of 256–512 tokens

**OPEN QUESTION:** MIMIC access timeline. If PhysioNet credentialing is not yet
complete, start with PubMed + PMC only. MIMIC adds clinical note coverage but
is not strictly necessary for the first model version.

#### 6.5.5 Generation batching strategy

```python
# Process in batches, roles distributed to create balanced training data
ROLES = ['physician', 'patient', 'researcher', 'nurse', 'student']
ROLE_WEIGHTS = [0.25, 0.30, 0.20, 0.15, 0.10]  # patient-centric bias

# For each passage:
# - Generate 1 query per passage (sufficient for scale)
# - Role sampled per ROLE_WEIGHTS
# - If passage has <3 unique CUIs with alternatives: skip
#   (ensures meaningful cross-vocab constraint is possible)

# Quality filter AFTER generation:
# - Query length: 5–100 characters
# - Query must not contain any 5+ character string from passage (approximate)
#   [catches cases where LLM ignored instructions]
# - Deduplicate with MinHash LSH (avoid near-duplicate queries)
```

### 6.6 Hard negative mining

Hard negatives are the second most important quality lever after the positive pairs.
Pipeline runs in stages:

#### Stage 1 negatives: BM25 (fast, run first, good enough for Stage 1 training)

```python
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('/path/to/index')

def mine_bm25_negatives(query, positive_id, k=20):
    hits = searcher.search(query, k=k)
    negatives = [h for h in hits if h.docid != positive_id]
    return negatives[:3]  # keep top-3 BM25 negatives
```

#### Stage 2 negatives: Dense retrieval (better, run after Stage 1 training)

```python
# After training a first-pass Stage 1 model, use it to mine harder negatives
# Passages that score high by the model but are NOT relevant

def mine_dense_negatives(query_embedding, positive_id, index, k=50):
    scores, ids = index.search(query_embedding, k=k)
    negatives = [(id, score) for id, score in zip(ids, scores)
                 if id != positive_id]
    # Top dense negatives are hard (model thinks they're relevant but they're not)
    return negatives[:3]
```

#### Stage 3 negatives: SNOMED siblings (already described in 6.4)

For the final training data, each triplet should have:
- 1 positive passage
- 1 BM25 hard negative (lexically similar, wrong answer)
- 1 dense hard negative (semantically similar by current model, wrong answer)
- 1 SNOMED sibling negative where available (structurally grounded, wrong answer)

The final loss uses all three negatives. This is called **multi-negative** training
and substantially improves model quality over single-negative training.

### 6.7 Final dataset construction

```python
# Merge all sources with quality filters
import pandas as pd

def build_final_dataset(sources, output_path):
    dfs = []
    for source_path, source_name in sources:
        df = pd.read_parquet(source_path)
        df['source'] = source_name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Quality filters
    combined = combined[
        (combined['query'].str.len() >= 5) &
        (combined['query'].str.len() <= 300) &
        (combined['positive'].str.len() >= 50) &
        (combined['negative'].str.len() >= 50) &
        (combined['query'] != combined['positive'].str[:combined['query'].str.len()])
    ]

    # Deduplicate on (query, positive) pairs
    combined = combined.drop_duplicates(subset=['query', 'positive'])

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    combined.to_parquet(output_path)
    print(f"Final dataset: {len(combined)} triplets")

    # Print breakdown
    print(combined['source'].value_counts())
```

**Target composition:**

| Source | Triplets | % of Total |
|---|---|---|
| Synthetic (PubMed, physician role) | 2.0M | 22% |
| Synthetic (PubMed, patient role) | 2.5M | 27% |
| Synthetic (PubMed, researcher role) | 1.5M | 16% |
| Synthetic (clinical notes, mixed roles) | 1.0M | 11% |
| UMLS pair → passage | 1.0M | 11% |
| SNOMED sibling triplets | 0.8M | 9% |
| Real curated (MedEmbed, BioASQ, TREC) | 0.5M | 5% |
| **Total** | **~9.3M** | 100% |

---

## 7. Training Pipeline

### 7.1 Framework decision

**Use:** `ragatouille` or the Stanford ColBERT library directly.

```bash
# Stanford ColBERT (official, most control)
pip install colbert-ai

# Alternative: pylate (newer, cleaner API)
pip install pylate
```

**Recommendation:** Use `pylate` for Stages 1 and 2 (cleaner training loop, better
HuggingFace integration), then switch to Stanford ColBERT for indexing and serving.

`pylate` supports:
- Custom backbone (any HuggingFace model)
- MRL loss natively
- Multi-GPU training via HuggingFace Accelerate
- Standard HuggingFace Trainer interface

**OPEN QUESTION:** Test both libraries on a small pilot (50K triplets, 1K steps)
before committing to either. Validate that BioClinical-ModernBERT loads correctly
and that MaxSim scores are numerically stable.

### 7.2 Stage 1: Contrastive warm-up on UMLS synonym pairs

**Goal:** warm the token embeddings toward concept-level synonym alignment before
introducing the complexity of passage retrieval and hard negatives.

**Data:** UMLS cross-vocabulary pairs only (not full triplets)
- Use `clinical_lay` pairs primarily (~800K)
- MultipleNegativesRankingLoss with in-batch negatives

**Config:**
```yaml
# configs/stage1.yaml
model: thomas-sounack/BioClinical-ModernBERT-base
colbert_dim: 128
query_maxlen: 64
doc_maxlen: 512

training:
  loss: mnrl                    # MultipleNegativesRankingLoss
  batch_size: 512               # large is critical — more in-batch negatives
  lr: 3e-5
  warmup_steps: 1000
  total_steps: 100000
  fp16: false
  bf16: true                    # BioClinical was trained in bf16

data:
  path: data/processed/umls_pairs/clinical_lay.parquet
  text_column_a: str1           # clinical term
  text_column_b: str2           # lay term / abbreviation
```

**What to monitor:**
- Loss should decrease smoothly. If it diverges: lower lr to 1e-5.
- Evaluate on a held-out synonym matching task every 10K steps:
  given `"heart attack"`, does the model rank `"myocardial infarction"` passages
  above `"cardiac arrhythmia"` passages?
- Save checkpoint every 20K steps.

**Expected duration:** ~12 hours on 4×A100 (batch 512, bf16)

### 7.3 Stage 2: ColBERT fine-tuning with hard negatives

**Goal:** main training stage. Teach the model full retrieval — not just synonym
matching but passage-level relevance with hard negatives.

**Data:** Full triplet dataset (9.3M triplets), all sources.
Oversample SNOMED sibling negatives (weight × 2) in the sampling strategy.

**Loss function:**
```python
import torch
import torch.nn.functional as F

class ColBERTLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_reps, pos_reps, neg_reps):
        # query_reps: [B, Q_len, dim]
        # pos_reps:   [B, D_len, dim]
        # neg_reps:   [B, D_len, dim] or [B, n_neg, D_len, dim]

        pos_scores = self.maxsim(query_reps, pos_reps)     # [B]
        neg_scores = self.maxsim(query_reps, neg_reps)     # [B] or [B, n_neg]

        if neg_scores.dim() == 2:
            # Multiple negatives per query — use max (hardest negative)
            neg_scores = neg_scores.max(dim=-1).values     # [B]

        # Pairwise softmax loss (same as ColBERTv2)
        scores = torch.stack([pos_scores, neg_scores], dim=-1) / self.temperature
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(scores, labels)

        return loss

    def maxsim(self, Q, D):
        # Q: [B, Q_len, dim], D: [B, D_len, dim]
        scores = torch.bmm(Q, D.transpose(1, 2))   # [B, Q_len, D_len]
        return scores.max(dim=-1).values.sum(dim=-1) # [B]
```

**Config:**
```yaml
# configs/stage2.yaml
model: checkpoints/stage1/final              # start from Stage 1
colbert_dim: 128
query_maxlen: 64
doc_maxlen: 512

training:
  loss: colbert_pairwise
  temperature: 1.0
  batch_size: 64                  # limited by GPU memory (full sequences)
  gradient_accumulation: 8        # effective batch = 512
  lr: 1e-5                        # lower than Stage 1
  warmup_steps: 2000
  total_steps: 400000             # ~4–5 epochs over 9M triplets
  bf16: true
  gradient_checkpointing: true    # needed for large model + 512 doc tokens

data:
  path: data/processed/final_triplets/
  n_negatives: 3                  # 1 BM25 + 1 dense + 1 SNOMED where available
  sampling_weights:
    synthetic_pubmed: 1.0
    synthetic_clinical: 1.5       # upweight clinical notes (harder, rarer)
    snomed_sibling: 2.0           # upweight structurally hard negatives
    real_curated: 2.0             # upweight high-quality real data
```

**What to monitor:**
- Eval on NFCorpus and BioASQ every 50K steps (quick proxies)
- If overfitting detected (train loss ↓, eval nDCG↓): increase weight decay or
  reduce training steps
- Save checkpoint every 50K steps, keep best-by-eval checkpoint

**Expected duration:** ~4 days on 4×A100 (base model), ~7 days (large model)

### 7.4 MRL (Matryoshka Representation Learning)

Add MRL loss as an auxiliary objective in Stage 2.

```python
class MRLColBERTLoss(torch.nn.Module):
    def __init__(self, dims=(32, 64, 128), weights=(0.1, 0.2, 0.7)):
        super().__init__()
        assert len(dims) == len(weights)
        self.dims = dims
        self.weights = weights
        self.colbert_loss = ColBERTLoss()

    def forward(self, query_reps, pos_reps, neg_reps):
        # query_reps, pos_reps, neg_reps are full dim=128 representations
        total_loss = 0.0
        for dim, weight in zip(self.dims, self.weights):
            # Truncate to first `dim` dimensions
            q = F.normalize(query_reps[..., :dim], dim=-1)
            p = F.normalize(pos_reps[..., :dim], dim=-1)
            n = F.normalize(neg_reps[..., :dim], dim=-1)
            loss = self.colbert_loss(q, p, n)
            total_loss += weight * loss
        return total_loss
```

The weight distribution `(0.1, 0.2, 0.7)` heavily weights the full dimension.
This ensures full-dim quality is not sacrificed for small-dim efficiency.
Ablate this — some papers use equal weights.

### 7.5 Stage 3: KL distillation from cross-encoder teacher

**Goal:** close the quality gap between ColBERT and a full cross-encoder.
The teacher scores the same (query, passage) pairs; the student (ColBERT) learns
to reproduce those scores via KL divergence.

#### 7.5.1 Build the teacher

```python
# Fine-tune BioClinical-ModernBERT-large as a cross-encoder
# Input: "[CLS] query [SEP] passage [SEP]"
# Output: single relevance score via linear head on [CLS]

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

teacher = AutoModelForSequenceClassification.from_pretrained(
    "thomas-sounack/BioClinical-ModernBERT-large",
    num_labels=1
)
# Fine-tune on real curated triplets (850K) for ~3 epochs
# Use BCEWithLogitsLoss with binary relevance labels
```

**OPEN QUESTION:** Whether to use BioClinical-large or an existing medical
cross-encoder like `ncbi/MedCPT-Cross-Encoder`. MedCPT is already trained on
PubMed citation pairs. If its quality is sufficient, skip training a teacher
and save compute. Evaluate: run both on held-out BioASQ examples and pick whichever
has higher nDCG@10.

#### 7.5.2 Distillation training

```python
class KLDistillationLoss(torch.nn.Module):
    def forward(self, student_scores, teacher_scores):
        # student_scores: [B, n_passages] — ColBERT MaxSim scores
        # teacher_scores: [B, n_passages] — cross-encoder logits

        # Min-max normalize both distributions (JaColBERT/Jina trick)
        # This is CRITICAL — raw MaxSim scores range [0, n_query_tokens]
        # while cross-encoder scores are logits in [-10, 10]
        # Without normalization, KL divergence is numerically unstable

        s_min, s_max = student_scores.min(dim=-1, keepdim=True).values, \
                       student_scores.max(dim=-1, keepdim=True).values
        t_min, t_max = teacher_scores.min(dim=-1, keepdim=True).values, \
                       teacher_scores.max(dim=-1, keepdim=True).values

        student_norm = (student_scores - s_min) / (s_max - s_min + 1e-8)
        teacher_norm = (teacher_scores - t_min) / (t_max - t_min + 1e-8)

        student_log_probs = F.log_softmax(student_norm, dim=-1)
        teacher_probs     = F.softmax(teacher_norm, dim=-1)

        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
```

**Data for Stage 3:** Use only the highest-quality subset:
- All 850K real curated triplets
- Top 500K synthetic triplets by teacher score variance
  (teacher assigns a clear winner = more informative distillation signal)
- Total: ~1.35M triplets, ~3 epochs, ~100K steps

### 7.6 Multi-GPU training setup

```python
# launch command
# torchrun --nproc_per_node=4 scripts/training/stage2_colbert_finetune.py

from accelerate import Accelerator
accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=8)

# For ColBERT specifically: documents in a batch should NOT all come from the
# same document set. Shuffle at the batch level aggressively.
# In-batch negatives are only useful if they're actually different documents.
```

---

## 8. Evaluation Suite

> This section defines exactly what SOTA means for this project.
> Every metric here must appear in the paper.

### 8.1 Primary benchmarks

#### NFCorpus (BEIR)
- **What:** Nutrition/health queries → NutritionFacts.org documents
- **Why it matters:** Standard biomedical IR benchmark, well-understood baseline
- **Metric:** nDCG@10 (primary), Recall@100 (secondary)
- **Current SOTA reference:** dense models ~40–44, BM25 ~34
- **Data source:** `ir-datasets` library or BEIR HuggingFace datasets
- **Note:** This benchmark has high BM25 overlap — cross-vocab gains may be modest.
  Use it but don't anchor the paper narrative on it.

#### TREC-COVID (BEIR)
- **What:** 50 COVID-19 queries against CORD-19 corpus (192K papers)
- **Why it matters:** Clinical domain, time-constrained (good real-world proxy)
- **Metric:** nDCG@10
- **Current SOTA reference:** strong models ~85–88 nDCG@10
- **Note:** High baseline — harder to show gains. Parity with SOTA is acceptable here.

#### BioASQ (BEIR version)
- **What:** Biomedical expert questions → PubMed articles
- **Why it matters:** Expert-generated queries, real clinical information need
- **Metric:** nDCG@10
- **Current SOTA reference:** ~50–58 depending on model class
- **Note:** BioASQ queries are technical, biomedical-vocabulary heavy.
  Our model should show strong gains here because we train on UMLS clinical terms.

#### TREC Clinical Trials 2021 / 2022
- **What:** Patient descriptions (clinical notes) as queries → ClinicalTrials.gov documents
- **Why it matters:** Real-world clinical use case. Queries are long (full notes).
  Tests the 8192 token window directly.
- **Metric:** nDCG@10, NDCG@1000 (eligibility-based retrieval)
- **Current SOTA reference:** ~55–62 nDCG@10 for dense models
- **Note:** This benchmark specifically tests long-query clinical IR.
  BioClinical's long context + our training should give strong gains here.
  This may be the benchmark where we show the largest delta.

#### R2MED (NEW — May 2025)
- **What:** Reasoning-intensive medical retrieval. Relevance defined by whether
  a document supports a clinical conclusion NOT directly stated in the query.
- **Why it matters:** No late-interaction model has published results on this.
  Being first is a free contribution.
- **Metric:** nDCG@10, MAP@10
- **Data source:** https://arxiv.org/abs/2505.14558 — check for HuggingFace release
- **OPEN QUESTION:** Confirm R2MED data is publicly available for download.
  If not released yet, email authors or use it as a future work discussion point.

#### MIRAGE (end-to-end RAG benchmark)
- **What:** Medical QA accuracy when using a retriever + LLM (GPT-3.5 or open model)
- **Subsets:** MMLU-Med, MedQA-US, MedMCQA, PubMedQA, BioASQ-Y/N
- **Why it matters:** Proves retrieval quality translates to downstream task performance
- **Metric:** Accuracy (% correct answers)
- **Note:** Use a fixed LLM (e.g., GPT-3.5-turbo) so only the retriever varies.
  This isolates retrieval quality contribution.
- **Implementation:** `Xiong et al. 2024` MIRAGE benchmark — check for HuggingFace
  evaluation harness.

### 8.2 Cross-vocabulary analysis (novel evaluation — paper contribution)

This is the evaluation that no prior work has done and that directly tests the
core hypothesis.

**Construct a cross-vocabulary test set:**

```python
# For each eval query, determine whether it uses:
# - Same vocabulary as relevant documents (clinical ↔ clinical)
# - Different vocabulary (lay ↔ clinical, abbrev ↔ clinical)

# Method: annotate queries and documents with UMLS CUIs + vocabulary source
# Query "heart attack" → CHV (consumer health vocab)
# Document "acute myocardial infarction" → SNOMEDCT_US

def classify_query_doc_pair(query_cuis, query_sources, doc_cuis, doc_sources):
    shared_cuis = query_cuis & doc_cuis
    if not shared_cuis:
        return 'no_overlap'

    # Check if shared concepts use same vocabulary
    for cui in shared_cuis:
        q_src = query_sources[cui]
        d_src = doc_sources[cui]
        if q_src != d_src:
            return 'cross_vocab'   # ← this is the hard case
    return 'same_vocab'

# Report nDCG@10 broken down by:
# - same_vocab queries (baseline, should be similar across models)
# - cross_vocab queries (our model should show large gains here)
```

**This analysis directly proves the core hypothesis.** If MedColBERT shows large
gains on cross_vocab but similar performance on same_vocab, the paper's story is
airtight. If gains are uniform, the story shifts to "scale helps everywhere."

### 8.3 Evaluation code

```python
# Standardized evaluation using BEIR / ir-measures
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models

# Wrap MedColBERT in BEIR-compatible interface
class MedColBERTRetriever:
    def __init__(self, model_path, batch_size=128):
        self.model = MedColBERT.from_pretrained(model_path)
        self.batch_size = batch_size

    def search(self, corpus, queries, top_k, *args, **kwargs):
        # encode all documents
        doc_embeddings = self.encode_corpus(corpus)
        results = {}
        for qid, query in queries.items():
            q_emb = self.encode_query(query)
            scores = self.maxsim_scores(q_emb, doc_embeddings)
            top_k_ids = scores.topk(top_k).indices
            results[qid] = {doc_ids[i]: scores[i].item() for i in top_k_ids}
        return results

# Run eval
retriever = EvaluateRetrieval(MedColBERTRetriever("checkpoints/final"))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [10, 100, 1000])
```

---

## 9. Ablation Studies

> Ablations are the mechanism by which reviewers verify your claims.
> Design them before running experiments, not after.

### Table: Ablation matrix

| Model variant | Purpose | Expected result |
|---|---|---|
| BM25 | Lower bound | nDCG ~34 NFCorpus, ~68 TREC-COVID |
| MedEmbed-large-v1 (dense) | Your prior SOTA | ~38–42 |
| ColBERTv2 (generic, zero-shot) | Shows late-interaction baseline | ~35–38 |
| ModernBERT → ColBERT (no domain) | Isolates backbone contribution | ~39–44 |
| BioClinical → ColBERT (no UMLS data) | What you'd get without this paper | ~43–47 |
| BioClinical → ColBERT + real data only | Shows scale matters | ~45–49 |
| BioClinical → ColBERT + synthetic, no SNOMED neg | Isolates SNOMED contribution | ~46–50 |
| **MedColBERT-base (full system)** | **Your model** | **~50–55** |
| MedColBERT-large (full system) | Scale-up | **~54–60** |

Run every row. The most important gaps to demonstrate:
1. Row 5 → Row 6: synthetic data helps substantially
2. Row 6 → Row 7: SNOMED negatives help beyond random negatives
3. Row 5 → Row 8: full system vs domain backbone alone

### MRL ablation

| Dims | NFCorpus | TREC-CT | Latency (relative) |
|---|---|---|---|
| dim=32 | ? | ? | 0.25× |
| dim=64 | ? | ? | 0.50× |
| dim=128 (full) | ? | ? | 1.00× |

Show the quality-speed tradeoff. This is a practical contribution for deployment.

### Vocabulary-split ablation

| Query type | BM25 | Dense | MedColBERT |
|---|---|---|---|
| Same-vocab (clinical ↔ clinical) | ? | ? | ? |
| Cross-vocab (lay ↔ clinical) | ? | ? | ? |
| Abbreviation queries | ? | ? | ? |

This is the clearest demonstration of what your model uniquely solves.

---

## 10. Compute Budget

### Training compute

| Stage | Hardware | Duration | Cost estimate |
|---|---|---|---|
| Stage 1 (base) | 4×A100 40GB | 12 hours | ~$50 |
| Stage 2 (base) | 4×A100 40GB | 96 hours | ~$400 |
| Stage 3 (base) | 4×A100 40GB | 24 hours | ~$100 |
| Stage 1 (large) | 4×A100 80GB | 18 hours | ~$75 |
| Stage 2 (large) | 4×A100 80GB | 160 hours | ~$650 |
| Stage 3 (large) | 4×A100 80GB | 36 hours | ~$150 |
| **Total training** | | ~15 days | **~$1,425** |

Cost based on Lambda Labs / Vast.ai A100 pricing (~$1.50/GPU/hr for 80GB).
Significantly cheaper than cloud (AWS/GCP/Azure).

### Data generation compute

| Task | Scale | Duration | Cost |
|---|---|---|---|
| QuickUMLS annotation (PubMed) | 35M abstracts | 3 days CPU | ~$20 |
| LLM query generation (8B model) | 5M queries | 4 days on 2×A100 | ~$150 |
| BM25 negative mining | 9M triplets | 1 day CPU | ~$10 |
| Dense negative mining | 9M triplets | 1 day on 1×A100 | ~$25 |
| **Total data** | | ~9 days | **~$205** |

**Total project compute budget: ~$1,630**

This is extremely lean for a top-conference submission. Cloud alternatives
(AWS p4d.24xlarge) would cost ~6–8× more.

**OPEN QUESTION:** GPU access. Options:
1. Lambda Labs / Vast.ai (cheapest, ~$1.50/GPU/hr A100)
2. RunPod (similar pricing)
3. Academic compute grants (apply to NSF ACCESS or equivalent — free)
4. HuggingFace research access (worth emailing, you have MedEmbed credibility)

---

## 11. Open Questions and Risks

> These are items that require a human decision or additional investigation.
> Annotated with [BLOCKER] if they must be resolved before proceeding,
> [IMPORTANT] if they affect quality, [NICE-TO-HAVE] if optional.

### [BLOCKER] — GPU access

Must secure before starting. Pipeline will stall at Stage 2 training without it.
Minimum viable: 4×A100 40GB, 2 weeks of access.

### [BLOCKER] — UMLS license

UMLS requires a (free) license from the National Library of Medicine.
URL: https://www.nlm.nih.gov/research/umls/
Approval is automated and instant for US researchers, may take a few days
for international. Apply immediately if not already done.

### [BLOCKER] — Verify BioClinical-ModernBERT loads with ColBERT config

Before building the full pipeline, run this sanity check:
```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("thomas-sounack/BioClinical-ModernBERT-base",
                                   attn_implementation="flash_attention_2",
                                   torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("thomas-sounack/BioClinical-ModernBERT-base")

# Test forward pass
inputs = tok("Patient presented with chest pain.", return_tensors="pt")
outputs = model(**inputs)
# Should output: last_hidden_state of shape [1, seq_len, 768]
print(outputs.last_hidden_state.shape)  # expect [1, N, 768]
```

If this fails, debug before investing in the full pipeline.

### [IMPORTANT] — PhysioNet / MIMIC credentialing

Clinical notes from MIMIC are the most domain-specific training data available.
Begin credentialing immediately (takes 1–2 weeks). If not available in time,
proceed with PubMed + PMC only. MIMIC adds ~1M clinical note triplets.
The model will still be competitive without it.

### [IMPORTANT] — R2MED data availability

R2MED (arxiv 2505.14558, May 2025) may not have released their dataset yet.
Check: https://arxiv.org/abs/2505.14558 and contact authors if needed.
If unavailable, replace with TREC 2023 Clinical Trials as 5th benchmark.

### [IMPORTANT] — Hard negative quality validation

Before running Stage 2 at full scale, manually inspect 100 triplets from each
negative source (BM25, dense, SNOMED). Verify:
- Negatives are actually non-relevant (not false negatives)
- SNOMED sibling negatives make clinical sense
- Synthetic queries actually use different vocabulary than the positive passage

SNOMED siblings can sometimes be too easy (completely different medical domain)
or impossible to distinguish (very similar concepts). Spot check: sample 50
SNOMED triplets and label them manually. Target: >80% should be
"genuinely hard" — a human doctor would need to read carefully to distinguish.

### [IMPORTANT] — Decontamination check

Verify that no test set passages appear in training data.
```python
# Fingerprint all eval passages with MinHash
# Check against all training passages
# Any match above Jaccard 0.8 must be removed from training

from datasketch import MinHash, MinHashLSH

def decontaminate(eval_passages, train_passages):
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    # index eval passages
    for i, p in enumerate(eval_passages):
        m = MinHash(num_perm=128)
        for w in p.split(): m.update(w.encode())
        lsh.insert(f"eval_{i}", m)
    # query each training passage
    to_remove = set()
    for i, p in enumerate(train_passages):
        m = MinHash(num_perm=128)
        for w in p.split(): m.update(w.encode())
        if lsh.query(m):
            to_remove.add(i)
    return to_remove
```

### [NICE-TO-HAVE] — ICD-10 codes as additional ontology signal

ICD-10 provides billing code → clinical description mappings.
These are shorter, more structured than SNOMED descriptions and could add
another ~80K high-quality synonym pairs. Lower priority — add after core pipeline
is working.

### [NICE-TO-HAVE] — Drug name synonyms via RxNorm

RxNorm (part of UMLS) provides brand name ↔ generic name ↔ ingredient mappings.
Example: "Tylenol" ↔ "acetaminophen" ↔ "paracetamol".
Drug name disambiguation is a significant failure mode in clinical IR.
Mine RxNorm pairs separately and add as a data source.

### [NICE-TO-HAVE] — Triton kernel for inference

For production deployment, a fused Triton MaxSim kernel reduces HBM bandwidth
during batch scoring. Not needed for training or evaluation. Implement only if
the paper includes a latency benchmark section.

### Risk: Hypothesis falsification

If the ablation in Section 3.3 shows that domain pretraining alone (without UMLS
supervision) achieves similar performance, the paper contribution is weakened.

**Mitigation:** Run a quick 200K-step pilot of "BioClinical → ColBERT, no UMLS data"
on a 100K triplet subset before committing to the full pipeline. If it's within
1–2 nDCG points of the UMLS-supervised model on BioASQ, rethink the framing —
the contribution becomes the synthetic data scale, not the ontology structure.
That is still publishable, just with a different narrative.

---

## 12. Timeline

> Assumes full-time effort (one person or one agent at a time).
> Weeks are calendar weeks, not work weeks.

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Environment setup, UMLS license, BioClinical sanity check | Working dev environment |
| 1–2 | Extract UMLS pairs, SNOMED triplets | ~5M raw pairs in parquet |
| 2–3 | Build PubMed passage index + QuickUMLS annotation | Indexed + annotated corpus |
| 3–5 | Synthetic query generation (5M queries) | Raw synthetic dataset |
| 5 | BM25 negative mining, quality filtering, decontamination | 9M clean triplets |
| 6 | Stage 1 training (base model) | Stage 1 checkpoint |
| 6–8 | Stage 2 training (base model) | Base model checkpoint |
| 8 | Evaluate base model on all benchmarks | Preliminary numbers |
| 8 | Dense negative mining from base model | Improved triplets |
| 8–9 | Stage 2 re-training with dense negatives | Improved base model |
| 9 | Stage 3 KL distillation (base model) | Final base model |
| 9–11 | Repeat Stages 2–3 for large model | Final large model |
| 11 | Full evaluation suite, ablation table | All numbers |
| 11–12 | Cross-vocabulary analysis | Novel analysis results |
| 12–16 | Paper writing | Draft |
| 16 | SIGIR 2026 submission | Submitted |

**Critical path:** Data pipeline (Weeks 1–5) → Stage 2 training (Weeks 6–8) →
Evaluation (Week 8). Everything else is parallel or downstream.

---

## 13. References and Prior Art

### Must-read papers (read before writing any code)

| Paper | Why |
|---|---|
| ColBERT (Khattab & Zaharia, 2020) | Original late interaction paper |
| ColBERTv2 (Santhanam et al., 2021) | Denoised supervision + residual compression |
| PLAID (Santhanam et al., 2022) | Production ColBERT inference engine |
| BioClinical-ModernBERT (Sounack et al., 2025) | Your backbone. arxiv 2506.10896 |
| MedEmbed (Balachandran, 2024) | Your prior work, the baseline |
| BEIR (Thakur et al., 2021) | Primary benchmark suite |
| R2MED (arxiv 2505.14558, 2025) | New benchmark — read carefully |
| ModernBERT+ColBERT biomedical (arxiv 2510.04757, 2025) | Most related concurrent work — must differentiate |
| Jina-ColBERT-v2 (Jha et al., 2024) | Best practices for ColBERT training recipe |
| MRL (Kusupati et al., 2022) | Matryoshka representation learning |

### Key related work to differentiate from

**ModernBERT + ColBERT (Oct 2025, arxiv 2510.04757):**
Uses generic ColBERTv2 (untrained on medical data) as a reranker over
ModernBERT-retrieved candidates. Not a domain-specific trained model.
Not trained end-to-end on medical data. Our model is trained from scratch
specifically for medical IR, not adapted from a general ColBERT.

**MedCPT (Jin et al., 2023):**
Dense bi-encoder trained on PubMed click-through data. Strong medical dense
retrieval baseline. Does not use late interaction. Use MedCPT cross-encoder
as our distillation teacher (Section 7.5).

**BM25 + MetaMap / QuickUMLS query expansion:**
Traditional approach: expand query terms with UMLS synonyms before BM25 retrieval.
Our approach is fundamentally different — we learn the synonym alignment inside
the token embeddings rather than doing symbolic expansion at query time.
Symbolic expansion is brittle (can over-expand), ours is learned and context-aware.

### Implementation resources

| Resource | URL |
|---|---|
| Stanford ColBERT | https://github.com/stanford-futuredata/ColBERT |
| pylate (cleaner API) | https://github.com/lightonai/pylate |
| BEIR benchmark | https://github.com/beir-cellar/beir |
| ir-datasets | https://ir-datasets.com/ |
| QuickUMLS | https://github.com/Georgetown-IR-Lab/QuickUMLS |
| scispaCy | https://scispacy.apps.allenai.org/ |
| UMLS download | https://www.nlm.nih.gov/research/umls/ |
| SNOMED CT download | https://www.snomed.org/snomed-ct/get-snomed |
| MIMIC-IV | https://physionet.org/content/mimic-iv-note/ |
| PubMed bulk download | https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ |
| PMC bulk download | https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/ |
| BioClinical-ModernBERT | https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base |
| MedCPT cross-encoder | https://huggingface.co/ncbi/MedCPT-Cross-Encoder |
| TREC Clinical Trials | https://trec.nist.gov/data/clinical2021.html |

---

## Appendix A: Minimal Working Example

Before running anything at scale, verify the core training loop on 1000 triplets.

```python
"""
Minimal ColBERT training loop sanity check.
Run this before investing in the full pipeline.
Expected: loss decreases over 100 steps, no NaN/Inf values.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BACKBONE = "thomas-sounack/BioClinical-ModernBERT-base"
DIM = 128

tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
backbone = AutoModel.from_pretrained(BACKBONE, torch_dtype=torch.bfloat16).cuda()
projection = torch.nn.Linear(768, DIM, bias=False).cuda().to(torch.bfloat16)

optimizer = torch.optim.AdamW(
    list(backbone.parameters()) + list(projection.parameters()), lr=1e-5)

query    = "heart attack treatment"
positive = "Acute myocardial infarction management includes reperfusion therapy."
negative = "Atrial fibrillation is managed with rate control or rhythm control."

def encode(text, maxlen):
    enc = tokenizer(text, max_length=maxlen, truncation=True,
                    padding='max_length', return_tensors='pt').to('cuda')
    out = backbone(**enc).last_hidden_state
    out = projection(out)
    out = F.normalize(out, dim=-1)
    out = out * enc['attention_mask'].unsqueeze(-1)
    return out

def maxsim(q, d):
    s = torch.einsum('id,jd->ij', q.squeeze(0), d.squeeze(0))
    return s.max(dim=-1).values.sum()

for step in range(100):
    optimizer.zero_grad()
    q = encode(query, 64)
    p = encode(positive, 128)
    n = encode(negative, 128)
    pos_score = maxsim(q, p)
    neg_score = maxsim(q, n)
    scores = torch.stack([pos_score, neg_score]).unsqueeze(0)
    loss = F.cross_entropy(scores, torch.zeros(1, dtype=torch.long, device='cuda'))
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, "
              f"pos={pos_score.item():.3f}, neg={neg_score.item():.3f}")

# Expected: loss decreasing from ~0.7 toward ~0.1
# pos_score should become higher than neg_score
```

---

## Appendix B: Glossary

| Term | Definition |
|---|---|
| ColBERT | Contextualized Late Interaction over BERT. Keeps per-token embeddings, scores via MaxSim. |
| MaxSim | For each query token, find the max similarity to any doc token; sum these maxima. |
| Late interaction | Query and document encoded independently; token-level interaction happens at retrieval time. |
| UMLS | Unified Medical Language System. 3.5M+ biomedical concepts, each with multiple string forms. |
| CUI | Concept Unique Identifier. UMLS's ID for a concept. Same CUI = same concept. |
| SNOMED CT | Hierarchical clinical ontology. Used for its sibling structure to generate hard negatives. |
| CHV | Consumer Health Vocabulary. UMLS subset mapping clinical terms to lay patient language. |
| Hard negative | A training negative the model would likely score as relevant but is not. Essential for learning. |
| SNOMED sibling | Two concepts sharing the same parent in SNOMED (e.g., Type 1 and Type 2 diabetes). |
| MRL / Matryoshka | Training technique making first k dimensions independently meaningful. Enables variable-dim inference. |
| nDCG@10 | Normalized Discounted Cumulative Gain at rank 10. Primary IR metric. Higher = better. |
| BEIR | Benchmarking IR. Standardized zero-shot IR benchmark suite (Thakur et al., 2021). |
| KL distillation | Student model learns to reproduce teacher model's score distribution via KL divergence. |
| Bi-encoder | Query and document encoded independently into single vectors. Fast, limited expressiveness. |
| Cross-encoder | Query and document processed jointly. Very accurate, cannot be pre-indexed. |
| MIMIC | Large de-identified ICU clinical notes dataset. Requires PhysioNet credentialing. |
| In-batch negatives | Other examples in the same training batch used as negatives. Free and effective. |
| BM25 | Classical lexical retrieval. Strong baseline. Fails on synonym variance. |