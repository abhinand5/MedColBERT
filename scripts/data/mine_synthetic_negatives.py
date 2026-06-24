"""Synthetic hard-negative + real-anchor mining for MedColBERT ColBERT training.

Implements docs/TRIPLET_MINING_PLAN.md. For every positive (query, passage) in the
cleaned v3g dataset, produces ~11-15 hard negatives:

  * SYNTHETIC (LLM, taxonomy-routed, style-conditioned): each negative fails the query
    in a specified clinical-failure mode (confusable disease, wrong drug in class,
    abbreviation trap, lay/clinical mismatch, wrong context, ...). Includes a
    UMLS-group-grounded `category_aware_synth` tier (same semantic_group, different
    concept) that replaces the broken `category_aware_real` anchor.
  * REAL ANCHOR (no LLM, from the 66k UMLS-annotated PubMed corpus):
      - 2 BM25 hard negatives (real passages sharing query terms, different concept)

Every negative (synthetic AND real) passes a calibrated LLM audit gate that DROPS:
  * false negatives  (ANSWERS_QUERY == TRUE  -- a negative that actually answers the query)
  * incoherent/gibberish passages (GROUNDING <= 2)
plus an intra-sample near-duplicate drop (4-gram overlap >= 0.8).

A held-out 10% real eval split is reserved (its positives are NOT used to generate
negatives) and a BM25 recall@k sanity is computed on it.

TWO BACKENDS (--backend):
  * gemma     : local vLLM Gemma-4-31B at http://localhost:8000/v1. Generation
                temperature=1.0, top_k=64, top_p=0.95, enable_thinking=False (via
                chat_template_kwargs). Audit temperature=0.0.
  * deepseek  : DeepSeek-V4-Flash at https://api.deepseek.com (OpenAI-compatible REST).
                Generation temperature=1.0, top_p=0.95 (NO top_k -- not supported by the
                DeepSeek API), thinking OFF (the `thinking` field is omitted = opt-in).
                Audit temperature=0.0. API key from env DEEPSEEK_API_KEY or the gitignored
                file .deepseek_api_key. Captures token usage for cost estimation.

Usage (Gemma, the production backend):
  # incremental full run in 10k contiguous chunks (disjoint; union = full train set).
  # SAME --tag accumulates into one negatives_v1.parquet via the resumability partial.
  uv run python scripts/data/mine_synthetic_negatives.py --start 0      --limit 10000 --tag v1 --backend gemma
  uv run python scripts/data/mine_synthetic_negatives.py --start 10000  --limit 10000 --tag v1 --backend gemma
  uv run python scripts/data/mine_synthetic_negatives.py --start 20000  --limit 10000 --tag v1 --backend gemma
  ... (50000, then the final tail) ...
  uv run python scripts/data/mine_synthetic_negatives.py --start 50000  --limit 0     --tag v1 --backend gemma   # 0 = rest

  # future two-machine split (L40S + RTX PRO 6000, different vLLM APIs on different hosts):
  #   machine A (L40S, local vLLM):       --start 0     --limit 27000 --tag v1 --backend gemma
  #   machine B (PRO 6000, remote vLLM):  --start 27000 --limit 0     --tag v1 --backend gemma \
  #                                       --base-url http://PRO6000_HOST:8000/v1 --model gemma-4-31B

  # smoke / small test:
  uv run python scripts/data/mine_synthetic_negatives.py --limit 24 --tag smoke --backend gemma
(Resumable: re-running the same command + same --tag skips query_ids in negatives_<tag>_partial.parquet.
 A crashed mid-chunk run resumes by re-running the SAME --start/--limit command.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path("/workspace/MedColBERT")
V3G = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v3g.parquet"
CORPUS_JSON = ROOT / "data/processed/private/passages/real_annotated_500.json"
NEG_DIR = ROOT / "data/processed/private/synthetic/negatives"
KEYFILE = ROOT / ".deepseek_api_key"

# ---- shared sampling params (FIXED, owner-specified) ----
GEN_TEMPERATURE = 1.0
GEN_TOP_K = 64          # gemma only (deepseek omits top_k; uses cfg["gen_top_p"])
AUDIT_TEMPERATURE = 0.0
CHECKPOINT_EVERY = 500
DEV_FRACTION = 0.10
SEED = 20260621
OVERLAP_POSITIVE = 0.6   # 4-gram overlap threshold to treat a corpus passage as the positive / near-dup
OVERLAP_NEAR_DUP = 0.8   # 4-gram overlap between two negatives of the same sample -> drop the dup
BM25_REAL_N = 2          # number of BM25 real anchors per positive (Mitigation 2: real anchor)

# ---- backend configs ----
BACKENDS = {
    "gemma": {
        "base_url": "http://localhost:8000/v1",
        "model": "gemma-4-31B",
        "auth": None,
        "default_workers": 64,
        "supports_top_k": True,
        # Owner-fixed gen sampling: temperature=1.0, top_k=64, top_p=0.95, thinking OFF
        "gen_top_p": 0.95,
        "gen_extra": {"chat_template_kwargs": {"enable_thinking": False}},
        "audit_extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "auth": "bearer_env",          # read key at runtime
        "default_workers": 256,        # DeepSeek allows 2500 concurrency
        "supports_top_k": False,       # DeepSeek API does not accept top_k
        # DeepSeek-recommended sampling for deepseek-v4-flash: temperature=1.0, top_p=1.0
        "gen_top_p": 1.0,
        # thinking ON for quality (DeepSeek is fast+cheap; speed no longer the constraint).
        # Thinking trace lands in message.reasoning_content; final answer in message.content.
        # Thinking tokens count against max_tokens, so gen/audit caps are raised for deepseek.
        "gen_extra": {"thinking": {"type": "enabled"}},
        "audit_extra": {"thinking": {"type": "enabled"}},
    },
}
# DeepSeek gen/audit max_tokens (raised to leave room for the thinking trace + output)
DS_GEN_MAX_TOKENS = 6000
DS_AUDIT_MAX_TOKENS = 1200
# DeepSeek pricing ($/1M tokens) for cost estimation only
DS_PRICE_IN_CACHE_HIT = 0.0028
DS_PRICE_IN_CACHE_MISS = 0.14
DS_PRICE_OUT = 0.28

BACKEND = None        # set in main()
_CLIENT = None        # shared httpx.Client (thread-safe across workers)
_MODEL = None
_TOKENS = {"prompt": 0, "completion": 0, "cached": 0}  # deepseek usage accumulator

# --------------------------------------------------------------------------- #
# Taxonomy: family -> list of (tier, n_negatives), and tier -> (label, spec)
# --------------------------------------------------------------------------- #
FAMILY_ROUTING = {
    # T2E (category_aware_synth, 1/positive) replaces the broken category_aware_real anchor
    # for every family -- UMLS-grounded "same semantic_group, different concept".
    "symptom_to_diagnosis":        [("T1", 2), ("T2A", 4), ("T2C", 3), ("T4B", 3), ("T2E", 1)],
    "biomedical_to_clinical":      [("T1", 2), ("T2A", 2), ("T3C", 4), ("T4A", 2), ("T5A", 2), ("T2E", 1)],
    "generic_clinical_retrieval":  [("T1", 3), ("T2A", 2), ("T3B", 2), ("T4A", 2), ("T5A", 2), ("T2E", 1)],
    "lay_to_clinical":             [("T1", 2), ("T2A", 2), ("T3A", 4), ("T4A", 3), ("T2E", 1)],
    "brand_generic":               [("T1", 2), ("T2B", 5), ("T2D", 3), ("T3B", 2), ("T4A", 1), ("T2E", 1)],
}

FAILURE_MODES = {
    "T1":  ("lexical_trap",
            "FAILURE_MODE: lexical_trap. Write a passage that shares 2-3 surface terms with the QUERY but is about a DIFFERENT condition/topic. A retriever fooled by word overlap would fetch it; a reader sees it is off-subject."),
    "T2A": ("confusable_disease",
            "FAILURE_MODE: confusable_disease. Write about a disease/condition clinically CONFUSABLE with the query's target -- overlapping presentation, same symptom cluster, or related pathophysiology -- but a DIFFERENT diagnosis. It must read like a plausible differential a clinician would weigh, then reject. It must share symptoms/terminology with the query so a retriever is fooled, but the diagnosis discussed is NOT the query's target."),
    "T2B": ("wrong_drug_same_class",
            "FAILURE_MODE: wrong_drug_same_class. Write about a DIFFERENT drug in the SAME therapeutic class as the query's target drug. Same class/indication family so it looks retrievable, but it is the wrong agent. Do not use the positive's drug."),
    "T2C": ("wrong_stage_severity",
            "FAILURE_MODE: wrong_stage_severity. Write about a WRONG stage / severity / variant of the query's target condition (e.g. acute vs chronic; mild vs severe; Type 1 vs Type 2; pediatric vs adult presentation). Same disease family, wrong specificity -- forces token-level discrimination."),
    "T2D": ("wrong_dose_route",
            "FAILURE_MODE: wrong_dose_route. Write about the query's target DRUG but with a WRONG dose, route, or administration detail (e.g. IV push vs oral; loading vs maintenance; pediatric vs adult dose). Same drug, wrong administration token."),
    "T3A": ("lay_clinical_mismatch",
            "FAILURE_MODE: lay_clinical_mismatch. Write in LAYPERSON wording but with CLINICALLY WRONG content for the query. The register matches a lay query but the answer is medically incorrect for what is asked. Do not use the positive's answer."),
    "T3B": ("abbreviation_trap",
            "FAILURE_MODE: abbreviation_trap. The query contains (or implies) an ambiguous abbreviation. Write a passage that answers a PLAUSIBLE BUT WRONG expansion of that abbreviation (e.g. 'MS' expanded as multiple sclerosis when the query meant morphine sulfate, or vice versa). It must look like a correct answer to the wrong interpretation. If the query contains no abbreviation, instead write a passage that conflates two same-acronym medical concepts."),
    "T3C": ("biomedical_clinical_mismatch",
            "FAILURE_MODE: biomedical_clinical_mismatch. Write in biomedical/literature phrasing but give the WRONG clinical answer for the query. The register matches a biomedical query but the content does not answer what is asked."),
    "T4A": ("wrong_context",
            "FAILURE_MODE: wrong_context. Write about the RIGHT topic/concept but the WRONG patient population or context (pediatric vs adult; acute vs chronic; animal model vs human; prophylaxis vs treatment). On-topic, contextually wrong."),
    "T4B": ("treatment_diagnosis_confound",
            "FAILURE_MODE: treatment_diagnosis_confound. If the QUERY asks about TREATMENT, write a passage that DIAGNOSES the condition instead (and vice versa: if it asks about diagnosis, write a passage that treats). Right condition, wrong clinical action axis -- intent vs content mismatch."),
    "T5A": ("domain_boundary",
            "FAILURE_MODE: domain_boundary. Write a NON-MEDICAL (or basic-science / chemistry) passage that is lexically on-topic for the query. For a drug query, a chemistry/synthesis passage; for a clinical query, a basic-science mechanism passage. It shares vocabulary but is outside the clinical retrieval domain."),
    # T2E is UMLS-grounded: the {semantic_group} is interpolated from the positive's own
    # UMLS group, so the negative stays in-category but on a different concept. This is the
    # synthetic replacement for the broken category_aware_real anchor (which sampled a
    # random passage from a too-broad group).
    "T2E": ("same_group_wrong_concept_synth",
            "FAILURE_MODE: category_aware_synth. The positive's intended concept belongs to the UMLS semantic group '{semantic_group}'. Write about a DIFFERENT concept that is ALSO in the '{semantic_group}' group (i.e. same broad medical category, e.g. a different disorder if the group is 'disorders', a different drug/chemical if 'drugs_chemicals', a different procedure if 'procedures', a different anatomical structure if 'anatomy'). Share enough surface terms with the QUERY that a retriever would fetch it, but discuss a genuinely different concept than the query's target. NOT a disease-confusable differential (that is a different tier) -- here the goal is a same-category, different-concept passage grounded in the '{semantic_group}' group."),
}
# T5B (garbage_refusal) is deferred per plan §7.3 -- trivially-easy junk; enable later if
# a robustness eval demands it. The dropped-rows pool is documented in the plan.

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM_GEN = """You are a senior medical-information-retrieval engineer writing HARD NEGATIVE passages for a contrastive retrieval trainer (ColBERT). A hard negative is a passage that LOOKS retrievable for a given query but does NOT answer that query's specific intent -- it is wrong in a precise, specified way.

You will be given: a QUERY, the POSITIVE PASSAGE (the correct answer), the FACT the positive establishes, the intended FAILURE_MODE for each negative, and how many to write.

HARD CONSTRAINTS for EVERY negative passage:
1. REGISTER MATCH -- Write in the SAME register, length, density, and source-voice as the POSITIVE PASSAGE. If the positive is a PubMed abstract, write a PubMed-style abstract (single paragraph, ~60-160 words, third-person, no headers). If clinical-note-style, match that. NO bullet points, NO headers, NO "as an AI", NO meta-commentary, NO preamble, NO "here is a negative". It must read as genuine medical literature/clinical text.
2. RETRIEVABLE -- The passage must plausibly be retrieved for the QUERY: share enough topical/lexical surface (terms, concept, domain) that a retriever would consider it a candidate. It is NOT a random off-topic passage.
3. DOES NOT ANSWER -- The passage must NOT satisfy the query's specific intent. It must be wrong in the exact FAILURE_MODE specified. It must not contain the positive's answer or the positive's FACT.
4. NO LEAKAGE -- Do not copy or lightly paraphrase the POSITIVE PASSAGE. Do not reuse its specific result, drug name, dose, or finding. A paraphrase of the positive is a FALSE negative (it corrupts training).
5. MEDICALLY COHERENT -- Realistic, internally consistent medical text. You may use real medical facts and real drug/disease names -- groundedness is good. The passage may be a real-sounding statement about a DIFFERENT concept than the query asks.
6. DIVERSITY -- When writing several negatives, make them genuinely distinct (different wrong concept, different failure mechanism), not minor rewordings of each other.

OUTPUT FORMAT -- exactly this, one block per negative, no prose before/after:
<<<NEG>>>
PASSAGE: <one paragraph, ~60-160 words, in the positive's voice>
WHY_WRONG: <one sentence: how it fails the query and in what way>
<<<END>>>

Write the requested number of negatives. Quality over quantity. A passage that accidentally ANSWERS the query is worthless and will be dropped."""

SYSTEM_AUDIT = """You are auditing a HARD NEGATIVE for a contrastive retrieval trainer. You are given the QUERY, the POSITIVE PASSAGE (the correct answer), a CANDIDATE NEGATIVE PASSAGE, and its intended FAILURE_MODE.

Score strictly but DO NOT over-flag. A hard negative is SUPPOSED to look retrievable and be wrong -- that is its job, not a flaw.

Judge:
- RELEVANT_LOOKS (1-5): would a retriever plausibly rank this passage for the query? (5=clearly on-topic; 3=borderline candidate; 1-2=too easy/irrelevant -- still usable but a weak negative)
- ANSWERS_QUERY (TRUE or FALSE): does the passage actually satisfy/answer the query's specific intent? TRUE = this is a FALSE NEGATIVE (a correct answer mislabeled as a negative -- it would corrupt training). Default FALSE unless the passage clearly and correctly answers what the query asks.
- GROUNDING (1-5): is the passage medically coherent/realistic? (5=realistic real-style medical text; 1-2=gibberish, self-contradictory, or hallucinated nonsense). A passage may contain REAL medical facts about a DIFFERENT concept than the query -- that is correct hard-negative behavior, score 4-5.
- IN_FAILURE_MODE (TRUE or FALSE): is the passage wrong in (or close to) the specified FAILURE_MODE? FALSE = mislabeled; it may still be a valid negative of some other kind (do not drop solely for this).

Respond EXACTLY (one line each, no prose before/after):
RELEVANT_LOOKS: <1-5>
ANSWERS_QUERY: <TRUE or FALSE>
GROUNDING: <1-5>
IN_FAILURE_MODE: <TRUE or FALSE>
NOTE: <one short line>"""

# --------------------------------------------------------------------------- #
# vLLM client
# --------------------------------------------------------------------------- #
def _post(messages, temperature, top_k, top_p, max_tokens, extra_body, timeout=300.0, retries=4):
    """POST a chat-completions request to the active backend via the shared client.
    Backend-aware: gemma sends top_k + chat_template_kwargs(thinking off); deepseek omits
    top_k (unsupported), uses top_p=1.0, thinking ON, and needs Bearer auth. With thinking on,
    the reasoning trace lands in message.reasoning_content and the final answer in message.content."""
    cfg = BACKENDS[BACKEND]
    payload = {
        "model": _MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if cfg["supports_top_k"]:
        payload["top_k"] = top_k
    payload.update(extra_body)
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    last = None
    for attempt in range(retries):
        try:
            r = _CLIENT.post(url, json=payload, timeout=timeout)
            if r.status_code == 429:  # rate limit (deepseek): back off harder
                time.sleep(min(2 ** attempt * 2, 30)); last = "HTTP 429"; continue
            r.raise_for_status()
            data = r.json()
            # accumulate token usage (both backends may report it)
            usage = data.get("usage")
            if usage:
                _TOKENS["prompt"] += usage.get("prompt_tokens", 0)
                _TOKENS["completion"] += usage.get("completion_tokens", 0)
                # prompt_cache_hit_tokens / cached_tokens (field name varies)
                _TOKENS["cached"] += (usage.get("prompt_cache_hit_tokens")
                                       or usage.get("cached_tokens") or 0)
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            # thinking-enabled models may leave content empty and put the answer in reasoning_content
            if not content.strip() and msg.get("reasoning_content"):
                content = msg["reasoning_content"]
            return content
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 8))
    return f"__ERROR__ {last}"


def gen_call(system, user, max_tokens=None):
    cfg = BACKENDS[BACKEND]
    if max_tokens is None:
        max_tokens = DS_GEN_MAX_TOKENS if BACKEND == "deepseek" else 1400
    return _post([{"role": "system", "content": system},
                  {"role": "user", "content": user}],
                 GEN_TEMPERATURE, GEN_TOP_K, cfg["gen_top_p"], max_tokens, cfg["gen_extra"])


def audit_call(system, user, max_tokens=None):
    cfg = BACKENDS[BACKEND]
    if max_tokens is None:
        max_tokens = DS_AUDIT_MAX_TOKENS if BACKEND == "deepseek" else 160
    return _post([{"role": "system", "content": system},
                  {"role": "user", "content": user}],
                 AUDIT_TEMPERATURE, -1, 1.0, max_tokens, cfg["audit_extra"])  # greedy audit


def backend_ok():
    """Health-check the active backend; return (ok, detail)."""
    cfg = BACKENDS[BACKEND]
    url = f"{cfg['base_url'].rstrip('/')}/models"
    try:
        r = _CLIENT.get(url, timeout=15)
        r.raise_for_status()
        ids = [x["id"] for x in r.json().get("data", [])]
        return (_MODEL in ids, ids)
    except Exception as e:  # noqa: BLE001
        return False, [str(e)]

# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
NEG_BLOCK_RE = re.compile(r"<<<NEG>>>(.*?)<<<END>>>", re.S)


def parse_negs(text):
    """Return list of (passage, why_wrong) from a generation response."""
    out = []
    for m in NEG_BLOCK_RE.finditer(str(text)):
        body = m.group(1).strip()
        pm = re.search(r"PASSAGE:\s*(.*?)(?:\n\s*WHY_WRONG:|$)", body, re.S)
        wm = re.search(r"WHY_WRONG:\s*(.*)", body, re.S)
        passage = pm.group(1).strip() if pm else ""
        # strip a trailing WHY_WRONG if the non-greedy caught it
        passage = re.sub(r"\n\s*WHY_WRONG:.*$", "", passage, flags=re.S).strip()
        why = wm.group(1).strip() if wm else ""
        # tidy whitespace
        passage = re.sub(r"\s+", " ", passage)
        if len(passage) >= 40:  # drop trivially short / empty
            out.append((passage, why))
    return out


def parse_audit(text):
    rl = ag = gr = ifm = None
    note = ""
    for line in str(text).splitlines():
        u = line.strip().upper()
        if u.startswith("RELEVANT_LOOKS:"):
            mm = re.search(r"([1-5])", line); rl = int(mm.group(1)) if mm else None
        elif u.startswith("ANSWERS_QUERY:"):
            ag = "TRUE" in u
        elif u.startswith("GROUNDING:"):
            mm = re.search(r"([1-5])", line); gr = int(mm.group(1)) if mm else None
        elif u.startswith("IN_FAILURE_MODE:"):
            ifm = "TRUE" in u
        elif u.startswith("NOTE:"):
            note = line.split(":", 1)[1].strip()
    return rl, ag, gr, ifm, note

# --------------------------------------------------------------------------- #
# N-gram overlap (4-gram, asymmetric-safe) + tokenizer
# --------------------------------------------------------------------------- #
_TOK_RE = re.compile(r"\b[a-z0-9][a-z0-9\-]{1,}\b")
_STOP = set("""the of and to a in for is with by on as at an be or from was were this that these
those it its we they their our you your not but are has have had can could should would will
which who whom whose what when where why how all any both each few more most other some such
no nor only own same so than too very also may might must shall do does did done being been
am if then there here into out up down over under again further once""".split())


def tokenize(text):
    return [t for t in _TOK_RE.findall(text.lower()) if t not in _STOP]


def _ngrams(text, n=4):
    toks = re.findall(r"\b\w+\b", text.lower())
    if len(toks) < n:
        return set(zip(*[toks])) if toks else set()  # fall back to unigrams
    return set(zip(*[toks[i:] for i in range(n)]))


def overlap(a, b):
    A, B = _ngrams(a), _ngrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))

# --------------------------------------------------------------------------- #
# Inline BM25 over the 66k corpus
# --------------------------------------------------------------------------- #
class BM25:
    def __init__(self, corpus_texts, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)   # term -> [(doc_idx, tf)]
        self.dl = []
        df = defaultdict(int)
        for i, t in enumerate(corpus_texts):
            toks = tokenize(t)
            self.dl.append(len(toks))
            for w, tf in Counter(toks).items():
                self.postings[w].append((i, tf))
                df[w] += 1
        self.N = len(corpus_texts)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0
        self.idf = {w: math.log(1 + (self.N - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def topk(self, query_toks, k=30):
        scores = defaultdict(float)
        for w in set(query_toks):
            post = self.postings.get(w)
            if not post:
                continue
            idf = self.idf[w]
            for i, tf in post:
                dl = self.dl[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        if not scores:
            return []
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
def build_user_gen(row, tier, n):
    label, spec = FAILURE_MODES[tier]
    # T2E's spec interpolates the positive's UMLS semantic_group
    spec = spec.format(semantic_group=row.get('semantic_group') or 'clinical')
    return (f"QUERY: {row['query']}\n"
            f"QUERY_FAMILY: {row['task_family']}  ({row['query_style']})\n"
            f"INTENDED CONCEPT: cui_label={row.get('cui_label')}  alt_term={row.get('alt_term')}  "
            f"semantic_group={row.get('semantic_group')}\n"
            f"FACT (what the POSITIVE establishes -- do NOT repeat this): {str(row.get('fact',''))[:400]}\n\n"
            f"POSITIVE PASSAGE (the correct answer -- match its voice/length; do NOT copy it):\n"
            f"{str(row['passage_text'])[:1600]}\n\n"
            f"{spec}\n\n"
            f"Write {n} such negatives, each a DIFFERENT wrong concept. Follow the output format exactly.")


def build_user_audit(query, positive_passage, negative_passage, failure_mode):
    return (f"QUERY: {query}\n"
            f"INTENDED FAILURE_MODE: {failure_mode}\n\n"
            f"POSITIVE PASSAGE (correct answer):\n{str(positive_passage)[:1500]}\n\n"
            f"CANDIDATE NEGATIVE PASSAGE:\n{str(negative_passage)[:1500]}\n\n"
            f"Judge per the rules above.")

# --------------------------------------------------------------------------- #
# Per-positive worker: generate + real-anchor + audit + drop
# --------------------------------------------------------------------------- #
def process_positive(idx_row, bm25, corpus_text):
    row_idx, row = idx_row
    qid = row["example_id"]
    pos_text = str(row["passage_text"])
    query = str(row["query"])
    qtoks = tokenize(query)
    raw = []  # list of dicts: negative_passage, negative_source, failure_mode, why_wrong

    # ---- synthetic negatives (per routed tier; T2E is the UMLS-grounded category anchor) ----
    for tier, n in FAMILY_ROUTING.get(row["task_family"], []):
        txt = gen_call(SYSTEM_GEN, build_user_gen(row, tier, n))
        if txt.startswith("__ERROR__"):
            continue
        for passage, why in parse_negs(txt):
            raw.append({"negative_passage": passage, "negative_source": "synthetic",
                        "failure_mode": FAILURE_MODES[tier][0], "why_wrong": why})

    # ---- REAL ANCHORS: BM25 hard negatives (exclude positive + near-dups), top-N ----
    # These are genuine real PubMed passages sharing query terms but a different concept.
    # Two per positive keeps the real-text anchor strong (Mitigation 2: train-on-synthetic).
    if qtoks:
        for k in (60, 200, 500):
            ranked = bm25.topk(qtoks, k=k)
            picked = []
            for doc_idx, _score in ranked:
                if overlap(corpus_text[doc_idx], pos_text) < OVERLAP_POSITIVE:
                    picked.append(doc_idx)
                    if len(picked) >= BM25_REAL_N:
                        break
            for doc_idx in picked[:BM25_REAL_N]:
                raw.append({"negative_passage": corpus_text[doc_idx],
                            "negative_source": "bm25_real",
                            "failure_mode": "lexical_trap_real", "why_wrong": ""})
            if len(picked) >= BM25_REAL_N:
                break

    # ---- audit every negative ----
    kept, dropped = [], []
    seen_grams = []
    for neg in raw:
        npass = neg["negative_passage"]
        atxt = audit_call(SYSTEM_AUDIT,
                          build_user_audit(query, pos_text, npass, neg["failure_mode"]))
        if atxt.startswith("__ERROR__"):
            neg.update({"audit_relevant_looks": None, "audit_answers_query": None,
                        "audit_grounding": None, "audit_in_failure_mode": None,
                        "audit_note": "audit_call_error", "drop_reason": "audit_call_error"})
            dropped.append(neg)
            continue
        rl, ag, gr, ifm, note = parse_audit(atxt)
        neg.update({"audit_relevant_looks": rl, "audit_answers_query": ag,
                    "audit_grounding": gr, "audit_in_failure_mode": ifm, "audit_note": note})
        # DROP: false negative, gibberish, OR an unparsed audit (conservative -- never ship an
        # unaudited negative, since a missed ANSWERS_QUERY:TRUE would be a false negative).
        if ag is True:
            neg["drop_reason"] = "answers_query_true"; dropped.append(neg); continue
        if gr is not None and gr <= 2:
            neg["drop_reason"] = "grounding_le2"; dropped.append(neg); continue
        if ag is None or gr is None:
            neg["drop_reason"] = "audit_unparsed"; dropped.append(neg); continue
        # intra-sample near-duplicate drop
        g = _ngrams(npass)
        if any(len(g & s) / min(len(g), len(s)) >= OVERLAP_NEAR_DUP if (g and s) else False
               for s in seen_grams):
            neg["drop_reason"] = "intra_sample_near_dup"
            dropped.append(neg)
            continue
        seen_grams.append(g)
        neg["weak"] = (rl is not None and rl <= 2)
        if ifm is False:
            neg["failure_mode"] = "unspecified"
        kept.append(neg)

    # attach query/positive ids to each
    for n in kept + dropped:
        n["query_id"] = qid
        n["query"] = query
        n["query_style"] = row["query_style"]
        n["task_family"] = row["task_family"]
        n["positive_passage_id"] = row["passage_id"]
        n["cui_label"] = row.get("cui_label")
        n["alt_term"] = row.get("alt_term")
        n["semantic_group"] = row.get("semantic_group")
    return kept, dropped

# --------------------------------------------------------------------------- #
# Stratified sampling
# --------------------------------------------------------------------------- #
def stratified_sample(df, n, col="task_family", seed=SEED):
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    counts = df[col].value_counts()
    total = counts.sum()
    for fam, c in counts.items():
        take = max(1, round(n * c / total))
        sub = df[df[col] == fam]
        parts.append(sub.sample(n=min(take, len(sub)), random_state=int(rng.integers(1e9))))
    out = pd.concat(parts).reset_index(drop=True)
    return out.head(n) if len(out) >= n else out

# --------------------------------------------------------------------------- #
# Dev split + BM25 recall sanity
# --------------------------------------------------------------------------- #
def make_dev_and_sanity(train_df, dev_df, bm25, corpus_text):
    rows = []
    for _, r in dev_df.iterrows():
        qtoks = tokenize(str(r["query"]))
        pos_text = str(r["passage_text"])
        ranked = bm25.topk(qtoks, k=200) if qtoks else []
        in_corpus = any(overlap(corpus_text[i], pos_text) >= OVERLAP_POSITIVE for i, _ in ranked)
        rec10 = any(overlap(corpus_text[i], pos_text) >= OVERLAP_POSITIVE
                    for i, _ in ranked[:10])
        rec100 = any(overlap(corpus_text[i], pos_text) >= OVERLAP_POSITIVE
                     for i, _ in ranked[:100])
        rows.append({"query": r["query"], "query_style": r["query_style"],
                     "task_family": r["task_family"],
                     "positive_passage_id": r["passage_id"],
                     "positive_passage_text": pos_text,
                     "positive_in_corpus": bool(in_corpus),
                     "bm25_recall_at_10": bool(rec10),
                     "bm25_recall_at_100": bool(rec100)})
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    global BACKEND, _CLIENT, _MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="positives to process in THIS chunk (0 = all from --start). With --start, "
                         "takes a contiguous slice train[start:start+limit] for incremental runs.")
    ap.add_argument("--start", type=int, default=0,
                    help="offset into the (sorted) train positives -- for 10k-increment or "
                         "multi-machine chunked runs. Default 0.")
    ap.add_argument("--tag", default="v1", help="output namespace (e.g. smoke, pilot, ds_pilot, v1)")
    ap.add_argument("--backend", choices=list(BACKENDS), default="gemma",
                    help="LLM backend (gemma=local vLLM Gemma-4-31B, deepseek=DeepSeek-V4-Flash API)")
    ap.add_argument("--base-url", default=None,
                    help="override the backend's base URL (e.g. for a remote vLLM on another "
                         "machine: http://192.168.1.50:8000/v1). Default = the backend's built-in URL.")
    ap.add_argument("--model", default=None,
                    help="override the served model id (e.g. if the remote vLLM serves the model "
                         "under a different name). Default = the backend's built-in model id.")
    ap.add_argument("--workers", type=int, default=0, help="concurrency (0 = backend default)")
    args = ap.parse_args()
    NEG_DIR.mkdir(parents=True, exist_ok=True)

    BACKEND = args.backend
    cfg = BACKENDS[BACKEND]
    if args.base_url:
        cfg["base_url"] = args.base_url.rstrip("/")   # apply override to the live config
    if args.model:
        cfg["model"] = args.model
    _MODEL = cfg["model"]
    workers = args.workers if args.workers > 0 else cfg["default_workers"]

    # ---- build the shared httpx client (backend-aware auth) ----
    headers = {"Content-Type": "application/json"}
    if cfg["auth"] == "bearer_env":
        key = os.environ.get("DEEPSEEK_API_KEY") or (KEYFILE.read_text().strip() if KEYFILE.exists() else "")
        if not key:
            print(f"DEEPSEEK_API_KEY not set and {KEYFILE.name} not found -- aborting."); return
        headers["Authorization"] = f"Bearer {key}"
    limits = httpx.Limits(max_connections=workers + 50, max_keepalive_connections=workers)
    _CLIENT = httpx.Client(headers=headers, limits=limits, timeout=httpx.Timeout(300.0, connect=15.0))

    ok, ids = backend_ok()
    if not ok:
        print(f"[{BACKEND}] health check FAILED ({_MODEL} not available). detail={ids} -- aborting."); return
    print(f"[{BACKEND}] OK: model={_MODEL} workers={workers}  ids={ids}")

    print("loading v3g positives...")
    df = pd.read_parquet(V3G).reset_index(drop=True)
    df = df[df["drop_reason"].isna()].copy()  # only kept positives (drop_reason is NaN in v3g)
    print(f"  {len(df):,} positives")

    # stratified 90/10 train/dev split (dev excluded from negative generation)
    dev_df = stratified_sample(df, int(len(df) * DEV_FRACTION), seed=SEED)
    dev_ids = set(dev_df["example_id"])
    train_df = df[~df["example_id"].isin(dev_ids)].reset_index(drop=True)
    print(f"  train={len(train_df):,}  dev={len(dev_df):,} (held out for real-eval)")

    # ---- contiguous chunk for incremental / multi-machine runs ----
    # Sort by example_id for a fully deterministic, stable ordering across machines and
    # restarts. --start (offset) + --limit (count) take a contiguous slice, so successive
    # 10k increments are disjoint and their union = the full train set. Same --tag across
    # chunks accumulates into one negatives_<tag>.parquet via the resumability partial.
    train_df = train_df.sort_values("example_id").reset_index(drop=True)
    if args.limit > 0:
        chunk = train_df.iloc[args.start:args.start + args.limit]
    else:
        chunk = train_df.iloc[args.start:]
    print(f"  chunk: start={args.start} count={len(chunk):,} (of {len(train_df):,} train positives, limit={args.limit})")
    train_df = chunk

    print("loading 66k real PubMed corpus + building BM25 index...")
    corpus = pd.DataFrame(json.loads(CORPUS_JSON.read_text()))
    corpus_text = [str(t) for t in corpus["text"].tolist()]
    bm25 = BM25(corpus_text)
    print(f"  corpus={len(corpus_text):,}  avgdl={bm25.avgdl:.0f}")

    # ---- resumability: load partial, skip done query_ids ----
    partial_path = NEG_DIR / f"negatives_{args.tag}_partial.parquet"
    done = set()
    kept_rows, dropped_rows = [], []
    if partial_path.exists():
        prev = pd.read_parquet(partial_path)
        done = set(prev["query_id"].unique())
        kept_rows = prev.to_dict("records")
        print(f"  resuming: {len(done):,} positives already done")

    todo = [(i, row) for i, row in train_df.iterrows() if row["example_id"] not in done]
    print(f"  to process: {len(todo):,}  (workers={workers})\n")

    t0 = time.time()
    done_count = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_positive, item, bm25, corpus_text)
                for item in todo]
        for fut in as_completed(futs):
            try:
                kept, dropped = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  worker error: {e}"); continue
            kept_rows.extend(kept)
            dropped_rows.extend(dropped)
            done_count += 1
            if done_count % 50 == 0:
                rate = done_count / max(1e-6, time.time() - t0)
                print(f"  {done_count}/{len(todo)} | kept={len(kept_rows)} dropped={len(dropped_rows)} "
                      f"| {rate:.2f}/s | ETA {((len(todo)-done_count)/max(1e-6,rate))/60:.0f}m")
            if done_count % CHECKPOINT_EVERY == 0 and kept_rows:
                pd.DataFrame(kept_rows).to_parquet(partial_path, index=False)
    # ---- write outputs ----
    kept_df = pd.DataFrame(kept_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    if "negative_id" not in kept_df.columns and len(kept_df):
        kept_df = kept_df.sort_values(["query_id"]).reset_index(drop=True)
        kept_df["negative_id"] = [f"{qid}__{i}" for i, qid in enumerate(kept_df["query_id"])]

    out_kept = NEG_DIR / f"negatives_{args.tag}.parquet"
    out_drop = NEG_DIR / f"negatives_{args.tag}_dropped.parquet"
    kept_df.to_parquet(out_kept, index=False)
    dropped_df.to_parquet(out_drop, index=False)

    # ---- QA sample: ~200 stratified by (task_family x failure_mode) ----
    if len(kept_df):
        kept_df["strat"] = kept_df["task_family"].astype(str) + "::" + kept_df["failure_mode"].astype(str)
        qa = stratified_sample(kept_df, 200, col="strat", seed=SEED)
        qa = qa.drop(columns=["strat"])
        # attach positive passage for review
        pos_map = df.set_index("example_id")["passage_text"].to_dict()
        qa["positive_passage_text"] = qa["query_id"].map(pos_map)
        qa.to_parquet(NEG_DIR / f"negatives_{args.tag}_qa_sample.parquet", index=False)

    # ---- dev split + BM25 recall sanity ----
    dev_df_full = dev_df  # always the full held-out 10%, regardless of --limit
    sanity = make_dev_and_sanity(train_df, dev_df_full, bm25, corpus_text)
    sanity.to_parquet(NEG_DIR / "triplets_dev_candidates.parquet", index=False)

    # ---- summary ----
    summary = {
        "tag": args.tag, "backend": BACKEND, "model": _MODEL, "limit": args.limit,
        "n_train_processed": int(done_count), "n_dev": int(len(dev_df_full)),
        "n_kept": int(len(kept_df)), "n_dropped": int(len(dropped_df)),
        "wall_seconds": round(time.time() - t0, 1),
    }
    # token usage + cost (both backends report usage; cost is meaningful only for deepseek)
    if _TOKENS["prompt"] or _TOKENS["completion"]:
        p_in = _TOKENS["prompt"]; p_cached = _TOKENS["cached"]; p_miss = max(0, p_in - p_cached)
        p_out = _TOKENS["completion"]
        summary["tokens"] = {"prompt": p_in, "prompt_cached": p_cached, "completion": p_out}
        if BACKEND == "deepseek":
            cost = (p_miss / 1e6) * DS_PRICE_IN_CACHE_MISS + (p_cached / 1e6) * DS_PRICE_IN_CACHE_HIT \
                 + (p_out / 1e6) * DS_PRICE_OUT
            summary["estimated_cost_usd"] = round(cost, 4)
    if len(kept_df):
        summary["negatives_per_positive"] = round(len(kept_df) / max(1, done_count), 2)
        summary["by_source"] = kept_df["negative_source"].value_counts().to_dict()
        summary["by_family"] = kept_df["task_family"].value_counts().to_dict()
        summary["by_failure_mode"] = kept_df["failure_mode"].value_counts().to_dict()
        summary["weak_rate"] = round(float(kept_df["weak"].mean()) if "weak" in kept_df else 0.0, 4)
        # pre-audit false-negative rate = dropped answers_query_true / (kept + dropped answers_query_true)
        if len(dropped_df) and "drop_reason" in dropped_df.columns:
            fn = int((dropped_df["drop_reason"] == "answers_query_true").sum())
            summary["false_negatives_dropped"] = fn
            summary["false_negative_pre_audit_rate"] = round(fn / max(1, len(kept_df) + fn), 4)
    if len(dropped_df):
        summary["drop_reasons"] = dropped_df["drop_reason"].value_counts().to_dict()
    if len(sanity):
        summary["dev_bm25_recall_at_10"] = round(float(sanity["bm25_recall_at_10"].mean()), 4)
        summary["dev_bm25_recall_at_100"] = round(float(sanity["bm25_recall_at_100"].mean()), 4)
        summary["dev_positive_in_corpus"] = round(float(sanity["positive_in_corpus"].mean()), 4)
    (NEG_DIR / f"negatives_{args.tag}_run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== DONE ({args.tag}) {time.time()-t0:.0f}s ===")
    print(f"  kept:   {len(kept_df):,} -> {out_kept.name}")
    print(f"  dropped:{len(dropped_df):,} -> {out_drop.name}")
    if len(kept_df):
        print(f"  neg/pos: {summary.get('negatives_per_positive')}")
        print(f"  by source: {summary.get('by_source')}")
        print(f"  by failure_mode: {summary.get('by_failure_mode')}")
        print(f"  weak_rate: {summary.get('weak_rate')}  false_neg_pre_audit_rate: {summary.get('false_negative_pre_audit_rate')}")
    if len(dropped_df):
        print(f"  drop reasons: {summary.get('drop_reasons')}")
    print(f"  dev BM25 recall@10={summary.get('dev_bm25_recall_at_10')} recall@100={summary.get('dev_bm25_recall_at_100')} "
          f"positive_in_corpus={summary.get('dev_positive_in_corpus')}")
    if "tokens" in summary:
        tk = summary["tokens"]
        cost_str = f"  | est cost ${summary.get('estimated_cost_usd', 0):.4f}" if "estimated_cost_usd" in summary else "  | local backend ($0 API cost)"
        print(f"  tokens: prompt={tk['prompt']:,} (cached {tk['prompt_cached']:,}) completion={tk['completion']:,}{cost_str}")
    print(f"  summary -> negatives_{args.tag}_run_summary.json")
    # tidy up partial on successful completion
    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    main()
