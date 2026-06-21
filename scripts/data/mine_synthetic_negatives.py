"""Synthetic hard-negative + real-anchor mining for MedColBERT ColBERT training.

Implements docs/TRIPLET_MINING_PLAN.md. For every positive (query, passage) in the
cleaned v3g dataset, produces ~11-15 hard negatives:

  * SYNTHETIC (Gemma-4-31B): taxonomy-routed, style-conditioned negatives, each failing
    the query in a specified clinical-failure mode (confusable disease, wrong drug in
    class, abbreviation trap, lay/clinical mismatch, wrong context, ...).
  * REAL ANCHOR (no LLM, from the 66k UMLS-annotated PubMed corpus):
      - 1 BM25 hard negative (real passage sharing query terms, different concept)
      - 1 category-aware real negative (same semantic_group, different concept)

Every negative (synthetic AND real) passes a calibrated Gemma audit gate that DROPS:
  * false negatives  (ANSWERS_QUERY == TRUE  -- a negative that actually answers the query)
  * incoherent/gibberish passages (GROUNDING <= 2)
plus an intra-sample near-duplicate drop (4-gram overlap >= 0.8).

A held-out 10% real eval split is reserved (its positives are NOT used to generate
negatives) and a BM25 recall@k sanity is computed on it.

Sampling params (FIXED, owner-specified): generation temperature=1.0, top_k=64,
top_p=0.95, enable_thinking=False (via chat_template_kwargs). Audit: temperature=0.0.

Usage:
  uv run python scripts/data/mine_synthetic_negatives.py --limit 24 --tag smoke   # smoke
  uv run python scripts/data/mine_synthetic_negatives.py --limit 2000 --tag pilot # 2k pilot
  uv run python scripts/data/mine_synthetic_negatives.py --limit 0 --tag v1       # full ~60k
(Resumable: re-running the same command skips query_ids already in the partial output.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

VLLM = "http://localhost:8000/v1"
MODEL = "gemma-4-31B"
GEN_TEMPERATURE = 1.0
GEN_TOP_K = 64
GEN_TOP_P = 0.95
AUDIT_TEMPERATURE = 0.0
MAX_WORKERS = 64
CHECKPOINT_EVERY = 500
DEV_FRACTION = 0.10
SEED = 20260621
OVERLAP_POSITIVE = 0.6   # 4-gram overlap threshold to treat a corpus passage as the positive / near-dup
OVERLAP_NEAR_DUP = 0.8   # 4-gram overlap between two negatives of the same sample -> drop the dup

# --------------------------------------------------------------------------- #
# Taxonomy: family -> list of (tier, n_negatives), and tier -> (label, spec)
# --------------------------------------------------------------------------- #
FAMILY_ROUTING = {
    "symptom_to_diagnosis":        [("T1", 2), ("T2A", 4), ("T2C", 3), ("T4B", 3)],
    "biomedical_to_clinical":      [("T1", 2), ("T2A", 2), ("T3C", 4), ("T4A", 2), ("T5A", 2)],
    "generic_clinical_retrieval":  [("T1", 3), ("T2A", 2), ("T3B", 2), ("T4A", 2), ("T5A", 2)],
    "lay_to_clinical":             [("T1", 2), ("T2A", 2), ("T3A", 4), ("T4A", 3)],
    "brand_generic":               [("T1", 2), ("T2B", 5), ("T2D", 3), ("T3B", 2), ("T4A", 1)],
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
def _post(messages, temperature, top_k, top_p, max_tokens, timeout=180.0, retries=3):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout) as c:
                r = c.post(f"{VLLM}/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 8))
    return f"__ERROR__ {last}"


def gen_call(system, user, max_tokens=1400):
    return _post([{"role": "system", "content": system},
                  {"role": "user", "content": user}],
                 GEN_TEMPERATURE, GEN_TOP_K, GEN_TOP_P, max_tokens)


def audit_call(system, user, max_tokens=160):
    return _post([{"role": "system", "content": system},
                  {"role": "user", "content": user}],
                 AUDIT_TEMPERATURE, -1, 1.0, max_tokens)  # top_k=-1 / top_p=1.0 = vLLM defaults


def vllm_ok():
    try:
        with httpx.Client(timeout=10) as c:
            ids = [x["id"] for x in c.get(f"{VLLM}/models").json().get("data", [])]
        return MODEL in ids, ids
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
def process_positive(idx_row, bm25, corpus_text, corpus_group, corpus_cui):
    row_idx, row = idx_row
    qid = row["example_id"]
    pos_text = str(row["passage_text"])
    query = str(row["query"])
    qtoks = tokenize(query)
    raw = []  # list of dicts: negative_passage, negative_source, failure_mode, why_wrong

    # ---- synthetic negatives (per routed tier) ----
    for tier, n in FAMILY_ROUTING.get(row["task_family"], []):
        txt = gen_call(SYSTEM_GEN, build_user_gen(row, tier, n))
        if txt.startswith("__ERROR__"):
            continue
        for passage, why in parse_negs(txt):
            raw.append({"negative_passage": passage, "negative_source": "synthetic",
                        "failure_mode": FAILURE_MODES[tier][0], "why_wrong": why})

    # ---- real anchor 1: BM25 hard negative (exclude positive + near-dups) ----
    if qtoks:
        for k in (30, 100, 300):
            ranked = bm25.topk(qtoks, k=k)
            picked = None
            for doc_idx, _score in ranked:
                if overlap(corpus_text[doc_idx], pos_text) < OVERLAP_POSITIVE:
                    picked = doc_idx
                    break
            if picked is not None:
                raw.append({"negative_passage": corpus_text[picked],
                            "negative_source": "bm25_real",
                            "failure_mode": "lexical_trap_real", "why_wrong": ""})
                break

    # ---- real anchor 2: category-aware (same semantic_group, different concept) ----
    grp = row.get("semantic_group")
    cands = corpus_group.get(grp, [])
    if cands:
        seed = int(hashlib.md5(qid.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        chosen = None
        order = list(range(len(cands)))
        rng.shuffle(order)
        cui = row.get("cui_label")
        for ci in order:
            doc_idx = cands[ci]
            if overlap(corpus_text[doc_idx], pos_text) >= OVERLAP_POSITIVE:
                continue
            # prefer a different cui_label, but accept any in-group different passage
            if corpus_cui[doc_idx] != cui or chosen is None:
                chosen = doc_idx
                if corpus_cui[doc_idx] != cui:
                    break
        if chosen is not None:
            raw.append({"negative_passage": corpus_text[chosen],
                        "negative_source": "category_aware_real",
                        "failure_mode": "same_group_wrong_concept_real", "why_wrong": ""})

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
        # DROP: false negative or gibberish
        if ag is True or (gr is not None and gr <= 2):
            neg["drop_reason"] = "answers_query_true" if ag is True else "grounding_le2"
            dropped.append(neg)
            continue
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="positives to process (0 = all train split)")
    ap.add_argument("--tag", default="v1", help="output namespace (e.g. smoke, pilot, v1)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()
    NEG_DIR.mkdir(parents=True, exist_ok=True)

    ok, ids = vllm_ok()
    if not ok:
        print(f"VLLM DOWN or {MODEL} not served. ids={ids} -- aborting."); return
    print(f"vLLM OK: {ids}")

    print("loading v3g positives...")
    df = pd.read_parquet(V3G).reset_index(drop=True)
    df = df[df["drop_reason"].isna()].copy()  # only kept positives (drop_reason is NaN in v3g)
    print(f"  {len(df):,} positives")

    # stratified 90/10 train/dev split (dev excluded from negative generation)
    dev_df = stratified_sample(df, int(len(df) * DEV_FRACTION), seed=SEED)
    dev_ids = set(dev_df["example_id"])
    train_df = df[~df["example_id"].isin(dev_ids)].reset_index(drop=True)
    print(f"  train={len(train_df):,}  dev={len(dev_df):,} (held out for real-eval)")

    if args.limit > 0:
        train_df = stratified_sample(train_df, args.limit, seed=SEED)
        print(f"  processing {len(train_df):,} train positives (limit={args.limit})")

    print("loading 66k real PubMed corpus + building BM25 index...")
    corpus = pd.DataFrame(json.loads(CORPUS_JSON.read_text()))
    corpus_text = [str(t) for t in corpus["text"].tolist()]
    corpus_cui = [str(c) for c in corpus["cui_label"].tolist()]
    corpus_group = defaultdict(list)
    for i, g in enumerate(corpus["semantic_group"].tolist()):
        corpus_group[str(g)].append(i)
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
    print(f"  to process: {len(todo):,}  (workers={args.workers})\n")

    t0 = time.time()
    done_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_positive, item, bm25, corpus_text, corpus_group, corpus_cui)
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
        "tag": args.tag, "model": MODEL, "limit": args.limit,
        "n_train_processed": int(done_count), "n_dev": int(len(dev_df_full)),
        "n_kept": int(len(kept_df)), "n_dropped": int(len(dropped_df)),
        "wall_seconds": round(time.time() - t0, 1),
    }
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
    print(f"  summary -> negatives_{args.tag}_run_summary.json")
    # tidy up partial on successful completion
    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    main()
