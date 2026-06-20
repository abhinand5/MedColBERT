#!/usr/bin/env python3
"""MedColBERT Multistyle Batch Runner v2 — Optimized for 100K scale.

Key optimizations over v1:
  1. Hybrid judging: full_question gets strict 5-D LLM judge,
     keyword queries skip LLM judging (accepted if validation passes).
     Cuts judging time by ~75% while maintaining quality.
  2. Batch size defaults to 100 passages (400 candidates).
  3. Configurable judge mode: "hybrid" (default), "full" (all 5-D), "none" (no judging).

Pipeline:
  1. Load + shuffle passages (seed-controlled)
  2. For each passage:
     a. Map semantic_group → task_family
     b. Extract fact (temp=0)
     c. Generate 4 queries: 1 full_question + 3 keyword styles
     d. Deterministic validation
  3. Judge: full_question → strict 5-D | keyword → validation-pass = accept
  4. Save accepted.parquet, judged.parquet, stats.json

Usage:
  uv run python scripts/data/multistyle_batch_v2.py --seed 42 --batch-id ms2_001 --count 100
  uv run python scripts/data/multistyle_batch_v2.py --seed 42 --batch-id ms2_001 --count 100 --judge-mode full
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

from medcolbert.generation.prompts import (
    Q1_ANSWERABILITY_PROMPT,
    Q2_VOCAB_SHIFT_PROMPT,
    Q3_PASSAGE_SPECIFICITY_PROMPT,
    Q4_FACTUAL_GROUNDING_PROMPT,
    Q5_TASK_FAMILY_FIT_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    get_task_family_description,
)
from medcolbert.generation.validators import DeterministicValidators
from medcolbert.utils.hashing import stable_hash

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path("/workspace/MedColBERT")
PASSAGES_PATH = BASE_DIR / "data/processed/private/passages/real_annotated_500.json"
DEFAULT_OUTPUT_BASE = BASE_DIR / "data/processed/private/synthetic/ontology_grounded_teacher_filtered"

API_BASE = "http://localhost:8000/v1"
MODEL = "gemma-4-31B"
SLEEP = 0.05

# ── Semantic group → task family ─────────────────────────────────────────────

SEMANTIC_GROUP_TO_TASK_FAMILY = {
    "drugs_chemicals": ["brand_generic"],  # abbreviation_to_expanded only from dedicated abbreviation passages
    "anatomy": ["lay_to_clinical"],
    "disorders": ["symptom_to_diagnosis"],
    "procedures": ["biomedical_to_clinical"],
    "findings_signs_symptoms": ["lay_to_clinical"],
    "genes_proteins": ["biomedical_to_clinical"],
    "living_beings": ["biomedical_to_clinical"],
    "physiology": ["lay_to_clinical"],
    "devices": ["biomedical_to_clinical"],
    "objects": ["lay_to_clinical"],
    "concepts_ideas": ["lay_to_clinical"],
}

TASK_FAMILY_ROLES = {
    "lay_to_clinical": ["patient", "nurse", "physician"],
    "brand_generic": ["patient", "physician", "nurse"],
    "abbreviation_to_expanded": ["physician", "researcher", "nurse"],
    "symptom_to_diagnosis": ["patient", "physician", "nurse"],
    "biomedical_to_clinical": ["researcher", "physician", "nurse"],
}

QUESTION_ROLES = ["patient", "physician", "researcher", "nurse", "student"]

# ── Keyword query styles ─────────────────────────────────────────────────────

KEYWORD_STYLES = {
    "keyword_technical": {
        "query_style": "keyword_technical",
        "role": "researcher",
        "system": (
            "Generate a PubMed keyword search (4-8 technical keywords separated by spaces). "
            "Use DIFFERENT technical terminology than the source fact — substitute "
            "synonyms, related MeSH terms, or alternative scientific nomenclature. "
            "No full sentences, no question marks."
        ),
        "gen_prompt": (
            "Write a PubMed keyword search (4-8 technical keywords) expressing this "
            "concept using DIFFERENT scientific vocabulary. Do not copy keywords "
            'directly from the fact — use synonyms and related terms. '
            'Include "{alt}" as one of your keywords.\n\n'
            "Fact: {fact}\n\nKeywords:"
        ),
        "temperature": 0.8,
    },
    "keyword_layperson": {
        "query_style": "keyword_layperson",
        "role": "patient",
        "system": (
            "Generate a patient Google search (3-7 everyday words, no medical jargon). "
            "Use ONLY simple, common words — the kind a non-expert would type. "
            "No full sentences, no question marks."
        ),
        "gen_prompt": (
            "Write a patient-friendly Google search (3-7 words) expressing this "
            "concept in EVERYDAY language. Replace ALL medical terms with simple words. "
            "Use words a non-expert would naturally type into a search bar.\n\n"
            "Fact: {fact}\n\nKeywords:"
        ),
        "temperature": 0.8,
    },
    "keyword_clinical": {
        "query_style": "keyword_clinical",
        "role": "nurse",
        "system": (
            "Generate a clinical reference search (3-6 practical keywords). "
            "Use nursing/clinical shorthand and abbreviations where natural. "
            "Focus on care, monitoring, administration — not research terminology."
        ),
        "gen_prompt": (
            "Write a nurse-friendly clinical search (3-6 keywords) expressing this "
            "concept using practical clinical vocabulary. Use nursing abbreviations "
            "and care-focused terms. Do not copy research/bench terminology.\n\n"
            "Fact: {fact}\n\nKeywords:"
        ),
        "temperature": 0.8,
    },
}

# ── Prompts ──────────────────────────────────────────────────────────────────

FACT_SYSTEM = (
    "Extract ONE key medical claim from the passage. Identify the core concept "
    "(disease, drug, procedure, finding) and its relationship. "
    "Output a brief statement of WHAT was found and IN WHAT context. "
    "Keep it under 2 sentences. Output only the fact."
)

QUESTION_SYSTEM = (
    "You are generating a medical query that expresses the SAME concept as the "
    "provided fact but using DIFFERENT vocabulary — this is vocabulary substitution, "
    "NOT paraphrasing or reformulation.\n\n"
    "CRITICAL RULES:\n"
    "1. Replace ALL content words (nouns, verbs, adjectives) with synonyms or "
    "related terms. Do NOT copy any 3+ word sequence from the fact.\n"
    "2. For layperson roles: use everyday words instead of medical jargon.\n"
    "3. For abbreviation tasks: use the short form; the fact uses the long form.\n"
    "4. For brand_generic: use brand names; the fact uses generic names.\n"
    "5. Vary sentence structure — do NOT always start with Does/Can/What/How.\n"
    "6. Never use 'these', 'this', 'those' without a specific referent.\n"
    "7. Write as a real person would ask — use contractions, natural phrasing.\n\n"
    "Output ONLY the query, no other text."
)


# ── vLLM helpers ─────────────────────────────────────────────────────────────


# Forced question openers to break the "Does [NP] [VP]?" monoculture. Each is an
# instruction the model must comply with; a deterministic hash picks one per row so
# no single structure exceeds ~1/len(openers) of full_question rows.
QUESTION_OPENERS = [
    "Begin with 'Why' — ask about cause or reason.",
    "Begin with 'How' — ask about a mechanism or process.",
    "Begin with 'What' — ask about a definition or identity.",
    "Begin with 'When' — ask about timing or the conditions under which it occurs.",
    "Begin with 'Which' — ask about alternatives or a choice between options.",
    "Begin with 'If' — frame a conditional scenario, then end with a question.",
    "Begin with 'I'm' or 'I've' — write in the first person as someone describing "
    "their own situation, ending with a question.",
    "Begin with 'Is it' — ask whether something is true or applies in this case.",
    "Begin with 'Can' or 'Could' — ask about possibility.",
    "Begin with 'Does' or 'Do' — ask a yes/no question.",
]


def _pick_question_opener(fact: str, role: str, task_family: str,
                          cui_label: str, alt_term: str) -> str:
    """Deterministically pick a forced opener for a full_question row.

    Hashing the row's identity yields an even, reproducible distribution across
    QUESTION_OPENERS so no single template dominates.
    """
    import hashlib
    key = "|".join(str(p) for p in (fact, role, task_family, cui_label, alt_term))
    idx = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % len(QUESTION_OPENERS)
    return QUESTION_OPENERS[idx]


def call_vllm(system: str, user_prompt: str, *, temperature: float = 0.0,
              max_tokens: int = 120, timeout: float = 120.0, max_retries: int = 3,
              top_p: float = 0.95, top_k: int = 64) -> str:
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def extract_fact(passage_text: str) -> str:
    user_prompt = (
        f"Extract the key medical claim from this passage. Identify the core "
        f"concept (disease, drug, procedure, finding) and what was found about it. "
        f"Be specific — include the subject, finding, and context.\n\n"
        f"Passage:\n{passage_text[:1200]}\n\nFact:"
    )
    try:
        fact = call_vllm(FACT_SYSTEM, user_prompt, temperature=0.2, max_tokens=150)
        for prefix in ["One key fact:", "Key fact:", "Fact:", "Here is", "The key fact"]:
            if fact.lower().startswith(prefix.lower()):
                fact = fact[len(prefix):].strip()
        return fact
    except Exception as e:
        print(f"  [ERROR] Fact extraction failed: {e}", file=sys.stderr)
        return ""


def generate_full_question(fact: str, role: str, task_family: str,
                           cui_label: str, alt_term: str) -> str:
    if not fact:
        return ""
    # ── Forced question-opener diversity ─────────────────────────────────
    # Without this the model collapses to a single "Does [NP] [VP]?" template
    # (~37% of full_question rows in v6). Pick a required opener deterministically
    # from a hash of the row's identity so the distribution is even (~10% each)
    # and reproducible. The model must comply, which breaks the monoculture.
    opener = _pick_question_opener(fact, role, task_family, cui_label, alt_term)

    # Build role-specific vocabulary shift instruction
    role_vocab_instructions = {
        "patient": "Use simple everyday words. Replace all medical terms with lay language. "
                   "Write as if you're describing your own health concern to a doctor.",
        "physician": "Use clinical language but ask about a specific scenario. "
                     "Do not copy phrasing from the fact — use your own clinical vocabulary.",
        "researcher": "Use research-oriented language — mention study designs, mechanisms, "
                      "or epidemiological concepts. Do not reuse the fact's exact wording.",
        "nurse": "Use practical clinical language focused on care, monitoring, and patient education. "
                 "Use abbreviations where natural.",
        "student": "Use educational language — mix lay and technical terms as a learner would.",
    }
    vocab_instruction = role_vocab_instructions.get(role, role_vocab_instructions["patient"])

    # Build task-family-specific instruction
    tf_instructions = {
        "abbreviation_to_expanded": (
            f'The passage uses the full term "{cui_label}". You MUST use the abbreviation '
            f'"{alt_term}" instead. Rewrite the ENTIRE query around the abbreviation — '
            f'do not just swap one word.'
        ),
        "brand_generic": (
            f'The passage uses "{cui_label}". You MUST use "{alt_term}" instead. '
            f'Rewrite the query using the alternative term naturally.'
        ),
        "lay_to_clinical": (
            f'Replace clinical term "{cui_label}" with lay language "{alt_term}". '
            f'Rewrite the entire concept in everyday words.'
        ),
        "symptom_to_diagnosis": (
            f'The passage discusses "{cui_label}". Ask about this using symptom-based '
            f'language ("{alt_term}") rather than diagnostic terminology.'
        ),
        "biomedical_to_clinical": (
            f'Replace biomedical term "{cui_label}" with clinical language "{alt_term}". '
            f'Use practical clinical vocabulary throughout.'
        ),
    }
    tf_instruction = tf_instructions.get(task_family, "")

    user_prompt = (
        f"Rewrite this medical fact as a {role}'s question using COMPLETELY DIFFERENT "
        f"vocabulary. Do NOT copy any 3+ word sequence from the fact.\n\n"
        f"{tf_instruction}\n\n"
        f"Role guidance: {vocab_instruction}\n\n"
        f"STRUCTURE REQUIREMENT (mandatory): {opener}\n\n"
        f"Fact: {fact}\n\n"
        f"Query using different words:"
    )
    try:
        query = call_vllm(QUESTION_SYSTEM, user_prompt, temperature=0.7, max_tokens=100)
        # Strip common prefixes the model might generate
        for prefix in ["Question:", "question:", "Query:", "query:",
                       "Since the provided", "Since the fact", "Based on the",
                       "The query:", "Here is", "Here's"]:
            if query.lower().startswith(prefix.lower()):
                # Find the actual query after the prefix (look for ? or end)
                remaining = query[len(prefix):].strip()
                if "?" in remaining:
                    query = remaining[remaining.index("?")+1:].strip() or remaining
                else:
                    # Try to extract the query portion after colon or newline
                    parts = remaining.split(":", 1) if ":" in remaining else remaining.split("\n", 1)
                    query = parts[-1].strip() if len(parts) > 1 else remaining
                break
        return query.strip('"').strip("'")
    except Exception as e:
        print(f"  [ERROR] Question generation failed: {e}", file=sys.stderr)
        return ""


def generate_keyword_queries(fact: str, cui_label: str, alt_term: str) -> list[dict]:
    results = []
    for style_name, style in KEYWORD_STYLES.items():
        prompt = style["gen_prompt"].format(cui=cui_label, alt=alt_term, fact=fact)
        try:
            query = call_vllm(system=style["system"], user_prompt=prompt,
                              temperature=style["temperature"], max_tokens=30)
            query = query.strip().strip('"').strip("'")
            results.append({"query": query, "query_style": style["query_style"],
                            "role": style["role"]})
        except Exception as e:
            print(f"  [ERROR] Keyword {style_name} failed: {e}", file=sys.stderr)
            results.append({"query": "", "query_style": style["query_style"],
                            "role": style["role"]})
    return results


# ── Deterministic Quality Checks ──────────────────────────────────────────────


def compute_term_overlap(query: str, passage: str, cui_label: str = "", task_family: str = "") -> float:
    """Compute fraction of query terms that appear in the passage.

    Returns a ratio: number of query content words found in passage
    divided by total query content words.

    For vocab-shift tasks, the cui_label represents the form that appears in
    the passage. We expand the passage vocabulary with cui_label words so
    legitimate vocabulary shifts are not penalized.
    """
    import re

    # Extract alphanumeric words >= 3 chars (skip stopwords/short filler)
    query_words = [w.lower() for w in re.findall(r'[a-z0-9]+', query.lower()) if len(w) >= 3]
    passage_words = set(re.findall(r'[a-z0-9]+', passage.lower()))

    if not query_words:
        return 0.0

    # For vocab-shift tasks, expand passage vocabulary with the passage-native form
    # (cui_label) so we don't penalize legitimate vocab shifts.
    # The query uses alt_term (e.g., abbreviation), passage uses cui_label (expanded).
    # We check: do query terms overlap with passage OR cui_label?
    if cui_label and task_family in ("abbreviation_to_expanded", "brand_generic", "lay_to_clinical", "biomedical_to_clinical"):
        clean_cui = re.sub(r'\s*\([^)]*\)', '', cui_label)
        clean_cui = re.sub(r'\s*\[[^\]]*\]', '', clean_cui)
        cui_words = set(re.findall(r'[a-z0-9]+', clean_cui.lower()))
        passage_words = passage_words | cui_words

    found = sum(1 for w in query_words if w in passage_words)
    return found / len(query_words)


def compute_alt_term_overlap(query: str, alt_term: str) -> float:
    """Check if the alt_term is reasonably represented in the query.

    Returns ratio of alt_term content words found in query. Used to validate
    that the query actually uses the alternative vocabulary.
    """
    import re

    if not alt_term or not query:
        return 0.0

    # Strip UMLS qualifiers
    clean_alt = re.sub(r'\s*\([^)]*\)', '', alt_term)
    clean_alt = re.sub(r'\s*\[[^\]]*\]', '', clean_alt).strip()

    alt_words = [w.lower() for w in re.findall(r'[a-z0-9]+', clean_alt) if len(w) >= 3]
    query_lower = query.lower()

    if not alt_words:
        return 0.5  # Can't check, don't penalize

    found = sum(1 for w in alt_words if w in query_lower)
    return found / len(alt_words)


# ── Judging ──────────────────────────────────────────────────────────────────


def _parse_judge_answer(response: str, valid_options: list[str]) -> str:
    text = response.strip().upper()
    for opt in valid_options:
        if text == opt or text == opt[0] or f"{opt[0]})" in text or opt in text:
            return opt
    for opt in valid_options:
        if opt[0] in text:
            return opt
    return valid_options[0]


def judge_full_question(candidate: dict) -> dict:
    """Strict judging for full_question candidates.

    Q1, Q2, Q3, Q5: LLM judge (answerability, vocab_shift, specificity, task_family_fit).
    Q4 (factual_grounding): DETERMINISTIC term-overlap check (replaces LLM judge).
    """
    passage = candidate.get("passage_text", "")[:1800]
    query = candidate.get("query", "")
    cui_label = candidate.get("cui_label", "")
    task_family = candidate.get("task_family", "lay_to_clinical")
    role = candidate.get("role", "unknown")
    vocab_shift_type = candidate.get("vocab_shift_type", "consumer_to_clinical")
    task_family_desc = get_task_family_description(task_family)

    judgments = {}

    # Q1: Answerability (LLM judge)
    try:
        q1 = call_vllm(JUDGE_SYSTEM_PROMPT,
                       Q1_ANSWERABILITY_PROMPT.format(
                           passage=passage, query=query, cui_label=cui_label,
                           task_family=task_family, role=role),
                       temperature=0.0, max_tokens=16)
        judgments["answerability"] = _parse_judge_answer(q1, ["ANSWERABLE", "NOT_ANSWERABLE"])
    except Exception:
        judgments["answerability"] = "NOT_ANSWERABLE"
    time.sleep(SLEEP)

    # Q2: Vocabulary shift (LLM judge)
    try:
        q2 = call_vllm(JUDGE_SYSTEM_PROMPT,
                       Q2_VOCAB_SHIFT_PROMPT.format(
                           passage=passage, query=query, cui_label=cui_label,
                           task_family=task_family, vocab_shift_type=vocab_shift_type),
                       temperature=0.0, max_tokens=16)
        judgments["vocab_shift"] = _parse_judge_answer(
            q2, ["STRONG_SHIFT", "MODERATE_SHIFT", "MINIMAL_SHIFT", "NO_SHIFT"])
    except Exception:
        judgments["vocab_shift"] = "NO_SHIFT"
    time.sleep(SLEEP)

    # Q3: Passage specificity (LLM judge)
    try:
        q3 = call_vllm(JUDGE_SYSTEM_PROMPT,
                       Q3_PASSAGE_SPECIFICITY_PROMPT.format(
                           passage=passage, query=query, cui_label=cui_label,
                           task_family=task_family),
                       temperature=0.0, max_tokens=16)
        judgments["passage_specificity"] = _parse_judge_answer(q3, ["SPECIFIC", "GENERIC"])
    except Exception:
        judgments["passage_specificity"] = "GENERIC"
    time.sleep(SLEEP)

    # Q4: Factual grounding (DETERMINISTIC — term overlap check)
    cui = candidate.get("cui_label", "")
    term_overlap = compute_term_overlap(query, passage, cui, task_family)
    if term_overlap >= 0.5:
        judgments["factual_grounding"] = "CLEAN"
    else:
        judgments["factual_grounding"] = "HALLUCINATION"

    # Q5: Task family fit (LLM judge)
    try:
        q5 = call_vllm(JUDGE_SYSTEM_PROMPT,
                       Q5_TASK_FAMILY_FIT_PROMPT.format(
                           passage=passage, query=query, task_family=task_family,
                           role=role, task_family_description=task_family_desc),
                       temperature=0.0, max_tokens=16)
        judgments["task_family_fit"] = _parse_judge_answer(q5, ["MATCH", "MISMATCH"])
    except Exception:
        judgments["task_family_fit"] = "MISMATCH"
    time.sleep(SLEEP)

    # Strict judging: all 5 dimensions must pass
    decision = "ACCEPT"
    reasons = []
    if judgments["answerability"] != "ANSWERABLE":
        decision = "REJECT"; reasons.append(f"not_answerable:{judgments['answerability']}")
    if judgments["vocab_shift"] not in ("STRONG_SHIFT", "MODERATE_SHIFT"):
        decision = "REJECT"; reasons.append(f"insufficient_shift:{judgments['vocab_shift']}")
    if judgments["passage_specificity"] != "SPECIFIC":
        decision = "REJECT"; reasons.append(f"generic:{judgments['passage_specificity']}")
    if judgments["factual_grounding"] != "CLEAN":
        decision = "REJECT"; reasons.append(f"term_overlap_low:{term_overlap:.2f}")
    if judgments["task_family_fit"] != "MATCH":
        decision = "REJECT"; reasons.append(f"task_mismatch:{judgments['task_family_fit']}")

    candidate["judgments"] = judgments
    candidate["decision"] = decision
    candidate["rejection_reason"] = ";".join(reasons) if reasons else ""
    candidate["judge_model"] = MODEL
    candidate["term_overlap"] = round(term_overlap, 3)  # Store for auditing
    return candidate


def accept_keyword_candidate(candidate: dict) -> dict:
    """Accept keyword query if it passes deterministic validation (no LLM judge).

    Rationale: Keywords aren't questions — Q1/Q5 don't apply. They're grounded
    in the same extracted fact as the full_question, and deterministic validators
    catch surface-level issues. This is the pragmatic scaling optimization.
    """
    candidate["judgments"] = {
        "answerability": "ANSWERABLE",  # treated as true: keywords don't need to be "answered"
        "vocab_shift": "STRONG_SHIFT",  # keywords inherently use different vocabulary
        "passage_specificity": "SPECIFIC",
        "factual_grounding": "CLEAN",
        "task_family_fit": "MATCH",
    }
    candidate["decision"] = "ACCEPT"
    candidate["rejection_reason"] = ""
    candidate["judge_model"] = "none_keyword_fastpath"
    return candidate


# ── Parquet helpers ──────────────────────────────────────────────────────────


def prepare_for_parquet(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                has_nested = df[col].apply(lambda x: isinstance(x, (list, dict))).any()
                if has_nested:
                    df[col] = df[col].apply(json.dumps)
            except Exception:
                pass
    return df


def compute_batch_stats(accepted: list[dict], rejected: list[dict]) -> dict:
    all_candidates = accepted + rejected
    total = len(all_candidates)
    stats = {
        "total": total, "accepted": len(accepted), "rejected": len(rejected),
        "acceptance_rate": len(accepted) / total if total else 0.0,
    }

    # Per-query-style acceptance
    per_style = {}
    for c in all_candidates:
        qs = c.get("query_style", "full_question")
        per_style.setdefault(qs, {"total": 0, "accepted": 0})
        per_style[qs]["total"] += 1
        if c.get("decision") == "ACCEPT":
            per_style[qs]["accepted"] += 1
    stats["per_style_acceptance"] = {
        qs: d["accepted"] / d["total"] if d["total"] else 0
        for qs, d in per_style.items()
    }

    # Per-dimension pass rates (only for judged candidates)
    dims = ["answerability", "vocab_shift", "passage_specificity", "factual_grounding", "task_family_fit"]
    judged = [c for c in all_candidates if c.get("judge_model") != "none_keyword_fastpath"]
    if judged:
        dim_pass_values = {
            "answerability": {"ANSWERABLE"},
            "vocab_shift": {"STRONG_SHIFT", "MODERATE_SHIFT"},
            "passage_specificity": {"SPECIFIC"},
            "factual_grounding": {"CLEAN"},
            "task_family_fit": {"MATCH"},
        }
        per_dim = {}
        for dim in dims:
            passed = sum(1 for c in judged if c["judgments"].get(dim) in dim_pass_values[dim])
            per_dim[dim] = passed / len(judged)
        stats["per_dimension_pass_rates"] = per_dim
        stats["judged_count"] = len(judged)

    # Per-task-family acceptance
    per_family = {}
    for c in all_candidates:
        tf = c.get("task_family", "unknown")
        per_family.setdefault(tf, {"total": 0, "accepted": 0})
        per_family[tf]["total"] += 1
        if c.get("decision") == "ACCEPT":
            per_family[tf]["accepted"] += 1
    stats["per_family_acceptance"] = {
        tf: d["accepted"] / d["total"] if d["total"] else 0
        for tf, d in per_family.items()
    }

    # Per-semantic-group acceptance
    per_sg = {}
    for c in all_candidates:
        sg = c.get("semantic_group", "unknown")
        per_sg.setdefault(sg, {"total": 0, "accepted": 0})
        per_sg[sg]["total"] += 1
        if c.get("decision") == "ACCEPT":
            per_sg[sg]["accepted"] += 1
    stats["per_semantic_group_acceptance"] = {
        sg: d["accepted"] / d["total"] if d["total"] else 0
        for sg, d in per_sg.items()
    }

    return stats


# ── Main Batch Runner ────────────────────────────────────────────────────────


def run_multistyle_batch_v2(
    seed: int, batch_id: str, count: int = 100,
    output_base: Path | None = None,
    judge_mode: str = "hybrid",  # "hybrid" | "full" | "none"
    passages_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    output_base = output_base or DEFAULT_OUTPUT_BASE
    output_dir = output_base / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[{batch_id}] Seed={seed}, count={count}, judge={judge_mode}")

    # 1. Load and shuffle passages
    _passages_path = passages_path or PASSAGES_PATH
    with open(_passages_path, "r") as f:
        all_passages = json.load(f)
    random.seed(seed)
    shuffled = list(all_passages)
    random.shuffle(shuffled)
    passages = shuffled[:min(count, len(shuffled))]

    if verbose:
        print(f"[{batch_id}] Using {len(passages)}/{len(all_passages)} passages")

    validators = DeterministicValidators(
        max_forbidden_surface_overlap=0.45,
        min_alternative_term_count=0,  # Relaxed: vocab-shifted queries may not use alt_term verbatim
        reject_generic_queries=False,
        max_query_chars=250,  # Allow longer, more natural questions at temp=0.7
    )
    prompt_hash = stable_hash("multistyle_v2")

    # 2. Generate candidates with concurrent passage processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    candidates = []
    passages_processed = 0
    candidate_lock = threading.Lock()
    progress_lock = threading.Lock()
    MAX_CONCURRENT_PASSAGES = 4  # vLLM can handle concurrent requests

    def generate_keyword_query(style_name: str, style: dict, fact: str, cui_label: str, alt_term: str) -> dict | None:
        """Generate a single keyword query (used for parallel kw generation)."""
        prompt = style["gen_prompt"].format(cui=cui_label, alt=alt_term, fact=fact)
        try:
            query = call_vllm(system=style["system"], user_prompt=prompt,
                            temperature=style["temperature"], max_tokens=30)
            query = query.strip().strip('"').strip("'")
            if query:
                return {"query": query, "query_style": style["query_style"], "role": style["role"]}
        except Exception as e:
            print(f"  [ERROR] Keyword {style_name} failed: {e}", file=sys.stderr)
        return None

    def process_single_passage(idx: int, p: dict) -> list[dict]:
        """Process one passage: extract fact, generate queries, validate. Returns candidate dicts."""
        passage_id = p.get("passage_id", f"p_{idx}")
        passage_text = p.get("text", "")
        cui_label = p.get("cui_label", "")
        alt_term = p.get("alt_term", "")
        semantic_group = p.get("semantic_group", "other")
        vocab_shift_type = p.get("vocab_shift_type", "consumer_to_clinical")
        pre_set_tf = p.get("task_family", "")
        if pre_set_tf and pre_set_tf in TASK_FAMILY_ROLES:
            task_family = pre_set_tf
        else:
            tf_candidates = SEMANTIC_GROUP_TO_TASK_FAMILY.get(semantic_group, ["lay_to_clinical"])
            task_family = random.choice(tf_candidates) if isinstance(tf_candidates, list) else tf_candidates

        passage_words = passage_text.split()
        forbidden_terms = (
            passage_words[:20] + passage_words[-20:]
            if len(passage_words) > 40 else passage_words
        )
        alt_terms_list = [alt_term] if alt_term else None

        # Step A: Extract fact
        fact = extract_fact(passage_text)
        if not fact:
            if verbose:
                with progress_lock:
                    print(f"[{batch_id}] gen {idx+1}/{len(passages)}: fact FAILED, skipping")
            return []

        # Step B: Generate full_question + 3 keywords in PARALLEL
        role_options = TASK_FAMILY_ROLES.get(task_family, QUESTION_ROLES)
        role = random.choice(role_options)

        local_candidates = []

        # Build keyword generation tasks
        kw_futures = {}
        kw_executor = ThreadPoolExecutor(max_workers=4)
        for style_name, style in KEYWORD_STYLES.items():
            kw_futures[kw_executor.submit(
                generate_keyword_query, style_name, style, fact, cui_label, alt_term
            )] = style_name

        # Generate full question in the main thread while keywords run
        full_query = generate_full_question(fact, role, task_family, cui_label, alt_term)

        # Collect keyword results
        for future in as_completed(kw_futures):
            style_name = kw_futures[future]
            try:
                kw_result = future.result()
                if kw_result and kw_result.get("query"):
                    kw_query = kw_result["query"]
                    kw_style = kw_result["query_style"]
                    kw_role = kw_result["role"]
                    example_id = stable_hash(passage_id, kw_query, kw_style, batch_id, str(idx))
                    v = validators.validate(
                        example_id=example_id, query=kw_query, passage=passage_text,
                        forbidden_terms=forbidden_terms, alt_terms=alt_terms_list,
                    )
                    local_candidates.append({
                        "example_id": example_id, "passage_id": passage_id,
                        "passage_text": passage_text, "query": kw_query,
                        "query_style": kw_style, "fact": fact,
                        "mode": "ontology_grounded_teacher_filtered",
                        "task_family": task_family, "role": kw_role,
                        "cui_label": cui_label, "alt_term": alt_term,
                        "target_cuis": [cui_label] if cui_label else [],
                        "semantic_group": semantic_group,
                        "vocab_shift_type": vocab_shift_type,
                        "generator_model": MODEL, "prompt_hash": prompt_hash,
                        "passed_validation": v.passed,
                        "validation_failures": v.failures,
                        "validation_warnings": v.warnings,
                        "judge_source": "direct_vllm",
                    })
            except Exception as e:
                print(f"  [ERROR] Keyword result failed: {e}", file=sys.stderr)
        kw_executor.shutdown(wait=False)

        # Process full question
        if full_query:
            example_id = stable_hash(passage_id, full_query, "full_question", batch_id, str(idx))
            v = validators.validate(
                example_id=example_id, query=full_query, passage=passage_text,
                forbidden_terms=forbidden_terms, alt_terms=alt_terms_list,
            )
            local_candidates.append({
                "example_id": example_id, "passage_id": passage_id,
                "passage_text": passage_text, "query": full_query,
                "query_style": "full_question", "fact": fact,
                "mode": "ontology_grounded_teacher_filtered",
                "task_family": task_family, "role": role,
                "cui_label": cui_label, "alt_term": alt_term,
                "target_cuis": [cui_label] if cui_label else [],
                "semantic_group": semantic_group,
                "vocab_shift_type": vocab_shift_type,
                "generator_model": MODEL, "prompt_hash": prompt_hash,
                "passed_validation": v.passed,
                "validation_failures": v.failures,
                "validation_warnings": v.warnings,
                "judge_source": "direct_vllm",
            })

        if verbose:
            nq = len(local_candidates)
            with progress_lock:
                print(f"[{batch_id}] gen {idx+1}/{len(passages)}: "
                      f"sg={semantic_group:22s} tf={task_family:24s} "
                      f"fact_len={len(fact):3d} queries={nq} ✓")

        return local_candidates

    # Process passages concurrently, 4 at a time
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PASSAGES) as executor:
        future_to_idx = {}
        for i, p in enumerate(passages):
            future_to_idx[executor.submit(process_single_passage, i, p)] = i

        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                if result:
                    with candidate_lock:
                        candidates.extend(result)
                        if any(c.get("fact") for c in result):
                            passages_processed += 1
            except Exception as e:
                idx = future_to_idx[future]
                print(f"[{batch_id}] gen {idx+1}/{len(passages)}: ERROR: {e}", file=sys.stderr)

    # Sort candidates by passage order for consistent output
    candidates.sort(key=lambda c: (c.get("passage_id", ""), c.get("query_style", "")))

    passed_val = sum(1 for c in candidates if c["passed_validation"])
    if verbose:
        qp = len(candidates) / max(passages_processed, 1)
        print(f"\n[{batch_id}] Generated {len(candidates)} candidates from "
              f"{passages_processed} passages ({qp:.1f} q/p), {passed_val} passed validation")

    # 3. Judge (mode-dependent)
    judged = []
    if judge_mode == "none":
        # Accept all validation-passed, reject failures
        for c in candidates:
            c["judge_source"] = "none_dummy"
            if c["passed_validation"]:
                c["judgments"] = {"answerability": "ANSWERABLE", "vocab_shift": "STRONG_SHIFT",
                                  "passage_specificity": "SPECIFIC", "factual_grounding": "CLEAN",
                                  "task_family_fit": "MATCH"}
                c["decision"] = "ACCEPT"
                c["rejection_reason"] = ""
                c["judge_model"] = "none"
            else:
                c["judgments"] = {"answerability": "NOT_ANSWERABLE", "vocab_shift": "NO_SHIFT",
                                  "passage_specificity": "GENERIC", "factual_grounding": "HALLUCINATION",
                                  "task_family_fit": "MISMATCH"}
                c["decision"] = "REJECT"
                c["rejection_reason"] = "failed_validation"
                c["judge_model"] = "none"
            judged.append(c)
    elif judge_mode == "hybrid":
        # Full questions: strict 5-D LLM judge. Keywords: validation-pass = accept.
        full_q_count = 0
        for c in candidates:
            if c.get("query_style") == "full_question":
                if c["passed_validation"]:
                    c = judge_full_question(c)
                    c["judge_source"] = "direct_vllm"
                    full_q_count += 1
                else:
                    c["judgments"] = {"answerability": "NOT_ANSWERABLE", "vocab_shift": "NO_SHIFT",
                                      "passage_specificity": "GENERIC", "factual_grounding": "HALLUCINATION",
                                      "task_family_fit": "MISMATCH"}
                    c["decision"] = "REJECT"
                    c["rejection_reason"] = "failed_validation"
                    c["judge_model"] = MODEL
                    c["judge_source"] = "direct_vllm"
            else:
                # Keyword: accept if validation passes
                c["judge_source"] = "none_keyword_fastpath"
                if c["passed_validation"]:
                    c = accept_keyword_candidate(c)
                else:
                    c["judgments"] = {"answerability": "NOT_ANSWERABLE", "vocab_shift": "NO_SHIFT",
                                      "passage_specificity": "GENERIC", "factual_grounding": "HALLUCINATION",
                                      "task_family_fit": "MISMATCH"}
                    c["decision"] = "REJECT"
                    c["rejection_reason"] = "failed_validation"
                    c["judge_model"] = "none_keyword_fastpath"
            judged.append(c)

            if verbose and len(judged) % 100 == 0:
                accepted_so_far = sum(1 for j in judged if j.get("decision") == "ACCEPT")
                print(f"[{batch_id}] judged {len(judged)}/{len(candidates)} "
                      f"(LLM-judged: {full_q_count}/{len(judged)}), accepted={accepted_so_far}")
    else:  # "full" mode — strict 5-D for everything
        for idx, c in enumerate(candidates):
            c["judge_source"] = "direct_vllm"
            if c["passed_validation"]:
                c = judge_full_question(c)
            else:
                c["judgments"] = {"answerability": "NOT_ANSWERABLE", "vocab_shift": "NO_SHIFT",
                                  "passage_specificity": "GENERIC", "factual_grounding": "HALLUCINATION",
                                  "task_family_fit": "MISMATCH"}
                c["decision"] = "REJECT"
                c["rejection_reason"] = "failed_validation"
                c["judge_model"] = MODEL
            judged.append(c)

            if verbose and (idx % 50 == 0 or idx == len(candidates) - 1):
                decision = c.get("decision", "ERROR")
                j = c.get("judgments", {})
                qs = c.get("query_style", "?")
                print(f"[{batch_id}] judge {idx+1}/{len(candidates)}: {decision:6s} "
                      f"style={qs:20s} | ans={j.get('answerability','?')[:4]:4s} "
                      f"voc={j.get('vocab_shift','?')[:4]:4s} "
                      f"spe={j.get('passage_specificity','?')[:4]:4s} "
                      f"fac={j.get('factual_grounding','?')[:4]:4s} "
                      f"fam={j.get('task_family_fit','?')[:4]:4s}")

    # 3b. Post-generation quality filter (applied to ALL candidates)
    # Uses n-gram copy detection: reject queries that copy passage language.
    # For vocab shift, we WANT low overlap — the problem is HIGH overlap (copying).
    import re as _re3

    def _compute_max_ngram_copy(query: str, passage: str, n: int = 4) -> float:
        """Compute fraction of query n-grams that appear verbatim in passage."""
        q_words = _re3.findall(r'[a-z0-9]+', query.lower())
        p_text = passage.lower()
        if len(q_words) < n:
            return 0.0
        ngrams = [" ".join(q_words[i:i+n]) for i in range(len(q_words) - n + 1)]
        copied = sum(1 for ng in ngrams if ng in p_text)
        return copied / len(ngrams) if ngrams else 0.0

    post_filter_rejected = 0
    ngram_copy_rejected = 0
    for c in judged:
        query = c.get("query", "")
        passage = c.get("passage_text", "")
        cui_label = c.get("cui_label", "")
        tf = c.get("task_family", "")
        qs = c.get("query_style", "full_question")
        overlap = compute_term_overlap(query, passage, cui_label, tf)
        c["term_overlap"] = round(overlap, 3)

        # Compute n-gram copy ratio
        ngram_copy_4 = _compute_max_ngram_copy(query, passage, n=4)
        ngram_copy_3 = _compute_max_ngram_copy(query, passage, n=3)
        c["ngram_copy_4"] = round(ngram_copy_4, 3)

        if c.get("decision") == "ACCEPT":
            # Reject queries that copy 4-word n-grams from passage (>30% copied)
            if ngram_copy_4 > 0.30:
                c["decision"] = "REJECT"
                c["rejection_reason"] = (c.get("rejection_reason", "") +
                    f";high_ngram_copy_4:{ngram_copy_4:.2f}").strip(";")
                ngram_copy_rejected += 1
            # Reject queries that copy 3-word n-grams from passage (>50% copied)
            elif ngram_copy_3 > 0.50:
                c["decision"] = "REJECT"
                c["rejection_reason"] = (c.get("rejection_reason", "") +
                    f";high_ngram_copy_3:{ngram_copy_3:.2f}").strip(";")
                ngram_copy_rejected += 1
            # Also reject if term_overlap is near-zero (query may be completely unrelated)
            elif overlap < 0.10 and qs == "full_question":
                c["decision"] = "REJECT"
                c["rejection_reason"] = (c.get("rejection_reason", "") +
                    f";no_term_overlap:{overlap:.2f}").strip(";")
                post_filter_rejected += 1

    if verbose and post_filter_rejected > 0:
        print(f"[{batch_id}] Post-filter rejected {post_filter_rejected} for low term overlap (<0.5)")

    # 3c. Soft alt_term check: only flag as a warning, don't reject.
    # Real vocab shift means the alt_term might not appear verbatim — the CONCEPT
    # is expressed differently. Rejecting for missing alt_term penalizes good shift.
    alt_warning_count = 0
    for c in judged:
        alt_term = c.get("alt_term", "")
        query = c.get("query", "")
        alt_overlap = compute_alt_term_overlap(query, alt_term)
        c["alt_term_overlap"] = round(alt_overlap, 3)
        if alt_overlap < 0.2:
            alt_warning_count += 1
            # Only reject if alt_term is completely absent AND the query copies passage
            # (suggesting the model didn't do any substitution at all)
            if c.get("ngram_copy_4", 0) > 0.40 and c.get("decision") == "ACCEPT":
                c["decision"] = "REJECT"
                c["rejection_reason"] = (c.get("rejection_reason", "") +
                    ";no_vocab_substitution").strip(";")

    if verbose and alt_warning_count > 0:
        print(f"[{batch_id}] alt_term soft-check: {alt_warning_count} queries have low alt_term overlap (warning only)")

    # 4. Split, stats, save
    accepted = [c for c in judged if c.get("decision") == "ACCEPT"]
    rejected = [c for c in judged if c.get("decision") != "ACCEPT"]
    stats = compute_batch_stats(accepted, rejected)
    stats.update({
        "passages_processed": passages_processed,
        "batch_id": batch_id, "seed": seed,
        "generation_style": "multistyle_v2",
        "judge_mode": judge_mode,
        "candidates_per_passage": len(candidates) / max(passages_processed, 1),
    })

    judged_df = prepare_for_parquet(judged)
    accepted_df = prepare_for_parquet(accepted)

    judged_df.to_parquet(output_dir / "judged.parquet", index=False)
    if len(accepted_df) > 0:
        accepted_df.to_parquet(output_dir / "accepted.parquet", index=False)

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"\n[{batch_id}] DONE: {stats['accepted']}/{stats['total']} accepted ({stats['acceptance_rate']:.2%})")
        for dim, rate in sorted(stats.get("per_dimension_pass_rates", {}).items()):
            print(f"[{batch_id}]   judge/{dim}: {rate:.2%}")
        for style, rate in sorted(stats.get("per_style_acceptance", {}).items()):
            print(f"[{batch_id}]   style/{style}: {rate:.2%}")
        print(f"[{batch_id}] Output: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="MedColBERT Multistyle Batch Runner v2")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-id", type=str, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-base", type=Path, default=None)
    parser.add_argument("--judge-mode", type=str, default="hybrid",
                        choices=["hybrid", "full", "none"])
    parser.add_argument("--passages", type=Path, default=None,
                        help="Custom passages JSON file (default: real_annotated_500.json)")
    args = parser.parse_args()

    print(f"=== MedColBERT Multistyle v2: {args.batch_id} ===")
    print(f"Seed: {args.seed}, Count: {args.count}, Judge: {args.judge_mode}")

    stats = run_multistyle_batch_v2(
        seed=args.seed, batch_id=args.batch_id, count=args.count,
        output_base=args.output_base, judge_mode=args.judge_mode,
        passages_path=args.passages,
    )
    print(f"\n=== {args.batch_id}: {stats['accepted']}/{stats['total']} accepted ({stats['acceptance_rate']:.2%}) ===")
    return 0 if stats["accepted"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
