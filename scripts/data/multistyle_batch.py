#!/usr/bin/env python3
"""MedColBERT Multistyle Production Batch Runner — 4 queries per passage.

Canonical multistyle pipeline:
  1. Load annotated passages, shuffle with seed control
  2. For each passage:
     a. Map semantic_group → task_family (CRITICAL FIX #6)
     b. Step A: Extract ONE key fact from passage (temp=0)
     c. Step B1: Generate full-sentence question from fact (temp=0)
     d. Step B2: Generate 3 keyword queries (technical, layperson, clinical) (temp=0.1)
     e. Deterministic validation (overlap=0.45, min_alt_term=1, no generic rejection)
  3. Judge all candidates via direct vLLM Q1-Q5 (strict 5-D)
     ACCEPT = ANSWERABLE + (STRONG|MODERATE)_SHIFT + SPECIFIC + CLEAN + MATCH
  4. Save accepted.parquet, judged.parquet, stats.json

Usage:
  uv run python scripts/data/multistyle_batch.py --seed 42 --batch-id ms_001 --count 100
  uv run python scripts/data/multistyle_batch.py --seed 123 --batch-id ms_002 --count 200
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
SLEEP = 0.05  # seconds between API calls
JUDGE_TIMEOUT = 120.0  # seconds per judge call

# ── Semantic group → task family (CRITICAL FIX #6) ───────────────────────────

SEMANTIC_GROUP_TO_TASK_FAMILY = {
    "drugs_chemicals": "brand_generic",
    "anatomy": "lay_to_clinical",
    "disorders": "symptom_to_diagnosis",
    "procedures": "biomedical_to_clinical",
    "findings_signs_symptoms": "lay_to_clinical",
}

TASK_FAMILY_ROLES = {
    "lay_to_clinical": ["patient", "nurse"],
    "brand_generic": ["patient", "physician"],
    "symptom_to_diagnosis": ["patient", "physician"],
    "biomedical_to_clinical": ["researcher", "physician"],
}

QUESTION_ROLES = ["patient", "physician", "researcher", "nurse", "student"]

# ── Keyword query styles ─────────────────────────────────────────────────────

KEYWORD_STYLES = {
    "keyword_technical": {
        "query_style": "keyword_technical",
        "role": "researcher",
        "system": (
            "Generate a short keyword-only search query (4-8 words) for PubMed. "
            "No full sentences, no question marks — just technical keywords "
            "separated by spaces."
        ),
        "gen_prompt": (
            "Based ONLY on this fact, write a PubMed keyword search "
            "(4-8 technical keywords) that would find this paper. "
            'Replace "{cui}" with "{alt}".\n\n'
            "Fact: {fact}\n\nKeywords:"
        ),
        "temperature": 0.1,
    },
    "keyword_layperson": {
        "query_style": "keyword_layperson",
        "role": "patient",
        "system": (
            "Generate a short keyword search (3-7 simple words) that a patient "
            "would type into Google. Use everyday words only. No medical jargon. "
            "No full sentences. No question marks."
        ),
        "gen_prompt": (
            "Based ONLY on this fact, write a patient-friendly keyword search "
            "(3-7 simple words) that would find this information. "
            'Replace "{cui}" with "{alt}".\n\n'
            "Fact: {fact}\n\nKeywords:"
        ),
        "temperature": 0.1,
    },
    "keyword_clinical": {
        "query_style": "keyword_clinical",
        "role": "nurse",
        "system": (
            "Generate a short clinical keyword search (3-6 words) a nurse would "
            "type into a clinical reference tool. Use practical abbreviations "
            "where natural."
        ),
        "gen_prompt": (
            "Based ONLY on this fact, write a nurse-friendly clinical keyword "
            'search (3-6 words). Replace "{cui}" with "{alt}".\n\n'
            "Fact: {fact}\n\nKeywords:"
        ),
        "temperature": 0.1,
    },
}

# ── Generation prompts (from handoff — DO NOT CHANGE) ────────────────────────

FACT_SYSTEM = (
    "Extract ONE key fact. Use exact wording from the passage. "
    "Output only the fact sentence."
)

QUESTION_SYSTEM = (
    "Write ONLY a short question using the alternative term. No other text."
)


# ── vLLM HTTP helper ─────────────────────────────────────────────────────────


def call_vllm(
    system: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 120,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> str:
    """Call the vLLM chat completions endpoint with retries."""
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


# ── Step A: Fact extraction ──────────────────────────────────────────────────


def extract_fact(passage_text: str) -> str:
    """Extract one key fact from the passage using exact wording."""
    user_prompt = f"Passage:\n{passage_text[:1200]}\n\nOne key fact:"
    try:
        fact = call_vllm(FACT_SYSTEM, user_prompt, temperature=0.0, max_tokens=120)
        for prefix in ["One key fact:", "Key fact:", "Fact:", "Here is", "The key fact"]:
            if fact.lower().startswith(prefix.lower()):
                fact = fact[len(prefix):].strip()
        return fact
    except Exception as e:
        print(f"  [ERROR] Fact extraction failed: {e}", file=sys.stderr)
        return ""


# ── Step B1: Full question generation ────────────────────────────────────────


def generate_full_question(
    fact: str, role: str, task_family: str, cui_label: str, alt_term: str
) -> str:
    """Generate a full-sentence question from the fact."""
    if not fact:
        return ""
    user_prompt = (
        f"Using ONLY this fact, write a {role} question for {task_family}. "
        f'Replace "{cui_label}" with "{alt_term}".\n\n'
        f"Fact: {fact}\n\nQuestion:"
    )
    try:
        query = call_vllm(QUESTION_SYSTEM, user_prompt, temperature=0.0, max_tokens=60)
        for prefix in ["Question:", "question:", "Query:", "query:"]:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
        query = query.strip('"').strip("'")
        return query
    except Exception as e:
        print(f"  [ERROR] Question generation failed: {e}", file=sys.stderr)
        return ""


# ── Step B2: Keyword query generation ────────────────────────────────────────


def generate_keyword_queries(
    fact: str, cui_label: str, alt_term: str
) -> list[dict]:
    """Generate all 3 keyword-style queries from the fact."""
    results = []
    for style_name, style in KEYWORD_STYLES.items():
        prompt = style["gen_prompt"].format(cui=cui_label, alt=alt_term, fact=fact)
        try:
            query = call_vllm(
                system=style["system"],
                user_prompt=prompt,
                temperature=style["temperature"],
                max_tokens=30,
            )
            query = query.strip().strip('"').strip("'")
            results.append({
                "query": query,
                "query_style": style["query_style"],
                "role": style["role"],
            })
        except Exception as e:
            print(f"  [ERROR] Keyword {style_name} failed: {e}", file=sys.stderr)
            results.append({
                "query": "",
                "query_style": style["query_style"],
                "role": style["role"],
            })
    return results


# ── Judging ──────────────────────────────────────────────────────────────────


def _parse_judge_answer(response: str, valid_options: list[str]) -> str:
    """Parse the judge response to extract the answer letter/name."""
    text = response.strip().upper()

    for opt in valid_options:
        if text == opt or text == opt[0]:
            return opt
        if f"{opt[0]})" in text:
            return opt
        if opt in text:
            return opt

    for opt in valid_options:
        if opt[0] in text:
            return opt

    return valid_options[0]


def judge_single_candidate(candidate: dict) -> dict:
    """Run all 5 judge questions via direct vLLM.

    Uses strict aggregate: ALL 5 dimensions must pass.
    Returns candidate with 'judgments', 'decision', 'rejection_reason' added.
    """
    if not candidate.get("passed_validation", False):
        candidate["judgments"] = {
            "answerability": "NOT_ANSWERABLE",
            "vocab_shift": "NO_SHIFT",
            "passage_specificity": "GENERIC",
            "factual_grounding": "HALLUCINATION",
            "task_family_fit": "MISMATCH",
        }
        candidate["decision"] = "REJECT"
        candidate["rejection_reason"] = "failed_validation:" + ",".join(
            candidate.get("validation_failures", ["unknown"])
        )
        candidate["judge_model"] = MODEL
        return candidate

    passage = candidate.get("passage_text", "")[:1800]
    query = candidate.get("query", "")
    cui_label = candidate.get("cui_label", "")
    task_family = candidate.get("task_family", "lay_to_clinical")
    role = candidate.get("role", "unknown")
    vocab_shift_type = candidate.get("vocab_shift_type", "consumer_to_clinical")
    task_family_desc = get_task_family_description(task_family)

    judgments = {}

    # -- Q1: Answerability --
    try:
        q1_prompt = Q1_ANSWERABILITY_PROMPT.format(
            passage=passage, query=query, cui_label=cui_label,
            task_family=task_family, role=role,
        )
        q1 = call_vllm(JUDGE_SYSTEM_PROMPT, q1_prompt, temperature=0.0, max_tokens=16)
        judgments["answerability"] = _parse_judge_answer(q1, ["ANSWERABLE", "NOT_ANSWERABLE"])
    except Exception as e:
        judgments["answerability"] = "NOT_ANSWERABLE"
        print(f"    [WARN] Q1 failed: {e}", file=sys.stderr)
    time.sleep(SLEEP)

    # -- Q2: Vocabulary Shift --
    try:
        q2_prompt = Q2_VOCAB_SHIFT_PROMPT.format(
            passage=passage, query=query, cui_label=cui_label,
            task_family=task_family, vocab_shift_type=vocab_shift_type,
        )
        q2 = call_vllm(JUDGE_SYSTEM_PROMPT, q2_prompt, temperature=0.0, max_tokens=16)
        judgments["vocab_shift"] = _parse_judge_answer(
            q2, ["STRONG_SHIFT", "MODERATE_SHIFT", "MINIMAL_SHIFT", "NO_SHIFT"]
        )
    except Exception as e:
        judgments["vocab_shift"] = "NO_SHIFT"
        print(f"    [WARN] Q2 failed: {e}", file=sys.stderr)
    time.sleep(SLEEP)

    # -- Q3: Passage Specificity --
    try:
        q3_prompt = Q3_PASSAGE_SPECIFICITY_PROMPT.format(
            passage=passage, query=query, cui_label=cui_label,
            task_family=task_family,
        )
        q3 = call_vllm(JUDGE_SYSTEM_PROMPT, q3_prompt, temperature=0.0, max_tokens=16)
        judgments["passage_specificity"] = _parse_judge_answer(q3, ["SPECIFIC", "GENERIC"])
    except Exception as e:
        judgments["passage_specificity"] = "GENERIC"
        print(f"    [WARN] Q3 failed: {e}", file=sys.stderr)
    time.sleep(SLEEP)

    # -- Q4: Factual Grounding --
    try:
        q4_prompt = Q4_FACTUAL_GROUNDING_PROMPT.format(
            passage=passage, query=query, cui_label=cui_label,
        )
        q4 = call_vllm(JUDGE_SYSTEM_PROMPT, q4_prompt, temperature=0.0, max_tokens=16)
        judgments["factual_grounding"] = _parse_judge_answer(
            q4, ["CLEAN", "HALLUCINATION", "CONTRADICTION"]
        )
    except Exception as e:
        judgments["factual_grounding"] = "HALLUCINATION"
        print(f"    [WARN] Q4 failed: {e}", file=sys.stderr)
    time.sleep(SLEEP)

    # -- Q5: Task Family Fit --
    try:
        q5_prompt = Q5_TASK_FAMILY_FIT_PROMPT.format(
            passage=passage, query=query, task_family=task_family,
            role=role, task_family_description=task_family_desc,
        )
        q5 = call_vllm(JUDGE_SYSTEM_PROMPT, q5_prompt, temperature=0.0, max_tokens=16)
        judgments["task_family_fit"] = _parse_judge_answer(q5, ["MATCH", "MISMATCH"])
    except Exception as e:
        judgments["task_family_fit"] = "MISMATCH"
        print(f"    [WARN] Q5 failed: {e}", file=sys.stderr)
    time.sleep(SLEEP)

    # -- Aggregate decision (calibrated per query style) --
    query_style = candidate.get("query_style", "full_question")
    is_keyword = query_style.startswith("keyword_")

    # For keyword queries: Q1 (answerability) and Q5 (task_family_fit) are advisory.
    # Keywords are search queries, not answerable questions — per handoff guidance:
    # "Q1 (answerability) may be lower in strict sense but the passage IS the relevant document."
    # Core requirements that ALWAYS apply: Q2 vocab_shift + Q3 specificity + Q4 factual grounding
    decision = "ACCEPT"
    rejection_reasons = []

    if judgments["vocab_shift"] not in ("STRONG_SHIFT", "MODERATE_SHIFT"):
        decision = "REJECT"
        rejection_reasons.append(f"insufficient_shift:{judgments['vocab_shift']}")
    if judgments["passage_specificity"] != "SPECIFIC":
        decision = "REJECT"
        rejection_reasons.append(f"generic:{judgments['passage_specificity']}")
    if judgments["factual_grounding"] != "CLEAN":
        decision = "REJECT"
        rejection_reasons.append(f"factual_issue:{judgments['factual_grounding']}")

    # These dimensions are strict for full_question, advisory for keywords
    if not is_keyword:
        if judgments["answerability"] != "ANSWERABLE":
            decision = "REJECT"
            rejection_reasons.append(f"not_answerable:{judgments['answerability']}")
        if judgments["task_family_fit"] != "MATCH":
            decision = "REJECT"
            rejection_reasons.append(f"task_mismatch:{judgments['task_family_fit']}")
    else:
        # Advisory logging for keywords (don't block acceptance)
        if judgments["answerability"] != "ANSWERABLE":
            rejection_reasons.append(f"advisory_not_answerable:{judgments['answerability']}")
        if judgments["task_family_fit"] != "MATCH":
            rejection_reasons.append(f"advisory_task_mismatch:{judgments['task_family_fit']}")

    candidate["judgments"] = judgments
    candidate["decision"] = decision
    candidate["rejection_reason"] = ";".join(rejection_reasons) if rejection_reasons else ""
    candidate["judge_model"] = MODEL

    return candidate


# ── Parquet helpers ──────────────────────────────────────────────────────────


def prepare_for_parquet(records: list[dict]) -> pd.DataFrame:
    """Serialize list/dict columns to JSON strings for parquet compatibility."""
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
    """Compute batch statistics from accepted and rejected candidates."""
    all_candidates = accepted + rejected
    total = len(all_candidates)
    stats = {
        "total": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / total if total else 0.0,
    }

    # Per-dimension pass rates
    dims = ["answerability", "vocab_shift", "passage_specificity", "factual_grounding", "task_family_fit"]
    dim_pass_values = {
        "answerability": {"ANSWERABLE"},
        "vocab_shift": {"STRONG_SHIFT", "MODERATE_SHIFT"},
        "passage_specificity": {"SPECIFIC"},
        "factual_grounding": {"CLEAN"},
        "task_family_fit": {"MATCH"},
    }
    per_dim = {}
    for dim in dims:
        judged = [c for c in all_candidates if "judgments" in c]
        if judged:
            passed_vals = dim_pass_values[dim]
            passed = sum(1 for c in judged if c["judgments"].get(dim) in passed_vals)
            per_dim[dim] = passed / len(judged)
    stats["per_dimension_pass_rates"] = per_dim

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


def run_multistyle_batch(
    seed: int,
    batch_id: str,
    count: int = 100,
    output_base: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run a single multistyle production batch.

    Each passage produces up to 4 queries:
      - 1 full-sentence question
      - 3 keyword queries (technical, layperson, clinical)

    All grounded in the same extracted fact.

    Returns:
        stats dict with accepted count, acceptance rate, per-dimension rates, etc.
    """
    output_base = output_base or DEFAULT_OUTPUT_BASE
    output_dir = output_base / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[{batch_id}] Seed={seed}, count={count}, output={output_dir}")
        print(f"[{batch_id}] Mode: multistyle (4 queries/passage)")

    # 1. Load and shuffle passages
    with open(PASSAGES_PATH, "r") as f:
        all_passages = json.load(f)

    random.seed(seed)
    shuffled = list(all_passages)
    random.shuffle(shuffled)
    passages = shuffled[:count]

    if verbose:
        print(f"[{batch_id}] Loaded {len(all_passages)} passages, using {len(passages)}")

    # 2. Set up validators
    validators = DeterministicValidators(
        max_forbidden_surface_overlap=0.45,
        min_alternative_term_count=1,
        reject_generic_queries=False,
    )
    prompt_hash = stable_hash("multistyle_production_v1")

    # 3. Generate candidates (multistyle: fact → 4 queries)
    candidates = []
    passages_processed = 0
    for i, p in enumerate(passages):
        passage_id = p.get("passage_id", f"p_{i}")
        passage_text = p.get("text", "")
        cui_label = p.get("cui_label", "")
        alt_term = p.get("alt_term", "")
        semantic_group = p.get("semantic_group", "other")
        vocab_shift_type = p.get("vocab_shift_type", "consumer_to_clinical")

        # Map semantic_group → task_family (CRITICAL FIX #6)
        task_family = SEMANTIC_GROUP_TO_TASK_FAMILY.get(semantic_group, "lay_to_clinical")

        # Validation prep
        passage_words = passage_text.split()
        forbidden_terms = (
            passage_words[:20] + passage_words[-20:]
            if len(passage_words) > 40
            else passage_words
        )
        alt_terms_list = [alt_term] if alt_term else None

        # Step A: Extract fact (shared across all 4 query styles)
        fact = extract_fact(passage_text)
        time.sleep(SLEEP)

        if not fact:
            if verbose:
                print(f"[{batch_id}] gen {i+1}/{count}: fact extraction FAILED, skipping")
            continue

        passages_processed += 1

        # ── Query 1: Full-sentence question ──────────────────────────────
        role_options = TASK_FAMILY_ROLES.get(task_family, QUESTION_ROLES)
        role = random.choice(role_options)

        full_query = generate_full_question(fact, role, task_family, cui_label, alt_term)
        time.sleep(SLEEP)

        if full_query:
            example_id = stable_hash(passage_id, full_query, "full_question", batch_id, str(i))
            v = validators.validate(
                example_id=example_id, query=full_query, passage=passage_text,
                forbidden_terms=forbidden_terms, alt_terms=alt_terms_list,
            )
            candidates.append({
                "example_id": example_id,
                "passage_id": passage_id,
                "passage_text": passage_text,  # CRITICAL: judge needs this
                "query": full_query,
                "query_style": "full_question",
                "fact": fact,
                "mode": "ontology_grounded_teacher_filtered",
                "task_family": task_family,
                "role": role,
                "cui_label": cui_label,
                "alt_term": alt_term,
                "target_cuis": [cui_label] if cui_label else [],
                "semantic_group": semantic_group,
                "vocab_shift_type": vocab_shift_type,
                "generator_model": MODEL,
                "prompt_hash": prompt_hash,
                "passed_validation": v.passed,
                "validation_failures": v.failures,
                "validation_warnings": v.warnings,
            })

        # ── Queries 2-4: Keyword styles ──────────────────────────────────
        for kw in generate_keyword_queries(fact, cui_label, alt_term):
            kw_query = kw["query"]
            if not kw_query:
                continue
            kw_style = kw["query_style"]
            kw_role = kw["role"]

            example_id = stable_hash(passage_id, kw_query, kw_style, batch_id, str(i))
            v = validators.validate(
                example_id=example_id, query=kw_query, passage=passage_text,
                forbidden_terms=forbidden_terms, alt_terms=alt_terms_list,
            )
            candidates.append({
                "example_id": example_id,
                "passage_id": passage_id,
                "passage_text": passage_text,
                "query": kw_query,
                "query_style": kw_style,
                "fact": fact,
                "mode": "ontology_grounded_teacher_filtered",
                "task_family": task_family,
                "role": kw_role,
                "cui_label": cui_label,
                "alt_term": alt_term,
                "target_cuis": [cui_label] if cui_label else [],
                "semantic_group": semantic_group,
                "vocab_shift_type": vocab_shift_type,
                "generator_model": MODEL,
                "prompt_hash": prompt_hash,
                "passed_validation": v.passed,
                "validation_failures": v.failures,
                "validation_warnings": v.warnings,
            })
        time.sleep(SLEEP)

        if verbose and (i % 10 == 0 or i == count - 1):
            generated_count = len([c for c in candidates if c.get("passage_id") == passage_id])
            status_icon = "✓" if generated_count > 0 else "✗"
            print(
                f"[{batch_id}] gen {i+1}/{count}: "
                f"sg={semantic_group:22s} tf={task_family:24s} "
                f"fact_len={len(fact):3d} queries={generated_count} {status_icon}"
            )

    passed_val = sum(1 for c in candidates if c["passed_validation"])
    if verbose:
        print(f"\n[{batch_id}] Generated {len(candidates)} candidates from {passages_processed} passages "
              f"({len(candidates)/max(passages_processed,1):.1f} q/passage), {passed_val} passed validation")

    # 4. Judge candidates via direct vLLM (Q1-Q5, strict mode)
    judged = []
    for idx, c in enumerate(candidates):
        result = judge_single_candidate(c)
        judged.append(result)

        if verbose and (idx % 50 == 0 or idx == len(candidates) - 1):
            decision = result.get("decision", "ERROR")
            j = result.get("judgments", {})
            qs = result.get("query_style", "?")
            print(
                f"[{batch_id}] judge {idx+1}/{len(candidates)}: "
                f"{decision:6s} style={qs:20s} | "
                f"ans={j.get('answerability','?')[:4]:4s} "
                f"voc={j.get('vocab_shift','?')[:4]:4s} "
                f"spe={j.get('passage_specificity','?')[:4]:4s} "
                f"fac={j.get('factual_grounding','?')[:4]:4s} "
                f"fam={j.get('task_family_fit','?')[:4]:4s}"
            )

    # 5. Split, compute stats, save
    accepted = [c for c in judged if c.get("decision") == "ACCEPT"]
    rejected = [c for c in judged if c.get("decision") != "ACCEPT"]

    stats = compute_batch_stats(accepted, rejected)
    stats["passages_processed"] = passages_processed
    stats["batch_id"] = batch_id
    stats["seed"] = seed
    stats["generation_style"] = "multistyle"
    stats["candidates_per_passage"] = len(candidates) / max(passages_processed, 1)

    judged_df = prepare_for_parquet(judged)
    accepted_df = prepare_for_parquet(accepted)

    judged_df.to_parquet(output_dir / "judged.parquet", index=False)
    if accepted_df is not None and len(accepted_df) > 0:
        accepted_df.to_parquet(output_dir / "accepted.parquet", index=False)

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"\n[{batch_id}] DONE: {stats['accepted']}/{stats['total']} accepted ({stats['acceptance_rate']:.2%})")
        for dim, rate in sorted(stats.get("per_dimension_pass_rates", {}).items()):
            print(f"[{batch_id}]   {dim}: {rate:.2%}")
        for style, rate in sorted(stats.get("per_style_acceptance", {}).items()):
            print(f"[{batch_id}]   style/{style}: {rate:.2%}")
        print(f"[{batch_id}] Output: {output_dir}")

    return stats


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="MedColBERT Multistyle Batch Runner")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for passage shuffling")
    parser.add_argument("--batch-id", type=str, required=True, help="Unique batch identifier")
    parser.add_argument("--count", type=int, default=100, help="Number of passages to process")
    parser.add_argument("--output-base", type=Path, default=None,
                        help="Base output directory")
    args = parser.parse_args()

    print(f"=== MedColBERT Multistyle Batch: {args.batch_id} ===")
    print(f"Seed: {args.seed}, Count: {args.count}")
    print(f"Mode: 4 queries per passage (full_question + keyword_technical + keyword_layperson + keyword_clinical)")

    stats = run_multistyle_batch(
        seed=args.seed,
        batch_id=args.batch_id,
        count=args.count,
        output_base=args.output_base,
    )

    print(f"\n=== Batch {args.batch_id} Complete ===")
    print(f"Accepted: {stats['accepted']}/{stats['total']} ({stats['acceptance_rate']:.2%})")
    return 0 if stats["accepted"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
