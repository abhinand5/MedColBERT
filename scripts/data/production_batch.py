#!/usr/bin/env python3
"""MedColBERT Production Batch Runner — scalable, parameterized batch generation.

Canonical pipeline (from run_batch_mf15.py — DO NOT DEVIATE):
  1. Load + shuffle annotated passages (seed-controlled)
  2. Two-step generation via direct vLLM httpx:
     Step A: Extract ONE key fact from passage (temp=0)
     Step B: Generate question from fact using alternative term (temp=0)
  3. Deterministic validation (overlap=0.45, min_alt_term=1, no generic rejection)
  4. Strict 5-D judging via direct vLLM:
     ACCEPT = ANSWERABLE + (STRONG|MODERATE)_SHIFT + SPECIFIC + CLEAN + MATCH
  5. Save accepted.parquet, judged.parquet, stats.json

Usage:
  uv run python scripts/data/production_batch.py --seed 42 --batch-id prod_001 --count 100
  uv run python scripts/data/production_batch.py --seed 123 --batch-id prod_002 --count 200
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

# ── Configuration (overridable via CLI) ────────────────────────────────────────────

BASE_DIR = Path("/workspace/MedColBERT")
PASSAGES_PATH = BASE_DIR / "data/processed/private/passages/real_annotated_500.json"
DEFAULT_OUTPUT_BASE = BASE_DIR / "data/processed/private/synthetic/ontology_grounded_teacher_filtered"

API_BASE = "http://localhost:8000/v1"
MODEL = "gemma-4-31B"
SLEEP = 0.05  # seconds between API calls
JUDGE_TIMEOUT = 120.0  # seconds

# ── Semantic group → task family mapping (CRITICAL FIX #6) ─────────────────────────

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

ALL_ROLES = ["patient", "physician", "researcher", "nurse", "student"]

# ── Two-step generation prompts (from handoff — DO NOT CHANGE) ─────────────────────

FACT_SYSTEM = (
    "Extract ONE key fact. Use exact wording from the passage. "
    "Output only the fact sentence."
)

QGEN_SYSTEM = (
    "Write ONLY a short question using the alternative term. No other text."
)

# ── vLLM HTTP helper ────────────────────────────────────────────────────────────────


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
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            time.sleep(wait)
    return ""


# ── Step A: Fact extraction ─────────────────────────────────────────────────────────


def extract_fact(passage_text: str) -> str:
    """Extract one key fact from the passage using exact wording."""
    user_prompt = f"Passage:\n{passage_text[:1200]}\n\nOne key fact:"
    try:
        fact = call_vllm(FACT_SYSTEM, user_prompt, temperature=0.0, max_tokens=120)
        # Clean up common prefixes
        for prefix in ["One key fact:", "Key fact:", "Fact:", "Here is", "The key fact"]:
            if fact.lower().startswith(prefix.lower()):
                fact = fact[len(prefix) :].strip()
        return fact
    except Exception as e:
        print(f"  [ERROR] Fact extraction failed: {e}", file=sys.stderr)
        return ""


# ── Step B: Question generation ─────────────────────────────────────────────────────


def generate_question(fact: str, role: str, cui_label: str, alt_term: str) -> str:
    """Generate a question from the fact, replacing CUI label with alt term."""
    if not fact:
        return ""
    user_prompt = (
        f'Using ONLY this fact, write a {role} question. '
        f'Replace "{cui_label}" with "{alt_term}".\n\n'
        f"Fact: {fact}\n\nQuestion:"
    )
    try:
        query = call_vllm(QGEN_SYSTEM, user_prompt, temperature=0.0, max_tokens=60)
        # Clean up common prefixes
        for prefix in ["Question:", "question:", "Query:", "query:"]:
            if query.startswith(prefix):
                query = query[len(prefix) :].strip()
        query = query.strip('"').strip("'")
        return query
    except Exception as e:
        print(f"  [ERROR] Question generation failed: {e}", file=sys.stderr)
        return ""


# ── Direct vLLM Judging (Q1-Q5) ─────────────────────────────────────────────────────


def _parse_judge_answer(response: str, valid_options: list[str]) -> str:
    """Parse the judge response to extract the answer letter/name."""
    text = response.strip().upper()

    # Try direct match for valid option strings or their first letters
    for opt in valid_options:
        if text == opt or text == opt[0]:
            return opt
        # Check for "A)" pattern
        if f"{opt[0]})" in text:
            return opt
        # Check for full option name
        if opt in text:
            return opt

    # Fallback: find first valid letter
    for opt in valid_options:
        if opt[0] in text:
            return opt

    return valid_options[0]  # Default to first option


def judge_single_candidate(candidate: dict) -> dict:
    """Run all 5 judge questions via direct vLLM httpx calls.

    CRITICAL: Uses the CORRECT task family (from semantic_group mapping).
    Returns candidate with 'judgments', 'decision', 'rejection_reason' added.
    """
    # Skip judging for validation failures
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
    cui_label = candidate.get("target_cuis", [""])[0] if candidate.get("target_cuis") else ""
    task_family = candidate.get("task_family", "lay_to_clinical")
    role = candidate.get("role", "")
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

    # -- Strict aggregate decision --
    # ACCEPT = ANSWERABLE + (STRONG|MODERATE)_SHIFT + SPECIFIC + CLEAN + MATCH
    decision = "ACCEPT"
    rejection_reasons = []

    if judgments["answerability"] != "ANSWERABLE":
        decision = "REJECT"
        rejection_reasons.append(f"not_answerable:{judgments['answerability']}")
    if judgments["vocab_shift"] not in ("STRONG_SHIFT", "MODERATE_SHIFT"):
        decision = "REJECT"
        rejection_reasons.append(f"insufficient_shift:{judgments['vocab_shift']}")
    if judgments["passage_specificity"] != "SPECIFIC":
        decision = "REJECT"
        rejection_reasons.append(f"generic:{judgments['passage_specificity']}")
    if judgments["factual_grounding"] != "CLEAN":
        decision = "REJECT"
        rejection_reasons.append(f"hallucination:{judgments['factual_grounding']}")
    if judgments["task_family_fit"] != "MATCH":
        decision = "REJECT"
        rejection_reasons.append(f"task_mismatch:{judgments['task_family_fit']}")

    candidate["judgments"] = judgments
    candidate["decision"] = decision
    candidate["rejection_reason"] = ";".join(rejection_reasons) if rejection_reasons else ""
    candidate["judge_model"] = MODEL

    return candidate


# ── Parquet helper ──────────────────────────────────────────────────────────────────


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
    total = len(accepted) + len(rejected)
    stats = {
        "total": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / total if total else 0.0,
    }

    # Per-dimension pass rates (on judged candidates)
    dims = ["answerability", "vocab_shift", "passage_specificity", "factual_grounding", "task_family_fit"]
    per_dim = {}
    for dim in dims:
        all_judged = [c for c in accepted + rejected if "judgments" in c]
        if all_judged:
            passed = sum(
                1 for c in all_judged
                if c.get("judgments", {}).get(dim) in (
                    "ANSWERABLE", "STRONG_SHIFT", "MODERATE_SHIFT", "SPECIFIC", "CLEAN", "MATCH"
                )
            )
            per_dim[dim] = passed / len(all_judged)
    stats["per_dimension_pass_rates"] = per_dim

    # Per-family acceptance
    per_family = {}
    for c in accepted:
        tf = c.get("task_family", "unknown")
        per_family[tf] = per_family.get(tf, 0) + 1
    for tf in per_family:
        total_fam = sum(1 for c in (accepted + rejected) if c.get("task_family") == tf)
        per_family[tf] = per_family[tf] / total_fam if total_fam else 0
    stats["per_family_acceptance"] = per_family

    # Per-semantic-group acceptance
    per_sg = {}
    for c in accepted:
        sg = c.get("semantic_group", "unknown")
        per_sg[sg] = per_sg.get(sg, 0) + 1
    for sg in per_sg:
        total_sg = sum(1 for c in (accepted + rejected) if c.get("semantic_group") == sg)
        per_sg[sg] = per_sg[sg] / total_sg if total_sg else 0
    stats["per_semantic_group_acceptance"] = per_sg

    return stats


# ── Main ─────────────────────────────────────────────────────────────────────────────


def run_batch(
    seed: int,
    batch_id: str,
    count: int = 100,
    output_base: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run a single production batch.

    Args:
        seed: Random seed for passage shuffling.
        batch_id: Unique batch identifier (used for output directory).
        count: Number of passages to process.
        output_base: Base output directory. Defaults to DEFAULT_OUTPUT_BASE.
        verbose: Print progress messages.

    Returns:
        stats dict with 'accepted', 'total', 'acceptance_rate', etc.
    """
    output_base = output_base or DEFAULT_OUTPUT_BASE
    output_dir = output_base / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[{batch_id}] Seed={seed}, count={count}, output={output_dir}")

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

    # 3. Generate candidates (two-step: fact extract + question generate)
    candidates = []
    for i, p in enumerate(passages):
        passage_id = p.get("passage_id", f"p_{i}")
        passage_text = p.get("text", "")
        cui_label = p.get("cui_label", "")
        alt_term = p.get("alt_term", "")
        semantic_group = p.get("semantic_group", "other")
        vocab_shift_type = p.get("vocab_shift_type", "consumer_to_clinical")

        # Map semantic_group → task_family (CRITICAL FIX #6)
        task_family = SEMANTIC_GROUP_TO_TASK_FAMILY.get(semantic_group, "lay_to_clinical")

        # Select role based on task family
        role_options = TASK_FAMILY_ROLES.get(task_family, ALL_ROLES)
        role = random.choice(role_options)

        # Step A: Extract fact (temp=0)
        fact = extract_fact(passage_text)
        time.sleep(SLEEP)

        # Step B: Generate question from fact (temp=0)
        query = generate_question(fact, role, cui_label, alt_term)
        time.sleep(SLEEP)

        # Build example_id
        example_id = stable_hash(
            passage_id, query, "ontology_grounded_teacher_filtered",
            f"prod_{seed}", task_family
        )

        # Deterministic validation
        passage_words = passage_text.split()
        forbidden_terms = (
            passage_words[:20] + passage_words[-20:]
            if len(passage_words) > 40
            else passage_words
        )
        alt_terms_list = [alt_term] if alt_term else None

        validation = validators.validate(
            example_id=example_id,
            query=query,
            passage=passage_text,
            forbidden_terms=forbidden_terms,
            alt_terms=alt_terms_list,
        )

        candidate = {
            "example_id": example_id,
            "passage_id": passage_id,
            "passage_text": passage_text,  # CRITICAL FIX #1: judge needs this
            "query": query,
            "fact": fact,
            "mode": "ontology_grounded_teacher_filtered",
            "task_family": task_family,
            "role": role,
            "target_cuis": [cui_label] if cui_label else [],
            "alt_term": alt_term,
            "cui_label": cui_label,
            "semantic_group": semantic_group,
            "vocab_shift_type": vocab_shift_type,
            "generator_model": MODEL,
            "passed_validation": validation.passed,
            "validation_failures": validation.failures,
            "validation_warnings": validation.warnings,
        }
        candidates.append(candidate)

        if verbose and (i % 20 == 0 or i == count - 1):
            val_status = "PASS" if validation.passed else f"FAIL"
            print(
                f"[{batch_id}] gen {i+1}/{count}: "
                f"sg={semantic_group:22s} tf={task_family:24s} "
                f"role={role:12s} qlen={len(query):3d} val={val_status}"
            )

    passed_val = sum(1 for c in candidates if c["passed_validation"])
    if verbose:
        print(f"[{batch_id}] Generated {len(candidates)} candidates, {passed_val} passed validation")

    # 4. Judge candidates via direct vLLM (Q1-Q5, strict mode)
    judged = []
    for idx, c in enumerate(candidates):
        result = judge_single_candidate(c)
        judged.append(result)

        if verbose and (idx % 20 == 0 or idx == len(candidates) - 1):
            decision = result.get("decision", "ERROR")
            j = result.get("judgments", {})
            print(
                f"[{batch_id}] judge {idx+1}/{len(candidates)}: "
                f"{decision:6s} | "
                f"ans={j.get('answerability','?')[:4]:4s} "
                f"voc={j.get('vocab_shift','?')[:4]:4s} "
                f"spe={j.get('passage_specificity','?')[:4]:4s} "
                f"fac={j.get('factual_grounding','?')[:4]:4s} "
                f"fam={j.get('task_family_fit','?')[:4]:4s}"
            )

    # 5. Split and save
    accepted = [c for c in judged if c.get("decision") == "ACCEPT"]
    rejected = [c for c in judged if c.get("decision") != "ACCEPT"]

    stats = compute_batch_stats(accepted, rejected)

    judged_df = prepare_for_parquet(judged)
    accepted_df = prepare_for_parquet(accepted)

    judged_df.to_parquet(output_dir / "judged.parquet", index=False)
    accepted_df.to_parquet(output_dir / "accepted.parquet", index=False)

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"[{batch_id}] DONE: {stats['accepted']}/{stats['total']} accepted ({stats['acceptance_rate']:.2%})")
        for dim, rate in sorted(stats.get("per_dimension_pass_rates", {}).items()):
            print(f"[{batch_id}]   {dim}: {rate:.2%}")
        print(f"[{batch_id}] Output: {output_dir}")

    return stats


# ── CLI ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="MedColBERT Production Batch Runner")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for passage shuffling")
    parser.add_argument("--batch-id", type=str, required=True, help="Unique batch identifier")
    parser.add_argument("--count", type=int, default=100, help="Number of passages to process")
    parser.add_argument("--output-base", type=Path, default=None,
                        help="Base output directory (default: ontology_grounded_teacher_filtered)")
    args = parser.parse_args()

    print(f"=== MedColBERT Production Batch: {args.batch_id} ===")
    print(f"Seed: {args.seed}, Count: {args.count}")

    stats = run_batch(
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
