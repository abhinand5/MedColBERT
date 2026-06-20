"""Direct vLLM generator — bypasses DSPy for full prompt control.

The DSPy Predict module formats prompts in its own way, which makes it hard to
enforce strict passage grounding. This module calls the vLLM API directly with
carefully crafted prompts that emphasize:
1. Only use information from the passage
2. Do not add or infer facts
3. Use the alternative term for vocabulary shift
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from medcolbert.generation.prompts import (
    get_task_family_description,
    get_role_instruction,
    hash_prompt,
)
from medcolbert.utils.hashing import stable_hash


def generate_query_direct(
    passage: str,
    api_base: str,
    model: str,
    *,
    role: str = "patient",
    task_family: str = "lay_to_clinical",
    cui_label: str = "",
    alt_term: str = "",
    semantic_group: str = "other",
    vocab_shift_type: str = "generic",
    temperature: float = 0.3,
    max_tokens: int = 128,
) -> dict[str, str]:
    """Generate a query by calling vLLM directly with a strict grounding prompt.

    Returns:
        {"query": str, "rationale": str}
    """
    # Truncate passage to control context length
    passage_short = passage[:1500]

    # Build prompt with passage as primary focus
    if cui_label and alt_term:
        concept_section = f"""\
MEDICAL CONCEPT (expressed clinically in the passage): "{cui_label}"
ALTERNATIVE TERM to use in your query: "{alt_term}\""""
    else:
        concept_section = ""

    prompt = f"""\
You are generating a short medical query for a biomedical retrieval training dataset.

CRITICAL RULES — follow exactly or the output will be rejected:
1. Read the passage below. Your query MUST be answerable from ONLY this passage.
2. Do NOT mention any fact, entity, mechanism, dosage, side effect, statistic,
   or study detail that is NOT explicitly written in this passage.
3. If the passage only says "substance X was studied", your query should only
   ask about X being studied — not about its effects, mechanisms, or outcomes
   unless those are stated in the passage.
4. Use simple, direct language appropriate for a {role}.
5. Keep the query under 150 characters.
{concept_section}

PASSAGE:
{passage_short}

TASK: Write a {role} asking a question about this passage using the alternative
terminology. The question must be answerable from ONLY the passage above.

Query:"""

    system_msg = "You are a medical query generator. Write ONLY the query text, nothing else. No explanations, no notes, just the query."

    # Call vLLM
    try:
        response = _call_vllm(
            api_base=api_base,
            model=model,
            system=system_msg,
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        query = response.strip().strip('"').strip("'").strip()
        # Remove any "Query:" prefix the model might add
        query = re.sub(r'^Query:\s*', '', query)
        # Truncate if too long
        if len(query) > 180:
            query = query[:180]
        return {"query": query, "rationale": f"Direct generation, temp={temperature}"}
    except Exception as e:
        return {"query": "", "rationale": f"Generation error: {e}"}


def generate_batch_direct(
    passages: list[dict],
    api_base: str = "http://localhost:8000/v1",
    model: str = "gemma-4-31B",
    mode: str = "ontology_grounded",
    temperature: float = 0.3,
) -> list[dict]:
    """Generate a batch of query candidates using direct vLLM calls.

    Args:
        passages: List of passage dicts with keys: passage_id, text, [cui_label, alt_term, ...]
        api_base: vLLM API base URL.
        model: Model name at vLLM endpoint.
        mode: Generation mode (for metadata).
        temperature: Generation temperature (lower = more grounded).

    Returns:
        List of candidate dicts (same schema as GenerationPipeline.generate_batch).
    """
    import random
    from medcolbert.generation.validators import DeterministicValidators
    from medcolbert.generation.teacher_filter import aggregate_decision

    task_families = [
        "lay_to_clinical", "abbreviation_to_expanded", "brand_generic",
        "symptom_to_diagnosis", "biomedical_to_clinical",
        "no_direct_lexical_overlap",
    ]
    roles = ["patient", "physician", "researcher", "nurse", "student"]
    validators = DeterministicValidators(
        max_forbidden_surface_overlap=0.45,
        min_alternative_term_count=1,
        reject_generic_queries=True,
    )

    candidates: list[dict] = []
    prompt_hash = hash_prompt("direct_generator_v1")

    for i, p in enumerate(passages):
        passage_id = p.get("passage_id", f"p_{i}")
        passage_text = p.get("text", "")
        cui_label = p.get("cui_label", "")
        alt_term = p.get("alt_term", "")
        semantic_group = p.get("semantic_group", "other")
        vocab_shift_type = p.get("vocab_shift_type", "consumer_to_clinical")
        task_family = task_families[i % len(task_families)]
        role = random.choices(roles, weights=[0.30, 0.25, 0.20, 0.15, 0.10], k=1)[0]

        result = generate_query_direct(
            passage=passage_text,
            api_base=api_base,
            model=model,
            role=role,
            task_family=task_family,
            cui_label=cui_label,
            alt_term=alt_term,
            semantic_group=semantic_group,
            vocab_shift_type=vocab_shift_type,
            temperature=temperature,
        )

        query = result["query"]
        example_id = stable_hash(passage_id, query, mode, str(i))

        # Deterministic validation
        passage_words = passage_text.split()
        forbidden = passage_words[:20] + passage_words[-20:] if len(passage_words) > 40 else passage_words
        validation = validators.validate(
            example_id=example_id,
            query=query,
            passage=passage_text,
            forbidden_terms=forbidden,
            alt_terms=[alt_term] if alt_term else None,
        )

        candidates.append({
            "example_id": example_id,
            "passage_id": passage_id,
            "query": query,
            "rationale": result.get("rationale", ""),
            "mode": mode,
            "task_family": task_family,
            "role": role,
            "target_cuis": [cui_label] if cui_label else [],
            "semantic_group": semantic_group,
            "vocab_shift_type": vocab_shift_type,
            "generator_model": model,
            "prompt_hash": prompt_hash,
            "passed_validation": validation.passed,
            "validation_failures": validation.failures,
            "validation_warnings": validation.warnings,
        })

        time.sleep(0.05)  # Small delay to avoid overwhelming vLLM

    return candidates


def generate_batch_multistyle(
    passages: list[dict],
    api_base: str = "http://localhost:8000/v1",
    model: str = "gemma-4-31B",
    mode: str = "ontology_grounded",
    temperature: float = 0.0,
    include_full_question: bool = True,
    include_keywords: bool = True,
) -> list[dict]:
    """Generate a batch with MULTIPLE query styles per passage.

    For each passage, produces:
      - 1 full-sentence question (if include_full_question=True)
      - 3 keyword queries: technical, layperson, clinical (if include_keywords=True)

    Total: 4 queries per passage — 4x the data density.

    All queries are grounded via the same extracted fact, ensuring passage specificity.
    """
    import random
    from medcolbert.generation.validators import DeterministicValidators

    task_families = [
        "lay_to_clinical", "abbreviation_to_expanded", "brand_generic",
        "symptom_to_diagnosis", "biomedical_to_clinical",
        "no_direct_lexical_overlap",
    ]
    question_roles = ["patient", "physician", "researcher", "nurse", "student"]
    validators = DeterministicValidators(
        max_forbidden_surface_overlap=0.45,
        min_alternative_term_count=1,
        reject_generic_queries=False,
    )
    prompt_hash = stable_hash("multistyle_v1")

    candidates: list[dict] = []

    for i, p in enumerate(passages):
        passage_id = p.get("passage_id", f"p_{i}")
        passage_text = p.get("text", "")
        cui_label = p.get("cui_label", "")
        alt_term = p.get("alt_term", "")
        semantic_group = p.get("semantic_group", "other")
        vocab_shift_type = p.get("vocab_shift_type", "consumer_to_clinical")
        task_family = task_families[i % len(task_families)]

        # ── Step 1: Extract one key fact (shared across all styles) ──────
        try:
            fact = _call_vllm(
                api_base=api_base,
                model=model,
                system="Extract ONE key fact. Use exact wording from the passage. Output only the fact sentence.",
                user_prompt=f"Passage:\n{passage_text[:1200]}\n\nOne key fact:",
                temperature=0.0,
                max_tokens=120,
            ).strip()
        except Exception:
            fact = ""

        if not fact:
            continue

        # ── Step 2a: Full-sentence question ──────────────────────────────
        if include_full_question:
            role = random.choices(question_roles, weights=[0.30, 0.25, 0.20, 0.15, 0.10], k=1)[0]
            try:
                query = _call_vllm(
                    api_base=api_base,
                    model=model,
                    system="Write ONLY a short question. No other text.",
                    user_prompt=(
                        f"Using ONLY this fact, write a {role} question for {task_family}. "
                        f'Replace "{cui_label}" with "{alt_term}".\n\n'
                        f"Fact: {fact}\n\nQuestion:"
                    ),
                    temperature=temperature,
                    max_tokens=60,
                ).strip().strip('"')
            except Exception:
                query = ""

            if query:
                example_id = stable_hash(passage_id, query, "full_question", str(i))
                passage_words = passage_text.split()
                forbidden = passage_words[:20] + passage_words[-20:] if len(passage_words) > 40 else passage_words
                v = validators.validate(
                    example_id=example_id, query=query, passage=passage_text,
                    forbidden_terms=forbidden, alt_terms=[alt_term] if alt_term else None,
                )
                candidates.append({
                    "example_id": example_id, "passage_id": passage_id,
                    "passage_text": passage_text, "query": query,
                    "query_style": "full_question", "fact": fact,
                    "mode": mode, "task_family": task_family, "role": role,
                    "target_cuis": [cui_label] if cui_label else [],
                    "semantic_group": semantic_group,
                    "vocab_shift_type": vocab_shift_type,
                    "generator_model": model, "prompt_hash": prompt_hash,
                    "cui_label": cui_label, "alt_term": alt_term,
                    "passed_validation": v.passed,
                    "validation_failures": v.failures,
                    "validation_warnings": v.warnings,
                })

        # ── Step 2b: Keyword queries (3 styles) ──────────────────────────
        if include_keywords:
            for kw in generate_keyword_queries(
                fact=fact, cui_label=cui_label, alt_term=alt_term,
                api_base=api_base, model=model, temperature=0.1,
            ):
                kw_query = kw["query"]
                if not kw_query:
                    continue

                example_id = stable_hash(passage_id, kw_query, kw["query_style"], str(i))
                passage_words = passage_text.split()
                forbidden = passage_words[:20] + passage_words[-20:] if len(passage_words) > 40 else passage_words
                v = validators.validate(
                    example_id=example_id, query=kw_query, passage=passage_text,
                    forbidden_terms=forbidden, alt_terms=[alt_term] if alt_term else None,
                )
                candidates.append({
                    "example_id": example_id, "passage_id": passage_id,
                    "passage_text": passage_text, "query": kw_query,
                    "query_style": kw["query_style"], "fact": fact,
                    "mode": mode, "task_family": task_family, "role": kw["role"],
                    "target_cuis": [cui_label] if cui_label else [],
                    "semantic_group": semantic_group,
                    "vocab_shift_type": vocab_shift_type,
                    "generator_model": model, "prompt_hash": prompt_hash,
                    "cui_label": cui_label, "alt_term": alt_term,
                    "passed_validation": v.passed,
                    "validation_failures": v.failures,
                    "validation_warnings": v.warnings,
                })

        time.sleep(0.05)

    return candidates


# ── Keyword Query Styles ─────────────────────────────────────────────────────

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
    },
}


def generate_keyword_queries(
    fact: str,
    cui_label: str,
    alt_term: str,
    api_base: str = "http://localhost:8000/v1",
    model: str = "gemma-4-31B",
    temperature: float = 0.1,
) -> list[dict[str, str]]:
    """Generate keyword-style queries from a single extracted fact.

    Produces one query per KEYWORD_STYLE (technical, layperson, clinical).

    Returns a list of dicts with keys: query, query_style, role.
    """
    results: list[dict[str, str]] = []
    for style_name, style in KEYWORD_STYLES.items():
        prompt = style["gen_prompt"].format(cui=cui_label, alt=alt_term, fact=fact)
        try:
            response = _call_vllm(
                api_base=api_base,
                model=model,
                system=style["system"],
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=30,
            )
            query = response.strip().strip('"').strip("'").strip()
            results.append({
                "query": query,
                "query_style": style["query_style"],
                "role": style["role"],
            })
        except Exception:
            results.append({
                "query": "",
                "query_style": style["query_style"],
                "role": style["role"],
            })
    return results


def _call_vllm(
    api_base: str,
    model: str,
    system: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 128,
    timeout: float = 60.0,
) -> str:
    """Call the vLLM chat completions endpoint and return the response text."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
