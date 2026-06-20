"""DSPy programs for synthetic query generation and judging.

Defines DSPy signatures and pipeline modules for:
- Generator: produces query from passage + ontology context
- 5 Judges: answerability, vocab_shift, specificity, factual_grounding, task_family_fit

Uses dspy.Predict (not ChainOfThought) — Gemma 4 31B reasons natively.
Separate LM instances: generator (temp=0.7), judge (temp=0.0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy
import pandas as pd

from medcolbert.generation.prompts import (
    GENERATOR_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    get_generator_prompt,
    get_judge_prompt,
    get_task_family_description,
    get_role_instruction,
    hash_prompt,
)
from medcolbert.generation.validators import DeterministicValidators
from medcolbert.generation.teacher_filter import (
    aggregate_decision,
    compute_batch_stats,
)
from medcolbert.utils.hashing import stable_hash
from medcolbert.utils.logging import Counts

# ── DSPy Signatures ───────────────────────────────────────────────────────────


class GenerateQuery(dspy.Signature):
    """Generate a synthetic medical query from a passage with vocabulary shift."""

    passage: str = dspy.InputField(desc="Source medical passage")
    role: str = dspy.InputField(desc="Role of the person asking (patient, physician, etc.)")
    task_family: str = dspy.InputField(desc="Task family for vocabulary shift type")
    cui_label: str = dspy.InputField(desc="Target CUI string label (if ontology-grounded)")
    alt_term: str = dspy.InputField(desc="Alternative term to use in query")
    semantic_group: str = dspy.InputField(desc="UMLS semantic group")
    vocab_shift_type: str = dspy.InputField(desc="Type of vocabulary shift")
    vocab_shift_instruction: str = dspy.InputField(desc="Instruction for vocabulary shift")

    query: str = dspy.OutputField(desc="Generated query (max 180 chars)")
    rationale: str = dspy.OutputField(desc="Brief reasoning about vocabulary choices")


class JudgeAnswerability(dspy.Signature):
    """Judge: Is the query answerable from ONLY the provided passage?"""

    passage: str = dspy.InputField()
    query: str = dspy.InputField()
    cui_label: str = dspy.InputField()
    task_family: str = dspy.InputField()
    role: str = dspy.InputField()

    answer: str = dspy.OutputField(desc="ANSWERABLE or NOT_ANSWERABLE")


class JudgeVocabShift(dspy.Signature):
    """Judge: Does the query use different vocabulary from the passage?"""

    passage: str = dspy.InputField()
    query: str = dspy.InputField()
    cui_label: str = dspy.InputField()
    task_family: str = dspy.InputField()
    vocab_shift_type: str = dspy.InputField()

    answer: str = dspy.OutputField(desc="STRONG_SHIFT, MODERATE_SHIFT, MINIMAL_SHIFT, or NO_SHIFT")


class JudgePassageSpecificity(dspy.Signature):
    """Judge: Is the query grounded in this specific passage?"""

    passage: str = dspy.InputField()
    query: str = dspy.InputField()
    cui_label: str = dspy.InputField()
    task_family: str = dspy.InputField()

    answer: str = dspy.OutputField(desc="SPECIFIC or GENERIC")


class JudgeFactualGrounding(dspy.Signature):
    """Judge: Does the query contain unsupported factual claims?"""

    passage: str = dspy.InputField()
    query: str = dspy.InputField()
    cui_label: str = dspy.InputField()

    answer: str = dspy.OutputField(desc="CLEAN, HALLUCINATION, or CONTRADICTION")


class JudgeTaskFamilyFit(dspy.Signature):
    """Judge: Does the query match its assigned task family and role?"""

    passage: str = dspy.InputField()
    query: str = dspy.InputField()
    task_family: str = dspy.InputField()
    role: str = dspy.InputField()
    task_family_description: str = dspy.InputField()

    answer: str = dspy.OutputField(desc="MATCH or MISMATCH")


# ── DSPy Pipeline Module ──────────────────────────────────────────────────────


STRICT_GROUNDING_INSTRUCTION = """\
Generate a short medical query that asks about ONLY what is stated in the passage.
CRITICAL RULES:
1. ONLY ask about facts explicitly written in the passage — never add or infer.
2. Do NOT mention dosages, side effects, mechanisms, statistics, or study methods unless they appear word-for-word in the passage.
3. If the passage only states "X causes Y", ask only "What causes Y?" or "How does X relate to Y?" using the alternative term for X.
4. Use the alternative term ({alt_term}) instead of the clinical term ({cui_label}).
5. The query must be answerable from THIS passage alone.
6. Keep the query under 150 characters.
7. Write as a {role} would ask."""


class QueryGenerator(dspy.Module):
    """DSPy module for generating a query from a passage."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateQuery)

    def forward(
        self,
        passage: str,
        role: str,
        task_family: str,
        cui_label: str = "",
        alt_term: str = "",
        semantic_group: str = "other",
        vocab_shift_type: str = "generic",
        vocab_shift_instruction: str = "Use different vocabulary",
    ) -> dict[str, Any]:
        # Build strict grounding instruction with filled placeholders
        instruction = STRICT_GROUNDING_INSTRUCTION.format(
            alt_term=alt_term or "different words",
            cui_label=cui_label or "the concept",
            role=role,
        )
        # Use the instruction to override the default prompt
        self.generate = dspy.Predict(GenerateQuery, instructions=instruction)
        result = self.generate(
            passage=passage[:1800],
            role=role,
            task_family=task_family,
            cui_label=cui_label,
            alt_term=alt_term,
            semantic_group=semantic_group,
            vocab_shift_type=vocab_shift_type,
            vocab_shift_instruction=vocab_shift_instruction,
        )
        return {"query": result.query, "rationale": result.rationale}


class JudgePipeline(dspy.Module):
    """DSPy module for judging a query-passage pair across all 5 dimensions."""

    def __init__(self):
        super().__init__()
        self.judge_answerability = dspy.Predict(JudgeAnswerability)
        self.judge_vocab_shift = dspy.Predict(JudgeVocabShift)
        self.judge_specificity = dspy.Predict(JudgePassageSpecificity)
        self.judge_factual = dspy.Predict(JudgeFactualGrounding)
        self.judge_family_fit = dspy.Predict(JudgeTaskFamilyFit)

    def forward(
        self,
        passage: str,
        query: str,
        cui_label: str = "",
        task_family: str = "",
        role: str = "",
        vocab_shift_type: str = "",
    ) -> dict[str, dict[str, str]]:
        """Run all 5 judge questions. Returns a dict of {dimension: answer}."""
        task_family_desc = get_task_family_description(task_family)

        return {
            "answerability": self.judge_answerability(
                passage=passage, query=query, cui_label=cui_label,
                task_family=task_family, role=role,
            ).answer,
            "vocab_shift": self.judge_vocab_shift(
                passage=passage, query=query, cui_label=cui_label,
                task_family=task_family, vocab_shift_type=vocab_shift_type,
            ).answer,
            "passage_specificity": self.judge_specificity(
                passage=passage, query=query, cui_label=cui_label,
                task_family=task_family,
            ).answer,
            "factual_grounding": self.judge_factual(
                passage=passage, query=query, cui_label=cui_label,
            ).answer,
            "task_family_fit": self.judge_family_fit(
                passage=passage, query=query, task_family=task_family,
                role=role, task_family_description=task_family_desc,
            ).answer,
        }


# ── LM Setup ──────────────────────────────────────────────────────────────────


def setup_generator_lm(
    api_base: str = "http://localhost:8000/v1",
    model: str = "gemma-4-31B",
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> dspy.LM:
    """Configure a DSPy LM for generation (creative, temp=0.7).

    Uses OpenAI-compatible API (vLLM).
    """
    lm = dspy.LM(
        f"openai/{model}",
        api_base=api_base,
        api_key="not-needed",  # vLLM doesn't require auth
        temperature=temperature,
        max_tokens=max_tokens,
        cache=True,
    )
    return lm


def setup_judge_lm(
    api_base: str = "http://localhost:8000/v1",
    model: str = "gemma-4-31B",
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> dspy.LM:
    """Configure a DSPy LM for judging (deterministic, temp=0.0).

    Uses OpenAI-compatible API (vLLM).
    """
    lm = dspy.LM(
        f"openai/{model}",
        api_base=api_base,
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
        cache=True,
    )
    return lm


# ── Generation Pipeline Orchestrator ──────────────────────────────────────────


@dataclass
class GenerationPipeline:
    """Orchestrates the full generation → validate → judge pipeline.

    Usage:
        gen_config = load_yaml("configs/generation.yaml")
        pipeline = GenerationPipeline.from_config(gen_config)
        results = pipeline.generate_batch(passages_df, mode="generic_synthetic")
    """

    generator_lm: dspy.LM | None = None
    judge_lm: dspy.LM | None = None
    generator: QueryGenerator | None = None
    judge_pipeline: JudgePipeline | None = None
    validators: DeterministicValidators | None = None
    config: dict = field(default_factory=dict)
    generator_prompt_hash: str = ""
    judge_prompt_hash: str = ""

    @classmethod
    def from_config(cls, config: dict) -> "GenerationPipeline":
        """Create a GenerationPipeline from a generation config dict."""
        dspy_cfg = config.get("dspy", {})
        gen_cfg = config.get("generation", {})
        val_cfg = config.get("validators", {})

        # Setup LMs
        model_name = dspy_cfg.get("lm", "gemma-4-31B")
        gen_lm = setup_generator_lm(
            api_base=dspy_cfg.get("api_base", "http://localhost:8000/v1"),
            model=model_name,
            temperature=dspy_cfg.get("temperature", 0.7),
            max_tokens=dspy_cfg.get("max_tokens", 256),
        )
        judge_lm = setup_judge_lm(
            api_base=dspy_cfg.get("api_base", "http://localhost:8000/v1"),
            model=model_name,
            temperature=0.0,
            max_tokens=64,
        )

        # Set DSPy defaults
        dspy.configure(lm=gen_lm)

        # Build modules
        generator = QueryGenerator()
        judge_pipeline = JudgePipeline()
        validators = DeterministicValidators(
            max_forbidden_surface_overlap=val_cfg.get("max_forbidden_surface_overlap", 0.25),
            min_alternative_term_count=val_cfg.get("min_alternative_term_count", 1),
            reject_generic_queries=val_cfg.get("reject_generic_queries", True),
            reject_near_duplicates=val_cfg.get("reject_near_duplicates", True),
            minhash_threshold=val_cfg.get("minhash_threshold", 0.86),
            min_query_chars=gen_cfg.get("min_query_chars", 5),
            max_query_chars=gen_cfg.get("max_query_chars", 180),
            max_passage_chars=gen_cfg.get("max_passage_chars", 1800),
        )

        # Hash prompts for reproducibility
        gen_hash = hash_prompt(GENERATOR_SYSTEM_PROMPT)
        judge_hash = hash_prompt(JUDGE_SYSTEM_PROMPT)

        return cls(
            generator_lm=gen_lm,
            judge_lm=judge_lm,
            generator=generator,
            judge_pipeline=judge_pipeline,
            validators=validators,
            config=config,
            generator_prompt_hash=gen_hash,
            judge_prompt_hash=judge_hash,
        )

    def generate_batch(
        self,
        passages: list[dict],
        mode: str = "generic_synthetic",
        task_families: list[str] | None = None,
        roles: list[str] | None = None,
    ) -> list[dict]:
        """Generate a batch of query-passage candidates.

        Args:
            passages: List of passage dicts with keys: passage_id, text, [cui_label, alt_term, ...]
            mode: Generation mode.
            task_families: List of task families to cycle through.
            roles: List of roles to cycle through.

        Returns:
            List of candidate dicts.
        """
        import random

        if task_families is None:
            task_families = [
                "lay_to_clinical", "abbreviation_to_expanded", "brand_generic",
                "symptom_to_diagnosis", "biomedical_to_clinical",
                "no_direct_lexical_overlap",
            ]
        if roles is None:
            roles = ["patient", "physician", "researcher", "nurse", "student"]

        candidates: list[dict] = []
        batch_id = stable_hash(str(len(candidates)), str(pd.Timestamp.now()))

        for i, passage_dict in enumerate(passages):
            passage_id = passage_dict.get("passage_id", f"p_{i}")
            passage_text = passage_dict.get("text", "")
            cui_label = passage_dict.get("cui_label", "")
            alt_term = passage_dict.get("alt_term", "")
            semantic_group = passage_dict.get("semantic_group", "other")
            vocab_shift_type = passage_dict.get("vocab_shift_type", "generic")

            task_family = task_families[i % len(task_families)]
            role = random.choices(
                roles,
                weights=[0.30, 0.25, 0.20, 0.15, 0.10],
                k=1,
            )[0]

            vocab_shift_instruction = (
                "Use entirely different words" if task_family == "no_direct_lexical_overlap"
                else "Use different vocabulary — avoid copying passage terms"
            )

            try:
                # Switch LM to generator (temp=0.7)
                with dspy.context(lm=self.generator_lm):
                    result = self.generator(
                        passage=passage_text[:1800],
                        role=role,
                        task_family=task_family,
                        cui_label=cui_label,
                        alt_term=alt_term,
                        semantic_group=semantic_group,
                        vocab_shift_type=vocab_shift_type,
                        vocab_shift_instruction=vocab_shift_instruction,
                    )
            except Exception as e:
                candidates.append({
                    "example_id": stable_hash(passage_id, str(i), mode, "error"),
                    "passage_id": passage_id,
                    "query": "",
                    "mode": mode,
                    "task_family": task_family,
                    "role": role,
                    "generation_error": str(e),
                    "passed_validation": False,
                })
                continue

            query = result["query"]
            rationale = result.get("rationale", "")

            example_id = stable_hash(passage_id, query, mode, str(i))

            # Deterministic validation
            # Build flat list of forbidden terms (first/last 20 words of passage)
            passage_words = passage_text.split()
            forbidden = passage_words[:20] + passage_words[-20:] if len(passage_words) > 40 else passage_words
            validation = self.validators.validate(
                example_id=example_id,
                query=query,
                passage=passage_text,
                forbidden_terms=forbidden,
                alt_terms=[alt_term] if alt_term else None,
            )

            candidates.append({
                "example_id": example_id,
                "passage_id": passage_id,
                "passage_text": passage_text,  # Include passage for judging
                "query": query,
                "rationale": rationale,
                "mode": mode,
                "task_family": task_family,
                "role": role,
                "target_cuis": [cui_label] if cui_label else [],
                "semantic_group": semantic_group,
                "vocab_shift_type": vocab_shift_type,
                "generator_model": "gemma-4-31b",
                "prompt_hash": self.generator_prompt_hash,
                "passed_validation": validation.passed,
                "validation_failures": validation.failures,
                "validation_warnings": validation.warnings,
            })

        return candidates

    def judge_candidates(
        self,
        candidates: list[dict],
        judge_model: str = "gemma-4-31b",
    ) -> list[dict]:
        """Judge all validation-passed candidates across 5 dimensions.

        Only judges candidates that passed deterministic validation.
        Returns the full candidate list with judgment fields added.
        """
        judged: list[dict] = []

        with dspy.context(lm=self.judge_lm):
            for candidate in candidates:
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
                    judged.append(candidate)
                    continue

                try:
                    judgments = self.judge_pipeline(
                        passage=candidate.get("passage_text", "")[:1800],
                        query=candidate["query"],
                        cui_label=candidate.get("target_cuis", [""])[0] if candidate.get("target_cuis") else "",
                        task_family=candidate.get("task_family", ""),
                        role=candidate.get("role", ""),
                        vocab_shift_type=candidate.get("vocab_shift_type", ""),
                    )

                    decision, rejection_reason = aggregate_decision(judgments, strict=True)

                    candidate["judgments"] = judgments
                    candidate["decision"] = decision
                    candidate["rejection_reason"] = rejection_reason
                    candidate["judge_model"] = judge_model
                    candidate["judge_prompt_hash"] = self.judge_prompt_hash

                except Exception as e:
                    candidate["judgments"] = {
                        "answerability": "NOT_ANSWERABLE",
                        "vocab_shift": "NO_SHIFT",
                        "passage_specificity": "GENERIC",
                        "factual_grounding": "HALLUCINATION",
                        "task_family_fit": "MISMATCH",
                    }
                    candidate["decision"] = "REJECT"
                    candidate["rejection_reason"] = f"judge_error:{e}"

                judged.append(candidate)

        return judged
