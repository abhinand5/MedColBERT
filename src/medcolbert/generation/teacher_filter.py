"""Teacher margin filtering for synthetic query-passage pairs.

Uses a judge LLM (Gemma 4 31B) to score candidates and compute a teacher margin.
Only candidates with margin >= minimum_teacher_margin are accepted.

The teacher filter is a second-pass quality gate applied after deterministic
validators and judge questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medcolbert.utils.logging import Counts


@dataclass
class TeacherVerdict:
    """A teacher's relevance judgment for a query-passage pair."""

    example_id: str
    relevance_score: float  # 0.0 to 1.0
    margin: float  # positive_score - negative_score
    teacher_model: str
    accepted: bool
    reasoning: str = ""


def compute_teacher_margin(
    judgments: dict[str, str],
    minimum_margin: float = 0.15,
) -> tuple[bool, float]:
    """Compute teacher margin from the 5 judge dimension answers.

    The teacher margin is a composite score:
    - Strong affirmative answers (ANSWERABLE, STRONG_SHIFT, SPECIFIC, CLEAN, MATCH)
      contribute +0.2 each
    - Moderate answers (MODERATE_SHIFT) contribute +0.1
    - Negative answers contribute 0.0
    - Margin is total_positive - total_negative (clamped to [0, 1])

    Returns:
        (accepted: bool, margin: float)
    """
    score = 0.0

    # Q1: Answerability
    q1 = judgments.get("answerability", "NOT_ANSWERABLE")
    if q1 == "ANSWERABLE":
        score += 0.2

    # Q2: Vocabulary shift
    q2 = judgments.get("vocab_shift", "NO_SHIFT")
    if q2 == "STRONG_SHIFT":
        score += 0.2
    elif q2 == "MODERATE_SHIFT":
        score += 0.1
    # MINIMAL_SHIFT or NO_SHIFT: 0

    # Q3: Passage specificity
    q3 = judgments.get("passage_specificity", "GENERIC")
    if q3 == "SPECIFIC":
        score += 0.2

    # Q4: Factual grounding
    q4 = judgments.get("factual_grounding", "HALLUCINATION")
    if q4 == "CLEAN":
        score += 0.2

    # Q5: Task family fit
    q5 = judgments.get("task_family_fit", "MISMATCH")
    if q5 == "MATCH":
        score += 0.2

    accepted = score >= (1.0 - minimum_margin)  # Need at least 0.85
    return accepted, score


def aggregate_decision(judgments: dict[str, str], strict: bool = True) -> tuple[str, str | None]:
    """Apply the aggregate decision rule from DS_GOAL.md.

    ACCEPT if (core dimensions):
      answerability == ANSWERABLE
      AND vocab_shift in (STRONG_SHIFT, MODERATE_SHIFT) OR not strict
      AND passage_specificity == SPECIFIC
      AND factual_grounding == CLEAN
      AND task_family_fit == MATCH OR not strict

    When strict=False, Q2 (vocab_shift) and Q5 (task_family_fit) are advisory
    but do not block acceptance.

    Returns:
        (decision: "ACCEPT" | "REJECT", rejection_reason: str | None)
    """
    reasons: list[str] = []

    # Core dimensions (always required)
    if judgments.get("answerability") != "ANSWERABLE":
        reasons.append(f"not_answerable:{judgments.get('answerability')}")

    if judgments.get("passage_specificity") != "SPECIFIC":
        reasons.append(f"not_specific:{judgments.get('passage_specificity')}")

    if judgments.get("factual_grounding") != "CLEAN":
        reasons.append(f"factual_issue:{judgments.get('factual_grounding')}")

    # Advisory dimensions (only required in strict mode)
    if strict:
        if judgments.get("vocab_shift") not in ("STRONG_SHIFT", "MODERATE_SHIFT"):
            reasons.append(f"insufficient_vocab_shift:{judgments.get('vocab_shift')}")

        if judgments.get("task_family_fit") != "MATCH":
            reasons.append(f"family_mismatch:{judgments.get('task_family_fit')}")

    if reasons:
        return "REJECT", "; ".join(reasons)
    return "ACCEPT", None


def compute_batch_stats(
    accepted: list[dict],
    rejected: list[dict],
) -> dict[str, Any]:
    """Compute aggregate statistics for a batch of judged examples.

    Args:
        accepted: List of accepted example dicts (with judgments).
        rejected: List of rejected example dicts (with judgments).

    Returns:
        Dict with per-dimension pass rates, per-family pass rates, etc.
    """
    total = len(accepted) + len(rejected)
    if total == 0:
        return {"total": 0, "accepted": 0, "acceptance_rate": 0.0}

    # Per-dimension pass rates
    dimensions = ["answerability", "vocab_shift", "passage_specificity", "factual_grounding", "task_family_fit"]

    dim_counts: dict[str, dict[str, int]] = {}
    for dim in dimensions:
        dim_counts[dim] = {"passed": 0, "total": total}

    # Per-family counts
    family_counts: dict[str, dict[str, int]] = {}
    role_counts: dict[str, dict[str, int]] = {}
    semantic_group_counts: dict[str, dict[str, int]] = {}

    for ex in accepted + rejected:
        judgments = ex.get("judgments", {})
        task_family = ex.get("task_family", "unknown")
        role = ex.get("role", "unknown")
        semantic_group = ex.get("semantic_group", "unknown")

        for dim in dimensions:
            val = judgments.get(dim, "")
            if dim == "answerability":
                dim_counts[dim]["passed"] += 1 if val == "ANSWERABLE" else 0
            elif dim == "vocab_shift":
                dim_counts[dim]["passed"] += 1 if val in ("STRONG_SHIFT", "MODERATE_SHIFT") else 0
            elif dim == "passage_specificity":
                dim_counts[dim]["passed"] += 1 if val == "SPECIFIC" else 0
            elif dim == "factual_grounding":
                dim_counts[dim]["passed"] += 1 if val == "CLEAN" else 0
            elif dim == "task_family_fit":
                dim_counts[dim]["passed"] += 1 if val == "MATCH" else 0

        family_counts.setdefault(task_family, {"accepted": 0, "total": 0})
        family_counts[task_family]["total"] += 1
        if ex in accepted:
            family_counts[task_family]["accepted"] += 1

        role_counts.setdefault(role, {"accepted": 0, "total": 0})
        role_counts[role]["total"] += 1
        if ex in accepted:
            role_counts[role]["accepted"] += 1

        semantic_group_counts.setdefault(semantic_group, {"accepted": 0, "total": 0})
        semantic_group_counts[semantic_group]["total"] += 1
        if ex in accepted:
            semantic_group_counts[semantic_group]["accepted"] += 1

    return {
        "total": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / total,
        "per_dimension_pass_rates": {
            dim: counts["passed"] / counts["total"]
            for dim, counts in dim_counts.items()
        },
        "per_family_acceptance": {
            fam: counts["accepted"] / counts["total"]
            for fam, counts in family_counts.items()
        },
        "per_role_acceptance": {
            role: counts["accepted"] / counts["total"]
            for role, counts in role_counts.items()
        },
        "per_semantic_group_acceptance": {
            sg: counts["accepted"] / counts["total"]
            for sg, counts in semantic_group_counts.items()
        },
    }
