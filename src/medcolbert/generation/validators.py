"""Deterministic validators for synthetic query generation.

Stage 3 checks from DS_GOAL.md:
- Surface overlap check (forbidden terms)
- Minimum alternative term check
- Generic query rejection
- Near-duplicate detection
- Length and format checks

Every validator returns a (passed: bool, reason: str) tuple.
Never silently drop examples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from medcolbert.utils.logging import Counts
from medcolbert.utils.text import token_overlap_ratio, tokenize_simple, normalize_text


@dataclass
class ValidationResult:
    """Result of running all validators on a single example."""

    example_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DeterministicValidators:
    """Collection of deterministic validation checks."""

    def __init__(
        self,
        max_forbidden_surface_overlap: float = 0.25,
        min_alternative_term_count: int = 1,
        reject_generic_queries: bool = True,
        reject_near_duplicates: bool = True,
        minhash_threshold: float = 0.86,
        min_query_chars: int = 5,
        max_query_chars: int = 180,
        max_passage_chars: int = 1800,
    ):
        self.max_forbidden_surface_overlap = max_forbidden_surface_overlap
        self.min_alternative_term_count = min_alternative_term_count
        self.reject_generic_queries = reject_generic_queries
        self.reject_near_duplicates = reject_near_duplicates
        self.minhash_threshold = minhash_threshold
        self.min_query_chars = min_query_chars
        self.max_query_chars = max_query_chars
        self.max_passage_chars = max_passage_chars

        self.counts = Counts()
        self.seen_queries: set[str] = set()  # For exact duplicate check
        self.seen_normalized: dict[str, str] = {}  # norm_hash → example_id

    def validate(
        self,
        example_id: str,
        query: str,
        passage: str,
        forbidden_terms: list[str] | None = None,
        target_cuis: list[str] | None = None,
        alt_terms: list[str] | None = None,
    ) -> ValidationResult:
        """Run all validators on a single example.

        Returns a ValidationResult with pass/fail and any failure reasons.
        Only REJECTS on hard failures; warnings are informational.
        """
        result = ValidationResult(example_id=example_id, passed=True)

        # 1. Length checks
        if len(query) < self.min_query_chars:
            result.passed = False
            result.failures.append(f"query_too_short:{len(query)}<{self.min_query_chars}")
            self.counts.add_rejected(reason="query_too_short")
            return result

        if len(query) > self.max_query_chars:
            result.passed = False
            result.failures.append(f"query_too_long:{len(query)}>{self.max_query_chars}")
            self.counts.add_rejected(reason="query_too_long")
            return result

        if len(passage) > self.max_passage_chars:
            result.warnings.append(f"passage_long:{len(passage)}>{self.max_passage_chars}")

        # 2. Surface overlap check
        if forbidden_terms:
            overlap_ratio = _compute_forbidden_overlap(query, passage, forbidden_terms)
            if overlap_ratio > self.max_forbidden_surface_overlap:
                result.passed = False
                result.failures.append(f"forbidden_overlap:{overlap_ratio:.2f}>{self.max_forbidden_surface_overlap}")
                self.counts.add_rejected(reason="forbidden_overlap")

        # 3. Alternative term check
        if self.min_alternative_term_count > 0 and alt_terms is not None:
            alt_count = _count_alternative_terms(query, alt_terms)
            if alt_count < self.min_alternative_term_count:
                result.passed = False
                result.failures.append(f"insufficient_alt_terms:{alt_count}<{self.min_alternative_term_count}")
                self.counts.add_rejected(reason="insufficient_alt_terms")

        # 4. Generic query rejection
        if self.reject_generic_queries:
            if _is_generic_query(query):
                result.passed = False
                result.failures.append("generic_query")
                self.counts.add_rejected(reason="generic_query")

        # 5. Exact duplicate check
        if query in self.seen_queries:
            result.passed = False
            result.failures.append("exact_duplicate")
            self.counts.add_rejected(reason="exact_duplicate")

        # 6. Near-duplicate check (simplified — full MinHash in teacher_filter)
        if self.reject_near_duplicates:
            norm = normalize_text(query)
            if norm in self.seen_normalized:
                result.passed = False
                result.failures.append("near_duplicate_normalized")
                self.counts.add_rejected(reason="near_duplicate_normalized")

        # Track for future duplicate checks
        if result.passed:
            self.seen_queries.add(query)
            self.seen_normalized[normalize_text(query)] = example_id
            self.counts.add_accepted()
        else:
            self.counts.add_rejected()

        return result

    def summary(self) -> dict[str, Any]:
        """Return aggregate counts."""
        return self.counts.summary()


def _compute_forbidden_overlap(
    query: str,
    passage: str,
    forbidden_terms: list[str],
) -> float:
    """Compute the fraction of forbidden terms that overlap between query and passage.

    Returns a ratio: number of forbidden terms that appear in BOTH query and passage
    divided by total number of unique forbidden terms.
    """
    query_lower = query.lower()
    passage_lower = passage.lower()

    overlapping = 0
    total_terms = 0

    for term in forbidden_terms:
        term_lower = term.lower()
        in_query = term_lower in query_lower
        in_passage = term_lower in passage_lower
        if in_query or in_passage:
            total_terms += 1
            if in_query and in_passage:
                overlapping += 1

    if total_terms == 0:
        return 0.0

    return overlapping / total_terms


def _strip_umls_qualifiers(term: str) -> str:
    """Strip UMLS parenthetical/bracket qualifiers for fuzzy matching."""
    if not term:
        return term
    import re as _re
    t = _re.sub(r'\s*\([^)]*\)', '', term)
    t = _re.sub(r'\s*\[[^\]]*\]', '', t)
    return _re.sub(r'\s+', ' ', t).strip()


def _count_alternative_terms(query: str, alt_terms: list[str]) -> int:
    """Count how many alternative terms appear in the query.

    Uses fuzzy matching: strips UMLS qualifiers from alt_terms, and checks
    normalized term overlap (individual words) rather than exact substring.
    This prevents forcing models to include UMLS codes like "(substance)"
    in their queries just to pass validation.
    """
    query_lower = query.lower()
    query_words = set(re.findall(r'[a-z0-9]+', query_lower))

    count = 0
    for term in alt_terms:
        if not term:
            continue
        # Strip UMLS qualifiers before matching
        clean_term = _strip_umls_qualifiers(term).lower()
        if not clean_term:
            continue

        # Strategy 1: Exact cleaned substring match
        if clean_term in query_lower:
            count += 1
            continue

        # Strategy 1b: For short alt_terms (≤4 chars total, like abbreviations "TD", "COPD"),
        # check if the term appears as a whole word in the query (case-insensitive)
        if len(clean_term) <= 6:
            query_words_set = set(re.findall(r'[a-z0-9]+', query_lower))
            if clean_term.lower() in query_words_set:
                count += 1
                continue

        # Strategy 2: Check if most of the term's content words appear in query
        term_words = set(re.findall(r'[a-z0-9]+', clean_term))
        if not term_words:
            continue

        # Words ≥3 chars count as "content" words; shorter words might be noise
        content_words = {w for w in term_words if len(w) >= 3}
        if not content_words:
            content_words = term_words

        # Require ≥60% content word overlap
        overlap = content_words & query_words
        if len(overlap) / len(content_words) >= 0.6:
            count += 1
            continue

        # Strategy 3: Check if any word from the clean term appears in query
        # (only for single-word terms that might be abbreviated)
        if any(w in query_words for w in term_words if len(w) >= 4):
            count += 1

    return count


# Generic query patterns — queries that start with these are likely too generic
GENERIC_QUERY_PATTERNS = [
    r"^what is ",
    r"^what are ",
    r"^define ",
    r"^how to ",
    r"^tell me about ",
    r"^explain ",
    r"^describe ",
    r"^list ",
    r"^what does ",
    r"^how does ",
]

# Phrases that indicate passage specificity
SPECIFIC_QUERY_MARKERS = [
    r"\bthis (study|trial|patient|case|finding|result|analysis)\b",
    r"\baccording to the (passage|study|article|text|trial)\b",
    r"\bmentioned in\b",
    r"\bthe (study|trial) (found|showed|reported|demonstrated)\b",
    r"\bin this (study|trial|case|analysis|cohort)\b",
]


def _is_generic_query(query: str) -> bool:
    """Heuristic check: does this query look like a generic question?

    Uses regex patterns to identify queries that could apply to many passages.
    """
    query_lower = query.lower().strip()

    # Check for passage-specific markers first
    for marker in SPECIFIC_QUERY_MARKERS:
        if re.search(marker, query_lower):
            return False

    # If the query starts with a generic pattern without any specific details,
    # it's likely generic
    for pattern in GENERIC_QUERY_PATTERNS:
        if re.match(pattern, query_lower):
            # But if it's long enough (has details), it might still be specific
            if len(query.split()) < 10:
                return True

    return False
