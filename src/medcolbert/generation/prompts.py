"""Prompt templates for synthetic query generation and judging.

Per-task-family generation prompts and per-dimension judge prompts.
All prompts are designed for Gemma 4 31B native reasoning (no ChainOfThought adapter).
"""

from __future__ import annotations

from medcolbert.utils.hashing import stable_hash

# ── Generation Prompts ────────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """\
You are a medical query generator for training biomedical retrieval systems.
Your job is to generate realistic, natural queries that a person might ask,
given a medical passage or concept pair. The query must express the same
medical concept using DIFFERENT vocabulary than the source — this is
vocabulary shift, not paraphrasing.

Do NOT copy terms from the passage directly. Use layperson language,
abbreviations, brand names, or related medical terminology as appropriate.

Vary question structure — avoid always starting with Does/Can/Do/Did/What/How.
Start some questions with Is, Are, Would, Could, Should, When, Why, or
use a different structure entirely. Do not use deictic references like
'these', 'this', 'those' without a clear, specific referent.

Output ONLY a valid JSON object with exactly these keys:
- "query": the generated query (max 180 characters)
- "rationale": brief reasoning about the vocabulary choices made (max 100 words)
"""

GENERATOR_TEMPLATES: dict[str, str] = {
    # ── generic_synthetic: query from passage, no ontology ────────────────
    "generic_synthetic": """\
Generate a realistic medical search query that could be answered by the following passage.
Use different vocabulary from the passage — a patient, nurse, or student might phrase this differently.

Passage:
{passage}

Role: {role}
Task family: {task_family}

Generate a query that:
- Is answerable from ONLY this passage
- Uses different wording than the passage
- Reflects how a {role} would naturally ask
""",

    # ── passage_grounded: query must be answerable ────────────────────────
    "passage_grounded": """\
Generate a medical query that can be answered using ONLY the information
in the following passage. The query must require specific content from this
exact passage — not be a generic question about the topic.

Passage:
{passage}

Role: {role}
Task family: {task_family}

Requirements:
- The query must be answerable from this specific passage
- The query should use different vocabulary than the passage ({vocab_shift_instruction})
- The query should reflect the perspective of a {role}
""",

    # ── ontology_grounded: targets CUIs with vocab shift ──────────────────
    "ontology_grounded": """\
Generate a medical query that expresses the medical concept below using
DIFFERENT terminology. This is a vocabulary-shift generation task.

CRITICAL: Use ONLY information explicitly stated in the passage below.
Do NOT add, infer, or expand on any facts beyond what the passage states.
Do not mention specific dosages, side effects, mechanisms, study designs,
or statistics unless they appear word-for-word in the passage.
If the passage only describes that concept X exists, your query should
only ask about concept X using the alternative vocabulary — nothing more.

Target concept: {cui_label}
Alternative term to use: {alt_term}
Semantic group: {semantic_group}

Passage:
{passage}

Role: {role}
Task family: {task_family}
Vocab shift type: {vocab_shift_type}

The passage contains the concept expressed in clinical/professional language.
Your query must express the SAME concept using the alternative terminology,
asking ONLY about information that IS in the passage.
""",

    # ── ontology_grounded_teacher_filtered: adds teacher margin ───────────
    "ontology_grounded_teacher_filtered": """\
Generate a high-quality medical query that expresses the target medical concept
using vocabulary that a {role} would use. This query will be evaluated by a
medical expert judge for quality.

Target concept: {cui_label}
Alternative term to use: {alt_term}
Semantic group: {semantic_group}

Passage:
{passage}

Role: {role}
Task family: {task_family}
Vocab shift type: {vocab_shift_type}

Requirements:
- Express the target concept using the alternative vocabulary
- Ensure the query is answerable from ONLY the provided passage
- Make the query specific to this passage (not generic)
- Do NOT hallucinate facts not in the passage
- Write naturally as a {role} would ask
- Vary question structure — avoid always starting with Does/Can/Do/Did/What/How. Use Is, Are, Would, Could, Should, When, Why, or other structures
- Do not use deictic references like 'these', 'this', 'those' without a clear, specific referent from the passage
""",
}

# ── Task Family Instructions ──────────────────────────────────────────────────

TASK_FAMILY_INSTRUCTIONS: dict[str, str] = {
    "lay_to_clinical": "Use patient/layperson wording to ask about a clinical concept. Replace clinical terms with everyday language.",
    "abbreviation_to_expanded": "Ask using a medical abbreviation; the passage uses the expanded/formal term.",
    "brand_generic": "Ask using a brand name or generic drug name; the passage uses the alternative.",
    "symptom_to_diagnosis": "Ask with symptom/malaise phrasing; the passage describes the diagnosis or treatment.",
    "trial_matching": "Write a patient vignette that matches a clinical trial's inclusion criteria.",
    "biomedical_to_clinical": "Ask using biomedical/research terminology; the passage uses clinical language.",
    "patient_similarity": "Write a patient summary; the passage describes a similar case or report.",
    "no_direct_lexical_overlap": "Paraphrase the concept completely — use entirely different words. No shared content words with the passage.",
}

# ── Role Instructions ─────────────────────────────────────────────────────────

ROLE_INSTRUCTIONS: dict[str, str] = {
    "patient": "You are a patient describing your condition or concern in everyday language. Use simple, non-technical words.",
    "physician": "You are a physician writing a clinical query. Use professional medical language but ask about specific clinical scenarios.",
    "researcher": "You are a biomedical researcher. Use research-oriented language — mention study types, mechanisms, or epidemiological concepts.",
    "nurse": "You are a nurse. Use practical clinical language — focus on care, monitoring, administration, or patient education.",
    "student": "You are a medical or nursing student. Your language is educational — you're learning and may mix lay and technical terms.",
}

# ── Judge Prompts ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are a medical relevance judge evaluating synthetic query-passage pairs
for a biomedical retrieval training dataset. Your job is to assess the quality
of each generated query against the provided passage.

For each question, choose EXACTLY ONE answer letter. Answer with the letter
only after a brief reasoning line.

Be fair but not overly strict. If a query asks about concepts clearly present
in the passage, mark it ANSWERABLE. A question is not HALLUCINATION just because
it asks about something — it must ASSERT a false claim. Reserve HALLUCINATION
for queries that state factual claims absent from the passage.
"""

# ── Q1: Answerability ─────────────────────────────────────────────────────────

Q1_ANSWERABILITY_PROMPT = """\
Is this query answerable from ONLY the provided passage?

Passage:
{passage}

Query:
{query}

Target concept (CUI label): {cui_label}
Task family: {task_family}
Role: {role}

A) ANSWERABLE — the passage contains the specific information needed to answer this query (even if the answer is negative, e.g., "no, X does not cause Y")
B) NOT_ANSWERABLE — the passage does not contain sufficient information to answer this query

Answer (A or B):\
"""

# ── Q2: Vocabulary Shift ──────────────────────────────────────────────────────

Q2_VOCAB_SHIFT_PROMPT = """\
Does the query use different vocabulary from the passage to express
the same medical concept?

Passage:
{passage}

Query:
{query}

Target concept (CUI label): {cui_label}
Task family: {task_family}
Vocab shift type: {vocab_shift_type}

A) STRONG_SHIFT — entirely different terms (e.g., "heart attack prevention pill" vs "myocardial infarction prophylaxis medication")
B) MODERATE_SHIFT — some overlap but meaningfully different terms
C) MINIMAL_SHIFT — mostly same words, rearranged
D) NO_SHIFT — direct copy or near-copy of passage language

Answer (A, B, C, or D):\
"""

# ── Q3: Passage Specificity ───────────────────────────────────────────────────

Q3_PASSAGE_SPECIFICITY_PROMPT = """\
Is this query grounded in the specific content of this passage, or is it
a generic question that could apply to many passages about this topic?

Passage:
{passage}

Query:
{query}

Target concept (CUI label): {cui_label}
Task family: {task_family}

A) SPECIFIC — the query references details, findings, or context unique to this passage
B) GENERIC — this query could be asked without ever reading this passage

Answer (A or B):\
"""

# ── Q4: Factual Grounding ────────────────────────────────────────────────────

Q4_FACTUAL_GROUNDING_PROMPT = """\
Does the query assert any factual claim that is NOT supported by the passage?

Key distinction: A question that ASKS about whether X causes Y (when X and Y are in the passage) is CLEAN. Only mark HALLUCINATION if the query STATES a fact not in the passage.

Passage:
{passage}

Query:
{query}

Target concept (CUI label): {cui_label}

A) CLEAN — the query asks about or references only things mentioned in the passage
B) HALLUCINATION — the query asserts a specific factual claim absent from the passage
C) CONTRADICTION — the query asserts something directly contradicted by the passage

Answer (A, B, or C):\
"""

# ── Q5: Task Family Fit ──────────────────────────────────────────────────────

Q5_TASK_FAMILY_FIT_PROMPT = """\
Does this query match its assigned task family and role?

Passage:
{passage}

Query:
{query}

Assigned task family: {task_family}
Assigned role: {role}

Task family expectation: {task_family_description}

A) MATCH — the query's language, perspective, and vocabulary level correctly reflect the assigned task family and role
B) MISMATCH — the query reads as a different task family or role than assigned

Answer (A or B):\
"""

# ── Prompt utilities ──────────────────────────────────────────────────────────


def get_generator_prompt(mode: str, **kwargs) -> str:
    """Return the generation prompt for a given mode, filled with kwargs."""
    template = GENERATOR_TEMPLATES.get(mode, GENERATOR_TEMPLATES["generic_synthetic"])
    return template.format(**kwargs)


def get_judge_prompt(question: str, **kwargs) -> str:
    """Return the judge prompt for a given quality dimension."""
    prompts = {
        "answerability": Q1_ANSWERABILITY_PROMPT,
        "vocab_shift": Q2_VOCAB_SHIFT_PROMPT,
        "passage_specificity": Q3_PASSAGE_SPECIFICITY_PROMPT,
        "factual_grounding": Q4_FACTUAL_GROUNDING_PROMPT,
        "task_family_fit": Q5_TASK_FAMILY_FIT_PROMPT,
    }
    template = prompts.get(question, "")
    return template.format(**kwargs)


def hash_prompt(prompt_text: str) -> str:
    """Deterministic hash of a prompt for versioning."""
    return stable_hash(prompt_text)


def get_task_family_description(task_family: str) -> str:
    """Return the task family instruction text."""
    return TASK_FAMILY_INSTRUCTIONS.get(task_family, "General medical query")


def get_role_instruction(role: str) -> str:
    """Return the role-specific instruction text."""
    return ROLE_INSTRUCTIONS.get(role, "General user")
