"""UMLS RRF file parser.

Parses MRCONSO.RRF, MRREL.RRF, MRSTY.RRF into structured records.
Keeps original UMLS strings private; never writes them to public outputs.

RRF format reference:
  MRCONSO: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
  MRSTY:   CUI|TUI|STN|STY|ATUI|CVF
  MRREL:   CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from medcolbert.utils.hashing import stable_hash
from medcolbert.utils.text import normalize_text, is_abbreviation_candidate

# Increase CSV field size limit for large UMLS string fields
csv.field_size_limit(2**31 - 1)

# ── MRCONSO columns (0-indexed) ──────────────────────────────────────────────
MRCONSO_COLS = [
    "cui", "lat", "ts", "lui", "stt", "sui", "ispref", "aui",
    "saui", "scui", "sdui", "sab", "tty", "code", "str", "srl",
    "suppress", "cvf",
]

# ── MRSTY columns ────────────────────────────────────────────────────────────
MRSTY_COLS = ["cui", "tui", "stn", "sty", "atui", "cvf"]

# ── MRREL columns ────────────────────────────────────────────────────────────
MRREL_COLS = [
    "cui1", "aui1", "stype1", "rel", "cui2", "aui2", "stype2",
    "rela", "rui", "srui", "sab", "sl", "rg", "dir", "suppress", "cvf",
]

# ── Source vocabulary groupings ──────────────────────────────────────────────
SOURCE_GROUPS = {
    "consumer": {"CHV"},
    "clinical": {"SNOMEDCT_US", "ICD10CM", "ICD9CM", "MSH", "NCI"},
    "drug": {"RXNORM", "VANDF"},
}

# ─── Semantic type → group mapping (UmLS Semantic Network groups) ────────────
# Based on the UMLS Semantic Network grouping at
# https://lhncbc.nlm.nih.gov/semanticnetwork/
SEMANTIC_GROUP_MAP: dict[str, str] = {
    # Disorders
    "T020": "disorders", "T049": "disorders", "T190": "disorders",
    "T191": "disorders", "T033": "disorders", "T037": "disorders",
    "T046": "disorders", "T047": "disorders", "T048": "disorders",
    "T184": "disorders", "T019": "disorders", "T050": "disorders",
    # Procedures
    "T060": "procedures", "T061": "procedures", "T059": "procedures",
    "T062": "procedures", "T063": "procedures", "T064": "procedures",
    # Drugs & Chemicals
    "T116": "drugs_chemicals", "T195": "drugs_chemicals", "T121": "drugs_chemicals",
    "T122": "drugs_chemicals", "T123": "drugs_chemicals", "T109": "drugs_chemicals",
    "T110": "drugs_chemicals", "T114": "drugs_chemicals", "T115": "drugs_chemicals",
    "T131": "drugs_chemicals", "T196": "drugs_chemicals", "T197": "drugs_chemicals",
    "T200": "drugs_chemicals",
    # Anatomy
    "T017": "anatomy", "T018": "anatomy", "T021": "anatomy", "T022": "anatomy",
    "T023": "anatomy", "T024": "anatomy", "T025": "anatomy", "T026": "anatomy",
    "T029": "anatomy", "T030": "anatomy",
    # Findings, Signs & Symptoms
    "T032": "findings_signs_symptoms", "T034": "findings_signs_symptoms",
    "T040": "findings_signs_symptoms", "T041": "findings_signs_symptoms",
    "T042": "findings_signs_symptoms", "T043": "findings_signs_symptoms",
    "T044": "findings_signs_symptoms", "T045": "findings_signs_symptoms",
    "T056": "findings_signs_symptoms", "T057": "findings_signs_symptoms",
    "T184": "disorders",
    # Genes / Proteins
    "T028": "genes_proteins", "T029": "genes_proteins", "T116": "genes_proteins",
    "T085": "genes_proteins", "T086": "genes_proteins", "T087": "genes_proteins",
    "T088": "genes_proteins", "T089": "genes_proteins",
    # Physiology
    "T039": "physiology", "T040": "physiology", "T041": "physiology",
    "T042": "physiology", "T043": "physiology", "T044": "physiology",
    # Objects
    "T071": "objects", "T072": "objects", "T073": "objects", "T074": "objects",
    "T075": "objects",
    # Concepts & Ideas
    "T077": "concepts_ideas", "T078": "concepts_ideas", "T079": "concepts_ideas",
    "T080": "concepts_ideas", "T082": "concepts_ideas", "T185": "concepts_ideas",
    # Living Beings
    "T005": "living_beings", "T007": "living_beings", "T008": "living_beings",
    "T009": "living_beings", "T010": "living_beings", "T011": "living_beings",
    "T012": "living_beings", "T013": "living_beings", "T014": "living_beings",
    "T015": "living_beings", "T016": "living_beings", "T098": "living_beings",
    "T099": "living_beings", "T100": "living_beings", "T101": "living_beings",
    # Activities & Behaviors
    "T051": "activities_behaviors", "T052": "activities_behaviors",
    "T053": "activities_behaviors", "T054": "activities_behaviors",
    "T055": "activities_behaviors", "T056": "activities_behaviors",
    "T057": "activities_behaviors",
    # Devices
    "T065": "devices", "T066": "devices", "T067": "devices", "T068": "devices",
    "T069": "devices", "T070": "devices",
    # Geographic Areas
    "T083": "geographic_areas",
    # Organizations
    "T090": "organizations", "T091": "organizations", "T092": "organizations",
    "T093": "organizations", "T094": "organizations", "T095": "organizations",
    "T096": "organizations", "T097": "organizations",
    # Occupations
    "T090": "organizations", "T091": "organizations",
    # Groups
    "T096": "organizations", "T097": "organizations", "T098": "living_beings",
    "T099": "living_beings", "T100": "living_beings", "T101": "living_beings",
    "T102": "groups",
}


def map_source_group(sab: str) -> str:
    """Map a source abbreviation (SAB) to a high-level source group."""
    for group, sabs in SOURCE_GROUPS.items():
        if sab in sabs:
            return group
    return "other"


def map_semantic_group(tui: str) -> str:
    """Map a semantic type UI (TUI) to its semantic group."""
    return SEMANTIC_GROUP_MAP.get(tui, "other")


@dataclass
class ConceptString:
    """A single UMLS concept string record (private)."""

    cui: str
    sab: str
    tty: str
    code: str
    scui: str | None
    string: str
    normalized_string: str
    is_abbreviation: bool
    semantic_types: list[str] = field(default_factory=list)
    semantic_group: str = "other"
    source_group: str = "other"
    string_hash: str = ""


@dataclass
class ConceptPair:
    """A pair of concept strings for vocabulary shift (private)."""

    pair_id: str
    cui: str
    left_string_id: str
    right_string_id: str
    left_source_group: str
    right_source_group: str
    vocab_shift_type: str
    semantic_group: str
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class ConceptEdge:
    """A relationship edge between two CUIs (private)."""

    edge_id: str
    source_cui: str
    target_cui: str
    relation: str
    rela: str | None
    sab: str
    negative_use: str
    confidence: float


# ── Parsing ──────────────────────────────────────────────────────────────────


def parse_mrconso(path: Path, config: dict | None = None) -> Iterator[dict]:
    """Parse MRCONSO.RRF and yield filtered concept string records.

    Filters:
      - Language: ENG only (configurable)
      - Suppress: N only
      - Min string chars: 3 (configurable)
      - Max string chars: 160 (configurable)
    """
    cfg = config or {}
    filters = cfg.get("filters", {})
    target_lang = filters.get("language", "ENG")
    min_chars = filters.get("min_string_chars", 3)
    max_chars = filters.get("max_string_chars", 160)
    allow_abbrev = filters.get("allow_abbreviations_only_if_unambiguous", True)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 15:
                continue
            lat = row[1].strip()
            suppress = row[16].strip() if len(row) > 16 else ""
            string = row[14].strip() if len(row) > 14 else ""

            # Language filter
            if lat != target_lang:
                continue
            # Suppress filter
            if suppress == "O":
                continue
            # Length filters
            if len(string) < min_chars:
                continue
            if len(string) > max_chars:
                continue

            cui = row[0].strip()
            sab = row[11].strip() if len(row) > 11 else ""
            tty = row[12].strip() if len(row) > 12 else ""
            code = row[13].strip() if len(row) > 13 else ""
            scui = row[9].strip() if len(row) > 9 else ""

            normalized = normalize_text(string)
            is_abbrev = is_abbreviation_candidate(string)

            source_group = map_source_group(sab)
            string_hash = stable_hash(cui, sab, tty, code, string)

            yield {
                "cui": cui,
                "sab": sab,
                "tty": tty,
                "code": code,
                "scui": scui if scui else None,
                "string": string,
                "normalized_string": normalized,
                "is_abbreviation": is_abbrev,
                "source_group": source_group,
                "string_hash": string_hash,
            }


def parse_mrsty(path: Path) -> dict[str, list[dict]]:
    """Parse MRSTY.RRF and return CUI → semantic type list mapping.

    Returns:
        Dict mapping CUI to list of {tui, stn, semantic_group} dicts.
    """
    cui_types: dict[str, list[dict]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 4:
                continue
            cui = row[0].strip()
            tui = row[1].strip() if len(row) > 1 else ""
            stn = row[2].strip() if len(row) > 2 else ""
            if cui and tui:
                cui_types[cui].append({
                    "tui": tui,
                    "stn": stn,
                    "semantic_group": map_semantic_group(tui),
                })
    return dict(cui_types)


def parse_mrrel(
    path: Path,
    config: dict | None = None,
) -> Iterator[dict]:
    """Parse MRREL.RRF and yield filtered concept edges.

    Filters:
      - Hierarchical relations (PAR, CHD, RB, RN) and related (RO, RQ)
    """
    cfg = config or {}
    relations_cfg = cfg.get("relations", {})
    hierarchical = set(relations_cfg.get("hierarchical", ["PAR", "CHD", "RB", "RN"]))
    related = set(relations_cfg.get("related", ["RO", "RQ"]))
    target_rels = hierarchical | related

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 16:
                continue
            rel = row[3].strip() if len(row) > 3 else ""
            if rel not in target_rels:
                continue
            suppress = row[14].strip() if len(row) > 14 else ""
            if suppress == "O":
                continue

            cui1 = row[0].strip()
            cui2 = row[4].strip() if len(row) > 4 else ""
            rela = row[7].strip() if len(row) > 7 else ""
            sab = row[10].strip() if len(row) > 10 else ""

            # Skip self-loops
            if cui1 == cui2:
                continue

            edge_id = stable_hash(cui1, cui2, rel, rela or "", sab)

            yield {
                "edge_id": edge_id,
                "source_cui": cui1,
                "target_cui": cui2,
                "relation": rel,
                "rela": rela if rela else None,
                "sab": sab,
                "negative_use": "concept_near_negative" if rel in hierarchical else "related_negative",
                "confidence": 1.0 if rel in hierarchical else 0.5,
            }
