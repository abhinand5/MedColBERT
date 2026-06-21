"""Round 2 re-audit per docs/CLEANUP_PLAN.md §9.3.
For each surviving family: stratified n=80 sample, Gemma judge for fact_grounding
and family_specific, Wilson 95% CI for hallucination rate, verdict from CI intervals.
Uses Gemma 4 31B via vLLM — NOT the orchestrator model."""
import json, re, hashlib, time
import pandas as pd
import numpy as np
import httpx
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from math import sqrt

ROOT = Path("/workspace/MedColBERT")
SRC = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v2.parquet"

VLLM = "http://localhost:8000/v1"
MODEL = "gemma-4-31B"
VLLM_CONCURRENCY = 16
VLLM_TIMEOUT = 120.0
N_PER_FAMILY = 80
N_PER_STYLE = 20

def gemma(system: str, user: str, max_tokens: int = 400, timeout: float = VLLM_TIMEOUT) -> str | None:
    """Call Gemma 4 31B via vLLM."""
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{VLLM}/chat/completions",
                       headers={"Content-Type": "application/json"},
                       json={"model": MODEL, "temperature": 0.0,
                             "max_tokens": max_tokens,
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [Gemma error] {e}")
        return None

def wilson_ci(n: int, k: int, z: float = 1.96) -> tuple:
    """Wilson 95% confidence interval for a proportion.
    Returns (point_estimate, lower, upper)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))

# ---- Family-specific judging criteria ----
FAMILY_PROMPTS = {
    "symptom_to_diagnosis": """FAMILY-SPECIFIC: This is a symptom_to_diagnosis family.
Criteria: The query should phrase as symptom/evidence leading to a diagnosis.
Score 5: query presents symptoms/findings clearly and the passage supports the diagnosis.
Score 4: good symptom framing, minor issues.
Score 3: acceptable but symptom-diagnosis relationship is weak.
Score 2: query reads as generic search, not symptom→diagnosis framing.
Score 1: no symptom/diagnosis framing at all, or complete mismatch.""",

    "biomedical_to_clinical": """FAMILY-SPECIFIC: This is a biomedical_to_clinical family.
Criteria: The query should use clinical/literature-search phrasing while the passage has biomedical/research content — a register shift.
Score 5: clear register shift from biomedical passage to clinical query phrasing.
Score 4: good shift, minor overlap in terminology.
Score 3: some shift but vocabulary very similar to passage.
Score 2: minimal register difference, mostly copied passage terms.
Score 1: no shift — query and passage use identical technical vocabulary.""",

    "lay_to_clinical": """FAMILY-SPECIFIC: This is a lay_to_clinical family.
Criteria: The query should use layperson-friendly wording while the passage is clinical/technical — a real register shift.
Score 5: excellent lay adaptation — patient-friendly terms replace all technical jargon.
Score 4: good lay phrasing, 1-2 technical terms remain.
Score 3: mixed — some lay terms but significant technical vocabulary remains.
Score 2: mostly technical phrasing with minimal lay adaptation.
Score 1: no lay adaptation — query reads like the clinical passage itself.""",

    "brand_generic": """FAMILY-SPECIFIC: This is a brand_generic (drug vocab-shift) family.
Criteria: The query should swap a brand name for a generic/ingredient name (or vice versa), not just paraphrase. alt_terms should be real drugs.
Score 5: clean brand↔generic swap with real drug names.
Score 4: clear drug-name substitution, alt_term is a real clinical drug.
Score 3: some drug vocabulary shift but alt_term is more chemical than drug.
Score 2: alt_term is a lab chemical/enzyme/cell-type, not really a drug.
Score 1: no brand↔generic relationship — query is just a paraphrase or alt_term is irrelevant.""",

    "generic_clinical_retrieval": """FAMILY-SPECIFIC: This is a generic_clinical_retrieval family.
Criteria: Is it a legitimate clinical/medical keyword retrieval query that matches its passage? This family is NOT expected to shift register — judge it as generic medical search. Only failures are hallucination, non-medical content, or irrelevance.
Score 5: query keywords are precise, clinically relevant, and fully passage-grounded.
Score 4: good clinical search terms, minor term mismatch with passage.
Score 3: acceptable medical keywords but some terms tangential to passage.
Score 2: keywords are mostly generic/non-clinical or poorly match passage.
Score 1: keywords are irrelevant, non-medical, or hallucinated."""
}

JUDGE_SYSTEM = """You are a strict medical-data quality auditor. For each row, you score two dimensions on a 1-5 scale and detect hallucination. You must output your response in this EXACT format:

FACT_GROUNDING: <1-5> <one-line reason>
FAMILY_SPECIFIC: <1-5> <one-line reason>
HALLUCINATED: <YES or NO>
UNGROUNDED_TERMS: <comma-separated list, or "none">

FACT_GROUNDING rubric:
5 = every query concept/claim is directly supported by the passage
4 = all major concepts supported, one minor detail not explicit
3 = mostly supported but 1-2 concepts not in passage
2 = several unsupported concepts
1 = most claims are fabricated/not in passage

The FAMILY_SPECIFIC rubric will be provided per row. Be strict and honest. Do NOT inflate scores."""

def build_judge_user(row, family: str) -> str:
    query = str(row["query"]) if pd.notna(row["query"]) else ""
    passage = str(row["passage_text"]) if pd.notna(row["passage_text"]) else ""
    fact = str(row["fact"]) if pd.notna(row["fact"]) else ""
    alt_term = str(row["alt_term"]) if pd.notna(row["alt_term"]) else "N/A"
    cui_label = str(row["cui_label"]) if pd.notna(row["cui_label"]) else "N/A"

    family_criteria = FAMILY_PROMPTS.get(family, "No family-specific criteria available.")

    return f"""{family_criteria}

PASSAGE: {passage}
QUERY: {query}
FACT (the ground-truth claim): {fact}
ALT_TERM: {alt_term}
CUI_LABEL: {cui_label}

Score FACT_GROUNDING and FAMILY_SPECIFIC on the 1-5 scales. Output exactly in the required format."""

def parse_judge_response(resp: str | None) -> dict:
    """Parse Gemma's response into structured scores. Returns dict with defaults on parse failure."""
    defaults = {"fact_grounding": 3, "family_specific": 3, "hallucinated": False,
                "ungrounded_terms": "parse_error", "raw": str(resp)[:500]}
    if resp is None:
        defaults["ungrounded_terms"] = "gemma_error"
        return defaults

    result = {"raw": resp[:500]}
    for line in resp.strip().split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("FACT_GROUNDING:"):
            content = line.split(":", 1)[1].strip()
            try:
                result["fact_grounding"] = int(content[0])
            except (ValueError, IndexError):
                # Try to find a digit
                digits = re.findall(r'\d', content)
                result["fact_grounding"] = int(digits[0]) if digits else 3
        elif upper.startswith("FAMILY_SPECIFIC:"):
            content = line.split(":", 1)[1].strip()
            try:
                result["family_specific"] = int(content[0])
            except (ValueError, IndexError):
                digits = re.findall(r'\d', content)
                result["family_specific"] = int(digits[0]) if digits else 3
        elif upper.startswith("HALLUCINATED:"):
            content = line.split(":", 1)[1].strip().upper()
            result["hallucinated"] = "YES" in content
        elif upper.startswith("UNGROUNDED_TERMS:"):
            content = line.split(":", 1)[1].strip()
            result["ungrounded_terms"] = content if content.lower() != "none" else ""

    # Fill defaults for missing keys
    for key, default in [("fact_grounding", 3), ("family_specific", 3),
                          ("hallucinated", False), ("ungrounded_terms", "")]:
        if key not in result:
            result[key] = default

    return result

def judge_row(row, family: str) -> dict:
    """Judge one row with Gemma. Returns parsed dict."""
    user = build_judge_user(row, family)
    resp = gemma(JUDGE_SYSTEM, user, max_tokens=400)
    return parse_judge_response(resp)

def monotony_check(df_family: pd.DataFrame) -> tuple:
    """Check if full_question openers are monotonous. Returns (flag, most_common_phrase, pct)."""
    fq = df_family[df_family["query_style"] == "full_question"]
    if len(fq) < 5:
        return (False, "", 0.0)

    # First word
    first_words = fq["query"].fillna("").str.strip().str.split(r"\s+").str[0]
    fw_counts = first_words.value_counts()
    top_word, top_count = fw_counts.index[0], fw_counts.iloc[0]
    top_pct = top_count / len(fq)
    if top_pct > 0.30:
        return (True, f"first word '{top_word}'", top_pct)

    # First 2 words
    first_two = fq["query"].fillna("").str.strip().str.split(r"\s+").str[:2].str.join(" ")
    ft_counts = first_two.value_counts()
    top_phrase, tp_count = ft_counts.index[0], ft_counts.iloc[0]
    tp_pct = tp_count / len(fq)
    if tp_pct > 0.30:
        return (True, f"first two words '{top_phrase}'", tp_pct)

    return (False, "", 0.0)

def vocab_shift_from_ngram(row: pd.Series) -> float:
    """Convert ngram_copy_4 to a 1-5 vocab_shift score.
    ngram_copy_4 is proportion of 4-grams copied.
    1 - ngram_copy_4 gives shift, scaled to 1-5."""
    nc = row.get("ngram_copy_4", 0.0)
    if pd.isna(nc):
        nc = 0.0
    # 1 - nc: 0=copies everything, 1=copies nothing
    shift_prop = 1.0 - nc
    # Scale to 1-5: map [0, 1] to [1, 5]
    # But ngram_copy_4 is usually very low (<0.01), so shift_prop ≈ 0.99
    # Use a more sensitive scale: map 0.00-0.05 ngram_copy to 1-5
    # ngram_copy=0.00 → 5.0, ngram_copy=0.05 → 1.0
    if nc <= 0.05:
        return 5.0 - 4.0 * (nc / 0.05)
    else:
        return 1.0

# ---- Main ----
print("Loading v2 cleaned file...")
df = pd.read_parquet(SRC)
print(f"  {len(df):,} rows")

families = sorted(df["task_family"].unique())
print(f"Families: {families}")

all_results = {}
all_worst_rows = []

for family in families:
    print(f"\n{'='*60}")
    print(f"Family: {family}")
    df_fam = df[df["task_family"] == family]
    print(f"  Rows: {len(df_fam):,}")
    print(f"  Query styles: {df_fam['query_style'].value_counts().to_dict()}")

    # ---- Stratified sampling: 20 per query_style, top up to 80 ----
    seed = int(hashlib.md5(family.encode()).hexdigest()[:8], 16) % 2**31
    rng = np.random.default_rng(seed)

    sampled_rows = []
    styles = sorted(df_fam["query_style"].unique())
    style_counts = {}
    for style in styles:
        df_style = df_fam[df_fam["query_style"] == style]
        n_take = min(N_PER_STYLE, len(df_style))
        sample = df_style.sample(n=n_take, random_state=seed + hash(style) % 10000)
        sampled_rows.append(sample)
        style_counts[style] = n_take

    sampled = pd.concat(sampled_rows, ignore_index=True)

    # Top up to 80 if needed
    if len(sampled) < N_PER_FAMILY and len(df_fam) > len(sampled):
        remaining = df_fam.drop(sampled.index.intersection(df_fam.index), errors="ignore")
        n_extra = N_PER_FAMILY - len(sampled)
        extra = remaining.sample(n=min(n_extra, len(remaining)), random_state=seed + 9999)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    n = len(sampled)
    print(f"  Sampled {n} rows (styles: {style_counts})")
    print(f"  Sample IDs: {[str(x)[:12] for x in sampled['example_id'].tolist()]}")

    # ---- Judge each row with Gemma ----
    print(f"  Judging {n} rows with Gemma...")
    counter = [0]
    counter_lock = threading.Lock()

    def judge_with_counter(row, fam):
        result = judge_row(row, fam)
        with counter_lock:
            counter[0] += 1
            if counter[0] % 20 == 0:
                print(f"    [{counter[0]}/{n}]")
        # Attach row metadata
        result["example_id"] = row["example_id"]
        result["query"] = str(row["query"])[:200]
        result["query_style"] = row["query_style"]
        result["ngram_copy_4"] = row.get("ngram_copy_4", 0.0)
        result["vocab_shift_score"] = vocab_shift_from_ngram(row)
        return result

    start_time = time.time()
    row_results = []
    with ThreadPoolExecutor(max_workers=VLLM_CONCURRENCY) as executor:
        futures = {executor.submit(judge_with_counter, row, family): i
                   for i, (_, row) in enumerate(sampled.iterrows())}
        for future in as_completed(futures):
            row_results.append(future.result())

    elapsed = time.time() - start_time
    print(f"  Judging complete in {elapsed:.0f}s ({elapsed/n:.1f}s per row)")

    # ---- Compute metrics ----
    fg_scores = [r.get("fact_grounding", 3) for r in row_results]
    fs_scores = [r.get("family_specific", 3) for r in row_results]
    vs_scores = [r.get("vocab_shift_score", 3) for r in row_results]
    halluc_count = sum(1 for r in row_results if r.get("hallucinated", False))
    pt_est, ci_low, ci_high = wilson_ci(n, halluc_count)

    print(f"\n  === {family} Results ===")
    print(f"  n = {n}")
    print(f"  Mean fact_grounding: {np.mean(fg_scores):.2f}")
    print(f"  Mean family_specific: {np.mean(fs_scores):.2f}")
    print(f"  Mean vocab_shift (from ngram_copy): {np.mean(vs_scores):.2f}")
    print(f"  Hallucination: {halluc_count}/{n} = {pt_est*100:.1f}%")
    print(f"  95% Wilson CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")

    # ---- Monotony check ----
    mono, mono_detail, mono_pct = monotony_check(df_fam)
    print(f"  Monotony: {'FLAG' if mono else 'OK'} {mono_detail + f' ({mono_pct*100:.0f}%)' if mono else ''}")

    # ---- Ngram copy from audit script ----
    # We'll get this from the ngram_copy_audit.py run, placeholder for now
    ngram_mean = df_fam["ngram_copy_4"].mean() if "ngram_copy_4" in df_fam.columns else 0.0
    print(f"  Mean ngram_copy_4: {ngram_mean:.6f}")

    # ---- Verdict from CI ----
    if ci_high < 0.10 and np.mean(fg_scores) >= 4.0 and np.mean(fs_scores) >= 3.5 and not mono:
        verdict = "PASS"
    elif ci_low >= 0.10:
        verdict = "FAIL"
    else:
        verdict = "BORDERLINE"

    print(f"  VERDICT: {verdict}")

    # ---- Worst rows ----
    worst = []
    for r in row_results:
        fg = r.get("fact_grounding", 5)
        fs = r.get("family_specific", 5)
        hall = r.get("hallucinated", False)
        if fg <= 2 or fs <= 2 or hall:
            worst.append({
                "example_id": r.get("example_id", ""),
                "fact_grounding": fg,
                "family_specific": fs,
                "hallucinated": hall,
                "ungrounded_terms": r.get("ungrounded_terms", ""),
                "query_preview": r.get("query", "")[:150],
                "dimension": "fact_grounding" if fg <= 2 else ("family_specific" if fs <= 2 else "hallucination"),
                "problem": r.get("ungrounded_terms", f"fg={fg}, fs={fs}")
            })
    worst.sort(key=lambda x: (x["fact_grounding"], x["family_specific"]))
    displayed_worst = worst[:5]
    all_worst_rows.extend([{**w, "family": family} for w in displayed_worst])

    if displayed_worst:
        print(f"\n  Worst rows:")
        for w in displayed_worst:
            print(f"    [{w['example_id'][:12]}...] fg={w['fact_grounding']} fs={w['family_specific']} "
                  f"hall={w['hallucinated']} | {w['dimension']}: {w['problem'][:120]}")

    all_results[family] = {
        "rows_in_family": len(df_fam),
        "n_sampled": n,
        "style_counts": style_counts,
        "mean_fact_grounding": np.mean(fg_scores),
        "mean_family_specific": np.mean(fs_scores),
        "mean_vocab_shift": np.mean(vs_scores),
        "hallucination_count": halluc_count,
        "hallucination_rate": pt_est,
        "wilson_ci_low": ci_low,
        "wilson_ci_high": ci_high,
        "monotony": mono,
        "monotony_detail": mono_detail,
        "ngram_copy_mean": ngram_mean,
        "verdict": verdict,
        "worst_rows": displayed_worst,
        "sample_ids": [str(x)[:12] for x in sampled["example_id"].tolist()],
    }

# ---- Aggregate table ----
print(f"\n{'='*60}")
print("AGGREGATE TABLE")
print(f"{'='*60}")
print(f"{'Family':<32} {'Rows':>6} {'VS':>5} {'FG':>5} {'FS':>5} {'Hall%':>7} {'95% CI':>16} {'Mono':>5} {'Ngram':>8} {'Verdict':>12}")
print("-" * 120)
overall_pass = True
for family in families:
    r = all_results[family]
    ci_str = f"[{r['wilson_ci_low']*100:.1f}%-{r['wilson_ci_high']*100:.1f}%]"
    print(f"{family:<32} {r['rows_in_family']:>6,} {r['mean_vocab_shift']:>5.2f} {r['mean_fact_grounding']:>5.2f} "
          f"{r['mean_family_specific']:>5.2f} {r['hallucination_rate']*100:>6.1f}% {ci_str:>16} "
          f"{'Y' if r['monotony'] else 'N':>5} {r['ngram_copy_mean']:>8.6f} {r['verdict']:>12}")
    if r["verdict"] == "FAIL":
        overall_pass = False
    # BORDERLINE is not FAIL but also not PASS — note it
print(f"\nOverall: {'PASS' if overall_pass and all(r['verdict'] != 'FAIL' for r in all_results.values()) else 'NOT ALL PASS'}")

# ---- Detailed worst rows ----
print(f"\n{'='*60}")
print("ALL WORST ROWS")
print(f"{'='*60}")
for w in sorted(all_worst_rows, key=lambda x: (x.get('family', ''), x.get('fact_grounding', 5), x.get('family_specific', 5))):
    print(f"  [{w.get('family', '')}] {w['example_id'][:12]}... fg={w['fact_grounding']} fs={w['family_specific']} "
          f"hall={w['hallucinated']} | {w['dimension']}: {w['problem'][:150]}")

# ---- Save results ----
out_json = ROOT / "data/processed/private/synthetic/consolidated/reaudit_v2_results.json"
with open(out_json, "w") as f:
    # Convert numpy types
    import json as _json
    class NpEncoder(_json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.ndarray,)): return obj.tolist()
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return super().default(obj)
    _json.dump(all_results, f, indent=2, cls=NpEncoder, default=str)

print(f"\nResults saved to {out_json}")
print("Done.")
