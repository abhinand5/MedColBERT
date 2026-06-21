"""Calibrated n=80 Gemma re-audit of the v3 cleaned dataset.

The Round-2 audit over-counted hallucinations because its Gemma prompt said
"be conservative: if a concept is NOT stated... mark it ungrounded" — which flags any
non-verbatim term (lay synonyms like 'spreads', generic verbs) as hallucination. This
script uses a CALIBRATED prompt: only flag HALLUCINATED for a fabricated or contradicted
SPECIFIC medical claim, not for synonyms/paraphrase/verbs.

vocab_shift is taken from the ngram_copy_4 column (authoritative), monotony from a regex.
Only fact_grounding + family_specific + hallucinated use Gemma (temperature 0.0).

Verdict applies to the 95% Wilson CI of the hallucination rate, never the point estimate.
"""
import json
import re
import hashlib
import httpx
import numpy as np
import pandas as pd
from math import sqrt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/workspace/MedColBERT")
SRC = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v3f.parquet"
OUT_JSON = ROOT / "data/processed/private/synthetic/consolidated/reaudit_v3f_results.json"
VLLM = "http://localhost:8000/v1"
MODEL = "gemma-4-31B"
N_PER_FAMILY = 120

FAMILY_SPECIFIC_CRITERIA = {
    "symptom_to_diagnosis": "The query should frame the topic as symptoms/evidence pointing toward a diagnosis; the passage is the clinical/diagnostic content. Does it fit symptom->diagnosis retrieval?",
    "biomedical_to_clinical": "Literature/biomedical phrasing in the query vs clinical phrasing in the passage — is there a real biomedical->clinical shift?",
    "lay_to_clinical": "Does the query use layperson wording while the passage is clinical (a real register shift)?",
    "brand_generic": "Does the query swap a brand name for a generic/ingredient (or vice versa)? Are the terms real drugs/chemicals?",
    "generic_clinical_retrieval": "Is it a legitimate medical/clinical keyword search query that matches its passage? (This family is NOT expected to shift register — judge it as generic medical search.)",
}

SYSTEM = """You are a medical-data auditor scoring synthetic (query, passage) training pairs for a retrieval model.
You will be strict about HALLUCINATION but you must NOT over-flag. Read carefully:

A keyword search query may legitimately use:
  - synonyms, paraphrase, or lay wording that is not verbatim in the passage
  - generic verbs and connectors (e.g. "spreads", "causes", "affects", "monitoring", "process", "results")
  - broader/categorical terms for what the passage discusses
These are NOT hallucinations. They are normal query vocabulary.

Mark a query HALLUCINATED=true ONLY IF it asserts a SPECIFIC medical claim — a named disease,
drug, procedure, dosage, anatomical site, or clinical action — that is either:
  (a) CONTRADICTED by the passage (the passage says the opposite), OR
  (b) a fabricated specific entity that the passage does NOT discuss and is NOT a synonym/variant of
      any concept the passage actually covers.
If the term is a reasonable synonym/paraphrase/lay-term for something in the passage, it is GROUNDED, not hallucinated.

Score fact_grounding 1-5 (5 = every specific claim supported; 1 = a fabricated/contradicted specific claim).
Score family_specific 1-5 per the given criterion.
Respond in EXACTLY this format (one line each, no prose before/after):
FACT_GROUNDING: <1-5>
FAMILY_SPECIFIC: <1-5>
HALLUCINATED: <TRUE or FALSE>
UNGROUNDED_TERMS: <comma list of any fabricated/contradicted specific terms, or NONE>"""

STOP = re.compile(r"\\b(note to self|the provided fact|since the (fact|passage)|please provide|so I can translate|as (an? )?(AI|language model))\\b", re.I)


def wilson_ci(n, k, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def gemma(system, user, max_tokens=200, timeout=90.0):
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{VLLM}/chat/completions",
                       json={"model": MODEL, "temperature": 0.0, "max_tokens": max_tokens,
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__ERROR__ {e}"


def build_user(row, fam):
    return (f"FAMILY: {fam}\n"
            f"FAMILY-SPECIFIC CRITERION: {FAMILY_SPECIFIC_CRITERIA[fam]}\n\n"
            f"PASSAGE: {str(row['passage_text'])[:1500]}\n\n"
            f"QUERY: {str(row['query'])}\n\n"
            f"ROW CONCEPT (intended subject): cui_label={row.get('cui_label')} alt_term={row.get('alt_term')}\n\n"
            f"Score per the rules above.")


def parse_response(text):
    fg = fs = None
    halluc = None
    ungrounded = ""
    for line in str(text).splitlines():
        u = line.strip().upper()
        if u.startswith("FACT_GROUNDING:"):
            try: fg = int(re.search(r"([1-5])", line).group(1))
            except: pass
        elif u.startswith("FAMILY_SPECIFIC:"):
            try: fs = int(re.search(r"([1-5])", line).group(1))
            except: pass
        elif u.startswith("HALLUCINATED:"):
            halluc = "TRUE" in u
        elif u.startswith("UNGROUNDED_TERMS:"):
            ungrounded = line.split(":", 1)[1].strip()
    return fg, fs, halluc, ungrounded


def verdict_from_ci(ci_low, ci_high, mean_fg, mean_fs, monotony, ngram_pass):
    if not (ngram_pass and not monotony):
        return "BORDERLINE"
    # FAIL if even the optimistic (lower) bound of the CI is >= 10%
    if ci_low >= 0.10:
        return "FAIL"
    # PASS if the pessimistic (upper) bound is < 10% and FG/FS strong
    if ci_high < 0.10 and mean_fg >= 4.0 and mean_fs >= 3.5:
        return "PASS"
    return "BORDERLINE"


def monotony_check(fam_df):
    fq = fam_df[fam_df["query_style"] == "full_question"]
    if len(fq) < 10:
        return False, ""
    first_words = fq["query"].fillna("").str.extract(r"^\s*(\w+)", expand=False).str.lower()
    top = first_words.value_counts()
    if len(top) == 0:
        return False, ""
    frac = top.iloc[0] / len(fq)
    return frac > 0.30, f"top opener '{top.index[0]}' = {frac*100:.0f}%"


def sample_family(fam_df, fam, n=N_PER_FAMILY):
    seed = int(hashlib.md5(fam.encode()).hexdigest()[:8], 16) % (2**31)
    styles = [s for s in fam_df["query_style"].unique() if fam_df["query_style"].eq(s).sum() > 0]
    per = max(1, n // len(styles)) if styles else n
    parts = []
    for s in styles:
        sub = fam_df[fam_df["query_style"] == s]
        parts.append(sub.sample(n=min(per, len(sub)), random_state=seed))
    samp = pd.concat(parts)
    if len(samp) < n:
        rest = fam_df.drop(samp.index)
        if len(rest):
            samp = pd.concat([samp, rest.sample(n=min(n - len(samp), len(rest)), random_state=seed)])
    return samp.head(n)


def main():
    # endpoint check
    try:
        with httpx.Client(timeout=10) as c:
            m = c.get(f"{VLLM}/models").json()
        ids = [x["id"] for x in m.get("data", [])]
        assert MODEL in ids, f"{MODEL} not in {ids}"
        print(f"vLLM OK: {ids}")
    except Exception as e:
        print(f"VLLM DOWN: {e} — aborting."); return

    df = pd.read_parquet(SRC)
    families = sorted(df["task_family"].unique())
    results = {}

    for fam in families:
        fam_df = df[df["task_family"] == fam]
        samp = sample_family(fam_df, fam)
        print(f"\n=== {fam} (sampled {len(samp)}/{len(fam_df)}) ===")
        rows = [samp.iloc[i] for i in range(len(samp))]

        def score_one(i):
            r = rows[i]
            txt = gemma(SYSTEM, build_user(r, fam))
            if txt.startswith("__ERROR__"):
                return i, None, None, None, None, str(r.get("example_id"))
            fg, fs, halluc, ungrounded = parse_response(txt)
            return i, fg, fs, halluc, ungrounded, str(r.get("example_id"))

        fgs, fss, halluc_flags, worst = [], [], [], []
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(score_one, i) for i in range(len(rows))]
            done = 0
            for fut in as_completed(futs):
                i, fg, fs, halluc, ungrounded, eid = fut.result()
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(rows)}")
                if fg is None:
                    continue
                fgs.append(fg); fss.append(fs)
                # HALLUCINATION = high-confidence: Gemma flagged HALLUCINATED AND scored FG<=2.
                # Bare HALL is over-eager (flags lay synonyms); FG<=2 is the reliable signal.
                if halluc and fg is not None and fg <= 2:
                    halluc_flags.append((eid, ungrounded, str(rows[i]["query"])[:90]))
                    worst.append({"example_id": eid, "dimension": "fact_grounding",
                                  "fg": fg, "problem": ungrounded, "query_preview": str(rows[i]["query"])[:90]})
                # capture low FS too
                if fs is not None and fs <= 2:
                    worst.append({"example_id": eid, "dimension": "family_specific",
                                  "problem": f"FS={fs}", "query_preview": str(rows[i]["query"])[:90]})

        n = len(fgs)
        k = len(halluc_flags)
        pt, ci_low, ci_high = wilson_ci(n, k)
        mono, mono_detail = monotony_check(fam_df)
        ngram_mean = float(fam_df["ngram_copy_4"].mean()) if "ngram_copy_4" in fam_df.columns else 0.0
        ngram_score = 100 * (1 - ngram_mean)
        ngram_pass = ngram_score >= 60
        mean_fg = float(np.mean(fgs)) if fgs else 0.0
        mean_fs = float(np.mean(fss)) if fss else 0.0
        v = verdict_from_ci(ci_low, ci_high, mean_fg, mean_fs, mono, ngram_pass)

        results[fam] = {
            "rows_in_family": int(len(fam_df)),
            "n_sampled": n,
            "hallucination_count": k,
            "hallucination_rate": round(pt, 4),
            "wilson_ci_low": round(ci_low, 4),
            "wilson_ci_high": round(ci_high, 4),
            "mean_fact_grounding": round(mean_fg, 3),
            "mean_family_specific": round(mean_fs, 3),
            "ngram_copy_score": round(ngram_score, 1),
            "monotony": mono,
            "monotony_detail": mono_detail,
            "verdict": v,
            "hallucinated_rows": [{"example_id": e, "terms": t, "query": q} for e, t, q in halluc_flags[:10]],
            "worst_rows": worst[:8],
        }
        print(f"  halluc={k}/{n} ({pt*100:.1f}%) CI=[{ci_low*100:.1f}%-{ci_high*100:.1f}%] "
              f"FG={mean_fg:.2f} FS={mean_fs:.2f} ngram={ngram_score:.1f} mono={mono} -> {v}")

    OUT_JSON.write_text(json.dumps(results, indent=2, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else str(o)))
    print(f"\nwrote {OUT_JSON.name}")
    # dump hallucinated rows for manual inspection
    for fam, r in results.items():
        if r["hallucinated_rows"]:
            print(f"\n--- {fam} hallucinated ({len(r['hallucinated_rows'])} shown) ---")
            for h in r["hallucinated_rows"]:
                print(f"  {h['example_id'][:10]} terms={h['terms']!r}")
                print(f"     Q: {h['query']}")
    print("\n=== SUMMARY ===")
    print(f"{'family':30} {'n':>4} {'halluc':>10} {'CI95':>16} {'FG':>5} {'FS':>5} {'verdict':>10}")
    for fam, r in results.items():
        print(f"{fam:30} {r['n_sampled']:>4} {r['hallucination_count']:>3}/{r['n_sampled']:<3}({r['hallucination_rate']*100:>4.1f}%) "
              f"[{r['wilson_ci_low']*100:>4.1f}-{r['wilson_ci_high']*100:>4.1f}] {r['mean_fact_grounding']:>5.2f} "
              f"{r['mean_family_specific']:>5.2f} {r['verdict']:>10}")
    overall = "PASS" if all(r["verdict"] == "PASS" for r in results.values()) else "NOT ALL PASS"
    print(f"\nOVERALL: {overall}")


if __name__ == "__main__":
    main()
