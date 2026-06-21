"""Final assembly: calibrated-Gemma hallucination filter on generic_clinical_retrieval
+ brand_generic downsample, producing the final v3f dataset.

The Round-2 UMLS+Gemma filter over-fired (87.6% drop) because its judge was over-strict.
This filter uses the CALIBRATED judge and drops ONLY high-confidence hallucinations:
Gemma returns HALLUCINATED=TRUE AND fact_grounding <= 2. (Bare HALLUCINATED over-flags
lay synonyms; FG<=2 is the reliable discriminator, verified on a held-out sample.)

Then re-audit at n=120/family so Wilson CIs are tight enough for clean families to PASS.
"""
import json
import re
import httpx
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/workspace/MedColBERT")
SRC = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v3.parquet"
OUT = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v3f.parquet"
OUT_DROP = ROOT / "data/processed/private/synthetic/consolidated/cleaned_v3f_dropped_rows.parquet"
VLLM = "http://localhost:8000/v1"
MODEL = "gemma-4-31B"
BRAND_GENERIC_FINAL = 1_500
RANDOM_STATE = 20260621

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
Respond in EXACTLY this format (one line each, no prose before/after):
FACT_GROUNDING: <1-5>
HALLUCINATED: <TRUE or FALSE>
UNGROUNDED_TERMS: <comma list of any fabricated/contradicted specific terms, or NONE>"""


def gemma(user, max_tokens=150, timeout=90.0):
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{VLLM}/chat/completions",
                       json={"model": MODEL, "temperature": 0.0, "max_tokens": max_tokens,
                             "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__ERROR__ {e}"


def build_user(row):
    return (f"PASSAGE: {str(row['passage_text'])[:1500]}\n\n"
            f"QUERY: {str(row['query'])}\n\n"
            f"ROW CONCEPT: cui_label={row.get('cui_label')} alt_term={row.get('alt_term')}\n\n"
            f"Score per the rules above.")


def parse(text):
    fg = None; halluc = None; ungrounded = ""
    for line in str(text).splitlines():
        u = line.strip().upper()
        if u.startswith("FACT_GROUNDING:"):
            try: fg = int(re.search(r"([1-5])", line).group(1))
            except: pass
        elif u.startswith("HALLUCINATED:"):
            halluc = "TRUE" in u
        elif u.startswith("UNGROUNDED_TERMS:"):
            ungrounded = line.split(":", 1)[1].strip()
    return fg, halluc, ungrounded


def main():
    # endpoint check
    with httpx.Client(timeout=10) as c:
        ids = [x["id"] for x in c.get(f"{VLLM}/models").json().get("data", [])]
    assert MODEL in ids, f"{MODEL} not up"; print(f"vLLM OK")

    df = pd.read_parquet(SRC).reset_index(drop=True)
    print(f"loaded v3: {len(df):,} rows")

    gcr_mask = df["task_family"] == "generic_clinical_retrieval"
    gcr = df[gcr_mask].copy()
    print(f"\n=== filtering generic_clinical_retrieval: {len(gcr):,} rows ===")
    rows = [gcr.iloc[i] for i in range(len(gcr))]

    def score(i):
        fg, halluc, ungrounded = parse(gemma(build_user(rows[i])))
        return i, fg, halluc, ungrounded

    drop_idx = []
    counter = [0]
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = [ex.submit(score, i) for i in range(len(rows))]
        for fut in as_completed(futs):
            i, fg, halluc, ungrounded = fut.result()
            counter[0] += 1
            if counter[0] % 500 == 0:
                print(f"  {counter[0]}/{len(rows)} | dropping {len(drop_idx)}")
            if fg is not None and halluc and fg <= 2:
                drop_idx.append(i)

    print(f"  dropping {len(drop_idx)} high-confidence hallucinations ({100*len(drop_idx)/len(gcr):.1f}%)")
    drop_labels = gcr.index[drop_idx].tolist()

    kept = df.drop(index=drop_labels).copy()
    dropped = df.loc[drop_labels].copy()
    dropped["drop_reason"] = "gemma_hallucinated_fg_le2"

    # brand_generic downsample 3k -> 1.5k
    print(f"\n=== downsample brand_generic -> {BRAND_GENERIC_FINAL:,} ===")
    bg_idx = kept.index[kept["task_family"] == "brand_generic"]
    print(f"  before: {len(bg_idx):,}")
    if len(bg_idx) > BRAND_GENERIC_FINAL:
        rng = np.random.default_rng(RANDOM_STATE)
        keep = set(rng.choice(bg_idx.to_numpy(), size=BRAND_GENERIC_FINAL, replace=False).tolist())
        drop_bg = bg_idx[~bg_idx.isin(keep)]
        bg_rows = kept.loc[drop_bg].copy()
        bg_rows["drop_reason"] = "downsample_brand_generic_final"
        dropped = pd.concat([dropped, bg_rows])
        kept = kept.drop(index=drop_bg)
        print(f"  kept {BRAND_GENERIC_FINAL:,}, dropped {len(drop_bg):,}")

    kept = kept.reset_index(drop=True)
    dropped = dropped.reset_index(drop=True)
    kept.to_parquet(OUT, index=False)
    dropped.to_parquet(OUT_DROP, index=False)
    print(f"\nwrote {len(kept):,} -> {OUT.name}")
    print(f"wrote {len(dropped):,} dropped -> {OUT_DROP.name}")
    print("\n=== v3f composition ===")
    print(pd.crosstab(kept["task_family"], kept["query_style"], margins=True).to_string())
    print("\n=== drop reasons ===")
    print(dropped["drop_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
