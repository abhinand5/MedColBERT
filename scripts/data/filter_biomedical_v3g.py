"""Apply the calibrated-Gemma hallucination filter to biomedical_to_clinical
(the one family still straddling 10% hallucination at n=120).

Same rule as filter_grounded_v3f.py: drop rows where Gemma returns
HALLUCINATED=TRUE AND fact_grounding <= 2 (high-confidence only).
Reads v3f, writes the FINAL v3g dataset.
"""
import re
import httpx
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/workspace/MedColBERT")
SRC = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v3f.parquet"
OUT = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v3g.parquet"
OUT_DROP = ROOT / "data/processed/private/synthetic/consolidated/cleaned_v3g_dropped_rows.parquet"
VLLM = "http://localhost:8000/v1"; MODEL = "gemma-4-31B"
FAMILY = "biomedical_to_clinical"

SYSTEM = """You are a medical-data auditor scoring synthetic (query, passage) training pairs for a retrieval model.
You will be strict about HALLUCINATION but you must NOT over-flag. Read carefully:

A keyword search query may legitimately use:
  - synonyms, paraphrase, or lay wording that is not verbatim in the passage
  - generic verbs and connectors (e.g. "spreads", "causes", "affects", "monitoring", "process", "results")
  - broader/categorical terms for what the passage discusses
These are NOT hallucinations.

Mark a query HALLUCINATED=true ONLY IF it asserts a SPECIFIC medical claim — a named disease,
drug, procedure, dosage, anatomical site, or clinical action — that is either:
  (a) CONTRADICTED by the passage, OR
  (b) a fabricated specific entity the passage does NOT discuss and is NOT a synonym/variant of
      any concept the passage actually covers.
If the term is a reasonable synonym/paraphrase/lay-term for something in the passage, it is GROUNDED.

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
            r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__ERROR__ {e}"

def build_user(row):
    return (f"PASSAGE: {str(row['passage_text'])[:1500]}\n\nQUERY: {str(row['query'])}\n\n"
            f"ROW CONCEPT: cui_label={row.get('cui_label')} alt_term={row.get('alt_term')}\n\nScore per the rules above.")

def parse(text):
    fg=None; halluc=None; ungrounded=""
    for line in str(text).splitlines():
        u=line.strip().upper()
        if u.startswith("FACT_GROUNDING:"):
            try: fg=int(re.search(r"([1-5])",line).group(1))
            except: pass
        elif u.startswith("HALLUCINATED:"): halluc="TRUE" in u
        elif u.startswith("UNGROUNDED_TERMS:"): ungrounded=line.split(":",1)[1].strip()
    return fg,halluc,ungrounded

def main():
    with httpx.Client(timeout=10) as c:
        ids=[x["id"] for x in c.get(f"{VLLM}/models").json().get("data",[])]
    assert MODEL in ids; print("vLLM OK")
    df=pd.read_parquet(SRC).reset_index(drop=True)
    fam=df[df["task_family"]==FAMILY].copy()
    print(f"filtering {FAMILY}: {len(fam):,} rows")
    rows=[fam.iloc[i] for i in range(len(fam))]
    def score(i): return i,*parse(gemma(build_user(rows[i])))
    drop_idx=[]; cnt=[0]
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs=[ex.submit(score,i) for i in range(len(rows))]
        for fut in as_completed(futs):
            i,fg,halluc,_=fut.result(); cnt[0]+=1
            if cnt[0]%500==0: print(f"  {cnt[0]}/{len(rows)} | drop {len(drop_idx)}")
            if fg is not None and halluc and fg<=2: drop_idx.append(i)
    print(f"dropping {len(drop_idx)} ({100*len(drop_idx)/len(fam):.1f}%)")
    drop_labels=fam.index[drop_idx].tolist()
    kept=df.drop(index=drop_labels).reset_index(drop=True)
    dropped=df.loc[drop_labels].copy(); dropped["drop_reason"]="gemma_hallucinated_fg_le2"
    kept.to_parquet(OUT,index=False); dropped.to_parquet(OUT_DROP,index=False)
    print(f"\nwrote {len(kept):,} -> {OUT.name}")
    print(pd.crosstab(kept["task_family"],kept["query_style"],margins=True).to_string())

if __name__=="__main__": main()
