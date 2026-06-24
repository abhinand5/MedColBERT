"""Head-to-head comparison: Gemma pilot vs DeepSeek pilot negatives.

Reads negatives_<tag>.parquet + negatives_<tag>_dropped.parquet + run_summary.json for two
tags (default: gemma 'pilot' vs deepseek 'ds_pilot') and prints a side-by-side table on:
  wall-time, est cost, neg/pos, pre-audit false-neg rate, weak rate, drop reasons,
  by-source, by-failure_mode, audit-score distributions (grounding / relevant_looks).

Also writes a per-family stratified QA JSONL for the DeepSeek pilot (60/family, with the
positive passage joined from v3g) in the SAME format as the Gemma QA samples, so the same
5-subagent QA can run on it for an apples-to-apples false-negative comparison.

Usage:
  uv run python scripts/data/compare_pilots.py --gemma pilot --deepseek ds_pilot
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace/MedColBERT")
NEG = ROOT / "data/processed/private/synthetic/negatives"
V3G = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v3g.parquet"


def load(tag):
    s = json.loads((NEG / f"negatives_{tag}_run_summary.json").read_text())
    k = pd.read_parquet(NEG / f"negatives_{tag}.parquet")
    d = pd.read_parquet(NEG / f"negatives_{tag}_dropped.parquet") if (NEG / f"negatives_{tag}_dropped.parquet").exists() else pd.DataFrame()
    return s, k, d


def row(label, s, k, d):
    fn = int((d["drop_reason"] == "answers_query_true").sum()) if len(d) and "drop_reason" in d else 0
    fn_rate = fn / max(1, len(k) + fn)
    weak = float(k["weak"].mean()) if len(k) and "weak" in k else 0.0
    gr = float(k["audit_grounding"].mean()) if len(k) and "audit_grounding" in k else 0.0
    rl = float(k["audit_relevant_looks"].mean()) if len(k) and "audit_relevant_looks" in k else 0.0
    return {
        "tag": label, "backend": s.get("backend"), "model": s.get("model"),
        "wall_min": round(s.get("wall_seconds", 0) / 60, 1),
        "cost_usd": s.get("estimated_cost_usd", "local"),
        "n_kept": len(k), "n_dropped": len(d),
        "neg/pos": s.get("negatives_per_positive"),
        "pre_audit_fn_rate": round(fn_rate, 4),
        "fn_dropped": fn,
        "weak_rate": round(weak, 4),
        "mean_grounding": round(gr, 2), "mean_relevant": round(rl, 2),
        "tokens_in_M": round(s.get("tokens", {}).get("prompt", 0) / 1e6, 2) if "tokens" in s else "-",
        "tokens_out_M": round(s.get("tokens", {}).get("completion", 0) / 1e6, 2) if "tokens" in s else "-",
        "cache_hit_pct": round(100 * s.get("tokens", {}).get("prompt_cached", 0) / max(1, s.get("tokens", {}).get("prompt", 1)), 1) if "tokens" in s else "-",
    }


def build_qa_jsonl(tag, outdir, n_per_family=60, seed=20260621):
    """Stratified 60/family QA JSONL with positive passage joined — same format as Gemma QA."""
    k = pd.read_parquet(NEG / f"negatives_{tag}.parquet")
    pos_map = pd.read_parquet(V3G).set_index("example_id")["passage_text"].to_dict()
    outdir = Path(outdir); outdir.mkdir(exist_ok=True)
    total = 0
    for fam in sorted(k.task_family.unique()):
        kf = k[k.task_family == fam].copy()
        modes = kf.failure_mode.value_counts()
        parts = []
        for fm, c in modes.items():
            take = max(2, round(n_per_family * c / len(kf)))
            sub = kf[kf.failure_mode == fm]
            parts.append(sub.sample(n=min(take, len(sub)), random_state=seed))
        samp = pd.concat(parts).drop_duplicates("negative_id").head(n_per_family)
        recs = []
        for _, r in samp.iterrows():
            recs.append({"negative_id": r["negative_id"], "query": str(r["query"]),
                         "query_style": str(r["query_style"]), "task_family": fam,
                         "positive_passage": str(pos_map.get(r["query_id"], "")),
                         "negative_passage": str(r["negative_passage"]),
                         "failure_mode": str(r["failure_mode"]),
                         "negative_source": str(r["negative_source"]),
                         "why_wrong": str(r.get("why_wrong", "")),
                         "audit_relevant_looks": int(r["audit_relevant_looks"]) if pd.notna(r.get("audit_relevant_looks")) else None,
                         "audit_answers_query": bool(r["audit_answers_query"]),
                         "audit_grounding": int(r["audit_grounding"]) if pd.notna(r.get("audit_grounding")) else None})
        out = outdir / f"qa_{fam}.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in recs))
        total += len(recs)
        print(f"  {fam}: {len(recs)} -> {out}")
    print(f"TOTAL QA samples ({tag}): {total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", default="pilot")
    ap.add_argument("--deepseek", default="ds_pilot")
    ap.add_argument("--build-ds-qa", action="store_true", help="write the DeepSeek QA JSONL for subagent review")
    ap.add_argument("--qa-dir", default="/tmp/neg_qa_ds_pilot")
    args = ap.parse_args()

    sg, kg, dg = load(args.gemma)
    sd, kd, dd = load(args.deepseek)
    rg, rd = row(args.gemma, sg, kg, dg), row(args.deepseek, sd, kd, dd)

    cols = ["tag", "backend", "wall_min", "cost_usd", "n_kept", "neg/pos",
            "pre_audit_fn_rate", "fn_dropped", "weak_rate", "mean_grounding", "mean_relevant",
            "tokens_in_M", "tokens_out_M", "cache_hit_pct"]
    print("=== HEAD-TO-HEAD ===")
    print(pd.DataFrame([rg, rd])[cols].to_string(index=False))

    print("\n=== by source ===")
    print("GEMMA:    ", kg.negative_source.value_counts().to_dict() if len(kg) else {})
    print("DEEPSEEK: ", kd.negative_source.value_counts().to_dict() if len(kd) else {})

    print("\n=== drop reasons ===")
    print("GEMMA:    ", dg.drop_reason.value_counts().to_dict() if len(dg) and "drop_reason" in dg else {})
    print("DEEPSEEK: ", dd.drop_reason.value_counts().to_dict() if len(dd) and "drop_reason" in dd else {})

    print("\n=== by failure_mode ===")
    print("GEMMA:    ", kg.failure_mode.value_counts().to_dict() if len(kg) else {})
    print("DEEPSEEK: ", kd.failure_mode.value_counts().to_dict() if len(kd) else {})

    print("\n=== grounding distribution (kept) ===")
    print("GEMMA:    ", kg.audit_grounding.value_counts().sort_index().to_dict() if len(kg) else {})
    print("DEEPSEEK: ", kd.audit_grounding.value_counts().sort_index().to_dict() if len(kd) else {})

    # cost projection to full 60k
    if "estimated_cost_usd" in sd:
        scale = 59670 / max(1, sd.get("n_train_processed", 1))
        print(f"\n=== DEEPSEEK full-60k cost projection: ~${sd['estimated_cost_usd']*scale:.2f} "
              f"(={sd['estimated_cost_usd']:.4f}/2k x {scale:.1f}x) ===")

    if args.build_ds_qa:
        print(f"\n=== building DeepSeek QA JSONL ({args.qa_dir}) ===")
        build_qa_jsonl(args.deepseek, args.qa_dir)


if __name__ == "__main__":
    main()
