"""Score a local checkpoint on MTEB medical retrieval + R2MED.

    uv run --with mteb python scripts/eval/run_mteb_local.py \\
        --model runs/base_stage2/final --arch colbert \\
        --name fierysurf/MedColBERT-base --out runs/eval/mteb_local/colbert-base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medcolbert.eval.mteb_local import (  # noqa: E402
    MEDICAL_RETRIEVAL_TASKS,
    R2MED_TASKS,
    load_local_model,
    main_scores,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local checkpoint dir or Hub id.")
    p.add_argument("--arch", choices=["colbert", "dense"], required=True)
    p.add_argument("--name", required=True, help="Model name recorded in results.")
    p.add_argument("--out", required=True, help="Result cache directory.")
    p.add_argument("--tasks", default="", help="Comma-separated override.")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    import mteb

    args = parse_args()
    names = [t for t in args.tasks.split(",") if t] or (
        MEDICAL_RETRIEVAL_TASKS + R2MED_TASKS
    )
    tasks = mteb.get_tasks(tasks=names, languages=["eng"])
    model = load_local_model(args.model, args.arch, args.name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=mteb.ResultCache(cache_path=out),
        encode_kwargs={"batch_size": args.batch_size},
        raise_error=False,
    )
    scores = main_scores(results)
    (out / "summary.json").write_text(json.dumps(scores, indent=2))
    for task, score in scores.items():
        print(f"{task:40s} {score:.4f}")
    missing = sorted(set(names) - set(scores))
    if missing:
        print(f"[mteb] failed/missing tasks: {missing}")


if __name__ == "__main__":
    main()
