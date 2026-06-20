"""MedColBERT CLI — thin Typer entrypoint.

CLI commands delegate to library functions; they do not contain core logic.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="medcolbert",
    help="MedColBERT — ontology-controlled biomedical retrieval.",
    no_args_is_help=True,
)

console = Console()


# ── Sub-apps (registered lazily to avoid heavy imports at CLI load) ──────────

umls_app = typer.Typer(help="UMLS vocabulary operations.", no_args_is_help=True)
app.add_typer(umls_app, name="umls")

data_app = typer.Typer(help="Data processing commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")

synth_app = typer.Typer(help="Synthetic data generation.", no_args_is_help=True)
app.add_typer(synth_app, name="synth")

eval_app = typer.Typer(help="Evaluation commands.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")

train_app = typer.Typer(help="Training commands.", no_args_is_help=True)
app.add_typer(train_app, name="train")


# ── UMLS commands ─────────────────────────────────────────────────────────────


@umls_app.command()
def extract_pairs(
    config_path: str = typer.Option(
        "configs/data.yaml", "--config", "-c", help="Path to data config YAML."
    ),
) -> None:
    """Parse UMLS RRF files and build the ontology control layer.

    Writes private parquet files to data/processed/private/ontology/
    and public-safe manifests to data/processed/public/manifests/.
    """
    from pathlib import Path

    from medcolbert.utils.config import load_yaml
    from medcolbert.data.ontology import build_ontology, run_ontology_gate

    config = load_yaml(config_path)
    cfg_path = Path(config_path)

    umls_cfg = config.get("umls", {})
    umls_dir = Path(umls_cfg.get("local_dir", "data/raw/umls"))

    mrconso = umls_dir / "MRCONSO.RRF"
    mrsty = umls_dir / "MRSTY.RRF"
    mrrel = umls_dir / "MRREL.RRF"

    missing = [f.name for f in [mrconso, mrsty, mrrel] if not f.exists()]
    if missing:
        console.print(f"[red]Missing required UMLS files: {missing}[/red]")
        console.print(f"[yellow]Expected in: {umls_dir}[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Building ontology from UMLS in {umls_dir}[/bold]\n")

    summary = build_ontology(mrconso, mrsty, mrrel, config)

    console.print("\n[bold cyan]Build Summary[/bold cyan]")
    console.print(f"  CUIs: {summary['unique_cuis']:,}")
    console.print(f"  Concept strings: {summary['total_concept_strings']:,}")
    console.print(f"  Concept pairs: {summary['total_pairs']:,}")
    console.print(f"  Edges: {summary['total_edges']:,}")

    passed = run_ontology_gate(summary)

    if passed:
        console.print("\n[bold green]Phase 2 gate PASSED — ready for Phase 3.[/bold green]")
    else:
        console.print("\n[bold red]Phase 2 gate FAILED — review logs above.[/bold red]")
        raise typer.Exit(code=1)


# ── Data commands ─────────────────────────────────────────────────────────────


@data_app.command()
def build_passages(
    config_path: str = typer.Option(
        "configs/data.yaml", "--config", "-c", help="Path to data config YAML."
    ),
) -> None:
    """Ingest public corpora and build the passage store.

    Chunks long documents, computes fingerprints, and writes
    passages.parquet + fingerprints.parquet to the private data directory.
    """
    from pathlib import Path

    from medcolbert.utils.config import load_yaml
    from medcolbert.data.passages import build_passage_store

    config = load_yaml(config_path)

    console.print("[bold]Building passage store...[/bold]\n")
    summary = build_passage_store(config=config)

    console.print(f"\n[bold cyan]Passage Store Summary[/bold cyan]")
    console.print(f"  Total passages: {summary['total_passages']:,}")
    console.print(f"  Total documents: {summary['total_documents']:,}")
    for source, stats in summary.get("coverage", {}).items():
        console.print(f"  {source}: {stats['num_passages']:,} passages from {stats['num_source_docs']:,} docs")

    console.print(f"\n[bold green]Passage store built.[/bold green]")


@data_app.command()
def annotate_passages(
    concepts_path: str = typer.Option(
        ..., "--concepts", help="Path to concept_strings.parquet from Phase 2."
    ),
    passages_path: str = typer.Option(
        ..., "--passages", help="Path to passages.parquet."
    ),
    output: str = typer.Option(
        ..., "--output", help="Path for output passage_cui_matches.parquet."
    ),
) -> None:
    """Annotate passages with UMLS CUI matches (private output)."""
    from pathlib import Path

    import pandas as pd

    from medcolbert.data.passages import annotate_passages_with_cuis

    concepts_df = pd.read_parquet(concepts_path)
    console.print(f"Loaded {len(concepts_df):,} concept strings")

    df_annotations = annotate_passages_with_cuis(
        Path(passages_path),
        concepts_df,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_annotations.to_parquet(output_path, index=False)

    console.print(f"[green]Annotated {len(df_annotations):,} CUI matches → {output_path}[/green]")


# ── Generation commands ───────────────────────────────────────────────────────


@synth_app.command()
def generate(
    config_path: str = typer.Option(
        "configs/generation.yaml", "--config", "-c", help="Path to generation config."
    ),
    mode: str = typer.Option(
        "generic_synthetic", "--mode", "-m", help="Generation mode."
    ),
    limit: int = typer.Option(
        5, "--limit", "-n", help="Number of examples to generate."
    ),
    batch_id: str = typer.Option(
        "0", "--batch-id", help="Batch identifier."
    ),
    output_dir: str = typer.Option(
        "data/processed/private/synthetic", "--output-dir", "-o", help="Output directory."
    ),
) -> None:
    """Generate synthetic query-passage pairs.

    Generation modes (in order):
      generic_synthetic, passage_grounded, ontology_grounded,
      ontology_grounded_teacher_filtered
    """
    from pathlib import Path

    console.print(f"[bold]Generating {limit} examples in mode '{mode}'...[/bold]")
    console.print("[yellow]Generation pipeline not yet implemented — see Phase 4.[/yellow]")


@synth_app.command()
def judge(
    candidates_path: str = typer.Option(
        ..., "--candidates", help="Path to candidates.parquet."
    ),
    output: str = typer.Option(
        ..., "--output", help="Path for judged output."
    ),
) -> None:
    """Judge candidate query-passage pairs across all 5 quality dimensions."""
    console.print("[yellow]Judging pipeline not yet implemented — see Phase 4.[/yellow]")


# ── Doctor command ───────────────────────────────────────────────────────────


@app.command()
def doctor(
    check_models: bool = typer.Option(
        False, "--models", help="Run model-load smoke tests (may trigger downloads)."
    ),
) -> None:
    """Diagnose the MedColBERT environment and report readiness."""
    errors: list[str] = []
    warnings: list[str] = []
    ok: list[str] = []

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if (3, 11) <= sys.version_info[:2] <= (3, 12):
        ok.append(f"Python {py_ver} (supported)")
    else:
        errors.append(f"Python {py_ver} — need 3.11 or 3.12")

    # Package version
    try:
        pkg_ver = importlib.metadata.version("medcolbert")
        ok.append(f"medcolbert {pkg_ver} installed")
    except importlib.metadata.PackageNotFoundError:
        warnings.append("medcolbert package not installed (editable install may be needed)")

    # CUDA
    try:
        import torch

        if torch.cuda.is_available():
            ok.append(f"CUDA available: {torch.cuda.get_device_name(0) or 'GPU'}")
        else:
            warnings.append("CUDA not available — running on CPU")
    except ImportError:
        warnings.append("torch not installed — CUDA check skipped")

    # Data paths
    from medcolbert.utils.io import private_dir, public_dir

    priv = private_dir()
    pub = public_dir()
    if priv.exists():
        ok.append(f"Private dir: {priv}")
    else:
        warnings.append(f"Private dir missing: {priv}")

    if pub.exists():
        ok.append(f"Public dir: {pub}")
    else:
        warnings.append(f"Public dir missing: {pub}")

    # UMLS discoverability
    from medcolbert.utils.config import load_yaml

    data_cfg_path = Path("configs/data.yaml")
    if data_cfg_path.exists():
        cfg = load_yaml(data_cfg_path)
        umls_dir = Path(cfg.get("umls", {}).get("local_dir", "data/raw/umls"))
        if umls_dir.exists():
            ok.append(f"UMLS dir found: {umls_dir}")
        else:
            warnings.append(f"UMLS dir not found: {umls_dir} (download required)")
    else:
        warnings.append("configs/data.yaml not found")

    # vLLM endpoint
    import httpx

    try:
        r = httpx.get("http://localhost:8000/health", timeout=5)
        if r.is_success:
            ok.append("vLLM endpoint reachable at http://localhost:8000")
        else:
            warnings.append(f"vLLM health check returned {r.status_code}")
    except Exception:
        warnings.append("vLLM endpoint not reachable at http://localhost:8000")

    # Optional model smoke tests
    if check_models:
        _smoke_test_models(ok, warnings, errors)

    # Report
    console.print("\n[bold cyan]MedColBERT Doctor Report[/bold cyan]\n")

    console.print(f"[green]{len(ok)} checks passed[/green]")
    for msg in ok:
        console.print(f"  ✅ {msg}")

    if warnings:
        console.print(f"\n[yellow]{len(warnings)} warnings[/yellow]")
        for msg in warnings:
            console.print(f"  ⚠️  {msg}")

    if errors:
        console.print(f"\n[red]{len(errors)} errors[/red]")
        for msg in errors:
            console.print(f"  ❌ {msg}")
        raise typer.Exit(code=1)

    console.print("\n[bold green]Environment looks good![/bold green]")


def _smoke_test_models(ok: list[str], warnings: list[str], errors: list[str]) -> None:
    """Run optional model-load smoke tests."""
    try:
        from transformers import AutoModel, AutoTokenizer
        from medcolbert.utils.config import load_yaml

        base_cfg = load_yaml("configs/base.yaml")
        backbone = base_cfg.get("model", "thomas-sounack/BioClinical-ModernBERT-base")

        tokenizer = AutoTokenizer.from_pretrained(backbone)
        model = AutoModel.from_pretrained(backbone)
        ok.append(f"BioClinical-ModernBERT loaded: {backbone}")
        del model, tokenizer
    except Exception as e:
        warnings.append(f"Model smoke test skipped: {e}")

    try:
        from pylate import ColBERT

        ok.append("PyLate ColBERT importable")
    except ImportError:
        warnings.append("PyLate not importable (training extras may be incomplete)")


# ── CLI entrypoint ───────────────────────────────────────────────────────────


def main() -> None:
    app()


if __name__ == "__main__":
    main()
