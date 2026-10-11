"""Command-line entry points for data preparation and baseline experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from trialign.config import load_settings
from trialign.data import load_corpus, load_parquet_corpus, write_corpus
from trialign.data.splits import assert_no_group_leakage, assign_group_splits
from trialign.models import TfidfEncoder
from trialign.pairing import generate_candidate_pairs
from trialign.schemas import AgentJudgment, CriterionRecord, HumanDecision, PairCandidate


app = typer.Typer(no_args_is_help=True, help="TriAlign data and experiment commands.")
console = Console()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


@app.command("export-schemas")
def export_schemas(
    output_dir: Path = typer.Option(Path("artifacts/schemas"), "--output-dir"),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Export the versioned JSON contracts used at pipeline boundaries."""
    settings = load_settings(config)
    destination = settings.resolve(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    models = {
        "criterion": CriterionRecord,
        "pair_candidate": PairCandidate,
        "agent_judgment": AgentJudgment,
        "human_decision": HumanDecision,
    }
    for name, model in models.items():
        _write_json(destination / f"{name}.schema.json", model.model_json_schema())
    console.print(f"[green]Wrote {len(models)} schemas to {destination}[/green]")


@app.command("inspect")
def inspect_corpus(
    input_path: Path = typer.Option(None, "--input", help="Modern AutoCrit review workbook."),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Validate the source workbook and print review readiness."""
    settings = load_settings(config)
    source = input_path or settings.resolve(settings.data.source_workbook)
    _, report = load_corpus(source, sheet=settings.data.review_sheet)
    table = Table(title="TriAlign corpus inspection")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in (
        ("Rows", report.rows_read),
        ("Schema-valid rows", report.valid_rows),
        ("Invalid rows", report.invalid_rows),
        ("Trials", report.trials),
        ("Workbook SHA-256", report.source_sha256),
    ):
        table.add_row(key, str(value))
    console.print(table)
    console.print("Review statuses:", report.review_statuses)
    if report.errors:
        console.print(f"[red]First validation errors:[/red] {report.errors[:5]}")


@app.command("freeze")
def freeze_corpus(
    input_path: Path = typer.Option(None, "--input"),
    output: Path = typer.Option(Path("data/processed/criteria_reviewed.parquet"), "--output"),
    config: Path = typer.Option(None, "--config"),
    allow_partial: bool = typer.Option(False, help="Freeze accepted/corrected rows before all review ends."),
) -> None:
    """Create the versioned model-ready corpus after human review."""
    settings = load_settings(config)
    source = input_path or settings.resolve(settings.data.source_workbook)
    all_records, all_report = load_corpus(
        source, sheet=settings.data.review_sheet, fail_on_invalid=True
    )
    reviewed, _ = load_corpus(
        source,
        sheet=settings.data.review_sheet,
        reviewed_only=True,
        reviewed_statuses=settings.data.reviewed_statuses,
        fail_on_invalid=True,
    )
    if not allow_partial and len(reviewed) != len(all_records):
        raise typer.BadParameter(
            f"Review is incomplete: {len(reviewed)} of {len(all_records)} rows are accepted/corrected."
        )
    destination = output if output.is_absolute() else settings.resolve(output)
    write_corpus(
        reviewed,
        destination,
        metadata={
            "source_workbook": str(source),
            "source_sha256": all_report.source_sha256,
            "review_statuses": settings.data.reviewed_statuses,
            "partial": len(reviewed) != len(all_records),
        },
    )
    console.print(f"[green]Wrote {len(reviewed)} reviewed criteria to {destination}[/green]")


@app.command("split")
def split_corpus(
    corpus: Path = typer.Option(Path("data/processed/criteria_reviewed.parquet")),
    output: Path = typer.Option(Path("data/processed/trial_splits.json")),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Create trial-level train/validation/test assignments."""
    settings = load_settings(config)
    records = load_parquet_corpus(settings.resolve(corpus))
    assignments = assign_group_splits(
        records,
        seed=settings.seed,
        train=settings.splits.train,
        validation=settings.splits.validation,
    )
    assert_no_group_leakage(assignments)
    destination = settings.resolve(output)
    _write_json(destination, {"seed": settings.seed, "assignments": assignments})
    counts = pd.Series(assignments).value_counts().to_dict()
    console.print(f"[green]Wrote trial splits to {destination}[/green] {counts}")


@app.command("pairs")
def build_pairs(
    corpus: Path = typer.Option(Path("data/processed/criteria_reviewed.parquet")),
    splits: Path = typer.Option(Path("data/processed/trial_splits.json")),
    output: Path = typer.Option(Path("data/processed/pair_candidates.parquet")),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Generate random controls and same-entity candidate pairs."""
    settings = load_settings(config)
    records = load_parquet_corpus(settings.resolve(corpus))
    split_payload = json.loads(settings.resolve(splits).read_text(encoding="utf-8"))
    pairs = generate_candidate_pairs(
        records,
        seed=settings.seed,
        random_pairs=int(settings.pairing.get("random_pairs", 1000)),
        same_entity_pairs=int(settings.pairing.get("same_entity_pairs", 2000)),
        split_by_trial=split_payload["assignments"],
    )
    destination = settings.resolve(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([pair.model_dump(mode="json") for pair in pairs]).to_parquet(
        destination, index=False
    )
    console.print(f"[green]Wrote {len(pairs)} pair candidates to {destination}[/green]")


@app.command("baseline-tfidf")
def baseline_tfidf(
    corpus: Path = typer.Option(Path("data/processed/criteria_reviewed.parquet")),
    output_dir: Path = typer.Option(Path("artifacts/baselines/tfidf")),
    top_k: int = typer.Option(20, min=1),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Fit the lexical reference baseline and save cross-trial neighbors."""
    settings = load_settings(config)
    records = load_parquet_corpus(settings.resolve(corpus))
    texts = [row.comparison_text or row.source for row in records]
    encoder = TfidfEncoder().fit(texts)
    embeddings = encoder.encode(texts)
    similarities = embeddings @ embeddings.T
    neighbors = []
    for index, row in enumerate(records):
        order = np.argsort(-similarities[index])
        rank = 0
        for candidate_index in order:
            candidate = records[int(candidate_index)]
            if candidate.criterion_id == row.criterion_id or candidate.trial_id == row.trial_id:
                continue
            rank += 1
            neighbors.append({
                "query_id": row.criterion_id,
                "candidate_id": candidate.criterion_id,
                "rank": rank,
                "score": float(similarities[index, candidate_index]),
                "model": encoder.name,
            })
            if rank >= top_k:
                break
    destination = settings.resolve(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(neighbors).to_parquet(destination / "neighbors.parquet", index=False)
    np.save(destination / "embeddings.npy", embeddings)
    encoder.save(str(destination / "vectorizer.joblib"))
    _write_json(destination / "run.json", {
        "model": encoder.name,
        "criteria": len(records),
        "top_k": top_k,
        "seed": settings.seed,
    })
    console.print(f"[green]Wrote TF-IDF baseline artifacts to {destination}[/green]")

