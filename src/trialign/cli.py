"""Command-line entry points for data preparation and baseline experiments."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from scipy.sparse import save_npz

from trialign.config import load_settings
from trialign.data import load_corpus, load_parquet_corpus, write_corpus
from trialign.data.partitions import load_partition_release, sha
from trialign.models import TfidfEncoder
from trialign.models.retrieval import retrieve_neighbors
from trialign.pairing import generate_candidate_pairs
from trialign.schemas import AgentJudgment, CriterionRecord, HumanDecision, PairCandidate

app = typer.Typer(no_args_is_help=True, help="TriAlign data and experiment commands.")
console = Console()


@app.command("annotations-check")
def annotations_check(
    workbook: Path,
    output: Path | None = None,
    packet: list[Path] | None = typer.Option(None, "--packet"),
):
    """Read-only audit of source, identities, human approval and review completeness."""
    from trialign.config import ROOT
    from trialign.data.annotations import inspect_annotations

    packets = (
        [
            *[
                p
                for p in [
                    ROOT / "data/processed/pair_pilot_v1/pairs.jsonl",
                    ROOT / "data/processed/pair_pilot_v1/training_review_v2.jsonl",
                ]
                if p.is_file()
            ],
            *packet,
        ]
        if packet
        else None
    )
    report = inspect_annotations(ROOT, workbook, packets=packets)
    report.pop("records")
    if output:
        _write_json(output, report)
    console.print(
        {
            "rows": report["rows"],
            "approved": report["approved_rows"],
            "errors": len(report["errors"]),
            "pending": len(report["pending"]),
            "excluded": len(report["excluded"]),
        }
    )
    if report["errors"]:
        raise typer.Exit(1)


@app.command("annotations-release")
def annotations_release(
    workbook: Path,
    output: Path,
    approve_research_use: bool = False,
    allow_partial: bool = False,
    packet: list[Path] | None = typer.Option(None, "--packet"),
):
    """Freeze human-approved labels; never promotes assistant proposals."""
    from trialign.config import ROOT
    from trialign.data.annotations import release_annotations

    try:
        report = release_annotations(
            ROOT,
            workbook,
            output,
            approve_research_use=approve_research_use,
            allow_partial=allow_partial,
            packets=(
                [
                    *[
                        p
                        for p in [
                            ROOT / "data/processed/pair_pilot_v1/pairs.jsonl",
                            ROOT / "data/processed/pair_pilot_v1/training_review_v2.jsonl",
                        ]
                        if p.is_file()
                    ],
                    *packet,
                ]
                if packet
                else None
            ),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(report["counts"])


@app.command("train")
def train(
    release: Path,
    output: Annotated[Path | None, typer.Argument()] = None,
    training_config: Path = Path("configs/training.yaml"),
    allow_draft_labels: bool = False,
    resume: bool = False,
    checkpoint_every_updates: int = typer.Option(10, min=1, max=10000),
):
    """Train encoder/head from approved labels or an explicitly opted-in draft pilot."""
    import yaml

    from trialign.config import ROOT
    from trialign.models.joint import TrainingConfig
    from trialign.training import train_model

    output = output or Path("docs/logging") / release.name / "model"
    try:
        config = TrainingConfig.model_validate(yaml.safe_load(training_config.read_text()))
        report = train_model(
            release,
            output,
            config,
            root=ROOT,
            allow_draft_labels=allow_draft_labels,
            resume=resume,
            checkpoint_every_updates=checkpoint_every_updates,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        {
            "output": str(output),
            "epochs": len(report["history"]),
            "synthetic": report["synthetic"],
            "draft_labels": report["draft_labels"],
            "status": report.get("status", "completed"),
        }
    )


@app.command("train-pause")
def train_pause(output: Path):
    """Request a checkpoint and graceful exit at the next optimizer-update boundary."""
    from trialign.training_control import request_pause

    try:
        console.print(request_pause(output))
    except (OSError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("annotations-pilot")
def annotations_pilot(workbook: Path, output: Path):
    """Snapshot all eligible workbook drafts for provisional, family-separated training."""
    from trialign.config import ROOT
    from trialign.data.annotations import release_draft_pilot

    try:
        report = release_draft_pilot(ROOT, workbook, output)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print({k: report[k] for k in ["counts", "usable_targets", "excluded_rows"]})


@app.command("train-smoke")
def train_smoke(
    output: Annotated[Path, typer.Argument()] = Path("docs/logging/learning_smoke_v1"),
):
    """Constructed tiny-encoder training/checkpoint/evaluation check; no clinical claims."""
    from trialign.evaluation.learned import evaluate_model
    from trialign.models.joint import TrainingConfig
    from trialign.training import synthetic_release, train_model

    if output.exists():
        raise typer.BadParameter("Smoke directory exists; use a new output directory.")
    synthetic_release(output / "labels")
    config = TrainingConfig(
        epochs=2,
        batch_size=4,
        hidden_size=16,
        max_seq_length=64,
        encoder_learning_rate=0.01,
        head_learning_rate=0.01,
        deterministic=True,
    )
    train_model(output / "labels", output / "model", config, synthetic=True)
    report = evaluate_model(output / "model", output / "labels", allow_synthetic=True)
    _write_json(output / "execution_check.json", report)
    console.print({"synthetic": True, "rows": report["rows"], "output": str(output)})


@app.command("lock-model-evaluation")
def lock_model_evaluation(artifact: Path, release: Path, output: Path):
    """Lock model, label release and evaluator fingerprints before opening test labels."""
    from trialign.evaluation.learned import lock_evaluation

    console.print(lock_evaluation(artifact, release, output))


@app.command("evaluate-model")
def evaluate_model_command(
    artifact: Path,
    release: Path,
    output: Annotated[Path | None, typer.Argument()] = None,
    split: str = "validation",
    test_lock: Path | None = None,
    allow_draft_labels: bool = False,
):
    """Evaluate relationship predictions and pair discrimination separately."""
    from trialign.evaluation.learned import evaluate_model

    output = output or Path("docs/logging") / (
        artifact.parent.name if artifact.name == "model" else artifact.name
    ) / ("fit.json" if split == "train" else split + ".json")
    try:
        report = evaluate_model(
            artifact,
            release,
            split=split,
            test_lock=test_lock,
            allow_draft_labels=allow_draft_labels,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _write_json(output, report)
    console.print({"output": str(output), **report})


@app.command("evaluate-retrieval")
def evaluate_retrieval_command(
    rankings: Path, release: Path, queries: Path, output: Path, split: str = "validation"
):
    """Pooled P@5/bounds/nDCG for declared query IDs, not a full-corpus recall claim.

    Rankings JSON: {model: {query_criterion_id: [candidate_criterion_id, ...]}}.
    Queries JSON: a preselected list of query criterion IDs.
    """
    from trialign.config import ROOT
    from trialign.data.annotations import load_label_split
    from trialign.data.partitions import load_partition_release
    from trialign.evaluation.learned import pooled_metrics

    if split != "validation":
        raise typer.BadParameter(
            "This pooled command is validation-only; test retrieval needs a separately "
            "frozen full-system evaluation configuration."
        )
    rows, _ = load_label_split(release, split)
    frozen = load_partition_release(ROOT)
    corpus = {
        r.criterion_id: r for r in load_parquet_corpus(ROOT / "data/processed/trial100_v1.parquet")
    }
    selected = json.loads(queries.read_text())
    provided = json.loads(rankings.read_text())
    inventory = json.loads(
        (ROOT / "data/processed/family_audit_v1/trial_inventory.json").read_text()
    )
    if not selected or len(selected) != len(set(selected)):
        raise typer.BadParameter("Declare distinct preselected query IDs.")
    for model, mapping in provided.items():
        if not set(selected) <= set(mapping):
            raise typer.BadParameter(f"Missing declared queries in {model}.")
        for q in selected:
            for identifier in [q, *mapping[q]]:
                if (
                    identifier not in corpus
                    or frozen["assignments"][corpus[identifier].trial_id] != split
                ):
                    raise typer.BadParameter(
                        "Ranking uses an unknown or cross-partition criterion."
                    )
            if any(corpus[c].trial_id == corpus[q].trial_id for c in mapping[q]):
                raise typer.BadParameter("Ranking contains a same-trial candidate.")
    judgments = {
        tuple(sorted((r["a"]["criterion_id"], r["b"]["criterion_id"]))): r["relevance_grade"]
        for r in rows
    }
    groups = json.loads((ROOT / "data/processed/family_audit_v1/family_mapping.json").read_text())[
        "split_group_by_trial"
    ]
    metadata = {
        q: {
            "cohort": inventory[corpus[q].trial_id]["cohort"],
            "split_group": groups[corpus[q].trial_id],
        }
        for q in selected
    }
    report = pooled_metrics(
        {m: {q: v[q] for q in selected} for m, v in provided.items()},
        judgments,
        query_metadata=metadata,
    )
    report.update(
        query_file_sha256=sha(queries),
        rankings_sha256=sha(rankings),
        label_release_sha256=sha(release / "manifest.json"),
    )
    _write_json(output, report)
    console.print(report)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


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
    allow_partial: bool = typer.Option(
        False, help="Freeze accepted/corrected rows before all review ends."
    ),
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
            f"Review is incomplete: {len(reviewed)} of {len(all_records)} "
            "rows are accepted/corrected."
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
    corpus: Path = typer.Option(Path("data/processed/trial100_v1.parquet")),
    output: Path = typer.Option(Path("artifacts/splits/trial_splits.json")),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Export the frozen family-aware assignments; never reshuffle trials."""
    settings = load_settings(config)
    release = load_partition_release(settings.resolve(Path(".")))
    if sha(settings.resolve(corpus)) != release["corpus_sha256"]:
        raise typer.BadParameter(
            "This partition release applies only to the frozen trial100_v1 corpus."
        )
    _write_json(
        settings.resolve(output),
        {
            "partition_release_id": release["partition_release_id"],
            "seed": release["seed"],
            "assignments": release["assignments"],
            "corpus_sha256": release["corpus_sha256"],
        },
    )
    console.print(release["counts"])


@app.command("pairs")
def build_pairs(
    corpus: Path = typer.Option(Path("data/processed/trial100_v1.parquet")),
    splits: Path = typer.Option(Path("data/processed/partitions_v1/partitions.json")),
    output: Path = typer.Option(Path("data/processed/pair_candidates.parquet")),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Generate training-only random/same-entity controls under frozen partitions."""
    settings = load_settings(config)
    release = load_partition_release(settings.resolve(Path(".")))
    if sha(settings.resolve(corpus)) != release["corpus_sha256"]:
        raise typer.BadParameter(
            "This partition release applies only to the frozen trial100_v1 corpus."
        )
    supplied = json.loads(settings.resolve(splits).read_text())
    if supplied.get("assignments") != release["assignments"]:
        raise typer.BadParameter(
            "Supplied assignments differ from the frozen family-aware release."
        )
    records = [
        r
        for r in load_parquet_corpus(settings.resolve(corpus))
        if release["assignments"][r.trial_id] == "train"
    ]
    destination = settings.resolve(output)
    if destination.exists():
        raise typer.BadParameter("Pair output already exists; choose a new output version.")
    pairs = generate_candidate_pairs(
        records,
        seed=release["seed"],
        random_pairs=int(settings.pairing.get("random_pairs", 1000)),
        same_entity_pairs=int(settings.pairing.get("same_entity_pairs", 2000)),
        split_by_trial=release["assignments"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([pair.model_dump(mode="json") for pair in pairs]).to_parquet(
        destination, index=False
    )
    console.print(
        f"[green]Wrote {len(pairs)} training-only control candidates to {destination}[/green]"
    )


@app.command("baseline-tfidf")
def baseline_tfidf(
    corpus: Path = typer.Option(Path("data/processed/criteria_reviewed.parquet")),
    output_dir: Path = typer.Option(Path("artifacts/baselines/tfidf")),
    top_k: int = typer.Option(20, min=1),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Fit the lexical reference baseline and save cross-trial neighbors."""
    started = perf_counter()
    settings = load_settings(config)
    corpus_path = settings.resolve(corpus)
    records = load_parquet_corpus(corpus_path)

    if not records:
        raise typer.BadParameter(
            "The corpus is empty. Supply a corpus containing criterion records.",
            param_hint="--corpus",
        )

    texts = [row.comparison_text or row.source for row in records]
    encoder = TfidfEncoder().fit(texts)
    embeddings = encoder.encode(texts)

    neighbors = retrieve_neighbors(
        records,
        embeddings,
        top_k=top_k,
        model=encoder.name,
    )

    destination = settings.resolve(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    neighbor_columns = [
        "query_id",
        "candidate_id",
        "rank",
        "score",
        "model",
        "model_revision",
        "text_field",
    ]
    pd.DataFrame(
        neighbors,
        columns=neighbor_columns,
    ).to_parquet(destination / "neighbors.parquet", index=False)

    save_npz(destination / "embeddings.npz", embeddings)
    encoder.save(str(destination / "vectorizer.joblib"))

    manifest_path = corpus_path.with_suffix(corpus_path.suffix + ".manifest.json")
    corpus_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )

    _write_json(
        destination / "run.json",
        {
            "model": encoder.name,
            "model_revision": None,
            "corpus": str(corpus_path),
            "text_field": "comparison_text",
            "criteria": len(records),
            "trials": len({row.trial_id for row in records}),
            "top_k": top_k,
            "seed": settings.seed,
            "neighbors": len(neighbors),
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "embedding_format": embeddings.format,
            "runtime_seconds": perf_counter() - started,
            "development_only": corpus_manifest.get("development_only"),
            "human_review_complete": corpus_manifest.get("human_review_complete"),
            "source_sha256": corpus_manifest.get("source_sha256"),
        },
    )

    console.print(f"[green]Wrote TF-IDF baseline artifacts to {destination}[/green]")


@app.command("baseline-dense")
def baseline_dense(
    corpus: Path = typer.Option(Path("data/processed/criteria_reviewed.parquet")),
    model_name: str = typer.Option("sentence-transformers/all-mpnet-base-v2", "--model-name"),
    output_dir: Path = typer.Option(Path("artifacts/baselines/dense")),
    text_field: str = typer.Option("comparison_text", "--text-field"),
    batch_size: int = typer.Option(32, min=1),
    device: str = typer.Option("auto", "--device"),
    top_k: int = typer.Option(20, min=1),
    revision: str = typer.Option(None, "--revision"),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Encode criteria with a dense model and save cross-trial neighbors."""
    import numpy as np

    from trialign.models.dense import DenseEncoder

    started = perf_counter()

    if text_field not in {"comparison_text", "source"}:
        raise typer.BadParameter(
            "Choose comparison_text or source.",
            param_hint="--text-field",
        )

    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise typer.BadParameter(
            "Choose auto, cpu, cuda, or mps.",
            param_hint="--device",
        )

    settings = load_settings(config)
    corpus_path = settings.resolve(corpus)
    records = load_parquet_corpus(corpus_path)

    if not records:
        raise typer.BadParameter(
            "The corpus is empty.",
            param_hint="--corpus",
        )

    ids = [record.criterion_id for record in records]
    if len(ids) != len(set(ids)):
        raise typer.BadParameter(
            "The corpus contains duplicate criterion IDs.",
            param_hint="--corpus",
        )

    # Explicit CPU needs no hardware probe. Keep torch optional until model loading.
    selected_device = device
    if device != "cpu":
        import torch

        if device == "auto":
            if torch.cuda.is_available():
                selected_device = "cuda"
            elif torch.backends.mps.is_available():
                selected_device = "mps"
            else:
                selected_device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise typer.BadParameter("CUDA is unavailable. Use cpu or auto.", param_hint="--device")
        elif device == "mps" and not torch.backends.mps.is_available():
            raise typer.BadParameter(
                "Apple MPS is unavailable. Use cpu or auto.", param_hint="--device"
            )

    manifest_path = corpus_path.with_suffix(corpus_path.suffix + ".manifest.json")
    corpus_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )

    texts = [
        (record.comparison_text or record.source)
        if text_field == "comparison_text"
        else record.source
        for record in records
    ]

    console.print(
        f"Encoding {len(records)} criteria with {model_name} "
        f"on {selected_device}. First use may download model weights."
    )

    encoder = DenseEncoder(
        model_name,
        device=selected_device,
        revision=revision,
        batch_size=batch_size,
    )
    embeddings = encoder.encode(texts)

    neighbors = retrieve_neighbors(
        records,
        embeddings,
        top_k=top_k,
        model=model_name,
        model_revision=revision,
        text_field=text_field,
    )

    destination = settings.resolve(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    neighbor_columns = [
        "query_id",
        "candidate_id",
        "rank",
        "score",
        "model",
        "model_revision",
        "text_field",
    ]
    pd.DataFrame(
        neighbors,
        columns=neighbor_columns,
    ).to_parquet(destination / "neighbors.parquet", index=False)

    np.save(destination / "embeddings.npy", embeddings)

    _write_json(
        destination / "run.json",
        {
            "model": model_name,
            "model_revision": revision,
            "revision_status": (
                "user_supplied; not independently resolved"
                if revision
                else "unpinned; resolved revision not yet recorded"
            ),
            "corpus": str(corpus_path),
            "text_field": text_field,
            "criteria": len(records),
            "trials": len({row.trial_id for row in records}),
            "top_k": top_k,
            "seed": settings.seed,
            "batch_size": batch_size,
            "device": selected_device,
            "neighbors": len(neighbors),
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "embedding_format": "dense",
            "normalized": True,
            "runtime_seconds": perf_counter() - started,
            "development_only": corpus_manifest.get("development_only"),
            "human_review_complete": corpus_manifest.get("human_review_complete"),
            "source_sha256": corpus_manifest.get("source_sha256"),
        },
    )

    console.print(f"[green]Wrote dense baseline artifacts to {destination}[/green]")


@app.command("serve")
def serve(
    port: int = typer.Option(8766, min=1024, max=65535),
    open_browser: bool = typer.Option(False, "--open-browser"),
    runtime_directory: Path = typer.Option(None, "--runtime-directory"),
    pilot_model: Path | None = typer.Option(None, "--pilot-model"),
) -> None:
    """Start the local trial-comparison UI and agent runtime."""
    from trialign.config import ROOT
    from trialign.web.server import run_browser_app

    if pilot_model is None:
        run_browser_app(
            ROOT, port=port, open_browser=open_browser, runtime_directory=runtime_directory
        )
        return

    import os

    from trialign.models.joint import read_manifest

    artifact = pilot_model.resolve()
    try:
        manifest = read_manifest(artifact)
        if manifest.get("synthetic") is not False:
            raise ValueError("Pilot runtime requires a real trained encoder, not a smoke fixture.")
    except (OSError, ValueError, KeyError) as exc:
        raise typer.BadParameter(f"Pilot artifact is unavailable or invalid: {exc}") from exc
    overrides = {
        "TRIALIGN_MODEL_PATH": str(artifact),
        "TRIALIGN_ALLOW_DRAFT_RUNTIME": "1",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        console.print(
            "[yellow]Pilot model test mode: unvalidated draft artifacts enabled "
            "for this server process only.[/yellow]"
        )
        run_browser_app(
            ROOT, port=port, open_browser=open_browser, runtime_directory=runtime_directory
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@app.command("compare-trials")
def compare_trials(
    trial_a: str = typer.Option(..., "--trial-a"),
    trial_b: str = typer.Option(..., "--trial-b"),
    max_pairs: int = typer.Option(16, min=1, max=80),
    model_calls: int = typer.Option(200, min=0),
    unlimited_model_calls: bool = typer.Option(False, "--unlimited-model-calls"),
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """Run the same persisted workflow used by the local browser UI."""
    from trialign.config import ROOT
    from trialign.runtime.service import ComparisonService

    service = ComparisonService(ROOT)
    job = service.create(
        {
            "trial_a": trial_a,
            "trial_b": trial_b,
            "max_pairs": max_pairs,
            "max_model_calls": None if unlimited_model_calls else model_calls,
            "use_llm": not offline,
        },
        synchronous=True,
    )
    console.print({"run_id": job["run_id"], "status": job["status"], **job["summary"]})


@app.command("harness")
def diagnostic_harness() -> None:
    """Run offline routing, polarity, endpoint and abstention diagnostics."""
    from trialign.config import ROOT
    from trialign.evaluation.harness import run_harness

    result = run_harness(ROOT, ROOT / "artifacts/demo/harness.json")
    console.print({"passed": result["passed"], "cases": result["cases"]})
    if result["passed"] != result["cases"]:
        console.print([r["id"] for r in result["results"] if not r["passed"]])
        raise typer.Exit(1)
