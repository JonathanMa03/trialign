"""Execution/approval regressions, deliberately separate from clinical evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest

from trialign.evaluation.harness import fixture


def criterion(side, **fields):
    return fixture(
        "learning", side, {"source": "Age >= 18 years", "comparator": ">=", "value": 18, **fields}
    )


def test_dense_cache_is_parameter_and_input_specific(tmp_path):
    from trialign.models.runtime import EncoderOptions, RuntimeEncoder

    class Backend:
        calls = 0

        def encode(self, texts):
            self.calls += 1
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    backend = Backend()
    one = RuntimeEncoder(tmp_path, "mpnet", EncoderOptions(), backend=backend)
    one.encode(["A"])
    one.encode(["A"])
    assert backend.calls == 1 and one.metadata()["cache_hit"]
    one.encode(["B"])
    assert backend.calls == 2
    two = RuntimeEncoder(tmp_path, "mpnet", EncoderOptions(max_seq_length=128), backend=backend)
    two.encode(["A"])
    assert backend.calls == 3


def test_encoder_input_distinguishes_selected_bounds_and_excludes_annotation_leakage():
    from trialign.models.runtime import serialize_criterion

    source = "Exclude BMI <18 or >45 at screening"
    lower = criterion(
        "A",
        entity="BMI",
        source=source,
        parent_statement=source,
        comparator="<",
        value=18,
        reviewer_id="SECRET REVIEWER",
        label="SECRET LABEL",
    )
    upper = lower.model_copy(update={"comparator": ">", "value": 45})
    one, two = serialize_criterion(lower), serialize_criterion(upper)
    assert one != two and "comparator: <" in one and "value: 45" in two
    assert "SECRET" not in one and lower.trial_id not in one and lower.criterion_id not in one


def test_selected_backend_and_options_are_saved_with_run(tmp_path, monkeypatch):
    from trialign.models.runtime import RuntimeEncoder
    from trialign.runtime.service import ComparisonService

    root = Path(__file__).resolve().parents[1]

    class Backend:
        def encode(self, texts):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    service = ComparisonService(root, directory=tmp_path / "runs")
    monkeypatch.setattr(
        service.encoder_pool,
        "get",
        lambda name, options: RuntimeEncoder(tmp_path, name, options, backend=Backend()),
    )
    run = service.create(
        {
            "trial_a": "NCT01841736",
            "trial_b": "NCT04310007",
            "retrieval_model": "mpnet",
            "encoder_options": {"batch_size": 7, "max_seq_length": 128},
            "max_pairs": 1,
            "use_llm": False,
        },
        synchronous=True,
    )
    assert run["provider_settings"]["retrieval_model"] == "mpnet"
    assert run["encoder_metadata"]["options"]["batch_size"] == 7
    assert len(run["encoder_metadata"]["revision"]) == 40
    assert len(run["results"]) == 1


def test_source_scope_blocks_heading_and_joint_exclusion_before_any_model():
    from trialign.agents.orchestrator import AgentOrchestrator
    from trialign.tools.source_quality import assess_source

    heading = criterion(
        "A",
        source="Chemotherapy",
        parent_statement="Chemotherapy",
        entity="Chemotherapy",
        comparator=None,
        value=None,
    )
    other = criterion("B", source="No chemotherapy within 3 weeks", entity="Chemotherapy")
    result = AgentOrchestrator().compare(heading, other)
    assert result.disposition == "source_review" and result.should_escalate
    assert any("heading_only" in x for x in result.critic.issues)
    joint = criterion("A", criterion_type="exclusion", logical_operator="and", parent_atom_count=2)
    assert any("joint_exclusion" in x for x in assess_source(joint).blockers)


def test_explicit_governing_header_can_be_sent_to_context_review():
    from trialign.tools.source_quality import assess_source

    row = criterion(
        "A",
        source="Unstable angina",
        parent_statement="Unstable angina",
        comparator=None,
        value=None,
        governing_context="No arterial event within six months, including Unstable angina",
    )
    result = assess_source(row)
    assert not result.blockers and result.warnings


def test_learned_proposal_is_not_auto_accepted():
    from trialign.agents.orchestrator import AgentOrchestrator
    from trialign.schemas import RelationshipLabel

    class Predictor:
        artifact_digest = "a" * 64

        def predict_pair(self, a, b):
            labels = [x.value for x in RelationshipLabel]
            return {
                "proposed_label": "equivalent",
                "top_label": "equivalent",
                "relevance_score": 0.9,
                "class_scores": {v: float(v == "equivalent") for v in labels},
                "calibrated": False,
                "requires_review": True,
                "artifact_sha256": self.artifact_digest,
                "serializer": "criterion-v1",
                "supported_class": True,
                "reason": "Uncalibrated proposal",
            }

    a = criterion("A", temporal="at screening")
    b = criterion("B", temporal="at consent")
    result = AgentOrchestrator(relationship_model=Predictor()).compare(a, b)
    assert result.model_prediction["proposed_label"] == "equivalent"
    assert result.label is None and result.should_escalate
    assert result.route == "trained_model_review"


@pytest.fixture
def synthetic_workbook(tmp_path, monkeypatch):
    from openpyxl import Workbook

    import trialign.data.annotations as annotations

    a, b = criterion("A"), criterion("B", source="Age >= 21 years", value=21)
    monkeypatch.setattr(
        annotations,
        "load_partition_release",
        lambda root: {"assignments": {a.trial_id: "train", b.trial_id: "train"}},
    )
    monkeypatch.setattr(annotations, "load_parquet_corpus", lambda path: [a, b])
    packet = tmp_path / "pairs.jsonl"
    packet.write_text(
        json.dumps(
            {
                "pair_id": "fixture-pair",
                "a": a.model_dump(mode="json"),
                "b": b.model_dump(mode="json"),
            }
        )
        + "\n"
    )
    values = {
        "Pair ID": "fixture-pair",
        "A criterion ID": a.criterion_id,
        "B criterion ID": b.criterion_id,
        "A trial ID": a.trial_id,
        "B trial ID": b.trial_id,
        "A source": a.source,
        "B source": b.source,
        "Relevance": "direct_counterpart",
        "Relationship": "b_more_restrictive",
        "Status": "adjudicated",
        "Uncertainty": "low",
        "Clinical question": "Minimum age",
        "Relevance rationale": "Same selected age question.",
        "Relationship rationale": "Age 19 is admitted only by A.",
        "Common comparison frame": "Minimum age in years.",
        "A supporting quote": a.source,
        "B supporting quote": b.source,
        "Governing scope": "Independent inclusion rules.",
        "Source issues": "none",
        "Assumptions": "none",
        "Reviewer": "Reviewer One",
        "Reviewed at UTC": "2026-11-09T12:00:00Z",
        "Adjudicator": "Reviewer Three",
        "Second review required": "yes",
    }
    headers = list(values)
    wb = Workbook()
    for name, reviewer in [("Label pairs", "Reviewer One"), ("Second review", "Reviewer Two")]:
        ws = wb.active if name == "Label pairs" else wb.create_sheet()
        ws.title = name
        for _ in range(4):
            ws.append([None])
        ws.append(headers)
        ws.append([{**values, "Reviewer": reviewer}[h] for h in headers])
    path = tmp_path / "constructed-review.xlsx"
    wb.save(path)
    return path, packet, headers


def test_import_requires_real_human_review_and_exact_quotes(synthetic_workbook, tmp_path):
    from openpyxl import load_workbook

    from trialign.data.annotations import inspect_annotations

    path, packet, headers = synthetic_workbook
    report = inspect_annotations(tmp_path, path, packets=[packet])
    assert report["approved_rows"] == 1
    wb = load_workbook(path)
    ws = wb["Second review"]
    ws.cell(6, headers.index("Reviewer") + 1, "Codex draft 2")
    wb.save(path)
    report = inspect_annotations(tmp_path, path, packets=[packet])
    assert not report["records"] and report["pending"]
    wb["Second review"].cell(6, headers.index("Reviewer") + 1, "Reviewer Two")
    wb["Label pairs"].cell(6, headers.index("A supporting quote") + 1, "Invented source")
    wb.save(path)
    report = inspect_annotations(tmp_path, path, packets=[packet])
    assert not report["records"] and any("exact frozen" in str(p) for p in report["pending"])


def test_human_release_is_versioned_and_never_overwrites(synthetic_workbook, tmp_path):
    from trialign.data.annotations import release_annotations

    root = Path(__file__).resolve().parents[1]
    path, packet, _ = synthetic_workbook
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="approval"):
        release_annotations(root, path, output, packets=[packet])
    manifest = release_annotations(root, path, output, packets=[packet], approve_research_use=True)
    assert manifest["approved_for_research_training"] and manifest["counts"] == {"train": 1}
    record = json.loads((output / "train.jsonl").read_text())
    assert record["relevance_mask"] and record["relationship_mask"]
    with pytest.raises(ValueError, match="exists"):
        release_annotations(root, path, output, packets=[packet], approve_research_use=True)


def test_candidate_preparation_remains_cancellable(tmp_path, monkeypatch):
    import threading
    import time

    import trialign.runtime.service as runtime

    entered = threading.Event()
    release = threading.Event()

    def slow(a, b, **kwargs):
        entered.set()
        assert release.wait(3)
        return []

    monkeypatch.setattr(runtime, "alignment_candidates", slow)
    root = Path(__file__).resolve().parents[1]
    service = runtime.ComparisonService(root, directory=tmp_path / "runs")
    run = service.create({"trial_a": "NCT01841736", "trial_b": "NCT04310007", "use_llm": False})
    try:
        assert entered.wait(3)
        assert service.cancel(run["run_id"])["status"] == "cancelling"
    finally:
        release.set()
    for _ in range(100):
        result = service.get(run["run_id"])
        if result["status"] == "cancelled":
            break
        time.sleep(0.02)
    assert result["status"] == "cancelled" and result["model_calls"] == 0


def test_source_conflicts_and_unknowns_are_not_negative_targets(synthetic_workbook, tmp_path):
    from openpyxl import load_workbook

    from trialign.data.annotations import inspect_annotations

    path, packet, headers = synthetic_workbook
    wb = load_workbook(path)
    for ws in wb:
        ws.cell(6, headers.index("Relevance") + 1, "uncertain")
        ws.cell(6, headers.index("Source issues") + 1, "Unresolved source scope")
    wb.save(path)
    record = inspect_annotations(tmp_path, path, packets=[packet])["records"][0]
    assert record["relevance_grade"] is None
    assert not record["relevance_mask"] and not record["relationship_mask"]


def test_unknown_labels_contribute_zero_gradient():
    torch = pytest.importorskip("torch")
    from trialign.models.joint import LABELS, TrainingConfig
    from trialign.training import masked_loss

    score = torch.tensor([0.7], requires_grad=True)
    logits = torch.randn(1, len(LABELS), requires_grad=True)
    rows = [
        {
            "relevance_label": "uncertain",
            "relationship_label": "ambiguous",
            "relevance_mask": False,
            "relationship_mask": False,
        }
    ]
    loss = masked_loss(score, logits, rows, TrainingConfig())
    loss.backward()
    assert loss.item() == 0 and score.grad.item() == 0 and torch.count_nonzero(logits.grad) == 0


def test_encoder_forward_preserves_non_tensor_tokenizer_metadata():
    torch = pytest.importorskip("torch")
    from trialign.models.joint import JointModel, TrainingConfig

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 4)

        def get_sentence_embedding_dimension(self):
            return 4

        def tokenize(self, texts):
            return {"input_ids": torch.tensor([[1, 2] for _ in texts]), "task": "sentence"}

        def forward(self, features):
            assert features["task"] == "sentence"
            return {"sentence_embedding": self.embedding(features["input_ids"]).mean(1)}

    model = JointModel(TrainingConfig(hidden_size=8), encoder=Encoder())
    score, logits = model(["A"], ["B"])
    (score.sum() + logits.square().mean()).backward()
    assert model.encoder.embedding.weight.grad is not None
    assert model.head[1].weight.grad is not None


def test_training_checkpoint_reload_and_sealed_test(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from trialign.evaluation.learned import evaluate_model, lock_evaluation
    from trialign.models.joint import TrainedModel, TrainingConfig
    from trialign.training import synthetic_release, train_model

    release = tmp_path / "labels"
    output = tmp_path / "model"
    synthetic_release(release)
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path == release / "test.jsonl":
            raise AssertionError("Training opened sealed test labels")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "read_text", guarded)
        manifest = train_model(
            release,
            output,
            TrainingConfig(epochs=2, batch_size=4, hidden_size=16, max_seq_length=64),
            synthetic=True,
        )
    assert manifest["synthetic"] and len(manifest["history"]) >= 1
    trained = TrainedModel(output)
    assert np.isfinite(trained.encode(["Age >= 18"])).all()
    with pytest.raises(ValueError, match="lock"):
        evaluate_model(output, release, split="test", allow_synthetic=True)
    lock = tmp_path / "lock.json"
    lock_evaluation(output, release, lock)
    result = evaluate_model(output, release, split="test", test_lock=lock, allow_synthetic=True)
    assert result["synthetic"] and result["rows"] == 12
    with (output / "model.safetensors").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        TrainedModel(output)


def test_pooled_metrics_preserve_unknowns_and_common_denominator():
    from trialign.evaluation.learned import pooled_metrics

    rankings = {"one": {"q": ["a", "b", "c", "d", "e"]}, "two": {"q": ["a", "b", "c", "d", "f"]}}
    judgments = {
        tuple(sorted(("q", c))): v for c, v in zip("abcdef", [2, 0, 1, 2, None, 0], strict=True)
    }
    result = pooled_metrics(rankings, judgments)
    assert result["fully_judged_queries"] == 0
    assert result["models"]["one"]["precision@k"] is None
    assert result["models"]["one"]["precision_lower_bound"] == 0.4
    assert result["models"]["one"]["precision_upper_bound"] == 0.6
    assert result["models"]["two"]["precision_upper_bound"] == 0.4
    judgments[("e", "q")] = 2
    result = pooled_metrics(
        rankings,
        judgments,
        query_metadata={"q": {"cohort": "constructed", "split_group": "family1"}},
    )
    assert result["models"]["one"]["precision@k"] == 0.6
    assert result["models"]["two"]["precision@k"] == 0.4
    assert result["bootstrap"]["replicates"] == 2000


def test_draft_import_masks_disagreements_without_claiming_human_approval(synthetic_workbook):
    from openpyxl import load_workbook

    from trialign.data.annotations import inspect_annotations

    path, packet, headers = synthetic_workbook
    wb = load_workbook(path)
    for name in ["Label pairs", "Second review"]:
        wb[name].cell(6, headers.index("Reviewer") + 1, "Codex draft")
        wb[name].cell(6, headers.index("Adjudicator") + 1).value = None
    wb["Second review"].cell(6, headers.index("Relationship") + 1, "equivalent")
    wb.save(path)
    human = inspect_annotations(path.parent, path, packets=[packet])
    assert not human["records"]
    draft = inspect_annotations(path.parent, path, packets=[packet], draft_pilot=True)
    assert not draft["errors"] and not draft["pending"]
    row = draft["records"][0]
    assert row["relevance_mask"] and not row["relationship_mask"]
    assert row["draft_relationship_disagreement"] and row["human_approved"] is False


def test_draft_training_without_validation_is_explicit_fixed_epoch_and_test_stays_sealed(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    from trialign.data.annotations import load_label_split
    from trialign.data.partitions import sha
    from trialign.evaluation.learned import evaluate_model
    from trialign.models.joint import TrainingConfig
    from trialign.training import synthetic_release, train_model

    release, output = tmp_path / "draft", tmp_path / "model"
    synthetic_release(release)
    (release / "validation.jsonl").write_text("")
    manifest = json.loads((release / "manifest.json").read_text())
    manifest.update(schema_version="draft-label-pilot-1", draft_labels=True, synthetic=False)
    manifest["files"]["validation.jsonl"] = sha(release / "validation.jsonl")
    (release / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="allow-draft-labels"):
        load_label_split(release, "train")
    with pytest.raises(ValueError, match="test evaluation is disabled"):
        load_label_split(release, "test", allow_draft_labels=True)
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path == release / "test.jsonl":
            raise AssertionError("Draft training opened sealed test labels")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", guarded)
        # Tiny encoder only for this execution regression; CLI pilot uses real MPNet.
        result = train_model(
            release,
            output,
            TrainingConfig(epochs=2, batch_size=4, hidden_size=16),
            synthetic=True,
            allow_draft_labels=True,
        )
    assert result["draft_labels"] and len(result["history"]) == 2
    assert result["checkpoint_selection"] == "final_fixed_epoch_no_validation"
    assert all(h["validation_loss"] is None for h in result["history"])
    fit = evaluate_model(output, release, split="train", allow_draft_labels=True)
    assert fit["in_sample"] and fit["draft_labels"] and not fit["clinical_validation_claimed"]
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(output, release, allow_draft_labels=True)


def test_backend_pilot_launch_is_temporary_and_runtime_marks_drafts(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    pytest.importorskip("torch")
    from typer.testing import CliRunner

    import trialign.models.joint as joint
    import trialign.web.server as server
    from trialign.cli import app
    from trialign.models.runtime import (
        EncoderOptions,
        RuntimeEncoder,
        encoder_capabilities,
        runtime_artifact_permitted,
    )

    artifact = tmp_path / "model"
    artifact.mkdir()
    manifest = {"schema_version": "trialign-joint-1", "synthetic": False, "draft_labels": True}
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.delenv("TRIALIGN_ALLOW_DRAFT_RUNTIME", raising=False)
    monkeypatch.setenv("TRIALIGN_MODEL_PATH", "original-artifact-path")
    assert not runtime_artifact_permitted(manifest)
    monkeypatch.setattr(joint, "read_manifest", lambda path: manifest)
    monkeypatch.setattr(
        joint,
        "TrainedModel",
        lambda *a, **kw: SimpleNamespace(manifest=manifest, artifact_digest="a" * 64),
    )
    with pytest.raises(ValueError, match="pilot-model"):
        RuntimeEncoder(tmp_path, "trained", EncoderOptions())

    observed = []

    def running(root, **kwargs):
        assert os.environ["TRIALIGN_MODEL_PATH"] == str(artifact)
        assert runtime_artifact_permitted(manifest)
        caps = next(c for c in encoder_capabilities(root) if c["id"] == "trained")
        assert caps["available"] and caps["testing_only"] and "draft pilot" in caps["label"]
        encoder = RuntimeEncoder(root, "trained", EncoderOptions())
        assert encoder.artifact_digest == "a" * 64
        observed.append(True)
        raise RuntimeError("Constructed server failure tests cleanup")

    monkeypatch.setattr(server, "run_browser_app", running)
    result = CliRunner().invoke(app, ["serve", "--pilot-model", str(artifact)])
    assert isinstance(result.exception, RuntimeError) and observed == [True]
    assert os.environ["TRIALIGN_MODEL_PATH"] == "original-artifact-path"
    assert "TRIALIGN_ALLOW_DRAFT_RUNTIME" not in os.environ
    assert not runtime_artifact_permitted(manifest)
    assert not runtime_artifact_permitted({**manifest, "synthetic": True})


def test_pilot_command_rejects_missing_artifact_without_starting_server(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    import trialign.web.server as server
    from trialign.cli import app

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid artifact must not start server")

    monkeypatch.setattr(server, "run_browser_app", forbidden)
    result = CliRunner().invoke(app, ["serve", "--pilot-model", str(tmp_path / "missing")])
    assert result.exit_code == 2 and "Pilot artifact is unavailable or invalid" in result.output


def test_training_pause_resume_matches_uninterrupted_optimizer_and_rng(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    from trialign.models.joint import TrainingConfig
    from trialign.training import synthetic_release, train_model
    from trialign.training_control import TrainingRun, request_pause, training_directory

    release = tmp_path / "labels"
    synthetic_release(release)
    config = TrainingConfig(
        epochs=3,
        batch_size=3,
        gradient_accumulation=2,
        hidden_size=16,
        max_seq_length=64,
        dropout=0.25,
    )
    uninterrupted, interrupted = tmp_path / "full", tmp_path / "resumed"
    expected = train_model(release, uninterrupted, config, synthetic=True)
    original = TrainingRun.progress
    requested = []

    def pause_mid_accumulation(run, **fields):
        original(run, **fields)
        if fields.get("phase") == "training" and fields.get("batch") == 1 and not requested:
            requested.append(True)
            request_pause(interrupted)

    with monkeypatch.context() as context:
        context.setattr(TrainingRun, "progress", pause_mid_accumulation)
        paused = train_model(release, interrupted, config, synthetic=True)
    assert paused["status"] == "paused" and not interrupted.exists()
    workspace = training_directory(interrupted)
    checkpoint = torch.load(workspace / "checkpoint.pt", weights_only=True)
    assert checkpoint["next_batch"] == 2 and checkpoint["completed_updates"] == 1
    assert len(checkpoint["losses"]) == 2
    assert json.loads((workspace / "progress.json").read_text())["status"] == "paused"
    with pytest.raises(ValueError, match="--resume"):
        train_model(release, interrupted, config, synthetic=True)
    mismatched = config.model_copy(update={"head_learning_rate": 0.1})
    with pytest.raises(ValueError, match="identical"):
        train_model(release, interrupted, mismatched, synthetic=True, resume=True)
    actual = train_model(release, interrupted, config, synthetic=True, resume=True)
    assert actual["history"] == expected["history"]
    expected_weights = load_file(str(uninterrupted / "model.safetensors"))
    actual_weights = load_file(str(interrupted / "model.safetensors"))
    assert all(torch.equal(expected_weights[k], actual_weights[k]) for k in expected_weights)
    assert json.loads((workspace / "progress.json").read_text())["status"] == "completed"
    events = [json.loads(line) for line in (workspace / "events.jsonl").read_text().splitlines()]
    assert any(e.get("eta_seconds") is not None for e in events)
    assert any(e["phase"] == "validation" for e in events)
    assert (workspace / "training.log").stat().st_size > 0


def test_training_pause_signal_and_exclusive_worker_lock(tmp_path):
    import signal

    from trialign.training_control import TrainingRun

    original = signal.getsignal(signal.SIGINT)
    output = tmp_path / "model"
    with TrainingRun(output) as run:
        run.save({"elapsed_seconds": 0.0})
        run._pause_signal()
        assert run.should_pause
        with pytest.raises(ValueError, match="Another training process"):
            with TrainingRun(output, resume=True):
                pass
    assert signal.getsignal(signal.SIGINT) is original


def test_training_cli_defaults_and_fit_report_use_docs_logging(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    import trialign.evaluation.learned as learned
    import trialign.training as training
    from trialign.cli import app

    settings = Path(__file__).resolve().parents[1] / "configs/training.yaml"
    calls = []

    def train(release, output, config, **kwargs):
        calls.append((output, kwargs))
        return {"history": [], "synthetic": False, "draft_labels": True, "status": "paused"}

    monkeypatch.setattr(training, "train_model", train)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "train",
            "labels/pilot",
            "--training-config",
            str(settings),
            "--allow-draft-labels",
            "--resume",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0][0] == Path("docs/logging/pilot/model")
    assert calls[0][1]["resume"] and calls[0][1]["allow_draft_labels"]
    result = CliRunner().invoke(
        app,
        ["train", "labels/pilot", "docs/logging/custom/model", "--training-config", str(settings)],
    )
    assert result.exit_code == 0, result.output
    assert calls[-1][0] == Path("docs/logging/custom/model")
    monkeypatch.setattr(learned, "evaluate_model", lambda *a, **kw: {"rows": 3})
    result = CliRunner().invoke(
        app, ["evaluate-model", "docs/logging/pilot/model", "labels/pilot", "--split", "train"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(Path("docs/logging/pilot/fit.json").read_text()) == {"rows": 3}
    result = CliRunner().invoke(
        app,
        [
            "evaluate-model",
            "docs/logging/pilot/model",
            "labels/pilot",
            "docs/logging/custom/fit.json",
            "--split",
            "train",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(Path("docs/logging/custom/fit.json").read_text()) == {"rows": 3}


def test_annotation_import_uses_tracked_workbook_without_local_jsonl(synthetic_workbook, tmp_path):
    import shutil

    from openpyxl import load_workbook

    from trialign.data.annotations import inspect_annotations

    source, _, headers = synthetic_workbook
    registry = tmp_path / "data/processed/pair_pilot_v1/trial100_v1_pair_labeling.xlsx"
    registry.parent.mkdir(parents=True)
    shutil.copyfile(source, registry)
    report = inspect_annotations(tmp_path, source)
    assert report["approved_rows"] == 1 and not report["errors"]
    assert set(report["packet_hashes"]) == {str(registry)}
    wb = load_workbook(source)
    for name in ["Label pairs", "Second review"]:
        wb[name].cell(6, headers.index("Pair ID") + 1, "unregistered-pair")
    wb.save(source)
    report = inspect_annotations(tmp_path, source)
    assert report["errors"] and not report["records"]
