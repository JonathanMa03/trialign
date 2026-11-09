"""Offline, masked multi-task training with sealed test data and versioned artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from trialign.data.annotations import load_label_split
from trialign.data.partitions import sha
from trialign.models.joint import LABELS, JointModel, device_name
from trialign.models.runtime import BASELINES, SERIALIZER, serialize_criterion
from trialign.schemas import CriterionRecord
from trialign.training_control import TrainingRun, training_directory


def batch_inputs(rows):
    return (
        [serialize_criterion(CriterionRecord.model_validate(r["a"])) for r in rows],
        [serialize_criterion(CriterionRecord.model_validate(r["b"])) for r in rows],
    )


def masked_loss(similarity, logits, rows, config):
    """Uncertain/source-blocked targets contribute zero loss, never negative labels."""
    device = logits.device
    positive = torch.tensor(
        [r["relevance_mask"] and r["relevance_label"] == "direct_counterpart" for r in rows],
        device=device,
    )
    negative = torch.tensor(
        [r["relevance_mask"] and r["relevance_label"] == "unrelated" for r in rows], device=device
    )
    related = torch.tensor(
        [r["relevance_mask"] and r["relevance_label"] == "related_topic" for r in rows],
        device=device,
    )
    weights = (
        positive.float() + negative.float() + related.float() * config.related_topic_negative_weight
    )
    terms = positive * F.relu(config.positive_margin - similarity).square()
    terms = (
        terms
        + (negative.float() + related.float() * config.related_topic_negative_weight)
        * F.relu(similarity - config.negative_margin).square()
    )
    relevance = terms.sum() / weights.sum().clamp_min(1)
    selected = torch.tensor(
        [r["relationship_mask"] and r["relationship_label"] != "ambiguous" for r in rows],
        device=device,
    )
    target = torch.tensor([LABELS.index(r["relationship_label"]) for r in rows], device=device)
    relationship = (
        F.cross_entropy(logits[selected], target[selected]) if selected.any() else logits.sum() * 0
    )
    return config.relevance_loss_weight * relevance + config.relationship_loss_weight * relationship


def train_model(
    release,
    output,
    config,
    *,
    root=None,
    synthetic=False,
    allow_draft_labels=False,
    resume=False,
    checkpoint_every_updates=10,
):
    output = Path(output)
    if output.exists():
        raise ValueError("Model directory exists; use a new experiment version.")
    if not resume and training_directory(output).exists():
        raise ValueError("Training workspace exists; use --resume or a new output directory.")
    with TrainingRun(
        output, resume=resume, checkpoint_every_updates=checkpoint_every_updates
    ) as run:
        run.progress(phase="validating data and configuration")
        result = _train_model(
            release,
            output,
            config,
            root=root,
            synthetic=synthetic,
            allow_draft_labels=allow_draft_labels,
            run=run,
        )
        run.progress(
            status=result.get("status", "completed"),
            phase="checkpoint saved" if result.get("status") == "paused" else "model exported",
            output=str(output),
        )
        return result


def _train_model(release, output, config, *, root, synthetic, allow_draft_labels, run):
    output, release = Path(output), Path(release)
    if output.exists():
        raise ValueError("Model directory exists; use a new experiment version.")
    train, manifest = load_label_split(
        release, "train", allow_synthetic=synthetic, allow_draft_labels=allow_draft_labels
    )
    validation, _ = load_label_split(
        release, "validation", allow_synthetic=synthetic, allow_draft_labels=allow_draft_labels
    )
    draft_fixed_epochs = bool(manifest.get("draft_labels") and not validation)
    if not train or (not validation and not draft_fixed_epochs):
        raise ValueError(
            "Approved training and validation pairs are required; no test fallback is permitted."
        )
    train_trials = {r[s]["trial_id"] for r in train for s in ["a", "b"]}
    validation_trials = {r[s]["trial_id"] for r in validation for s in ["a", "b"]}
    if train_trials & validation_trials:
        raise ValueError("Training and validation trials overlap.")
    if not synthetic:
        if root is None:
            raise ValueError(
                "Production training requires the repository's frozen partition release."
            )
        from trialign.data.partitions import load_partition_release

        partitions = load_partition_release(root)
        if manifest.get("corpus_sha256") != sha(
            Path(root) / "data/processed/trial100_v1.parquet"
        ) or manifest.get("partitions_sha256") != sha(
            Path(root) / "data/processed/partitions_v1/manifest.json"
        ):
            raise ValueError("Label corpus/partition fingerprints do not match this repository.")
        for split, rows in [("train", train), ("validation", validation)]:
            if any(
                partitions["assignments"].get(r[s]["trial_id"]) != split
                for r in rows
                for s in ["a", "b"]
            ):
                raise ValueError("Label records violate frozen family assignments.")

    def usable(row):
        relevance = row["relevance_mask"] and row["relevance_label"] in {
            "direct_counterpart",
            "unrelated",
        }
        relevance = relevance or (
            row["relevance_mask"]
            and row["relevance_label"] == "related_topic"
            and config.related_topic_negative_weight > 0
        )
        relationship = row["relationship_mask"] and row["relationship_label"] != "ambiguous"
        return (config.relevance_loss_weight > 0 and relevance) or (
            config.relationship_loss_weight > 0 and relationship
        )

    if not any(usable(r) for r in train) or (
        not draft_fixed_epochs and not any(usable(r) for r in validation)
    ):
        raise ValueError("Training and validation need usable targets for the enabled losses.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    device = device_name(config.device)
    run.progress(phase="loading encoder and relationship head")
    model = JointModel(
        config,
        synthetic=synthetic,
        cache_folder=Path(root) / "artifacts/model_cache/huggingface/hub" if root else None,
    ).to(device)
    truncation = {"train": 0, "validation": 0}
    if not synthetic:
        run.progress(phase="auditing source token lengths")
        for split, rows in [("train", train), ("validation", validation)]:
            texts = [t for row in rows for t in batch_inputs([row])[0] + batch_inputs([row])[1]]
            truncation[split] = sum(
                len(model.encoder.tokenizer(t, truncation=False)["input_ids"])
                > config.max_seq_length
                for t in texts
            )
        if any(truncation.values()) and config.truncation_policy == "error":
            raise ValueError(
                f"Source/context truncation detected {truncation}; review rendering/token limit "
                "or explicitly use allow_recorded."
            )
    if config.swap_augmentation:
        reverse = {
            "a_more_restrictive": "b_more_restrictive",
            "b_more_restrictive": "a_more_restrictive",
        }
        train = train + [
            {
                **r,
                "a": r["b"],
                "b": r["a"],
                "relationship_label": reverse.get(r["relationship_label"], r["relationship_label"]),
            }
            for r in train
        ]
    groups = [{"params": model.head.parameters(), "lr": config.head_learning_rate}]
    if not config.freeze_encoder:
        groups.append({"params": model.encoder.parameters(), "lr": config.encoder_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    updates = (
        math.ceil(math.ceil(len(train) / config.batch_size) / config.gradient_accumulation)
        * config.epochs
    )
    warmup = int(updates * config.warmup_fraction)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / max(1, warmup), max(0, (updates - step) / max(1, updates - warmup))
        ),
    )
    fingerprint = {
        "schema_version": "training-checkpoint-1",
        "config": config.model_dump(),
        "release_sha256": sha(release / "manifest.json"),
        "train_sha256": sha(release / "train.jsonl"),
        "validation_sha256": sha(release / "validation.jsonl"),
        "synthetic": synthetic,
        "device": device,
        "code": {
            name: sha(Path(__file__).parent / name)
            for name in [
                "training.py",
                "training_control.py",
                "models/joint.py",
                "models/runtime.py",
                "data/annotations.py",
            ]
        },
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "model_packages": {
            package: importlib.metadata.version(package)
            for package in ["sentence-transformers", "transformers", "safetensors"]
        },
    }
    rng, best, stale, history, best_state = random.Random(config.seed), float("inf"), 0, [], None
    start_epoch, next_batch, order_ids, losses, completed_updates = 0, 0, None, [], 0

    def save_checkpoint(epoch, batch, order):
        numpy_state = np.random.get_state()
        run.save(
            {
                "fingerprint": fingerprint,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": schedule.state_dict(),
                "best_state": best_state,
                "best": best,
                "stale": stale,
                "history": history,
                "epoch": epoch,
                "next_batch": batch,
                "order_ids": order,
                "losses": losses,
                "completed_updates": completed_updates,
                "shuffle_rng": rng.getstate(),
                "python_rng": random.getstate(),
                "numpy_rng": [
                    numpy_state[0],
                    numpy_state[1].tolist(),
                    numpy_state[2],
                    numpy_state[3],
                    numpy_state[4],
                ],
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
            }
        )

    if run.resume:
        run.progress(phase="restoring checkpoint")
        checkpoint = run.load()
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError(
                "Resume requires identical labels, configuration, code, device "
                "and recorded library versions; use a new run for changes."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        schedule.load_state_dict(checkpoint["schedule"])
        best_state, best = checkpoint["best_state"], checkpoint["best"]
        stale, history = checkpoint["stale"], checkpoint["history"]
        start_epoch, next_batch = checkpoint["epoch"], checkpoint["next_batch"]
        order_ids, losses = checkpoint["order_ids"], checkpoint["losses"]
        completed_updates = checkpoint["completed_updates"]
        rng.setstate(checkpoint["shuffle_rng"])
        random.setstate(checkpoint["python_rng"])
        n = checkpoint["numpy_rng"]
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(checkpoint["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if device == "mps":
            torch.mps.set_rng_state(checkpoint["mps_rng"])
        del checkpoint
    else:
        save_checkpoint(0, 0, None)

    def paused():
        return {
            "status": "paused",
            "history": history,
            "synthetic": synthetic,
            "draft_labels": manifest.get("draft_labels", False),
            "training_directory": str(run.directory),
        }

    if run.should_pause:
        save_checkpoint(start_epoch, next_batch, order_ids)
        return paused()
    resumed_work = completed_updates
    compute_started = time.perf_counter()
    for epoch in range(start_epoch, start_epoch if stale >= config.patience else config.epochs):
        if order_ids is None:
            order_ids = list(range(len(train)))
            rng.shuffle(order_ids)
            next_batch, losses = 0, []
        order = [train[i] for i in order_ids]
        model.train()
        optimizer.zero_grad()
        batches = [
            order[i : i + config.batch_size] for i in range(0, len(order), config.batch_size)
        ]
        for i in range(next_batch, len(batches)):
            rows = batches[i]
            a, b = batch_inputs(rows)
            score, logits = model(a, b)
            loss = masked_loss(score, logits, rows, config)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; no artifact released.")
            group_size = min(
                config.gradient_accumulation,
                len(batches) - (i // config.gradient_accumulation) * config.gradient_accumulation,
            )
            (loss / group_size).backward()
            losses.append(float(loss.detach()))
            boundary = (i + 1) % config.gradient_accumulation == 0 or i + 1 == len(batches)
            if boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                completed_updates += 1
            measured_updates = completed_updates - resumed_work
            eta = (
                (
                    (time.perf_counter() - compute_started)
                    / measured_updates
                    * (updates - completed_updates)
                )
                if measured_updates
                else None
            )
            run.progress(
                phase="training",
                epoch=epoch + 1,
                epochs=config.epochs,
                batch=i + 1,
                batches=len(batches),
                loss=losses[-1],
                optimizer_updates=completed_updates,
                total_optimizer_updates=updates,
                eta_seconds=eta,
            )
            if boundary and (run.should_pause or completed_updates % run.interval == 0):
                save_checkpoint(epoch, i + 1, order_ids)
                if run.should_pause:
                    return paused()
        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(validation), config.batch_size):
                rows = validation[i : i + config.batch_size]
                a, b = batch_inputs(rows)
                score, logits = model(a, b)
                val_sum += float(masked_loss(score, logits, rows, config)) * len(rows)
                val_count += len(rows)
                run.progress(
                    phase="validation",
                    epoch=epoch + 1,
                    epochs=config.epochs,
                    batch=i // config.batch_size + 1,
                    batches=math.ceil(len(validation) / config.batch_size),
                )
        value = val_sum / val_count if val_count else None
        if value is not None and not math.isfinite(value):
            raise ValueError("Non-finite validation loss.")
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": value}
        )
        run.progress(
            phase="epoch completed",
            epoch=epoch + 1,
            epochs=config.epochs,
            loss=history[-1]["train_loss"],
            validation_loss=value,
        )
        if draft_fixed_epochs or value < best:
            best, stale = value, 0
            best_state = {
                k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
        order_ids, losses, next_batch = None, [], 0
        save_checkpoint(epoch + 1, 0, None)
        if run.should_pause:
            return paused()
        if stale >= config.patience:
            break
    run.progress(phase="exporting selected model")
    model.load_state_dict(best_state)
    model.to(device).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "model"
        staged.mkdir()
        if synthetic:
            save_file(
                {k: v.cpu().contiguous() for k, v in model.state_dict().items()},
                str(staged / "model.safetensors"),
            )
        else:
            model.encoder.save_pretrained(
                str(staged / "encoder"), safe_serialization=True, create_model_card=False
            )
            save_file(
                {k: v.cpu().contiguous() for k, v in model.head.state_dict().items()},
                str(staged / "head.safetensors"),
            )
        (staged / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        packages = {}
        for package in ["torch", "sentence-transformers", "transformers", "numpy"]:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        metadata = {
            "schema_version": "trialign-joint-1",
            "serializer": SERIALIZER,
            "synthetic": synthetic,
            "draft_labels": manifest.get("draft_labels", False),
            "checkpoint_selection": "final_fixed_epoch_no_validation"
            if draft_fixed_epochs
            else "minimum_validation_loss",
            "label_provenance": "unvalidated_workbook_draft"
            if manifest.get("draft_labels")
            else "synthetic"
            if synthetic
            else "human_reviewed",
            "labels": LABELS,
            "training_config": config.model_dump(),
            "base_revision": BASELINES[config.base_model][1] if not synthetic else None,
            "label_release_sha256": sha(release / "manifest.json"),
            "corpus_sha256": manifest.get("corpus_sha256"),
            "trained_relationship_labels": sorted(
                {
                    r["relationship_label"]
                    for r in train
                    if config.relationship_loss_weight > 0
                    and r["relationship_mask"]
                    and r["relationship_label"] != "ambiguous"
                }
            ),
            "runtime_seconds": run.elapsed,
            "training_directory": str(run.directory),
            "effective_device": device,
            "history": history,
            "calibrated": False,
            "clinical_validation_claimed": False,
            "truncated_input_occurrences": truncation,
            "packages": packages,
            "training_source_sha256": {
                name: sha(Path(__file__).parent / name)
                for name in [
                    "training.py",
                    "training_control.py",
                    "models/joint.py",
                    "models/runtime.py",
                    "data/annotations.py",
                    "evaluation/learned.py",
                ]
            },
            "files": {str(p.relative_to(staged)): sha(p) for p in staged.rglob("*") if p.is_file()},
        }
        (staged / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        staged.rename(output)
    return metadata


def synthetic_release(output):
    """Constructed execution fixtures with disjoint trial IDs; not a clinical dataset."""
    from trialign.evaluation.harness import fixture

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in ["train", "validation", "test"]:
        rows = []
        for i in range(12):
            a = fixture(
                split + str(i), "A", {"trial_id": split + "-A", "source": "Age >= 18 years"}
            )
            unrelated = i % 3 == 0
            b = fixture(
                split + str(i),
                "B",
                {
                    "trial_id": split + "-B",
                    "source": "Platelets >= 100000 per microliter"
                    if unrelated
                    else "Age >= 21 years",
                    "entity": "Platelet Count" if unrelated else "Age",
                    "domain": "laboratory" if unrelated else "person",
                    "unit": "cells/microliter" if unrelated else "years",
                },
            )
            rows.append(
                {
                    "pair_id": split + str(i),
                    "partition": split,
                    "a": a.model_dump(mode="json"),
                    "b": b.model_dump(mode="json"),
                    "relevance_label": "unrelated" if unrelated else "direct_counterpart",
                    "relationship_label": "unrelated" if unrelated else "b_more_restrictive",
                    "relevance_mask": True,
                    "relationship_mask": True,
                }
            )
        path = output / (split + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[path.name] = sha(path)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "human-label-release-1",
                "synthetic": True,
                "approved_for_research_training": False,
                "purpose": "constructed execution checks only",
                "files": files,
            },
            indent=2,
        )
    )
