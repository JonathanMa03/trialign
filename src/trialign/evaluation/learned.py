"""Separate relationship, reviewed-pair discrimination, and pooled ranking metrics."""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from trialign.data.partitions import sha


def pair_metrics(model, rows, *, batch_size=16, progress=None):
    import torch
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

    from trialign.models.joint import LABELS
    from trialign.training import batch_inputs

    similarities, truth, predicted, swapped, relevance_truth = [], [], [], [], []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            part = rows[i : i + batch_size]
            a, b = batch_inputs(part)
            score, logits = model(a, b)
            _, reverse = model(b, a)
            for row, s, y, rev in zip(
                part,
                score.cpu().tolist(),
                logits.argmax(1).cpu().tolist(),
                reverse.argmax(1).cpu().tolist(),
                strict=True,
            ):
                if row["relevance_mask"] and row["relevance_label"] != "uncertain":
                    similarities.append(s)
                    relevance_truth.append(int(row["relevance_label"] == "direct_counterpart"))
                if row["relationship_mask"] and row["relationship_label"] != "ambiguous":
                    truth.append(row["relationship_label"])
                    predicted.append(LABELS[y])
                    swapped.append(LABELS[rev])
            if progress:
                processed = min(i + batch_size, len(rows))
                progress(
                    phase="evaluating pairs and A/B reversal",
                    batch=i // batch_size + 1,
                    batches=math.ceil(len(rows) / batch_size),
                    processed_pairs=processed,
                    total_pairs=len(rows),
                    eta_seconds=(time.perf_counter() - started)
                    / processed
                    * (len(rows) - processed),
                )
    if progress:
        progress(phase="computing metrics", processed_pairs=len(rows), total_pairs=len(rows))
    reverse_label = {
        "a_more_restrictive": "b_more_restrictive",
        "b_more_restrictive": "a_more_restrictive",
    }
    both = len(set(relevance_truth)) == 2
    return {
        "rows": len(rows),
        "relationship_judged": len(truth),
        "relationship_accuracy": float(accuracy_score(truth, predicted)) if truth else None,
        "relationship_macro_f1": float(
            f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)
        )
        if truth
        else None,
        "relationship_macro_f1_label_universe": LABELS,
        "swap_consistency": float(
            np.mean([b == reverse_label.get(a, a) for a, b in zip(predicted, swapped, strict=True)])
        )
        if truth
        else None,
        "relevance_judged": len(relevance_truth),
        "relevance_pair_roc_auc": float(roc_auc_score(relevance_truth, similarities))
        if both
        else None,
        "relevance_pair_average_precision": float(
            average_precision_score(relevance_truth, similarities)
        )
        if both
        else None,
        "task1_scope": "Reviewed-pair discrimination; not full-corpus retrieval quality or recall.",
        "calibrated": False,
        "clinical_validation_claimed": False,
    }


def lock_evaluation(artifact, release, output):
    from trialign.models.joint import read_manifest

    read_manifest(artifact)
    output = Path(output)
    if output.exists():
        raise ValueError("Evaluation lock exists; do not rewrite a locked configuration.")
    payload = {
        "schema_version": "model-evaluation-lock-1",
        "artifact_sha256": sha(Path(artifact) / "manifest.json"),
        "release_sha256": sha(Path(release) / "manifest.json"),
        "evaluator_sha256": sha(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def evaluate_model(
    artifact,
    release,
    *,
    split="validation",
    test_lock=None,
    allow_synthetic=False,
    allow_draft_labels=False,
    progress=None,
):
    from trialign.data.annotations import load_label_split
    from trialign.models.joint import TrainedModel

    if progress:
        progress(phase="validating evaluation inputs", split=split)
    if split == "test":
        if not test_lock:
            raise ValueError(
                "Test evaluation requires a model/release/evaluator lock "
                "made before opening test labels."
            )
        lock = json.loads(Path(test_lock).read_text())
        expected = {
            "schema_version": "model-evaluation-lock-1",
            "artifact_sha256": sha(Path(artifact) / "manifest.json"),
            "release_sha256": sha(Path(release) / "manifest.json"),
            "evaluator_sha256": sha(Path(__file__)),
        }
        if lock != expected:
            raise ValueError(
                "Evaluation lock no longer matches the artifact, label release or evaluator."
            )
    rows, manifest = load_label_split(
        release, split, allow_synthetic=allow_synthetic, allow_draft_labels=allow_draft_labels
    )
    if not rows:
        raise ValueError("Selected label partition is empty; no evaluation can be reported.")
    if progress:
        progress(phase="loading saved model", split=split, total_pairs=len(rows))
    trained = TrainedModel(artifact)
    if trained.manifest["label_release_sha256"] != sha(Path(release) / "manifest.json"):
        raise ValueError("Model was trained against a different label release.")
    return {
        "split": split,
        "synthetic": manifest["synthetic"],
        "draft_labels": manifest.get("draft_labels", False),
        "in_sample": split == "train",
        "clinical_validation_claimed": False,
        "interpretation": "Agreement with unvalidated draft labels; not clinical accuracy."
        if manifest.get("draft_labels")
        else "Performance against released reference labels.",
        "artifact_sha256": trained.artifact_digest,
        "label_release_sha256": sha(Path(release) / "manifest.json"),
        **pair_metrics(
            trained.model, rows, batch_size=trained.config.batch_size, progress=progress
        ),
    }


def pooled_metrics(
    rankings, judgments, *, k=5, query_metadata=None, bootstrap_replicates=2000, seed=20261011
):
    """rankings: model -> query ID -> ordered candidate IDs; unknowns stay unknown."""
    if k < 1 or not rankings:
        raise ValueError("At least one model and positive k are required.")
    models = sorted(rankings)
    queries = set.intersection(*(set(rankings[m]) for m in models))
    observations, complete = {}, []
    for query in sorted(queries):
        lists = {m: rankings[m][query][:k] for m in models}
        if any(len(c) != k or len(set(c)) != k or query in c for c in lists.values()):
            raise ValueError("Each query needs k distinct non-self candidates per model.")
        union = set().union(*map(set, lists.values()))
        grades = {
            candidate: judgments.get(tuple(sorted((query, candidate)))) for candidate in union
        }
        if any(v not in {0, 1, 2, None} for v in grades.values()):
            raise ValueError("Unknown relevance grade.")
        fully_judged = all(value is not None for value in grades.values())
        if fully_judged:
            complete.append(query)
        ideal = (
            sum(
                (2**v - 1) / np.log2(i + 2)
                for i, v in enumerate(sorted(grades.values(), reverse=True)[:k])
            )
            if fully_judged
            else None
        )
        observations[query] = {}
        for model, candidates in lists.items():
            positive = sum(grades[c] == 2 for c in candidates)
            unknown = sum(grades[c] is None for c in candidates)
            dcg = (
                sum((2 ** grades[c] - 1) / np.log2(i + 2) for i, c in enumerate(candidates))
                if fully_judged
                else None
            )
            observations[query][model] = {
                "lower": positive / k,
                "upper": (positive + unknown) / k,
                "precision": positive / k if fully_judged else None,
                "ndcg": dcg / ideal if fully_judged and ideal else None,
            }
    means = {}
    for model in models:
        values = [observations[q][model] for q in sorted(queries)]

        def avg(key, values=values):
            known = [v[key] for v in values if v[key] is not None]
            return float(np.mean(known)) if known else None

        means[model] = {
            "precision@k": avg("precision"),
            "pooled_ndcg@k": avg("ndcg"),
            "precision_lower_bound": avg("lower"),
            "precision_upper_bound": avg("upper"),
        }
    result = {
        "k": k,
        "common_queries": len(queries),
        "fully_judged_queries": len(complete),
        "incomplete_queries": sorted(queries - set(complete)),
        "models": means,
        "model_query_counts": {m: len(rankings[m]) for m in models},
        "scope": "Pooled candidate judgments; no complete-corpus recall claim.",
    }
    if query_metadata and complete:
        # Paired, cohort-stratified split-group bootstrap, same draws for every model.
        strata = {}
        for q in complete:
            metadata = query_metadata[q]
            strata.setdefault(metadata["cohort"], {}).setdefault(
                metadata["split_group"], []
            ).append(q)
        rng = np.random.default_rng(seed)
        samples = {m: [] for m in models}
        for _ in range(bootstrap_replicates):
            draw = []
            for groups in strata.values():
                keys = sorted(groups)
                for i in rng.integers(0, len(keys), size=len(keys)):
                    draw.extend(groups[keys[i]])
            for model in models:
                samples[model].append(np.mean([observations[q][model]["precision"] for q in draw]))
        result["precision_95pct_ci"] = {
            m: np.quantile(values, [0.025, 0.975]).tolist() for m, values in samples.items()
        }
        result["paired_precision_difference_95pct_ci"] = {
            f"{a} minus {b}": np.quantile(
                np.asarray(samples[a]) - samples[b], [0.025, 0.975]
            ).tolist()
            for a, b in itertools.combinations(models, 2)
        }
        result["bootstrap"] = {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "unit": "cohort-stratified split groups",
        }
    return result
