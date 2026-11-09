"""Prepare an unlabeled, training-only pair packet; never writes Excel or labels.

The working Excel workbook is a separate human-review interface. This command
records reproducible pair identities, frozen source context and sampling provenance.
Existing packets are verified, never overwritten. Dense caches must correspond
to the unchanged full-corpus order and their pinned baseline run metadata.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialign.data import load_parquet_corpus  # noqa: E402
from trialign.data.partitions import freeze_partitions  # noqa: E402
from trialign.models.retrieval import retrieve_neighbors  # noqa: E402
from trialign.models.tfidf import TfidfEncoder  # noqa: E402

PACKET = ROOT / "data/processed/pair_pilot_v1/training_review_v2.jsonl"
POLICY = ROOT / "configs/pair_review_sampling_v2.json"
CORPUS = ROOT / "data/processed/trial100_v1.parquet"
PINS = {
    "mpnet": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    "biolord": "167aab527b238a50ca65224e6319215d2ff4fc9f",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join((text or "").casefold().split())


def prepare():
    freeze_partitions(ROOT, verify=True)
    policy = json.loads(POLICY.read_text())
    partitions = json.loads((ROOT / "data/processed/partitions_v1/partitions.json").read_text())
    inventory = json.loads((ROOT / "data/processed/family_audit_v1/trial_inventory.json").read_text())
    records = load_parquet_corpus(CORPUS)
    train_indices = [i for i, r in enumerate(records) if partitions["assignments"][r.trial_id] == "train"]
    train = [records[i] for i in train_indices]
    by_id = {r.criterion_id: r for r in train}
    pilot = [json.loads(line) for line in (PACKET.parent / "pairs.jsonl").read_text().splitlines()]
    inputs = [CORPUS, POLICY, Path(__file__), PACKET.parent / "pairs.jsonl",
              ROOT / "data/processed/partitions_v1/manifest.json",
              ROOT / "configs/retrieval_relevance_v1.json"]
    rng = np.random.default_rng(policy["seed"])

    def key(a, b):
        return tuple(sorted((a.criterion_id, b.criterion_id)))

    def source_key(a, b):
        return tuple(sorted((normalized(a.source), normalized(b.source))))

    seen = {tuple(sorted((p["criterion_a_id"], p["criterion_b_id"]))) for p in pilot}
    source_seen = {source_key(type("S", (), p["a"])(), type("S", (), p["b"])()) for p in pilot}
    counts, trial_counts, trial_pairs, entity_counts, question_counts = (Counter() for _ in range(5))
    cohorts = Counter()
    rejected = Counter()
    selected = []

    def hash_order(value):
        return hashlib.sha256(f'{policy["seed"]}:{value}'.encode()).hexdigest()

    def eligible(a, b):
        if a.trial_id == b.trial_id:
            return "same_trial"
        if key(a, b) in seen:
            return "duplicate_or_pilot_pair"
        if source_key(a, b) in source_seen:
            return "duplicate_source_pair"
        if max(counts[a.criterion_id], counts[b.criterion_id]) >= policy["criterion_reuse_cap"]:
            return "criterion_cap"
        if max(trial_counts[a.trial_id], trial_counts[b.trial_id]) >= policy["trial_endpoint_cap"]:
            return "trial_cap"
        if trial_pairs[tuple(sorted((a.trial_id, b.trial_id)))] >= policy["trial_pair_reuse_cap"]:
            return "trial_pair_cap"
        if any(entity_counts[e] >= 300 * policy["normalized_entity_share_cap"]
               for e in {normalized(a.entity_normalized or a.entity), normalized(b.entity_normalized or b.entity)}):
            return "entity_cap"
        return None

    def accept(a, b, stratum, question=None, provenance=None):
        a, b = sorted((a, b), key=lambda r: r.criterion_id)
        seen.add(key(a, b)); source_seen.add(source_key(a, b))
        for r in (a, b):
            counts[r.criterion_id] += 1
            trial_counts[r.trial_id] += 1
            cohorts[inventory[r.trial_id]["cohort"]] += 1
        trial_pairs[tuple(sorted((a.trial_id, b.trial_id)))] += 1
        for e in {normalized(a.entity_normalized or a.entity), normalized(b.entity_normalized or b.entity)}:
            entity_counts[e] += 1
        question_counts[question or "baseline_or_control"] += 1
        group = hashlib.sha256("|".join(key(a, b)).encode()).hexdigest()[:20]
        selected.append({
            "schema_version": "training-review-2.0", "pair_id": "ordered-" + group,
            "pair_group_id": "group-" + group, "partition": "train",
            "corpus_sha256": sha(CORPUS), "sampling_policy": policy["policy_id"],
            "stratum": stratum, "search_question": question,
            "a": a.model_dump(mode="json"), "b": b.model_dump(mode="json"),
            "retrieval_occurrences": provenance or [], "constructed": False,
            "relevance_label": None, "relationship_label": None,
            "human_review_complete": False, "supervised_training_approved": False,
        })

    groups = defaultdict(list)
    lookup = {entity: q for q, entities in policy["clinical_question_entities"].items() for entity in entities}
    for r in train:
        q = lookup.get(normalized(r.entity_normalized or r.entity))
        if q:
            groups[q].append(r)
    clinical = [(a, b, q) for q, rs in groups.items() for a, b in itertools.combinations(rs, 2)
                if a.trial_id != b.trial_id]
    numeric = set(policy["numeric_questions"])
    pools = {True: [], False: []}
    for a, b, q in clinical:
        pools[q in numeric].append((a, b, q))
    clinical_pool_counts = {"numeric": len(pools[True]), "contextual": len(pools[False])}
    # Balance topics and trial exposure without assigning any clinical labels.
    for is_numeric, target in [(False, 140), (True, policy["clinical_search_numeric_pairs"])]:
        for _ in range(target):
            options = []
            for a, b, q in pools[is_numeric]:
                reason = eligible(a, b)
                if reason:
                    rejected[reason] += 1
                    continue
                cohort_load = sum(cohorts[inventory[r.trial_id]["cohort"]] for r in (a, b))
                load = (cohort_load, question_counts[q], max(trial_counts[a.trial_id], trial_counts[b.trial_id]),
                        trial_counts[a.trial_id] + trial_counts[b.trial_id], hash_order(key(a, b)))
                options.append((load, a, b, q))
            if not options:
                raise ValueError(f"Clinical search shortfall; selected {len(selected)}. Caps were not relaxed.")
            _, a, b, q = min(options, key=lambda x: x[0])
            accept(a, b, "clinical_question_search", q)

    occurrences = defaultdict(list)
    model_metadata = {}
    texts = [r.comparison_text or r.source for r in train]
    tfidf = TfidfEncoder().fit(texts).encode(texts)  # training-only vocabulary and IDF
    for name in ("tfidf", "mpnet", "biolord"):
        if name == "tfidf":
            vectors, revision = tfidf, None
        else:
            directory = ROOT / "artifacts/baselines" / name
            run_path, vector_path = directory / "run.json", directory / "embeddings.npy"
            run = json.loads(run_path.read_text())
            if run["model_revision"] != PINS[name] or run["text_field"] != "comparison_text":
                raise ValueError(f"Unexpected dense cache configuration: {name}")
            vectors_full = np.load(vector_path)
            if vectors_full.shape != (len(records), 768) or not np.isfinite(vectors_full).all():
                raise ValueError(f"Invalid dense cache: {name}")
            if not np.allclose(np.linalg.norm(vectors_full, axis=1), 1, atol=1e-5):
                raise ValueError(f"Unnormalized dense cache: {name}")
            # Validate each full-corpus cached query's top five against its saved IDs/scores.
            import pandas as pd
            neighbors_path = directory / "neighbors.parquet"
            saved = pd.read_parquet(neighbors_path)
            rebuilt = retrieve_neighbors(records, vectors_full, top_k=5, model=name, model_revision=PINS[name])
            if list(zip(saved.query_id, saved.candidate_id)) != [(r["query_id"], r["candidate_id"]) for r in rebuilt]:
                raise ValueError(f"Dense cache row-order mismatch: {name}")
            if not np.allclose(saved.score, [r["score"] for r in rebuilt], rtol=1e-6, atol=1e-8):
                raise ValueError(f"Dense cache score mismatch: {name}")
            vectors, revision = vectors_full[train_indices], PINS[name]
            inputs += [run_path, vector_path, neighbors_path]
        ranking = retrieve_neighbors(train, vectors, top_k=5, model=name, model_revision=revision)
        for row in ranking:
            occurrences[tuple(sorted((row["query_id"], row["candidate_id"])))].append(row)
        model_metadata[name] = {"revision": revision, "query_count": len(train), "neighbor_rows": len(ranking),
                                "fit_partition": "train" if name == "tfidf" else "frozen_pretrained",
                                "candidate_partition": "train", "text_field": "comparison_text"}

    baseline_pool_counts = Counter()
    for k, rows in occurrences.items():
        baseline_pool_counts["agreement" if len({r["model"] for r in rows}) >= 2 else "unique"] += 1
    for agreement, target in [(True, 30), (False, 30)]:
        for _ in range(target):
            options = []
            for k, rows in occurrences.items():
                if (len({r["model"] for r in rows}) >= 2) != agreement:
                    continue
                a, b = (by_id[c] for c in k)
                reason = eligible(a, b)
                if reason:
                    rejected[reason] += 1
                    continue
                load = (sum(cohorts[inventory[r.trial_id]["cohort"]] for r in (a, b)),
                        max(trial_counts[a.trial_id], trial_counts[b.trial_id]), hash_order(k))
                options.append((load, a, b, rows))
            if not options:
                raise ValueError("Baseline stratum shortfall; caps were not relaxed")
            _, a, b, rows = min(options, key=lambda x: x[0])
            accept(a, b, "baseline_agreement_top5" if agreement else "baseline_unique_top5", provenance=rows)

    by_trial = defaultdict(list)
    for r in train:
        by_trial[r.trial_id].append(r)
    trial_choices = list(itertools.combinations(sorted(by_trial), 2))
    draws = 0
    while len(selected) < 300:
        draws += 1
        if draws > 100000:
            raise ValueError("Random control shortfall; caps were not relaxed")
        ta, tb = trial_choices[int(rng.integers(len(trial_choices)))]
        a = by_trial[ta][int(rng.integers(len(by_trial[ta])))]
        b = by_trial[tb][int(rng.integers(len(by_trial[tb])))]
        reason = eligible(a, b)
        if reason:
            rejected["random_" + reason] += 1
            continue
        accept(a, b, "uniform_trial_pair_controls")

    selected.sort(key=lambda p: hash_order(p["pair_id"]))
    second = set(sorted((p["pair_id"] for p in selected), key=lambda k: hash_order("double:" + k))[:60])
    for i, row in enumerate(selected, 76):
        row["retrieval_occurrences"] = occurrences.get(
            tuple(sorted((row["a"]["criterion_id"], row["b"]["criterion_id"]))), [])
        row.update(pair_number=i, independent_second_review_required=row["pair_id"] in second,
                   initial_wave=i < 126)
    assert Counter(p["stratum"] for p in selected) == Counter(policy["strata"])
    raw = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in selected).encode()
    manifest = {
        "packet_id": policy["policy_id"], "pairs": len(selected), "labels_present": False,
        "partition": "train", "pilot_rows_preserved_in_workbook": 75, "initial_review_wave_pair_numbers": [76, 125],
        "independent_second_review_pairs": 60, "sampling_counts": dict(Counter(p["stratum"] for p in selected)),
        "clinical_search_counts": dict(question_counts), "eligible_clinical_search_pool": clinical_pool_counts,
        "baseline_pool_counts": dict(baseline_pool_counts), "rejection_checks": dict(rejected),
        "random_draws": draws, "criterion_reuse_max": max(counts.values()),
        "trial_pair_reuse_max": max(trial_pairs.values()), "trial_endpoint_max": max(trial_counts.values()),
        "entity_pair_count_max": max(entity_counts.values()), "trial_endpoint_cohorts": dict(cohorts),
        "trials": len(trial_counts), "criteria": len(counts), "models": model_metadata,
        "source_files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        "ordered_corpus_criterion_ids_sha256": hashlib.sha256("\n".join(r.criterion_id for r in records).encode()).hexdigest(),
        "packet_sha256": hashlib.sha256(raw).hexdigest(), "supervised_training_approved": False,
        "note": "Proposals only. Search aliases and baseline ranks are not labels. No held-out source packet created."
    }
    manifest_path = PACKET.with_suffix(".manifest.json")
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    for path, data in [(PACKET, raw), (manifest_path, manifest_raw)]:
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"Existing packet differs: {path}; use a versioned amendment")
        else:
            path.write_bytes(data)
    print(json.dumps({k: manifest[k] for k in ("pairs", "sampling_counts", "trials", "criteria", "trial_endpoint_cohorts")}))


if __name__ == "__main__":
    prepare()
