"""Frozen, exposure-aware trial-family partitions for the reviewed corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from trialign.data.corpus import load_parquet_corpus

PARTITIONS = ("train", "validation", "test")
POLICY = Path("configs/evaluation_policy_v1.json")
RELEASE = Path("data/processed/partitions_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def apportion(total, weights):
    denominator = sum(weights.values())
    exact = {key: total * value / denominator for key, value in weights.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    order = sorted(weights, key=lambda k: (-(exact[k] - counts[k]), list(weights).index(k)))
    for key in order[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def pair_partition(a, b, assignments):
    if a not in assignments or b not in assignments:
        raise ValueError("Every trial must have an explicit frozen partition assignment.")
    left, right = assignments[a], assignments[b]
    return left if a != b and left == right and left in PARTITIONS else None


def assign_partitions(records, family, inventory, policy):
    ids = {r.trial_id for r in records}
    groups = family["split_group_by_trial"]
    if set(groups) != ids or set(inventory) != ids or set(family["family_by_trial"]) != ids:
        raise ValueError("Family mapping and inventory must cover exactly the frozen corpus.")
    reserved = set(family["holdout_excluded_trial_ids"])
    for exposure in policy["relationship_development_exposures"]:
        reserved.update(exposure["trial_ids"])
    unresolved = set(family["unresolved_trial_ids"])
    if not reserved <= ids or not unresolved <= ids:
        raise ValueError("Unknown trial in exclusions or exposure ledger.")
    quarantine = {groups[t] for t in unresolved}
    forced = {groups[t] for t in reserved} - quarantine
    members = defaultdict(list)
    for trial in sorted(ids):
        members[groups[trial]].append(trial)
    cohorts = {}
    for group, trials in members.items():
        labels = {inventory[t]["cohort"] for t in trials}
        if len(labels) != 1:
            raise ValueError(
                "Mixed-cohort split groups require an explicit revised stratum policy."
            )
        cohorts[group] = labels.pop()
    eligible = set(members) - quarantine
    targets = apportion(len(eligible), policy["target_ratios"])
    cohort_counts = Counter(cohorts[g] for g in eligible)
    train_targets = apportion(targets["train"], dict(sorted(cohort_counts.items())))
    by_group = {g: "quarantine" for g in quarantine}

    def rank(phase, group):
        return hashlib.sha256(f"{policy['seed']}|{phase}|{group}".encode()).hexdigest()

    for cohort in sorted(cohort_counts):
        selected = {g for g in forced if cohorts[g] == cohort}
        count = train_targets[cohort]
        if len(selected) > count:
            raise ValueError("Development exposures exceed the fixed stratum training capacity.")
        available = sorted(
            (g for g in eligible - forced if cohorts[g] == cohort), key=lambda g: rank("train", g)
        )
        selected.update(available[: count - len(selected)])
        by_group.update({g: "train" for g in selected})
    remaining = eligible - set(by_group)
    validation_targets = apportion(
        targets["validation"], dict(sorted(Counter(cohorts[g] for g in remaining).items()))
    )
    for cohort, count in validation_targets.items():
        available = sorted(
            (g for g in remaining if cohorts[g] == cohort), key=lambda g: rank("validation", g)
        )
        by_group.update({g: "validation" for g in available[:count]})
    by_group.update({g: "test" for g in eligible - set(by_group)})
    assignments = {t: by_group[groups[t]] for t in sorted(ids)}
    if Counter(by_group.values()) != Counter({**targets, "quarantine": len(quarantine)}):
        raise ValueError("Partition group targets were not satisfied.")
    if any(assignments[t] in {"validation", "test"} for t in reserved):
        raise ValueError("Development or quarantine trial reached held-out evaluation.")
    for mapping in [groups, family["family_by_trial"]]:
        observed = defaultdict(set)
        for trial, group in mapping.items():
            observed[group].add(assignments[trial])
        if any(len(values) != 1 for values in observed.values()):
            raise ValueError("Clinical family or split-group leakage detected.")
    return assignments, by_group, sorted(reserved)


def fingerprint(record):
    row = record.model_dump(mode="json")
    fields = [
        "source",
        "parent_statement",
        "criterion_type",
        "entity",
        "comparator",
        "value",
        "unit",
        "lower_bound",
        "lower_inclusive",
        "upper_bound",
        "upper_inclusive",
        "negated",
        "temporal",
        "qualifier",
        "logical_operator",
        "repetition",
        "governing_context",
        "surrounding_context",
    ]
    data = {key: row.get(key) for key in fields}
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = re.sub(r"\s+", " ", value.casefold()).strip()
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def contract_text(root, policy):
    text = (root / policy["technical_document"]).read_text()
    start = text.index(r"\section{Retrieval relevance contract 1.0}")
    end = text.index("\n\\section{", start + 1)
    return text[start:end]


def payloads(root):
    policy = json.loads((root / POLICY).read_text())
    corpus = root / policy["corpus"]
    if sha(corpus) != policy["corpus_sha256"]:
        raise ValueError("Frozen corpus checksum changed.")
    metadata = json.loads(corpus.with_suffix(".parquet.manifest.json").read_text())
    if metadata["parquet_sha256"] != sha(corpus):
        raise ValueError("Corpus manifest does not match the snapshot.")
    family = json.loads((root / policy["family_mapping"]).read_text())
    inventory = json.loads((root / policy["inventory"]).read_text())
    if family["corpus_sha256"] != sha(corpus):
        raise ValueError("Family map was generated for a different corpus.")
    if sha(root / "configs/family_audit_decisions_v1.json") != family["decision_ledger_sha256"]:
        raise ValueError("Family evidence ledger changed after the audit.")
    records = load_parquet_corpus(corpus)
    assignments, groups, reservations = assign_partitions(records, family, inventory, policy)
    grouped = defaultdict(list)
    for record in records:
        grouped[fingerprint(record)].append(record)
    source_groups = defaultdict(list)
    for record in records:
        text = re.sub(r"\s+", " ", record.source.casefold()).strip()
        source_groups[hashlib.sha256(text.encode()).hexdigest()].append(record)
    source_partitions = {}
    source_duplicates = []
    for key, members in source_groups.items():
        values = {assignments[r.trial_id] for r in members}
        for record in members:
            source_partitions[record.criterion_id] = values
        if len(values) > 1:
            source_duplicates.append(
                {
                    "source_fingerprint": key,
                    "partitions": sorted(values),
                    "criterion_ids": sorted(r.criterion_id for r in members),
                    "trial_ids": sorted({r.trial_id for r in members}),
                }
            )
    flags, duplicates = {}, []
    for key, members in grouped.items():
        splits = {assignments[r.trial_id] for r in members}
        if len(splits) > 1:
            duplicates.append(
                {
                    "fingerprint": key,
                    "partitions": sorted(splits),
                    "criterion_ids": sorted(r.criterion_id for r in members),
                    "trial_ids": sorted({r.trial_id for r in members}),
                }
            )
        for record in members:
            split = assignments[record.trial_id]
            prior = (
                {"train"}
                if split == "validation"
                else {"train", "validation"}
                if split == "test"
                else set()
            )
            flags[record.criterion_id] = split in {"validation", "test"} and not bool(
                source_partitions[record.criterion_id] & prior
            )
    criterion_assignments = {r.criterion_id: assignments[r.trial_id] for r in records}
    counts = {
        part: {
            "trials": sum(v == part for v in assignments.values()),
            "groups": sum(v == part for v in groups.values()),
            "criteria": sum(v == part for v in criterion_assignments.values()),
            "cohorts": dict(
                sorted(
                    Counter(
                        inventory[t]["cohort"] for t, v in assignments.items() if v == part
                    ).items()
                )
            ),
        }
        for part in (*PARTITIONS, "quarantine")
    }
    output = {
        "partition_release_id": policy["policy_id"],
        "status": "frozen_family_aware_partitions",
        "seed": policy["seed"],
        "corpus_sha256": sha(corpus),
        "assignments": assignments,
        "split_group_assignments": groups,
        "criterion_assignments": criterion_assignments,
        "secondary_exact_source_eligible": flags,
        "counts": counts,
        "development_reserved_trial_ids": reservations,
        "quarantined_trial_ids": sorted(t for t, v in assignments.items() if v == "quarantine"),
        "external_assignments": {
            row["trial_id"]: groups[row["split_group_id"]]
            for row in family["rows"]
            if not row["in_current_corpus"] and row["split_group_id"] in groups
        },
        "final_partitions_assigned": True,
        "training_labels_approved": False,
        "heldout_labels_created": False,
        "limitations": [
            "Uses frozen audit v1 evidence; absence of a supported link "
            "does not prove independence.",
            "Prior whole-corpus baseline runs were exploratory and are not held-out evaluations.",
            "Primary family-disjoint evaluation can contain repeated clauses; "
            "secondary exact-source flags remove repeated wording only.",
            "New exposures, family links or source revisions require a new release; "
            "never silently resplit after viewing test results.",
        ],
    }
    duplicates_output = {
        "policy": "Report source-only and source-plus-scope duplication; retain original corpus. "
        "Secondary held-out pairs require both sides to be eligible.",
        "cross_partition_groups": sorted(duplicates, key=lambda d: d["fingerprint"]),
        "source_text_overlap_groups": sorted(
            source_duplicates, key=lambda d: d["source_fingerprint"]
        ),
        "secondary_eligible_counts": dict(
            Counter(criterion_assignments[c] for c, ok in flags.items() if ok)
        ),
    }
    return {
        "partitions.json": encode(output),
        "duplicate_audit.json": encode(duplicates_output),
    }, policy


def freeze_partitions(root, output=None, *, verify=False):
    root = Path(root)
    output = Path(output) if output else root / RELEASE
    generated, policy = payloads(root)
    sources = [
        POLICY,
        Path(policy["corpus"]),
        Path(policy["corpus"] + ".manifest.json"),
        Path(policy["family_mapping"]),
        Path(policy["inventory"]),
        Path(policy["relevance_contract"]),
        Path("configs/family_audit_decisions_v1.json"),
        Path("data/processed/pair_pilot_v1/manifest.json"),
        Path("data/processed/pair_pilot_v1/release_v1/manifest.json"),
    ]
    provenance = {str(path): sha(root / path) for path in sources}
    contract_hash = hashlib.sha256(contract_text(root, policy).encode()).hexdigest()
    if verify:
        manifest = json.loads((output / "manifest.json").read_text())
        if (
            manifest["source_sha256"] != provenance
            or manifest["relevance_contract_sha256"] != contract_hash
        ):
            raise ValueError("Partition sources or frozen relevance contract changed.")
        for name, content in generated.items():
            if (output / name).read_bytes() != content or sha(output / name) != manifest["files"][
                name
            ]:
                raise ValueError("Frozen partition output changed: " + name)
    else:
        if output.exists():
            raise ValueError(
                "Partition release already exists; use --verify or a new output version."
            )
        output.mkdir(parents=True)
        for name, content in generated.items():
            (output / name).write_bytes(content)
        manifest = {
            "partition_release_id": policy["policy_id"],
            "source_sha256": provenance,
            "relevance_contract_sha256": contract_hash,
            "generator_sha256": sha(Path(__file__)),
            "files": {name: sha(output / name) for name in generated},
            "status": "frozen",
        }
        (output / "manifest.json").write_bytes(encode(manifest))
    return json.loads(generated["partitions.json"])


def load_partition_release(root):
    return freeze_partitions(root, verify=True)
