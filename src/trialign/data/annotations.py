"""Read human workbook decisions into a source-verified, partitioned label release."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from openpyxl import load_workbook

from trialign.data.corpus import load_parquet_corpus
from trialign.data.partitions import load_partition_release, sha
from trialign.schemas import RelationshipLabel

RELEVANCE = {"direct_counterpart": 2, "related_topic": 1, "unrelated": 0, "uncertain": None}
AUTOMATED = re.compile(
    r"^(codex|assistant|gpt|claude|gemini|ai\b|test\b|synthetic\b|dummy\b)", re.I
)
REQUIRED = [
    "Relevance",
    "Relationship",
    "Uncertainty",
    "Clinical question",
    "Relevance rationale",
    "Relationship rationale",
    "Common comparison frame",
    "A supporting quote",
    "B supporting quote",
    "Governing scope",
    "Source issues",
    "Assumptions",
    "Reviewer",
    "Reviewed at UTC",
    "Adjudicator",
]


def _read_sheet(sheet):
    values = list(sheet.iter_rows(values_only=True))
    if len(values) < 5:
        raise ValueError("Annotation sheet has no row-5 headers.")
    headers = values[4]
    if len({h for h in headers if h}) != len([h for h in headers if h]):
        raise ValueError("Annotation headers must be unique.")
    result = {}
    for number, cells in enumerate(values[5:], 6):
        if not any(c is not None for c in cells):
            continue
        row = dict(zip(headers, cells, strict=True))
        pair_id = row.get("Pair ID")
        if not pair_id or pair_id in result:
            raise ValueError(f"Missing/duplicate Pair ID at row {number}.")
        row["excel_row"] = number
        result[pair_id] = row
    return result


def _human(row):
    for field in ["Reviewer", "Adjudicator"]:
        value = str(row.get(field) or "").strip()
        if not value or AUTOMATED.search(value):
            return False
    return row.get("Status") == "adjudicated"


def inspect_annotations(root: Path, workbook: Path, *, packets=None, draft_pilot=False):
    """Audit all rows. Draft/unresolved rows remain visible, never implicit negatives."""
    partition = load_partition_release(root)
    corpus_path = root / "data/processed/trial100_v1.parquet"
    corpus = {r.criterion_id: r for r in load_parquet_corpus(corpus_path)}
    default_packets = [
        root / "data/processed/pair_pilot_v1/pairs.jsonl",
        root / "data/processed/pair_pilot_v1/training_review_v2.jsonl",
    ]
    packet_paths = packets if packets is not None else [p for p in default_packets if p.is_file()]
    packet = {}
    for path in packet_paths:
        path = Path(path)
        construction_manifest = path.with_suffix(".manifest.json")
        if construction_manifest.is_file():
            expected = json.loads(construction_manifest.read_text()).get("packet_sha256")
            if expected and sha(path) != expected:
                raise ValueError("Construction packet checksum mismatch.")
        for line in Path(path).read_text().splitlines():
            item = json.loads(line)
            if item["pair_id"] in packet:
                raise ValueError("Duplicate pair ID in supplied construction packets.")
            packet[item["pair_id"]] = item
    # The tracked labeling workbook is the authoritative registry on a clean checkout;
    # construction JSONL packets are optional local provenance, not Git dependencies.
    registry_path = root / "data/processed/pair_pilot_v1/trial100_v1_pair_labeling.xlsx"
    registry_used = False
    if registry_path.is_file():
        registry_book = load_workbook(registry_path, read_only=True, data_only=False)
        try:
            registry = _read_sheet(registry_book["Label pairs"])
        finally:
            registry_book.close()
        for pair_id, registered in registry.items():
            if pair_id not in packet:
                packet[pair_id] = {
                    "a": {"criterion_id": registered["A criterion ID"]},
                    "b": {"criterion_id": registered["B criterion ID"]},
                }
                registry_used = True
    wb = load_workbook(workbook, read_only=True, data_only=False)
    try:
        primary = _read_sheet(wb["Label pairs"])
        second = _read_sheet(wb["Second review"])
    finally:
        wb.close()
    if set(second) - set(primary):
        raise ValueError("Second review contains a pair absent from primary review.")
    errors, pending, excluded, exported = [], [], [], []
    seen_pairs = set()
    for pair_id, row in primary.items():
        issues = []
        pair = packet.get(pair_id)
        if pair is None:
            errors.append(
                {
                    "pair_id": pair_id,
                    "issues": ["Unknown construction pair; supply its immutable packet."],
                }
            )
            continue
        identifiers = [
            pair.get("criterion_a_id", pair["a"]["criterion_id"]),
            pair.get("criterion_b_id", pair["b"]["criterion_id"]),
        ]
        if any(i not in corpus for i in identifiers):
            errors.append({"pair_id": pair_id, "issues": ["Unknown frozen criterion ID."]})
            continue
        a, b = (corpus[i] for i in identifiers)
        key = tuple(sorted(identifiers))
        if key in seen_pairs:
            issues.append("Duplicate unordered criterion pair.")
        seen_pairs.add(key)
        if a.trial_id == b.trial_id:
            issues.append("Pair is not cross-trial.")
        if row.get("Corpus version") not in {None, "trial100_v1"}:
            issues.append("Workbook corpus version differs from the frozen corpus.")
        for side, record in [("A", a), ("B", b)]:
            if (
                row.get(f"{side} criterion ID") != record.criterion_id
                or row.get(f"{side} trial ID") != record.trial_id
                or row.get(f"{side} source") != record.source
            ):
                issues.append(f"{side} identity/orientation/source differs from frozen corpus.")
        splits = {partition["assignments"][r.trial_id] for r in [a, b]}
        if "quarantine" in splits:
            excluded.append(
                {
                    "pair_id": pair_id,
                    "issues": ["Quarantined source; excluded from all supervised partitions."],
                }
            )
            continue
        if len(splits) != 1:
            issues.append("Cross-partition pair.")
        if issues:
            errors.append({"pair_id": pair_id, "issues": issues})
            continue
        if not draft_pilot and not _human(row):
            pending.append(
                {
                    "pair_id": pair_id,
                    "issues": [
                        "Actual human reviewer/adjudicator and adjudicated status required."
                    ],
                }
            )
            continue
        for candidate, role in [(row, "primary"), (second.get(pair_id), "second")]:
            if role == "second" and (
                (draft_pilot and candidate is None)
                or (not draft_pilot and row.get("Second review required") != "yes")
            ):
                continue
            if candidate is None or (not draft_pilot and not _human(candidate)):
                issues.append(f"Human {role} review is incomplete.")
                continue
            if (
                role == "second"
                and not draft_pilot
                and candidate["Reviewer"].strip().casefold() == row["Reviewer"].strip().casefold()
            ):
                issues.append("Independent second reviewer must differ from the primary reviewer.")
            for field in REQUIRED:
                if draft_pilot and field == "Adjudicator":
                    continue
                if not str(candidate.get(field) or "").strip():
                    issues.append(f"{role}: missing {field}.")
            try:
                reviewed = datetime.fromisoformat(
                    str(candidate.get("Reviewed at UTC", "")).replace("Z", "+00:00")
                )
                if reviewed.tzinfo is None:
                    raise ValueError()
            except ValueError:
                issues.append(f"{role}: review timestamp requires ISO time and timezone.")
            if candidate.get("Relevance") not in RELEVANCE:
                issues.append(f"{role}: unknown relevance label.")
            if candidate.get("Relationship") not in {x.value for x in RelationshipLabel}:
                issues.append(f"{role}: unknown relationship label.")
            if candidate.get("Uncertainty") not in {"low", "medium", "high"}:
                issues.append(f"{role}: unknown uncertainty.")
            if candidate.get("Relationship") == "ambiguous" and not candidate.get(
                "Ambiguity reason"
            ):
                issues.append(f"{role}: ambiguous disposition requires a reason.")
            for side, record in [("A", a), ("B", b)]:
                quote = str(candidate.get(f"{side} supporting quote") or "")
                if not quote or not any(
                    quote in (text or "")
                    for text in [record.source, record.parent_statement, record.surrounding_context]
                ):
                    issues.append(
                        f"{role}: {side} quote is not an exact frozen source/context span."
                    )
            if role == "second":
                for field in [
                    "Pair ID",
                    "A criterion ID",
                    "B criterion ID",
                    "A source",
                    "B source",
                ]:
                    if candidate.get(field) != row.get(field):
                        issues.append(f"Second-review {field} does not match the selected pair.")
                if (candidate.get("Relevance"), candidate.get("Relationship")) != (
                    row.get("Relevance"),
                    row.get("Relationship"),
                ):
                    # Keep both reviews and require a documented adjudication.
                    if (
                        not draft_pilot
                        and "adjudicat" not in str(row.get("Relationship rationale", "")).casefold()
                    ):
                        issues.append("Review disagreement needs explicit adjudication rationale.")
        if issues:
            pending.append({"pair_id": pair_id, "issues": issues})
            continue
        clean_source = str(row["Source issues"]).strip().casefold() == "none"
        second_row = second.get(pair_id) if draft_pilot else None
        if second_row:
            clean_source = clean_source and str(second_row["Source issues"]).casefold() == "none"
        relevance_agrees = not second_row or second_row["Relevance"] == row["Relevance"]
        relationship_agrees = not second_row or second_row["Relationship"] == row["Relationship"]
        relevance = row["Relevance"]
        relationship = row["Relationship"]
        exported.append(
            {
                "pair_id": pair_id,
                "partition": next(iter(splits)),
                "a": a.model_dump(mode="json"),
                "b": b.model_dump(mode="json"),
                "relevance_label": relevance,
                "relevance_grade": RELEVANCE[relevance],
                "relationship_label": relationship,
                "relevance_mask": relevance != "uncertain" and clean_source and relevance_agrees,
                "relationship_mask": relationship != "ambiguous"
                and relevance != "uncertain"
                and clean_source,
                "source_issues": row["Source issues"],
                "reviewer_id": row["Reviewer"],
                "adjudicator_id": row["Adjudicator"],
                "reviewed_at_utc": row["Reviewed at UTC"],
                "second_review": second.get(pair_id)
                if draft_pilot or row.get("Second review required") == "yes"
                else None,
                **(
                    {
                        "label_provenance": "unvalidated_workbook_draft",
                        "human_approved": False,
                        "primary_status": row.get("Status"),
                        "draft_relevance_disagreement": not relevance_agrees,
                        "draft_relationship_disagreement": not relationship_agrees,
                    }
                    if draft_pilot
                    else {}
                ),
                "retrieval_occurrences": pair.get("retrieval_occurrences", []),
            }
        )
        if draft_pilot:
            exported[-1]["relationship_mask"] &= relevance_agrees and relationship_agrees
    return {
        "schema_version": "annotation-audit-1",
        "rows": len(primary),
        "approved_rows": 0 if draft_pilot else len(exported),
        **({"draft_rows": len(exported)} if draft_pilot else {}),
        "errors": errors,
        "pending": pending,
        "excluded": excluded,
        "records": exported,
        "packet_hashes": {
            **{str(p): sha(Path(p)) for p in packet_paths},
            **({str(registry_path): sha(registry_path)} if registry_used else {}),
        },
        "supervised_training_approved": False,
    }


def release_annotations(
    root, workbook, output, *, approve_research_use=False, allow_partial=False, packets=None
):
    if not approve_research_use:
        raise ValueError(
            "An explicit research-use approval is required; drafts cannot be released."
        )
    if output.exists():
        raise ValueError("Release directory already exists; create a new version.")
    guide = (root / "docs/notes/trialign_technical.tex").read_text()
    begin = guide.index(r"\section{Pair labeling contract 0.2}")
    end = guide.index("\n\\section{", begin + 1)
    relationship_contract = hashlib.sha256(guide[begin:end].encode()).hexdigest()
    expected = json.loads(
        (root / "data/processed/pair_pilot_v1/release_v1/manifest.json").read_text()
    )["annotation_contract_sha256"]
    if relationship_contract != expected:
        raise ValueError("Frozen relationship annotation contract changed.")
    report = inspect_annotations(root, workbook, packets=packets)
    if report["errors"] or (report["pending"] and not allow_partial):
        raise ValueError(
            "Annotation audit is incomplete; inspect validation results before releasing."
        )
    rows = report["records"]
    if not rows or not any(
        r["partition"] == "train" and (r["relevance_mask"] or r["relationship_mask"]) for r in rows
    ):
        raise ValueError("No source-clean, human-approved training targets available.")
    # A partial development release must still meet the declared double-review minimum.
    development = [r for r in rows if r["partition"] == "train"]
    if sum(r["second_review"] is not None for r in development) < len(development) * 0.2:
        raise ValueError(
            "At least 20% of released training pairs require independent human double review."
        )
    if any(r["partition"] != "train" and r["second_review"] is None for r in rows):
        raise ValueError("Every held-out judgment requires independent human double review.")
    import tempfile

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "release"
        staged.mkdir()
        files = {}
        for split in ["train", "validation", "test"]:
            path = staged / (split + ".jsonl")
            path.write_text(
                "".join(
                    json.dumps(r, sort_keys=True, default=str) + "\n"
                    for r in rows
                    if r["partition"] == split
                )
            )
            files[path.name] = sha(path)
        manifest = {
            "schema_version": "human-label-release-1",
            "synthetic": False,
            "approved_for_research_training": True,
            "clinical_validation_claimed": False,
            "source_workbook_sha256": sha(workbook),
            "corpus_sha256": sha(root / "data/processed/trial100_v1.parquet"),
            "partitions_sha256": sha(root / "data/processed/partitions_v1/manifest.json"),
            "relevance_contract_sha256": sha(root / "configs/retrieval_relevance_v1.json"),
            "relationship_contract_sha256": relationship_contract,
            "annotation_contracts": ["pair-guide-0.2", "retrieval-relevance-1.0"],
            "construction_packets": report["packet_hashes"],
            "files": files,
            "counts": dict(Counter(r["partition"] for r in rows)),
            "pending_rows": len(report["pending"]),
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staged.rename(output)
    return manifest


def release_draft_pilot(root, workbook, output):
    """Freeze unvalidated workbook drafts, preserving family splits and review provenance."""
    import shutil
    import tempfile

    root, workbook, output = Path(root), Path(workbook), Path(output)
    if output.exists():
        raise ValueError("Pilot directory exists; use a new version.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "pilot"
        staged.mkdir()
        snapshot = staged / "workbook.xlsx"
        before = sha(workbook)
        shutil.copyfile(workbook, snapshot)
        if sha(snapshot) != before or sha(workbook) != before:
            raise ValueError("Workbook changed during snapshot; save and retry.")
        report = inspect_annotations(root, snapshot, draft_pilot=True)
        if report["errors"] or report["pending"]:
            raise ValueError(
                "Draft source/label audit failed: "
                + json.dumps({k: report[k] for k in ["errors", "pending"]})
            )
        rows = report.pop("records")
        for split in ["train", "validation", "test"]:
            (staged / (split + ".jsonl")).write_text(
                "".join(
                    json.dumps(r, sort_keys=True, default=str) + "\n"
                    for r in rows
                    if r["partition"] == split
                )
            )
        (staged / "audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        guide = (root / "docs/notes/trialign_technical.tex").read_text()
        begin = guide.index(r"\section{Pair labeling contract 0.2}")
        end = guide.index("\n\\section{", begin + 1)
        manifest = {
            "schema_version": "draft-label-pilot-1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "synthetic": False,
            "draft_labels": True,
            "approved_for_research_training": False,
            "clinical_validation_claimed": False,
            "purpose": "Provisional training on unvalidated workbook proposals only.",
            "source_workbook_sha256": before,
            "corpus_sha256": sha(root / "data/processed/trial100_v1.parquet"),
            "partitions_sha256": sha(root / "data/processed/partitions_v1/manifest.json"),
            "relevance_contract_sha256": sha(root / "configs/retrieval_relevance_v1.json"),
            "relationship_contract_sha256": hashlib.sha256(guide[begin:end].encode()).hexdigest(),
            "construction_packets": report["packet_hashes"],
            "counts": dict(Counter(r["partition"] for r in rows)),
            "usable_targets": {
                split: {
                    task: sum(bool(r[task + "_mask"]) for r in rows if r["partition"] == split)
                    for task in ["relevance", "relationship"]
                }
                for split in ["train", "validation", "test"]
            },
            "excluded_rows": len(report["excluded"]),
            "test_evaluation_enabled": False,
            "files": {p.name: sha(p) for p in staged.iterdir()},
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staged.rename(output)
    return manifest


def load_label_split(directory, split, *, allow_synthetic=False, allow_draft_labels=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    draft = manifest.get("schema_version") == "draft-label-pilot-1"
    if draft:
        if not allow_draft_labels:
            raise ValueError("Draft pilot requires explicit --allow-draft-labels opt-in.")
        if split == "test":
            raise ValueError("Draft-label test evaluation is disabled; reserve human test review.")
        if manifest.get("approved_for_research_training") or not manifest.get("draft_labels"):
            raise ValueError("Invalid draft-pilot provenance.")
        for name, expected in manifest["files"].items():
            if (
                name in {"train.jsonl", "validation.jsonl", "test.jsonl"}
                and name != split + ".jsonl"
            ):
                continue
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
                raise ValueError("Draft pilot snapshot checksum mismatch.")
    elif manifest.get("schema_version") != "human-label-release-1":
        raise ValueError("Training requires a versioned human-label release.")
    if manifest.get("synthetic"):
        if not allow_synthetic:
            raise ValueError("Synthetic fixtures are only allowed in the explicit smoke command.")
    elif not draft and not manifest.get("approved_for_research_training"):
        raise ValueError("Training data has not been approved for research use.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown label partition.")
    name = split + ".jsonl"
    if sha(directory / name) != manifest["files"][name]:
        raise ValueError("Label release checksum mismatch.")
    rows = [json.loads(line) for line in (directory / name).read_text().splitlines()]
    for row in rows:
        if row["partition"] != split or row["a"]["trial_id"] == row["b"]["trial_id"]:
            raise ValueError("Pair partition/trial identity mismatch.")
        if row["relevance_label"] not in {
            "direct_counterpart",
            "related_topic",
            "unrelated",
            "uncertain",
        } or row["relationship_label"] not in {label.value for label in RelationshipLabel}:
            raise ValueError("Unknown training target.")
    return rows, manifest
