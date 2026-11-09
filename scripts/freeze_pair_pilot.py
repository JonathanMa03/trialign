"""Freeze or verify the source-linked development pair-review release.

This exports decisions, not new source criteria or a held-out clinical benchmark.
Run from a checkout with the project dependencies installed. Excel is read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PILOT = Path("data/processed/pair_pilot_v1")
OUTPUT = PILOT / "release_v1"
CORPUS = Path("data/processed/trial100_v1.parquet")
AUDIT = Path("data/processed/family_audit_v1")
DECISIONS = Path("configs/family_audit_decisions_v1.json")
TECHNICAL = Path("docs/notes/trialign_technical.tex")
RELEASE_ID = "trial100_v1_pair_review_v1"
RESOLVED_SOURCE_CODES = {"source_typo_resolved", "source_boundary_corroborated"}
QUEUE_ACTIONS = {
    10: "Correct B inclusion/exclusion role in a new corpus release and migrate the pair.",
    16: "Clarify the complete illness-list scope; do not assume the examples are exhaustive.",
    18: "Obtain the leukemic-involvement exception policy and supported sample-time mapping.",
    24: "Replace the chemotherapy heading with a valid criterion in a new corpus/pair version.",
    47: "Clarify the exception ceiling and construct an independently interpretable unit.",
    48: "Obtain a controlling primary source for B strict versus inclusive age-18 boundary.",
    50: "Construct a separately identified whole-rule comparison instead of the exception child.",
    52: "Replace the lesion-measurability fragment in a new corpus/pair version.",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def jsonl(rows: list[dict]) -> bytes:
    return ("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                    for row in rows)).encode("utf-8")


def guide_text(root: Path) -> str:
    text = (root / TECHNICAL).read_text(encoding="utf-8")
    start = text.index(r"\section{Pair labeling contract 0.2}")
    end = text.index("\n\\section{", start + 1)
    return text[start:end]


def source_blockers(review: dict) -> list[str]:
    return sorted({issue["code"] for issue in review.get("source_issues", [])
                   if issue["code"] not in RESOLVED_SOURCE_CODES})


def validate_workbook(path: Path, pairs: list[dict], reviews: list[dict]) -> None:
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "Workbook ZIP integrity check failed")
    workbook = load_workbook(path, data_only=False)
    # The owner expanded the mutable Excel interface after release v1. Check
    # the historical 75 decisions by stable IDs without mistaking new rows for
    # frozen annotations. verify() still enforces the old release's byte hashes.
    if workbook["Label pairs"].cell(5, 6).value == "Relevance":
        sheet = workbook["Label pairs"]
        headers = {cell.value: cell.column for cell in sheet[5]}
        indexed = {}
        for row in range(6, sheet.max_row + 1):
            if sheet.cell(row, headers["Batch"]).value == "pilot_history":
                pair_id = sheet.cell(row, headers["Pair ID"]).value
                require(pair_id not in indexed, "Duplicate historical pair ID")
                indexed[pair_id] = row
        require(set(indexed) == {p["pair_id"] for p in pairs}, "Historical pilot coverage changed")
        fields = {
            "label": "Relationship", "status": "Status", "uncertainty": "Uncertainty",
            "rationale": "Relationship rationale", "common_frame": "Common comparison frame",
            "evidence": "Evidence cited", "assumptions": "Assumptions",
            "ambiguity_reason": "Ambiguity reason", "reviewer_id": "Reviewer",
            "review_date": "Review date (YYYY-MM-DD)", "adjudicator_id": "Adjudicator",
        }
        for pair, review in zip(pairs, reviews, strict=True):
            row = indexed[pair["pair_id"]]
            for header, expected in {
                "A source": pair["a"]["source"], "B source": pair["b"]["source"],
                "Pair group ID": pair["pair_group_id"],
                "A criterion ID": pair["criterion_a_id"], "B criterion ID": pair["criterion_b_id"],
                "A trial ID": pair["a"]["trial_id"], "B trial ID": pair["b"]["trial_id"],
                "Guide version": pair["guide_version"], "Corpus version": pair["dataset_version"],
            }.items():
                require(sheet.cell(row, headers[header]).value == expected,
                        f"Historical source/identity differs: {pair['pair_id']}, {header}")
            for field, header in fields.items():
                actual = sheet.cell(row, headers[header]).value
                if field == "review_date" and hasattr(actual, "strftime"):
                    actual = actual.strftime("%Y-%m-%d")
                require((actual or "") == (review[field] or ""),
                        f"Historical decision changed: {pair['pair_id']}, {field}")
        workbook.close()
        return
    require(workbook.sheetnames == ["Label pairs", "Evidence", "Guide"],
            "Unexpected workbook sheets")
    sheet = workbook["Label pairs"]
    require(sheet.max_row == len(pairs) + 5, "Unexpected workbook row count")
    fields = ["label", "status", "uncertainty", "rationale", "common_frame", "evidence",
              "assumptions", "ambiguity_reason", "reviewer_id", "review_date", "adjudicator_id"]
    for number, (pair, review) in enumerate(zip(pairs, reviews, strict=True), 1):
        row = number + 5
        require(sheet.cell(row, 1).value == number, f"Workbook order changed at {number}")
        for col, expected in [(2, pair["a"]["source"]), (3, pair["b"]["source"]),
                              (15, pair["pair_id"]), (16, pair["pair_group_id"]),
                              (17, pair["criterion_a_id"]), (18, pair["criterion_b_id"]),
                              (19, pair["a"]["trial_id"]), (20, pair["b"]["trial_id"]),
                              (21, pair["guide_version"]), (22, pair["dataset_version"])]:
            require(sheet.cell(row, col).value == expected,
                    f"Workbook source/identity differs at pair {number}, column {col}")
        for col, field in enumerate(fields, 4):
            actual = sheet.cell(row, col).value
            if field == "review_date" and hasattr(actual, "strftime"):
                actual = actual.strftime("%Y-%m-%d")
            require((actual or "") == (review[field] or ""),
                    f"Workbook/log mismatch: pair {number}, {field}")
    workbook.close()


def load_inputs(root: Path) -> tuple[list[dict], list[dict], dict, list[Path]]:
    from trialign.data.corpus import load_parquet_corpus

    manifest = read_json(root / PILOT / "manifest.json")
    corpus_manifest = read_json(root / CORPUS.with_suffix(".parquet.manifest.json"))
    require(sha(root / CORPUS) == manifest["corpus_sha256"]
            == corpus_manifest["parquet_sha256"], "Frozen corpus checksum changed")
    require(sha(root / CORPUS.with_suffix(".parquet.manifest.json"))
            == manifest["corpus_manifest_sha256"], "Corpus manifest changed")
    require(sha(root / corpus_manifest["source_workbook"])
            == corpus_manifest["source_sha256"], "Frozen source workbook changed")
    require(sha(root / PILOT / "pairs.jsonl") == manifest["files"]["pairs.jsonl"],
            "Original sampling packet changed")
    pairs = read_jsonl(root / PILOT / "pairs.jsonl")
    reviews = read_jsonl(root / PILOT / "adjudication_2026-11-07.jsonl")
    draft = read_jsonl(root / PILOT / "annotations_ai_draft_2026-11-07.jsonl")
    require(len(pairs) == len(reviews) == len(draft) == manifest["pairs"] == 75,
            "Pilot, history and review coverage must be exactly 75")
    require(len({p["pair_id"] for p in pairs}) == 75
            and len({p["pair_group_id"] for p in pairs}) == 75,
            "Duplicate ordered or unordered pairs")
    snapshot_path = root / PILOT / "source_checks_2026-11-07.json"
    snapshot = read_json(snapshot_path)
    family = read_json(root / AUDIT / "family_mapping.json")
    family_decisions = read_json(root / DECISIONS)
    screening = read_json(root / AUDIT / "screening_manifest.json")
    require(family["corpus_sha256"] == manifest["corpus_sha256"], "Family corpus mismatch")
    require(family["decision_ledger_sha256"] == sha(root / DECISIONS), "Stale family decisions")
    require(family["pilot_manifest_sha256"] == sha(root / PILOT / "manifest.json"),
            "Stale family reservations")
    for name, key in [("candidate_edges.json", "candidate_edges_sha256"),
                      ("registry_snapshot.json", "registry_snapshot_sha256"),
                      ("external_registry_snapshot.json", "external_registry_snapshot_sha256")]:
        require(sha(root / AUDIT / name) == family_decisions[key], f"Stale family input: {name}")
    require(sha(root / AUDIT / "trial_inventory.json") == screening["trial_inventory_sha256"],
            "Stale trial inventory")
    corpus = load_parquet_corpus(root / CORPUS)
    by_id = {record.criterion_id: record for record in corpus}
    require(len(by_id) == len(corpus) == corpus_manifest["rows"], "Corpus identity mismatch")
    trial_ids = {record.trial_id for record in corpus}
    require(set(family["family_by_trial"]) == set(family["split_group_by_trial"]) == trial_ids,
            "Incomplete family/split-group coverage")
    reserved = set(manifest["reserved_development_trial_ids"])
    require(reserved <= set(family["holdout_excluded_trial_ids"]), "Pilot leakage into holdout")
    allowed = set(manifest["allowed_labels"])
    required = ["rationale", "common_frame", "evidence", "assumptions", "reviewer_id",
                "review_date", "adjudicator_id"]
    for number, (pair, review, initial) in enumerate(zip(pairs, reviews, draft, strict=True), 1):
        require(pair["pair_id"] == review["pair_id"] == initial["pair_id"],
                f"Review order/identity mismatch at {number}")
        require(review["pair_number"] == number
                and review["pair_group_id"] == pair["pair_group_id"], "Review group mismatch")
        require(pair["a"]["trial_id"] != pair["b"]["trial_id"], "Within-trial example")
        require(pair["guide_version"] == review["construction_guide_version"] == "pair-guide-0.1"
                and review["annotation_guide_version"] == "pair-guide-0.2", "Guide mismatch")
        require(review["corpus_version"] == pair["dataset_version"] == "trial100_v1",
                "Review corpus version mismatch")
        require(review["label"] in allowed and review["uncertainty"] in {"low", "medium", "high"},
                "Invalid label or uncertainty")
        require(all(review.get(key) for key in required), f"Incomplete review at {number}")
        require(review["prior_label"] == initial["label"]
                and review["label_changed"] == (review["label"] != initial["label"]),
                "Prior-decision history mismatch")
        if review["label"] == "ambiguous":
            require(review["status"] == "needs_adjudication" and review["ambiguity_reason"],
                    "Ambiguous example has no unresolved disposition")
        else:
            require(review["status"] == "adjudicated", "Unreviewed resolved example")
        require(review["source_check_snapshot_sha256"] == sha(snapshot_path),
                "Source snapshot changed after review")
        for side in ["a", "b"]:
            selected = pair[side]
            record = by_id[selected["criterion_id"]]
            for key in ["source", "parent_statement", "criterion_type", "trial_id",
                        "protocol_hash", "source_version"]:
                require(getattr(record, key) == selected[key], f"Corpus/packet mismatch: {key}")
            require(selected["trial_id"] in reserved, "Pair outside development reservation")
            study = snapshot["studies"][selected["trial_id"]]
            text = study["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
            digest = hashlib.sha256(text.encode()).hexdigest()
            require("sha256:" + digest == selected["protocol_hash"],
                    "Primary eligibility snapshot differs from frozen source")
            require(study["live_text_identical_to_frozen"] is True, "Unmatched source snapshot")
    reuse = Counter(c for pair in pairs for c in [pair["criterion_a_id"], pair["criterion_b_id"]])
    trial_reuse = Counter(tuple(sorted([p["a"]["trial_id"], p["b"]["trial_id"]])) for p in pairs)
    require(max(reuse.values()) <= manifest["max_pairs_per_criterion"], "Criterion reuse exceeded")
    require(max(trial_reuse.values()) <= manifest["max_pairs_per_trial_pair"], "Trial reuse exceeded")
    validate_workbook(root / PILOT / "trial100_v1_pair_labeling.xlsx", pairs, reviews)
    paths = [CORPUS, CORPUS.with_suffix(".parquet.manifest.json"),
             Path(corpus_manifest["source_workbook"]), DECISIONS]
    paths += [PILOT / name for name in ["manifest.json", "pairs.jsonl",
              "annotations_ai_draft_2026-11-07.jsonl", "adjudication_2026-11-07.jsonl",
              "source_checks_2026-11-07.json", "trial100_v1_pair_labeling.xlsx"]]
    paths += [AUDIT / name for name in ["all_pair_scores.json", "shared_secondary_identifiers.json",
              "trial_inventory.json", "external_registry_snapshot.json", "screening_manifest.json",
              "candidate_edges.json", "family_mapping.json", "registry_snapshot.json"]]
    return pairs, reviews, family, paths


def build_outputs(pairs: list[dict], reviews: list[dict], family: dict) -> tuple[dict, dict]:
    annotations, comparisons, queue = [], [], []
    for pair, review in zip(pairs, reviews, strict=True):
        record = dict(review)
        blockers = source_blockers(review)
        clean = review["status"] == "adjudicated" and not blockers
        record.update({
            "release_id": RELEASE_ID,
            "comparison_scope": ("abstract_measurement_predicate"
                                 if "abstract" in review["common_frame"].lower()
                                 else "selected_criterion_with_inherited_context"),
            "clean_development_comparison": clean,
            "development_only": True,
            "held_out_evaluation_eligible": False,
            "supervised_training_approved": False,
            "source_correction_codes": blockers,
            "freeze_reviewed": bool(review["label_changed"]
                                    or review["status"] == "needs_adjudication" or blockers),
        })
        for side in ["a", "b"]:
            trial_id = pair[side]["trial_id"]
            record[f"criterion_{side}_id"] = pair[side]["criterion_id"]
            record[f"trial_{side}_id"] = trial_id
            record[f"family_{side}_id"] = family["family_by_trial"][trial_id]
            record[f"split_group_{side}_id"] = family["split_group_by_trial"][trial_id]
        annotations.append(record)
        if clean:
            comparisons.append(record)
        else:
            number = review["pair_number"]
            require(number in QUEUE_ACTIONS, f"New queue case needs an explicit action: {number}")
            queue.append({**record, "required_resolution": QUEUE_ACTIONS[number]})
    counts = {"all_pairs": len(annotations), "resolved_relationships": sum(
                  r["status"] == "adjudicated" for r in annotations),
              "clean_development_comparisons": len(comparisons), "review_queue": len(queue),
              "unresolved_relationships": sum(r["label"] == "ambiguous" for r in annotations),
              "label_changes_from_initial_round": sum(r["label_changed"] for r in annotations),
              "labels_all_pairs": dict(sorted(Counter(r["label"] for r in annotations).items())),
              "labels_clean_comparisons": dict(sorted(Counter(r["label"]
                                                             for r in comparisons).items()))}
    require((counts["all_pairs"], counts["resolved_relationships"], len(comparisons), len(queue))
            == (75, 68, 67, 8), "Freeze counts differ from reviewed disposition")
    return {"annotations.jsonl": jsonl(annotations), "resolved_pairs.jsonl": jsonl(comparisons),
            "review_queue.jsonl": jsonl(queue)}, counts


def freeze(output: Path, root: Path = ROOT) -> dict:
    require(not output.exists(), "Release already exists; use --verify or a new --output version")
    pairs, reviews, family, inputs = load_inputs(root)
    payloads, counts = build_outputs(pairs, reviews, family)
    contract_hash = hashlib.sha256(guide_text(root).encode()).hexdigest()
    # Hash inputs before writing; no existing sampling, corpus or review file is modified.
    files = {p.as_posix(): {"sha256": sha(root / p), "bytes": (root / p).stat().st_size}
             for p in inputs}
    require(output.resolve().is_relative_to(root.resolve()), "Output must be inside the checkout")
    output.mkdir(parents=True)
    for name, content in payloads.items():
        path = output / name
        path.write_bytes(content)
        files[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": len(content)}
    manifest = {
        "schema_version": "pair-review-release-1", "release_id": RELEASE_ID,
        "status": "frozen_development_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_guide": "pair-guide-0.2", "annotation_contract_sha256": contract_hash,
        "construction_guide": "pair-guide-0.1", "corpus_version": "trial100_v1",
        "reviewer_id": "Codex", "freeze_script_sha256": sha(Path(__file__)),
        "counts": counts, "files": files,
        "final_pass_pair_numbers": [r["pair_number"] for r in reviews
                                    if r["label_changed"] or r["status"] == "needs_adjudication"
                                    or source_blockers(r)],
        "unresolved_pair_numbers": [r["pair_number"] for r in reviews if r["label"] == "ambiguous"],
        "source_role_quarantine_pair_numbers": [10],
        "pilot_development_trial_ids": sorted(read_json(root / PILOT / "manifest.json")
                                               ["reserved_development_trial_ids"]),
        "family_holdout_excluded_trial_ids": family["holdout_excluded_trial_ids"],
        "family_mapping_status": family["status"], "final_partitions_assigned": False,
        "supervised_training_approved": False, "held_out_evaluation_eligible": False,
        "limitations": [
            "This is a frozen development review, not independent clinical validation.",
            "Resolved comparisons retain their stated frame; abstract thresholds do not align dates.",
            "Owner validation is required before promoting labels to supervised training targets.",
            "Family mappings remain proposed; all pilot exposures remain excluded from holdout.",
            "Seven unresolved relations and the source-role conflict are outside resolved_pairs.jsonl.",
            "The original sampler is absent; this command verifies and freezes, not resamples.",
        ],
        "historical_packet_note": "The original manifest references review.md, now archived in "
                                  "docs/trialign_technical.tex. Its original checksum is retained.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify(output, root)


def verify(output: Path, root: Path = ROOT) -> dict:
    manifest = read_json(output / "manifest.json")
    require(manifest["release_id"] == RELEASE_ID
            and manifest["status"] == "frozen_development_review", "Unknown release")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest path")
        path = root / relative
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
                f"Release checksum mismatch: {name}")
    require(hashlib.sha256(guide_text(root).encode()).hexdigest()
            == manifest["annotation_contract_sha256"], "Frozen annotation contract changed")
    pairs, reviews, family, inputs = load_inputs(root)
    payloads, counts = build_outputs(pairs, reviews, family)
    require(manifest["counts"] == counts, "Manifest counts do not match reviewed decisions")
    expected_files = {p.as_posix() for p in inputs}
    expected_files.update((output / name).relative_to(root).as_posix() for name in payloads)
    require(set(manifest["files"]) == expected_files, "Manifest file coverage changed")
    for name, content in payloads.items():
        require((output / name).read_bytes() == content, f"Release disposition changed: {name}")
    require(manifest["supervised_training_approved"] is False
            and manifest["held_out_evaluation_eligible"] is False
            and manifest["final_partitions_assigned"] is False, "Development-use gates changed")
    require(manifest["family_holdout_excluded_trial_ids"] == family["holdout_excluded_trial_ids"],
            "Frozen holdout exclusions changed")
    return {"release_id": RELEASE_ID, "verified_files": len(manifest["files"]), **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true", help="Read-only integrity verification")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = verify(output) if args.verify else freeze(output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
