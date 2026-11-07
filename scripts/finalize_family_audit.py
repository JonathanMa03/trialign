"""Turn a reviewed evidence ledger into complete family/split-group mappings.

This does not assign train/validation/test partitions or touch pair labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Components:
    def __init__(self, ids):
        self.parent = {x: x for x in ids}

    def root(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def finalize(audit, decisions_path, pilot):
    decisions = json.loads(decisions_path.read_text())
    if sha(audit / "candidate_edges.json") != decisions["candidate_edges_sha256"]:
        raise ValueError("Candidate review ledger is stale")
    if sha(audit / "registry_snapshot.json") != decisions["registry_snapshot_sha256"]:
        raise ValueError("Registry evidence changed after review")
    if sha(audit / "external_registry_snapshot.json") != decisions["external_registry_snapshot_sha256"]:
        raise ValueError("External registry evidence changed after review")
    screening = json.loads((audit / "screening_manifest.json").read_text())
    if sha(audit / "trial_inventory.json") != screening["trial_inventory_sha256"]:
        raise ValueError("Trial inventory changed after screening")
    inventory = json.loads((audit / "trial_inventory.json").read_text())
    registry_ids = set(json.loads((audit / "registry_snapshot.json").read_text())["studies"])
    if set(inventory) != registry_ids or len(inventory) != screening["trials"]:
        raise ValueError("Incomplete trial inventory")
    external = json.loads((audit / "external_registry_snapshot.json").read_text())["studies"]
    candidates = json.loads((audit / "candidate_edges.json").read_text())
    reviewed = {x["candidate_id"]: x for x in decisions["in_cohort_reviews"]}
    if (len(reviewed) != len(candidates) or len(reviewed) != len(decisions["in_cohort_reviews"])
            or set(reviewed) != {x["candidate_id"] for x in candidates}):
        raise ValueError("Missing or duplicate candidate decisions")
    if any(x["decision"] != "no_merge_evidence" for x in reviewed.values()):
        raise ValueError("Explicitly implement newly accepted cohort edges before finalizing")
    nodes = set(inventory) | set(external)
    families, split_groups = Components(nodes), Components(nodes)
    for edge in decisions["external_links"]:
        if edge["status"] != "source_supported":
            raise ValueError("Unreviewed external link")
        if edge["a"] not in nodes or edge["b"] not in nodes:
            raise ValueError("External link lacks source inventory")
        if edge["clinical_family_link"]:
            families.join(edge["a"], edge["b"])
        if edge["split_link"]:
            split_groups.join(edge["a"], edge["b"])
    pilot_manifest = json.loads(pilot.read_text())
    reserved = set(pilot_manifest["reserved_development_trial_ids"])
    if not reserved <= set(inventory):
        raise ValueError("Pilot reservations absent from inventory")
    quarantined = {x["trial_id"] for x in decisions["unresolved_links"]}
    if not quarantined <= set(inventory):
        raise ValueError("Quarantine missing from inventory")
    reserved_groups = {split_groups.root(t) for t in reserved}
    quarantine_groups = {split_groups.root(t) for t in quarantined}
    rows = []
    for t in sorted(nodes):
        group = split_groups.root(t)
        reasons = []
        if group in reserved_groups:
            reasons.append("pilot_development_exposure_or_linked_family")
        if group in quarantine_groups:
            reasons.append("unresolved_parentage_quarantine")
        rows.append({"trial_id": t, "in_current_corpus": t in inventory,
            "family_id": "family-" + families.root(t), "split_group_id": "group-" + group,
            "holdout_excluded": bool(reasons), "exclusion_reasons": reasons,
            "audit_status": "proposed_after_evidence_screen", "final_split": None})
    current = [x for x in rows if x["in_current_corpus"]]
    payload = {
        "audit_version": decisions["audit_version"], "status": decisions["status"],
        "corpus_sha256": json.loads((audit / "screening_manifest.json").read_text())["corpus_sha256"],
        "decision_ledger_sha256": sha(decisions_path), "pilot_manifest_sha256": sha(pilot),
        "generator_sha256": sha(__file__), "rows": rows,
        "family_by_trial": {x["trial_id"]: x["family_id"] for x in current},
        "split_group_by_trial": {x["trial_id"]: x["split_group_id"] for x in current},
        "holdout_excluded_trial_ids": [x["trial_id"] for x in current if x["holdout_excluded"]],
        "unresolved_trial_ids": sorted(quarantined),
        "counts": {"corpus_trials": len(current), "external_nodes": len(external),
            "clinical_families_in_corpus": len({x["family_id"] for x in current}),
            "split_groups_in_corpus": len({x["split_group_id"] for x in current}),
            "holdout_excluded_corpus_trials": sum(x["holdout_excluded"] for x in current),
            "holdout_excluded_corpus_criteria": sum(inventory[x["trial_id"]]["criteria"] for x in current if x["holdout_excluded"]),
            "cohort_composition": dict(Counter(x["cohort"] for x in inventory.values()))},
        "limitations": ["No confirmed in-cohort family link found is not proof that every pair is independent.",
            "Registry metadata may postdate frozen eligibility text.",
            "Complete protocol PDFs and unnamed external parents were not exhaustively audited.",
            "Use split groups and exclusions together; existing generic split CLI does not enforce this artifact.",
            "Do not assign partitions until the annotation guide/pilot review and split policy are frozen."]}
    path = audit / "family_mapping.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"]))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("data/processed/family_audit_v1"))
    parser.add_argument("--decisions", type=Path, default=Path("configs/family_audit_decisions_v1.json"))
    parser.add_argument("--pilot", type=Path, default=Path("data/processed/pair_pilot_v1/manifest.json"))
    args = parser.parse_args()
    finalize(args.audit, args.decisions, args.pilot)
