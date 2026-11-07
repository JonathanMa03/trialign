"""Inventory trial-family evidence without changing labels, corpus or splits.

Run from the repository root. Network is used only with --fetch-registry;
subsequent runs can reuse an explicitly supplied registry snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trialign.data.corpus import load_parquet_corpus
from sklearn.feature_extraction.text import TfidfVectorizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    # Keep numbers and mathematical operators. Formatting-only deduplication.
    return re.sub(r"\s+", " ", text.replace("\\", "").casefold()).strip()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def fetch(ids):
    url = "https://clinicaltrials.gov/api/v2/studies?" + urlencode({
        "filter.ids": ",".join(ids), "pageSize": 100, "format": "json",
    })
    with urlopen(Request(url, headers={"User-Agent": "TriAlign family audit/0.1"}), timeout=60) as response:
        payload = json.load(response)
    studies = payload.get("studies", [])
    found = {x["protocolSection"]["identificationModule"]["nctId"] for x in studies}
    if found != set(ids):
        raise ValueError(f"Registry coverage mismatch: missing={set(ids)-found}, extra={found-set(ids)}")
    return {x["protocolSection"]["identificationModule"]["nctId"]: {
        # Limit snapshot to relevant public study information, omitting contacts.
        "protocolSection": {k: v for k, v in x["protocolSection"].items() if k in {
            "identificationModule", "statusModule", "sponsorCollaboratorsModule",
            "descriptionModule", "conditionsModule", "designModule", "referencesModule",
        }},
        "source_url": "https://clinicaltrials.gov/api/v2/studies/" + x["protocolSection"]["identificationModule"]["nctId"],
    } for x in studies}


def main(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.corpus.with_suffix(".parquet.manifest.json").read_text())
    if sha(args.corpus) != manifest["parquet_sha256"]:
        raise ValueError("Frozen corpus hash mismatch")
    records = load_parquet_corpus(args.corpus)
    by_trial = defaultdict(list)
    for row in records:
        by_trial[row.trial_id].append(row)
    ids = sorted(by_trial)
    cohort = json.loads(args.cohort.read_text())
    cohort_by_id = {x["trial_id"]: x for x in cohort["trials"]}
    if set(ids) != set(cohort_by_id):
        raise ValueError("Cohort and frozen corpus trial IDs differ")
    snapshot_path = output / "registry_snapshot.json"
    if args.fetch_registry:
        if snapshot_path.exists():
            raise FileExistsError("Refusing to overwrite a registry snapshot")
        write(snapshot_path, {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "studies": fetch(ids)})
    else:
        snapshot_path = args.registry_snapshot or snapshot_path
    snapshot = json.loads(snapshot_path.read_text())
    if set(snapshot["studies"]) != set(ids):
        raise ValueError("Registry snapshot is incomplete")
    if args.external_ids:
        external_path = output / "external_registry_snapshot.json"
        if external_path.exists():
            raise FileExistsError("Refusing to overwrite external registry snapshot")
        write(external_path, {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "studies": fetch(sorted(args.external_ids.split(",")))})
    trial_records, parent_trials, parent_text = {}, defaultdict(set), {}
    for trial_id in ids:
        rows = by_trial[trial_id]
        job_ids = sorted({r.job_id for r in rows})
        jobs = []
        for job_id in job_ids:
            job = json.loads((args.jobs / f"{job_id}.json").read_text())
            if job["trial_id"] != trial_id:
                raise ValueError("Archived job trial ID mismatch")
            jobs.append(job)
        protocol_hashes = sorted({r.protocol_hash for r in rows})
        if set(protocol_hashes) != {j["protocol_hash"] for j in jobs}:
            raise ValueError("Archived source and frozen corpus hashes differ")
        text = "\n".join(j["protocol_text"] for j in jobs)
        meta = snapshot["studies"][trial_id]["protocolSection"]
        identification = meta["identificationModule"]
        source_versions = sorted({r.source_version for r in rows})
        paragraphs = {}
        for r in rows:
            key = r.criterion_type + ":" + normalized(r.parent_statement)
            paragraphs[key] = r.parent_statement
            parent_trials[key].add(trial_id)
            parent_text[key] = r.parent_statement
        frozen_refs = sorted(set(re.findall(r"NCT\d{8}", text, re.I)) - {trial_id})
        description = meta.get("descriptionModule", {})
        metadata_text = "\n".join(str(v) for v in description.values())
        registry_refs = sorted(set(re.findall(r"NCT\d{8}", metadata_text, re.I)) - {trial_id})
        trial_records[trial_id] = {
            "trial_id": trial_id, "cohort": cohort_by_id[trial_id]["cohort"],
            "cohort_title": cohort_by_id[trial_id]["title"], "criteria": len(rows),
            "frozen_source_versions": source_versions, "protocol_hashes": protocol_hashes,
            "frozen_protocol_text": text, "archived_job_ids": job_ids,
            "archived_job_sha256": [sha(args.jobs / f"{j}.json") for j in job_ids],
            "paragraphs": paragraphs, "identification": identification,
            "sponsors": meta.get("sponsorCollaboratorsModule", {}),
            "conditions": meta.get("conditionsModule", {}),
            "description": description, "design": meta.get("designModule", {}),
            "registry_last_update": meta.get("statusModule", {}).get("lastUpdatePostDateStruct", {}).get("date"),
            "frozen_nct_references": frozen_refs, "registry_nct_references": registry_refs,
            "source_url": snapshot["studies"][trial_id]["source_url"],
        }
    # Rare exact source-parent sharing catches copied templates, not just shared entities.
    rare = {p for p, trials in parent_trials.items() if 2 <= len(trials) <= 5 and len(p.split()) >= 12}
    texts = [trial_records[t]["frozen_protocol_text"] for t in ids]
    vectors = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
    cosines = (vectors @ vectors.T).toarray()
    shingles = []
    for text in texts:
        words = re.findall(r"[a-z0-9]+|>=|<=|>|<|=", text.casefold())
        shingles.append({tuple(words[k:k+8]) for k in range(len(words)-7)})
    frequency = Counter(s for document in shingles for s in document)
    rare_shingles = {s for s, count in frequency.items() if 2 <= count <= 5}
    edges, all_scores = [], []
    for i, j in itertools.combinations(range(len(ids)), 2):
        a, b = trial_records[ids[i]], trial_records[ids[j]]
        shared = sorted(set(a["paragraphs"]) & set(b["paragraphs"]) & rare)
        reasons = []
        if set(a["protocol_hashes"]) & set(b["protocol_hashes"]): reasons.append("shared_frozen_protocol_hash")
        if normalized(a["frozen_protocol_text"]) == normalized(b["frozen_protocol_text"]): reasons.append("identical_normalized_eligibility_text")
        ai, bi = a["identification"], b["identification"]
        asponsor = normalized(ai.get("organization", {}).get("fullName", ""))
        bsponsor = normalized(bi.get("organization", {}).get("fullName", ""))
        aprotocol = normalized(ai.get("orgStudyIdInfo", {}).get("id", ""))
        bprotocol = normalized(bi.get("orgStudyIdInfo", {}).get("id", ""))
        if aprotocol and aprotocol == bprotocol and asponsor and asponsor == bsponsor:
            reasons.append("same_org_protocol_id_and_organization")
        aliases = set(ai.get("nctIdAliases", [])) | set(bi.get("nctIdAliases", []))
        if a["trial_id"] in aliases or b["trial_id"] in aliases: reasons.append("registry_nct_alias")
        if b["trial_id"] in a["frozen_nct_references"] or a["trial_id"] in b["frozen_nct_references"]:
            reasons.append("cross_reference_in_frozen_eligibility")
        if b["trial_id"] in a["registry_nct_references"] or a["trial_id"] in b["registry_nct_references"]:
            reasons.append("cross_reference_in_registry_description")
        if len(shared) >= 3: reasons.append("three_or_more_rare_shared_parent_statements")
        if cosines[i,j] >= .60: reasons.append("eligibility_tfidf_cosine_at_least_0.60")
        shared_shingles = shingles[i] & shingles[j] & rare_shingles
        overlap = len(shared_shingles) / max(1, min(len(shingles[i]), len(shingles[j])))
        if len(shared_shingles) >= 20 and overlap >= .20:
            reasons.append("rare_8gram_coverage_at_least_0.20")
        score = {"trial_a": a["trial_id"], "trial_b": b["trial_id"], "cosine": round(float(cosines[i,j]), 6), "rare_shared_parents": len(shared),
            "shared_rare_8grams": len(shared_shingles), "rare_8gram_coverage": round(overlap, 6)}
        all_scores.append(score)
        if reasons:
            edges.append({**score, "candidate_id": "edge-" + a["trial_id"] + "-" + b["trial_id"],
                "title_a": ai.get("briefTitle"), "title_b": bi.get("briefTitle"),
                "org_protocol_a": aprotocol, "org_protocol_b": bprotocol,
                "organization_a": asponsor, "organization_b": bsponsor,
                "reasons": reasons, "shared_parent_evidence": [parent_text[p] for p in shared],
                "status": "pending_evidence_review", "family_link": None,
            })
    # Rank a review queue even when fixed thresholds find no strong candidates.
    # Selection for review must not be mistaken for family evidence.
    ranked = sorted(all_scores, key=lambda x: -x["cosine"])
    review_queue = ranked[:20] + sorted(all_scores, key=lambda x: -x["rare_8gram_coverage"])[:5]
    for score in review_queue:
        if not any(e["trial_a"] == score["trial_a"] and e["trial_b"] == score["trial_b"] for e in edges):
            a, b = trial_records[score["trial_a"]], trial_records[score["trial_b"]]
            edges.append({**score, "candidate_id": "edge-" + score["trial_a"] + "-" + score["trial_b"],
                "title_a": a["identification"].get("briefTitle"), "title_b": b["identification"].get("briefTitle"),
                "reasons": ["ranked_similarity_review_only"], "shared_parent_evidence": [],
                "status": "pending_evidence_review", "family_link": None})
    secondary = defaultdict(list)
    for trial_id, item in trial_records.items():
        for identifier in item["identification"].get("secondaryIdInfos", []):
            key = normalized(identifier.get("id", ""))
            if key:
                secondary[key].append({"trial_id": trial_id, **identifier})
    write(output / "shared_secondary_identifiers.json", {k: v for k, v in secondary.items() if len({x["trial_id"] for x in v}) > 1})
    edges.sort(key=lambda e: (-e["rare_shared_parents"], -e["cosine"], e["candidate_id"]))
    write(output / "trial_inventory.json", trial_records)
    write(output / "candidate_edges.json", edges)
    write(output / "all_pair_scores.json", sorted(all_scores, key=lambda x: -x["cosine"]))
    write(output / "screening_manifest.json", {
        "corpus_sha256": sha(args.corpus), "registry_snapshot_sha256": sha(snapshot_path),
        "corpus_manifest_sha256": sha(args.corpus.with_suffix(".parquet.manifest.json")),
        "cohort_manifest_sha256": sha(args.cohort),
        "trial_inventory_sha256": sha(output / "trial_inventory.json"),
        "registry_retrieved_at_utc": snapshot["retrieved_at_utc"], "generator_sha256": sha(__file__),
        "trials": len(ids), "criteria": len(records), "cohort_counts": dict(Counter(x["cohort"] for x in trial_records.values())),
        "all_trial_pairs_screened": len(all_scores), "candidate_edges": len(edges),
        "rare_paragraph_min_tokens": 12, "rare_paragraph_document_frequency": [2,5],
        "candidate_thresholds": {"rare_shared_parents": 3, "whole_text_cosine": .60, "rare_8gram_coverage": .20, "shared_rare_8grams": 20},
        "family_links_are_not_inferred_from_similarity_alone": True,
        "split_assignments_created": False, "registry_metadata_may_postdate_frozen_eligibility": True,
    })
    print(json.dumps({"trials": len(ids), "screened_pairs": len(all_scores), "candidate_edges": len(edges), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/trial100_v1.parquet"))
    parser.add_argument("--cohort", type=Path, default=Path("../modern-autocrit/config/trial_cohort_100.json"))
    parser.add_argument("--jobs", type=Path, default=Path("../modern-autocrit/runtime/criteria/jobs"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/family_audit_v1"))
    parser.add_argument("--fetch-registry", action="store_true")
    parser.add_argument("--registry-snapshot", type=Path)
    parser.add_argument("--external-ids", help="Comma-separated external parent/follow-up NCT IDs to verify")
    main(parser.parse_args())
