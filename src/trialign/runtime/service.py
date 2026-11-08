"""Trial alignment jobs, bounded execution, replay, cancellation and human review."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from trialign.agents.hooks import BudgetExceeded, RuntimeBudget
from trialign.agents.orchestrator import VERSION, AgentOrchestrator
from trialign.agents.provider import OpenAIProvider, local_configuration
from trialign.models import TfidfEncoder
from trialign.runtime.catalog import Catalog
from trialign.runtime.storage import atomic_write, read_run, run_path
from trialign.schemas import HumanDecision, RelationshipLabel
from trialign.tools.structured import canonical_entity


def now():
    return datetime.now(UTC).isoformat()


def alignment_candidates(a, b, *, top_k=2, floor=0.08, encoder=None):
    """Bidirectional candidate union; retrieval scores never supply a relation."""
    texts = [f"{r.entity} {r.source} {r.parent_statement}" for r in [*a, *b]]
    encoder = encoder or TfidfEncoder()
    if hasattr(encoder, "fit"):
        encoder.fit(texts)
    vectors = encoder.encode(texts)
    similarities = vectors[: len(a)] @ vectors[len(a) :].T
    similarities = (
        similarities.toarray() if hasattr(similarities, "toarray") else np.asarray(similarities)
    )
    candidates = {}
    for direction in ["a", "b"]:
        queries, corpus = (a, b) if direction == "a" else (b, a)
        for qi, query in enumerate(queries):
            scores = similarities[qi] if direction == "a" else similarities[:, qi]
            ranked = sorted(
                range(len(corpus)), key=lambda j: (-float(scores[j]), corpus[j].criterion_id)
            )
            # Entity agreement helps prioritize tools, without becoming a match label.
            ranked.sort(
                key=lambda j: canonical_entity(query.entity) != canonical_entity(corpus[j].entity)
            )
            taken = 0
            for j in ranked:
                entity_match = canonical_entity(query.entity) == canonical_entity(corpus[j].entity)
                if float(scores[j]) < floor and not entity_match:
                    continue
                left, right = (query, corpus[j]) if direction == "a" else (corpus[j], query)
                candidates[(left.criterion_id, right.criterion_id)] = {
                    "a": left.criterion_id,
                    "b": right.criterion_id,
                    "retrieval_score": float(scores[j]),
                    "entity_match": entity_match,
                }
                taken += 1
                if taken >= top_k:
                    break
    return sorted(
        candidates.values(),
        key=lambda x: (not x["entity_match"], -x["retrieval_score"], x["a"], x["b"]),
    )


class ComparisonService:
    def __init__(self, root: Path, *, directory: Path | None = None, encoder_factory=None):
        self.root = root
        self.encoder_factory = encoder_factory or TfidfEncoder
        self.directory = directory or root / "artifacts/demo/runs"
        self.import_directory = self.directory.parent / "imports"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.catalog = Catalog(root)
        digest = hashlib.sha256()
        for directory in ["agents", "tools", "runtime", "web"]:
            for path in sorted((root / "src/trialign" / directory).rglob("*")):
                if path.is_file() and path.suffix in {".py", ".yaml", ".html", ".css", ".js"}:
                    digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                    digest.update(path.read_bytes())
        self.source_fingerprint = digest.hexdigest()
        self.lock = threading.RLock()
        self.cancel_events = {}
        for path in sorted(self.import_directory.glob("import-*.json")):
            payload = json.loads(path.read_text())
            from trialign.schemas import CriterionRecord

            self.catalog.add(
                [CriterionRecord.model_validate(r) for r in payload["records"]],
                origin=payload["origin"],
            )
        # A process restart never silently resumes a model call or loses saved work.
        for path in self.directory.glob("run-*.json"):
            job = json.loads(path.read_text())
            owner_alive = False
            if job.get("owner_pid"):
                try:
                    os.kill(job["owner_pid"], 0)
                    owner_alive = True
                except (ProcessLookupError, PermissionError):
                    pass
            if job["status"] in {"queued", "running", "cancelling"} and not owner_alive:
                job["status"] = "interrupted"
                job["message"] = "Server restarted; completed pair results were preserved."
                atomic_write(path, job)

    def capabilities(self):
        config = local_configuration(self.root)
        planner = AgentOrchestrator().planner
        return {
            "version": VERSION,
            "llm_configured": bool(config["api_key"]),
            "model": config["model"],
            "configuration_origin": config["origin"],
            "encoder": "TF-IDF baseline (temporary encoder; no fine-tuned model)",
            "skills": [{"id": key, **value} for key, value in planner.skills.skills.items()],
            "skill_version": planner.skills.version,
            "tools": list(AgentOrchestrator().tools.names),
            "hooks": [
                "input_validation",
                "before_tool",
                "after_tool",
                "evidence_validation",
                "budget_stop",
                "before_finalize",
            ],
            "labels": [label.value for label in RelationshipLabel],
        }

    def import_file(self, filename: str, content: bytes):
        with self.lock:
            before = set(self.catalog.records)
            result = self.catalog.import_file(filename, content)
            added = [
                r.model_dump(mode="json")
                for key, r in self.catalog.records.items()
                if key not in before
            ]
            if added:
                atomic_write(
                    self.import_directory / ("import-" + result["sha256"] + ".json"),
                    {"origin": "import:" + result["sha256"][:12], "records": added},
                )
            return {**result, "new_records": len(added), "trials_available": self.catalog.trials()}

    def create(self, request: dict, *, synchronous=False, provider_override=None):
        trial_a, trial_b = str(request.get("trial_a", "")), str(request.get("trial_b", ""))
        if not trial_a or trial_a == trial_b:
            raise ValueError("Select two different trials.")
        a, b = self.catalog.for_trial(trial_a), self.catalog.for_trial(trial_b)
        max_pairs = int(request.get("max_pairs", 16))
        calls = int(request.get("max_model_calls", 8))
        seconds = int(request.get("max_seconds", 150))
        if not 1 <= max_pairs <= 80 or not 0 <= calls <= 24 or not 10 <= seconds <= 300:
            raise ValueError("Run limits are outside the permitted range.")
        with self.lock:
            if any(r["status"] in {"queued", "running", "cancelling"} for r in self.list_runs()):
                raise ValueError("A comparison is already running. Finish or cancel it first.")
            candidates = alignment_candidates(a, b, encoder=self.encoder_factory())
            run_id = "run-" + uuid4().hex
            inputs = [r.model_dump(mode="json") for r in [*a, *b]]
            job = {
                "schema_version": "comparison-run-1",
                "run_id": run_id,
                "created_at": now(),
                "updated_at": now(),
                "status": "queued",
                "owner_pid": os.getpid(),
                "trial_a": trial_a,
                "trial_b": trial_b,
                "title_a": self.catalog.titles.get(trial_a, trial_a),
                "title_b": self.catalog.titles.get(trial_b, trial_b),
                "criteria_a": [r.model_dump(mode="json") for r in a],
                "criteria_b": [r.model_dump(mode="json") for r in b],
                "input_sha256": hashlib.sha256(
                    json.dumps(inputs, sort_keys=True).encode()
                ).hexdigest(),
                "runtime_version": VERSION,
                "runtime_source_sha256": self.source_fingerprint,
                "encoder": "TF-IDF baseline; trained encoder not available",
                "max_pairs": max_pairs,
                "max_model_calls": calls,
                "max_seconds": seconds,
                "use_llm": bool(request.get("use_llm", True)),
                "candidates": candidates,
                "results": [],
                "reviews": {},
                "events": [],
                "revision": 0,
                "message": "Comparison queued",
                "model_calls": 0,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
            atomic_write(run_path(self.directory, run_id), job)
            self.cancel_events[run_id] = threading.Event()
        if synchronous:
            self._execute(run_id, provider_override=provider_override)
        else:
            threading.Thread(target=self._execute, args=(run_id,), daemon=True).start()
        return self.get(run_id)

    def _execute(self, run_id, *, provider_override=None):
        with self.lock:
            job = read_run(self.directory, run_id)
            job["status"] = "running"
            self._save(job)
        budget = RuntimeBudget(
            max_model_calls=job["max_model_calls"],
            max_seconds=job["max_seconds"],
            max_tool_calls=job["max_pairs"] * 3,
            cancelled=self.cancel_events[run_id],
        )
        provider = provider_override
        try:
            configuration = local_configuration(self.root)
            if provider is None and job["use_llm"] and configuration["api_key"]:
                provider = OpenAIProvider(configuration)
            runner = AgentOrchestrator(provider if job["use_llm"] else None)
            job["model"] = getattr(provider, "model", None) if job["use_llm"] else None
            job["skill_version"] = runner.planner.skills.version
            job["skill_sha256"] = runner.planner.skills.digest
            records = {r["criterion_id"]: r for r in [*job["criteria_a"], *job["criteria_b"]]}
            from trialign.schemas import CriterionRecord

            def event_hook(event):
                with self.lock:
                    job["events"].append(event)
                    if budget.cancelled.is_set():
                        job["status"] = "cancelling"
                    job["message"] = event["stage"].replace(".", " · ")
                    job["model_calls"] = budget.model_calls
                    self._save(job)

            for candidate in job["candidates"][: job["max_pairs"]]:
                budget.check()
                a = CriterionRecord.model_validate(records[candidate["a"]])
                b = CriterionRecord.model_validate(records[candidate["b"]])
                result = runner.compare(a, b, budget=budget, on_event=event_hook, run_id=run_id)
                with self.lock:
                    job["results"].append(
                        {
                            **result.model_dump(mode="json"),
                            "retrieval_score": candidate["retrieval_score"],
                        }
                    )
                    self._save(job)
            budget.check()
            job["status"] = "completed"
            job["message"] = "Comparison complete; review the proposed relationships."
        except BudgetExceeded as exc:
            job["status"] = "cancelled" if budget.cancelled.is_set() else "limited"
            job["message"] = str(exc)
        except Exception as exc:
            job["status"] = "failed"
            job["message"] = (
                f"Run could not finish ({type(exc).__name__}); saved results remain available."
            )
        finally:
            with self.lock:
                job["model_calls"] = budget.model_calls
                job["tool_calls"] = budget.tool_calls
                job["usage"] = getattr(provider, "usage", job["usage"])
                job["resolved_models"] = getattr(provider, "resolved_models", [])
                job["finished_at"] = now()
                self._save(job)
            if isinstance(provider, OpenAIProvider):
                provider.client.close()

    def _save(self, job):
        job["updated_at"] = now()
        atomic_write(run_path(self.directory, job["run_id"]), job)

    def get(self, run_id):
        with self.lock:
            job = read_run(self.directory, run_id)
        results = job["results"]
        candidate_a = {c["a"] for c in job["candidates"]}
        candidate_b = {c["b"] for c in job["candidates"]}
        examined_a = {r["criterion_a_id"] for r in results}
        examined_b = {r["criterion_b_id"] for r in results}
        supported = [
            r
            for r in results
            if not r["should_escalate"]
            and r["label"] not in {"unrelated", "related_not_comparable", "ambiguous"}
        ]
        matched_a = {r["criterion_a_id"] for r in supported}
        matched_b = {r["criterion_b_id"] for r in supported}
        job["summary"] = {
            "candidate_pairs": len(job["candidates"]),
            "compared_pairs": len(results),
            "supported_pairs": len(supported),
            "needs_review": sum(r["should_escalate"] for r in results),
            "reviewed_pairs": len(job["reviews"]),
            "labels": dict(Counter(r["label"] for r in results if r["label"])),
        }
        for rows, candidates, examined, matched in [
            (job["criteria_a"], candidate_a, examined_a, matched_a),
            (job["criteria_b"], candidate_b, examined_b, matched_b),
        ]:
            for row in rows:
                identifier = row["criterion_id"]
                row["alignment_status"] = (
                    "supported_candidate"
                    if identifier in matched
                    else "needs_review"
                    if identifier in examined
                    else "not_examined"
                    if identifier in candidates
                    else "no_candidate_found"
                )
        return job

    def list_runs(self):
        results = []
        for path in sorted(self.directory.glob("run-*.json"), reverse=True):
            job = json.loads(path.read_text())
            results.append(
                {
                    k: job.get(k)
                    for k in [
                        "run_id",
                        "created_at",
                        "status",
                        "trial_a",
                        "trial_b",
                        "title_a",
                        "title_b",
                    ]
                }
            )
        return sorted(results, key=lambda x: x["created_at"], reverse=True)

    def cancel(self, run_id):
        with self.lock:
            job = read_run(self.directory, run_id)
            if run_id in self.cancel_events and job["status"] in {
                "queued",
                "running",
                "cancelling",
            }:
                self.cancel_events[run_id].set()
                job["status"] = "cancelling"
                self._save(job)
        return self.get(run_id)

    def review(self, run_id, body):
        with self.lock:
            job = read_run(self.directory, run_id)
            if job["status"] in {"queued", "running", "cancelling"}:
                raise ValueError("Wait for the run to finish before saving review decisions.")
            if int(body.get("expected_revision", -1)) != job["revision"]:
                raise ValueError("This review changed in another window; reload before saving.")
            pair_id = str(body.get("pair_id", ""))
            if pair_id not in {r["pair_id"] for r in job["results"]}:
                raise ValueError("Pair is not part of this saved run.")
            reviewer = str(body.get("reviewer_id", "")).strip()
            rationale = str(body.get("rationale", "")).strip()
            if not reviewer or not rationale:
                raise ValueError("Provide reviewer name and rationale.")
            decision = HumanDecision(
                decision_id="review-" + uuid4().hex,
                pair_id=pair_id,
                reviewer_id=reviewer,
                status=str(body.get("status", "accepted")),
                final_label=RelationshipLabel(body["final_label"]),
                rationale=rationale,
            )
            history = job["reviews"].setdefault(pair_id, [])
            history.append(decision.model_dump(mode="json"))
            job["revision"] += 1
            self._save(job)
        return self.get(run_id)
