"""Trial alignment jobs, bounded execution, replay, cancellation and human review."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import threading
import time
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
from trialign.runtime.metrics import estimate_api_cost, pricing_settings
from trialign.runtime.storage import (
    atomic_write,
    migrate_legacy_runs,
    read_run,
    run_path,
    saved_run_files,
    trial_label,
    write_run,
)
from trialign.schemas import HumanDecision, RelationshipLabel
from trialign.tools.correspondence import VERSION as MATCHING_VERSION
from trialign.tools.correspondence import correspondence


def now():
    return datetime.now(UTC).isoformat()


def alignment_candidates(a, b, *, encoder=None):
    """Maximum-weight one-to-one assignment with explicit unmatched options.

    Shared parent prose never supplies a retrieval feature or a concept mapping.
    Scores rank eligible counterparts; they never supply a relationship label.
    """
    from scipy.optimize import linear_sum_assignment

    if not a or not b:
        return []
    a = sorted(a, key=lambda row: row.criterion_id)
    b = sorted(b, key=lambda row: row.criterion_id)
    if len({r.criterion_id for r in a + b}) != len(a) + len(b):
        raise ValueError("Comparison inputs must have unique criterion IDs.")
    texts = [f"{r.entity} {r.source}" for r in [*a, *b]]
    encoder = encoder or TfidfEncoder()
    if hasattr(encoder, "fit"):
        encoder.fit(texts)
    vectors = encoder.encode(texts)
    similarities = vectors[: len(a)] @ vectors[len(a) :].T
    similarities = (
        similarities.toarray() if hasattr(similarities, "toarray") else np.asarray(similarities)
    )
    # Dummy columns permit every A to remain unmatched, at zero weight.
    weights = np.zeros((len(a), len(b) + len(a)))
    weights[:, : len(b)] = -1
    reasons = {}
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            allowed, reason = correspondence(left, right)
            if allowed:
                score = float(similarities[i, j])
                weights[i, j] = 1 + max(0, min(1, score))
                reasons[i, j] = reason
    rows, columns = linear_sum_assignment(weights, maximize=True)
    candidates = []
    for i, j in zip(rows, columns, strict=True):
        if j >= len(b) or (i, j) not in reasons:
            continue
        candidates.append(
            {
                "a": a[i].criterion_id,
                "b": b[j].criterion_id,
                "retrieval_score": float(similarities[i, j]),
                "entity_match": True,
                "matching_reason": reasons[i, j],
            }
        )
    return sorted(candidates, key=lambda x: (-x["retrieval_score"], x["a"], x["b"]))


class ComparisonService:
    def __init__(self, root: Path, *, directory: Path | None = None, encoder_factory=None):
        self.root = root
        self.encoder_factory = encoder_factory or TfidfEncoder
        self.directory = directory or root / "runs"
        self.import_directory = self.directory / "imports"
        self.upload_directory = self.directory / "uploads"
        self.directory.mkdir(parents=True, exist_ok=True)
        if directory is None:
            migrate_legacy_runs(root, self.directory)
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
        for _path, job in saved_run_files(self.directory):
            job.setdefault("note", "")
            job.setdefault("display_status", "")
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
            if job["status"] not in {"queued", "running", "cancelling"} or not owner_alive:
                write_run(self.directory, job)

    def capabilities(self):
        config = local_configuration(self.root)
        planner = AgentOrchestrator().planner
        return {
            "version": VERSION,
            "llm_configured": bool(config["api_key"]),
            "model": config["model"],
            "critic_model": config["critic_model"],
            "configuration_origin": config["origin"],
            "encoder": "TF-IDF baseline (temporary encoder; no fine-tuned model)",
            "skills": [{"id": key, **value} for key, value in planner.skills.skills.items()],
            "skill_version": planner.skills.version,
            "tools": list(AgentOrchestrator().tools.names),
            "matching_policy": MATCHING_VERSION,
            "hooks": [
                "clinical counterpart gate",
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

    def import_files(self, files, **options):
        from trialign.runtime.uploads import parse_upload

        if sum(len(f["content"]) for f in files) > 15_000_000:
            raise ValueError("Upload limit is 15 MB total per batch.")
        payload = parse_upload(files, **options)
        with self.lock:
            atomic_write(self.upload_directory / (payload["dataset_id"] + ".json"), payload)
        return self.upload(payload["dataset_id"])

    def upload(self, dataset_id):
        if not re.fullmatch(r"upload-[a-f0-9]{32}", dataset_id):
            raise ValueError("Invalid uploaded dataset identifier.")
        path = self.upload_directory / (dataset_id + ".json")
        if not path.is_file():
            raise ValueError("Uploaded dataset was not found; upload the files again.")
        return json.loads(path.read_text())

    def create(self, request: dict, *, synchronous=False, provider_override=None):
        trial_a, trial_b = str(request.get("trial_a", "")), str(request.get("trial_b", ""))
        if not trial_a or trial_a == trial_b:
            raise ValueError("Select two different trials.")
        dataset_id = str(request.get("dataset_id") or "")
        dataset = self.upload(dataset_id) if dataset_id else None
        if dataset:
            from trialign.schemas import CriterionRecord

            a = [
                CriterionRecord.model_validate(r)
                for r in dataset["records"]
                if r["trial_id"] == trial_a
            ]
            b = [
                CriterionRecord.model_validate(r)
                for r in dataset["records"]
                if r["trial_id"] == trial_b
            ]
            if not a or not b:
                raise ValueError("Both selected trials must belong to the uploaded dataset.")
            titles = {t["trial_id"]: t["title"] for t in dataset["trials"]}
        else:
            a, b = self.catalog.for_trial(trial_a), self.catalog.for_trial(trial_b)
            titles = self.catalog.titles
        configuration = local_configuration(self.root)
        judge_model = request.get("judge_model", configuration["model"])
        critic_model = request.get("critic_model", configuration["critic_model"])
        for model in [judge_model, critic_model]:
            if not isinstance(model, str) or not model.strip() or len(model) > 200 or "\n" in model:
                raise ValueError("Provide a valid judge and critic model name.")
        if request.get("retrieval_model", "tfidf") != "tfidf":
            raise ValueError("This runtime currently supports TF-IDF retrieval only.")
        pricing = pricing_settings(request.get("pricing"))
        requested_pairs = int(request.get("max_pairs", 0))
        max_pairs = requested_pairs
        raw_calls = request.get("max_model_calls", 200)
        calls = None if raw_calls is None else int(raw_calls)
        seconds = int(request.get("max_seconds", 900))
        if not 0 <= max_pairs <= 5000:
            raise ValueError("Candidate-pair limit must be between 0 (all retrieved) and 5,000.")
        if calls is not None and calls < 0:
            raise ValueError("Call budget must be nonnegative or Unlimited in Provider settings.")
        if not 10 <= seconds <= 3600:
            raise ValueError("Time limit must be between 10 and 3600 seconds.")
        with self.lock:
            if any(r["status"] in {"queued", "running", "cancelling"} for r in self.list_runs()):
                raise ValueError("A comparison is already running. Finish or cancel it first.")
            retrieval_started = time.perf_counter()
            retrieval_cpu_started = time.thread_time()
            candidates = alignment_candidates(a, b, encoder=self.encoder_factory())
            retrieval_seconds = time.perf_counter() - retrieval_started
            retrieval_cpu_seconds = time.thread_time() - retrieval_cpu_started
            max_pairs = min(5000, len(candidates)) if requested_pairs == 0 else requested_pairs
            max_pairs = max(1, max_pairs)
            run_id = "run-" + uuid4().hex
            inputs = [r.model_dump(mode="json") for r in [*a, *b]]
            job = {
                "schema_version": "comparison-run-1",
                "note": str(request.get("note") or "").strip(),
                "display_status": "",
                "run_id": run_id,
                "created_at": now(),
                "updated_at": now(),
                "status": "queued",
                "owner_pid": os.getpid(),
                "trial_a": trial_a,
                "trial_b": trial_b,
                "title_a": titles.get(trial_a, trial_a),
                "title_b": titles.get(trial_b, trial_b),
                "dataset_id": dataset_id or None,
                "input_origin": "Uploaded files: " + ", ".join(dataset["files"])
                if dataset
                else "Saved criterion catalog",
                "requested_pair_limit": requested_pairs,
                "criteria_a": [r.model_dump(mode="json") for r in a],
                "criteria_b": [r.model_dump(mode="json") for r in b],
                "input_sha256": hashlib.sha256(
                    json.dumps(inputs, sort_keys=True).encode()
                ).hexdigest(),
                "provider_settings": {
                    "provider": "openai",
                    "judge_model": judge_model.strip(),
                    "critic_model": critic_model.strip(),
                    "retrieval_model": "tfidf",
                    "use_llm": bool(request.get("use_llm", True)),
                    "max_model_calls": calls,
                    "max_seconds": seconds,
                    "requested_pair_limit": requested_pairs,
                    "pricing": pricing,
                    "request_timeout_seconds": 35,
                    "max_output_tokens": 1800,
                },
                "metrics": {
                    "retrieval_seconds": round(retrieval_seconds, 6),
                    "retrieval_cpu_seconds": round(retrieval_cpu_seconds, 6),
                    "python_version": platform.python_version(),
                    "platform": platform.system(),
                    "machine": platform.machine(),
                    "local_device": "CPU; contextual model inference is remote",
                },
                "runtime_version": VERSION,
                "runtime_source_sha256": self.source_fingerprint,
                "encoder": "TF-IDF baseline; trained encoder not available",
                "max_pairs": max_pairs,
                "max_model_calls": calls,
                "max_seconds": seconds,
                "use_llm": bool(request.get("use_llm", True)),
                "candidates": candidates,
                "matching_policy": MATCHING_VERSION,
                "results": [],
                "reviews": {},
                "events": [],
                "revision": 0,
                "message": "Comparison queued",
                "model_calls": 0,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
            write_run(self.directory, job)
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
        execution_started = time.perf_counter()
        execution_cpu_started = time.thread_time()
        job["started_at"] = now()
        budget = RuntimeBudget(
            max_model_calls=job["max_model_calls"],
            max_seconds=job["max_seconds"],
            max_tool_calls=job["max_pairs"] * 3,
            cancelled=self.cancel_events[run_id],
        )
        provider = provider_override
        try:
            configuration = local_configuration(self.root)
            configuration["model"] = job["provider_settings"]["judge_model"]
            configuration["critic_model"] = job["provider_settings"]["critic_model"]
            if provider is None and job["use_llm"] and configuration["api_key"]:
                provider = OpenAIProvider(configuration)
            runner = AgentOrchestrator(provider if job["use_llm"] else None)
            job["model"] = getattr(provider, "model", None) if job["use_llm"] else None
            job["critic_model"] = (
                getattr(provider, "critic_model", job["model"]) if job["use_llm"] else None
            )
            job["provider_settings"]["effective_judge_model"] = job["model"]
            job["provider_settings"]["effective_critic_model"] = job["critic_model"]
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
            limited = len(job["results"]) < len(job["candidates"])
            job["status"] = "limited" if limited else "completed"
            job["message"] = (
                f"Pair limit reached: examined {len(job['results'])} of "
                f"{len(job['candidates'])} retrieved candidates. "
                "Remaining candidates were not examined."
                if limited
                else "All retrieved candidates processed; inspect unresolved cases and coverage."
            )
        except BudgetExceeded as exc:
            job["status"] = "cancelled" if budget.cancelled.is_set() else "limited"
            job["message"] = (
                f"{exc}: examined {len(job['results'])} of {len(job['candidates'])} "
                "assigned counterparts. Saved decisions remain available; unexamined "
                "counterparts have no relationship decision."
            )
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
                job["usage_ledger"] = getattr(provider, "usage_ledger", [])
                execution_seconds = time.perf_counter() - execution_started
                job["metrics"].update(
                    {
                        "execution_seconds": round(execution_seconds, 6),
                        "elapsed_seconds": round(
                            job["metrics"]["retrieval_seconds"] + execution_seconds, 6
                        ),
                        "worker_thread_cpu_seconds": round(
                            job["metrics"]["retrieval_cpu_seconds"]
                            + time.thread_time()
                            - execution_cpu_started,
                            6,
                        ),
                        "api_request_seconds": round(
                            sum(e["elapsed_seconds"] for e in job["usage_ledger"]), 6
                        ),
                    }
                )
                job["cost_estimate"] = estimate_api_cost(
                    job["usage_ledger"], job["provider_settings"]["pricing"], budget.model_calls
                )

                self._save(job)
            if isinstance(provider, OpenAIProvider):
                provider.client.close()

    def _save(self, job):
        job["updated_at"] = now()
        write_run(self.directory, job)

    def get(self, run_id):
        with self.lock:
            job = read_run(self.directory, run_id)
            job["file_name"] = run_path(self.directory, run_id).name
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
        from trialign.schemas import CriterionRecord

        typed = (
            {
                side: [CriterionRecord.model_validate(row) for row in job["criteria_" + side]]
                for side in ["a", "b"]
            }
            if job.get("matching_policy") == MATCHING_VERSION
            else {}
        )
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
                if job.get("matching_policy") == MATCHING_VERSION:
                    others = typed["b"] if rows is job["criteria_a"] else typed["a"]
                    record = CriterionRecord.model_validate(row)
                    has_counterpart = any(correspondence(record, other)[0] for other in others)
                    row["alignment_reason"] = (
                        "Assigned counterpart has not been examined because the run stopped."
                        if identifier in candidates and identifier not in examined
                        else (
                            "Compatible counterparts were assigned to other criteria "
                            "under the one-to-one policy."
                        )
                        if identifier not in candidates and has_counterpart
                        else "No supported common clinical concept in the other trial."
                        if identifier not in candidates
                        else "One-to-one counterpart examined; see its decision and review status."
                    )
        return job

    def list_runs(self):
        results = []
        with self.lock:
            for path, job in saved_run_files(self.directory):
                results.append(
                    {
                        **{
                            k: job.get(k)
                            for k in [
                                "run_id",
                                "created_at",
                                "status",
                                "trial_a",
                                "trial_b",
                                "title_a",
                                "title_b",
                                "note",
                                "display_status",
                            ]
                        },
                        "trial_label_a": trial_label(job, "a"),
                        "trial_label_b": trial_label(job, "b"),
                        "file_name": path.name,
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
                job["message"] = (
                    "Stop requested; completed results are preserved. "
                    "An in-flight model request may take up to 35 seconds to return."
                )
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
