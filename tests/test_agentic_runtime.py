"""Behavioral checks for source grounding, critic control and durable review."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from trialign.agents.contracts import Critique, Evidence, Judgment
from trialign.agents.hooks import RuntimeBudget
from trialign.agents.orchestrator import AgentOrchestrator
from trialign.evaluation.harness import fixture, run_harness
from trialign.runtime.catalog import Catalog
from trialign.runtime.service import ComparisonService
from trialign.schemas import RelationshipLabel

ROOT = Path(__file__).resolve().parents[1]


def age(side, value=18, **extra):
    fields = {"source": f"Age >= {value} years", "comparator": ">=", "value": value, **extra}
    return fixture("test", side, fields)


def contextual():
    return age("A", temporal="at consent"), age("B", 21, temporal="at screening")


def proposal(a, b, **extra):
    return Judgment(
        label=RelationshipLabel.RELATED_NOT_COMPARABLE,
        rationale="The assessment anchors are not operationally aligned.",
        common_frame="Age with separate consent and screening anchors.",
        uncertainty="medium",
        assumptions=[],
        evidence=[Evidence(side="A", quote=a.source), Evidence(side="B", quote=b.source)],
        **extra,
    )


class ScriptedProvider:
    model = "test-double-not-a-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.roles = []

    def generate(self, role, instructions, payload, schema, **kwargs):
        self.roles.append(role)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return schema.model_validate(result.model_dump())


def test_constructed_harness():
    report = run_harness(ROOT)
    assert report["passed"] == report["cases"] == 12


def test_exact_role_inversion_has_executed_tools_and_critic():
    a, b = age("A"), age("B", criterion_type="exclusion")
    result = AgentOrchestrator().compare(a, b)
    assert result.label == RelationshipLabel.CONTRADICTORY
    assert not result.should_escalate
    stages = [e["stage"] for e in result.trace]
    assert "tool.completed" in stages and "agent.critic.rule_audit" in stages
    assert stages[-1] == "hook.before_finalize"


def test_critic_veto_prevents_publishing_judge_label():
    a, b = contextual()
    provider = ScriptedProvider(
        [
            proposal(a, b),
            Critique(
                verdict="escalate",
                issues=["Insufficient mapping"],
                rationale="Cannot substantiate nesting.",
            ),
        ]
    )
    result = AgentOrchestrator(provider).compare(a, b)
    assert result.label is None and result.should_escalate
    assert provider.roles == ["criterion_judgment", "criterion_critique"]


def test_one_revision_then_final_critic():
    a, b = contextual()
    provider = ScriptedProvider(
        [
            proposal(a, b),
            Critique(verdict="revise", issues=["Clarify scope"], rationale="Repairable"),
            proposal(a, b),
            Critique(verdict="accept", issues=[], rationale="Scope is explicit"),
        ]
    )
    result = AgentOrchestrator(provider).compare(a, b, budget=RuntimeBudget(max_model_calls=4))
    assert result.label == RelationshipLabel.RELATED_NOT_COMPARABLE
    assert len(provider.roles) == 4
    assert sum(e["stage"] == "orchestrator.revision" for e in result.trace) == 1


def test_no_revision_without_budget_for_judge_and_critic():
    a, b = contextual()
    provider = ScriptedProvider(
        [
            proposal(a, b),
            Critique(verdict="revise", issues=["Needs repair"], rationale="Repairable"),
        ]
    )
    result = AgentOrchestrator(provider).compare(a, b, budget=RuntimeBudget(max_model_calls=2))
    assert result.label is None and len(provider.roles) == 2


def test_invented_quote_is_rejected_before_model_critic():
    a, b = contextual()
    invalid = proposal(a, b)
    invalid.evidence[0].quote = "The source never said this."
    provider = ScriptedProvider([invalid])
    result = AgentOrchestrator(provider).compare(a, b)
    assert result.label is None
    assert len(provider.roles) == 1
    assert "exact source span" in " ".join(result.critic.issues)


def test_provider_failure_is_not_a_clinical_label():
    a, b = contextual()
    provider = ScriptedProvider([RuntimeError("do not leak this request or credential")])
    result = AgentOrchestrator(provider).compare(a, b)
    assert result.label is None and result.disposition == "operational_failure"
    assert "do not leak" not in result.model_dump_json()


def test_known_source_defect_cannot_be_repaired_by_a_model():
    a = age("A", source_quality_issues=["source role conflicts with heading"])
    provider = ScriptedProvider([])
    result = AgentOrchestrator(provider).compare(a, age("B"))
    assert result.label == RelationshipLabel.AMBIGUOUS
    assert result.disposition == "source_review" and provider.roles == []


def test_cancel_hook_stops_before_tools_or_model():
    budget = RuntimeBudget()
    budget.cancelled.set()
    result = AgentOrchestrator().compare(age("A"), age("B"), budget=budget)
    assert result.label is None and not any(e["stage"] == "tool.completed" for e in result.trace)


def test_changed_source_inclusive_flag_does_not_gain_exact_label():
    a = age("A", lower_bound=18, lower_inclusive=False)
    assert AgentOrchestrator().compare(a, age("B")).label is None


def test_ecog_is_discrete():
    a = fixture(
        "ecog",
        "A",
        {"entity": "ECOG", "unit": "score", "source": "ECOG <= 1", "comparator": "<=", "value": 1},
    )
    b = fixture(
        "ecog",
        "B",
        {"entity": "ECOG", "unit": "score", "source": "ECOG < 2", "comparator": "<", "value": 2},
    )
    assert AgentOrchestrator().compare(a, b).label == RelationshipLabel.EQUIVALENT


def test_literal_ecog_ranges_compare_without_dropping_endpoints():
    fields = {
        "entity": "ECOG",
        "unit": "score",
        "comparator": "range",
        "value": "0-1",
        "lower_bound": 0,
        "lower_inclusive": True,
        "upper_inclusive": True,
    }
    a = fixture(
        "ecog-range", "A", {**fields, "source": "ECOG performance status 0-1", "upper_bound": 1}
    )
    b = fixture(
        "ecog-range", "B", {**fields, "source": "ECOG performance status 0-2", "upper_bound": 2}
    )
    assert AgentOrchestrator().compare(a, b).label == RelationshipLabel.A_MORE_RESTRICTIVE


def test_review_and_history_survive_reload(tmp_path):
    service = ComparisonService(tmp_path)
    service.catalog.add([age("A"), age("B", 21)], origin="test")
    job = service.create(
        {"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "use_llm": False}, synchronous=True
    )
    pair = job["results"][0]
    body = {
        "pair_id": pair["pair_id"],
        "reviewer_id": "Test reviewer",
        "status": "accepted",
        "final_label": pair["label"],
        "rationale": "Reviewed strict age containment.",
        "expected_revision": 0,
    }
    saved = service.review(job["run_id"], body)
    assert saved["revision"] == 1
    with pytest.raises(ValueError, match="another window"):
        service.review(job["run_id"], body)
    body["expected_revision"] = 1
    service.review(job["run_id"], body)
    loaded = ComparisonService(tmp_path).get(job["run_id"])
    assert len(loaded["reviews"][pair["pair_id"]]) == 2
    assert loaded["input_sha256"] == job["input_sha256"]


def test_import_is_all_or_nothing_and_ids_are_immutable(tmp_path):
    catalog = Catalog(tmp_path)
    row = age("A").model_dump(mode="json")
    with pytest.raises(ValueError, match="row 2"):
        catalog.import_file("criteria.json", json.dumps([row, {"invalid": True}]).encode())
    assert not catalog.records
    catalog.import_file("criteria.json", json.dumps([row]).encode())
    row["source"] = "Different source under the same ID"
    with pytest.raises(ValueError, match="conflicts"):
        catalog.import_file("criteria.json", json.dumps([row]).encode())
    assert catalog.records[row["criterion_id"]].source == "Age >= 18 years"


def test_imported_criteria_persist_without_writing_frozen_inputs(tmp_path):
    service = ComparisonService(tmp_path)
    rows = [age("A").model_dump(mode="json"), age("B").model_dump(mode="json")]
    service.import_file("criteria.json", json.dumps(rows).encode())
    loaded = ComparisonService(tmp_path)
    assert len(loaded.catalog.records) == 2
    assert not (tmp_path / "data").exists()


def test_not_examined_is_distinct_from_no_candidate(tmp_path):
    service = ComparisonService(tmp_path)
    a = age("A")
    second_a = age("A").model_copy(
        update={"criterion_id": "another-a", "value": 21, "source": "Age >= 21 years"}
    )
    service.catalog.add([a, second_a, age("B")], origin="test")
    job = service.create(
        {"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "max_pairs": 1, "use_llm": False},
        synchronous=True,
    )
    assert len(job["results"]) == 1
    assert any(r["alignment_status"] == "not_examined" for r in job["criteria_a"])
    assert not any(r["alignment_status"] == "no_candidate_found" for r in job["criteria_a"])


def test_configuration_is_independent_of_modern_autocrit(tmp_path, monkeypatch):
    from trialign.agents.provider import local_configuration

    for key in list(os.environ):
        if key.startswith(("OPENAI_", "TRIALIGN_")):
            monkeypatch.delenv(key)
    root = tmp_path / "trialign"
    sibling = tmp_path / "modern-autocrit"
    root.mkdir()
    sibling.mkdir()
    (sibling / ".env").write_text("OPENAI_API_KEY=sibling-test-key\n")
    assert not local_configuration(root)["api_key"]
    (root / ".env").write_text("OPENAI_API_KEY=trialign-test-key\nOPENAI_MODEL=local-model\n")
    assert local_configuration(root)["api_key"] == "trialign-test-key"
    assert local_configuration(root)["model"] == "local-model"
    explicit = tmp_path / "explicit.env"
    explicit.write_text("OPENAI_API_KEY=explicit-test-key\n")
    monkeypatch.setenv("TRIALIGN_ENV_FILE", str(explicit))
    assert local_configuration(root)["api_key"] == "explicit-test-key"
    monkeypatch.setenv("OPENAI_API_KEY", "environment-test-key")
    assert local_configuration(root)["api_key"] == "environment-test-key"


def test_relocated_checkout_runs_without_upstream_or_credentials(tmp_path):
    standalone = tmp_path / "standalone"
    shutil.copytree(ROOT / "src", standalone / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for relative in [
        "data/processed/trial100_v1.parquet",
        "data/processed/trial100_v1.parquet.manifest.json",
        "data/processed/family_audit_v1/trial_inventory.json",
    ]:
        destination = standalone / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OPENAI_", "TRIALIGN_", "PYTHONPATH"))
    }
    program = r"""
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
from trialign.config import ROOT
from trialign.agents.provider import local_configuration
from trialign.runtime.service import ComparisonService
from trialign.web.server import STATIC, handler
assert ROOT == root
assert not (root.parent / "modern-autocrit").exists()
assert not local_configuration(root)["api_key"]
assert (STATIC / "index.html").is_file()
service = ComparisonService(root)
assert len(service.catalog.trials()) == 100
job = service.create({"trial_a": "NCT05379712", "trial_b": "NCT02769832",
                      "max_pairs": 8, "max_model_calls": 0, "use_llm": False},
                     synchronous=True)
assert job["status"] == "completed"
assert job["summary"]["compared_pairs"] == 8
assert job["summary"]["supported_pairs"] >= 1
assert job["model_calls"] == 0
assert handler(service)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", program, str(standalone)],
        cwd=standalone,
        env=environment,
        check=True,
        timeout=45,
        capture_output=True,
        text=True,
    )
