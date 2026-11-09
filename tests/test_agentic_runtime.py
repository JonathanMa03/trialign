"""Behavioral checks for source grounding, critic control and durable review."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from trialign.agents.contracts import Critique, Evidence, Judgment
from trialign.agents.hooks import BudgetExceeded, RuntimeBudget
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
        self.payloads = []

    def generate(self, role, instructions, payload, schema, **kwargs):
        self.roles.append(role)
        self.payloads.append(payload)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return schema.model_validate(result.model_dump())


def test_constructed_harness():
    report = run_harness(ROOT)
    assert report["passed"] == report["cases"] == 15


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
    assert provider.roles == [
        "criterion_judgment",
        "criterion_critic_assessment",
        "criterion_critique",
    ]


def test_one_revision_then_final_critic():
    a, b = contextual()
    provider = ScriptedProvider(
        [
            proposal(a, b),
            proposal(a, b),
            Critique(verdict="revise", issues=["Clarify scope"], rationale="Repairable"),
            proposal(a, b),
            Critique(verdict="accept", issues=[], rationale="Scope is explicit"),
        ]
    )
    result = AgentOrchestrator(provider).compare(a, b, budget=RuntimeBudget(max_model_calls=5))
    assert result.label == RelationshipLabel.RELATED_NOT_COMPARABLE
    assert len(provider.roles) == 5
    assert sum(e["stage"] == "orchestrator.revision" for e in result.trace) == 1


def test_no_revision_without_budget_for_judge_and_critic():
    a, b = contextual()
    provider = ScriptedProvider(
        [
            proposal(a, b),
            proposal(a, b),
            Critique(verdict="revise", issues=["Needs repair"], rationale="Repairable"),
        ]
    )
    result = AgentOrchestrator(provider).compare(a, b, budget=RuntimeBudget(max_model_calls=3))
    assert result.label is None and len(provider.roles) == 3


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
    service.catalog.add(
        [a, second_a, age("B"), age("B", 21, criterion_id="second-b")], origin="test"
    )
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
assert job["status"] in {"limited", "completed"}
assert job["summary"]["compared_pairs"] == min(8, job["summary"]["candidate_pairs"])
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


def workbook_bytes(sheets):
    """Small XLSX fixtures written as XML to exercise sheet discovery."""
    import io
    import xml.etree.ElementTree as ET
    import zipfile

    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    doc = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package = "http://schemas.openxmlformats.org/package/2006/relationships"
    content = "application/vnd.openxmlformats-officedocument.spreadsheetml."
    types = ET.Element(
        "Types", xmlns="http://schemas.openxmlformats.org/package/2006/content-types"
    )
    ET.SubElement(types, "Default", Extension="xml", ContentType="application/xml")
    ET.SubElement(
        types,
        "Default",
        Extension="rels",
        ContentType="application/vnd.openxmlformats-package.relationships+xml",
    )
    ET.SubElement(
        types, "Override", PartName="/xl/workbook.xml", ContentType=content + "sheet.main+xml"
    )
    root_rels = ET.Element("Relationships", xmlns=package)
    ET.SubElement(
        root_rels, "Relationship", Id="rId1", Type=doc + "/officeDocument", Target="xl/workbook.xml"
    )
    workbook = ET.Element("workbook", xmlns=main)
    sheet_nodes = ET.SubElement(workbook, "sheets")
    rels = ET.Element("Relationships", xmlns=package)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index, (name, rows) in enumerate(sheets, 1):
            ET.SubElement(
                types,
                "Override",
                PartName=f"/xl/worksheets/sheet{index}.xml",
                ContentType=content + "worksheet+xml",
            )
            ET.SubElement(
                sheet_nodes,
                "sheet",
                name=name,
                sheetId=str(index),
                attrib={"{" + doc + "}id": f"rId{index}"},
            )
            ET.SubElement(
                rels,
                "Relationship",
                Id=f"rId{index}",
                Type=doc + "/worksheet",
                Target=f"worksheets/sheet{index}.xml",
            )
            worksheet = ET.Element("worksheet", xmlns=main)
            data = ET.SubElement(worksheet, "sheetData")
            for ri, values in enumerate(rows, 1):
                row = ET.SubElement(data, "row", r=str(ri))
                for ci, value in enumerate(values):
                    cell = ET.SubElement(row, "c", r=f"{chr(65 + ci)}{ri}", t="inlineStr")
                    ET.SubElement(ET.SubElement(cell, "is"), "t").text = str(value)
            archive.writestr(f"xl/worksheets/sheet{index}.xml", ET.tostring(worksheet))
        for path, tree in [
            ("[Content_Types].xml", types),
            ("_rels/.rels", root_rels),
            ("xl/workbook.xml", workbook),
            ("xl/_rels/workbook.xml.rels", rels),
        ]:
            archive.writestr(path, ET.tostring(tree))
    return buffer.getvalue()


def test_uploaded_subset_keeps_exact_inputs_and_survives_restart(tmp_path):
    service = ComparisonService(ROOT, directory=tmp_path / "runs")
    rows = [
        service.catalog.for_trial(t)[0].model_dump(mode="json")
        for t in ["NCT00003187", "NCT00025207"]
    ]
    rows[0]["source"] = "Uploaded source is different from the catalog."
    dataset = service.import_files(
        [{"filename": "subset.json", "content": json.dumps(rows).encode()}]
    )
    ids = [t["trial_id"] for t in dataset["trials"]]
    request = {
        "dataset_id": dataset["dataset_id"],
        "trial_a": ids[0],
        "trial_b": ids[1],
        "use_llm": False,
    }
    reloaded = ComparisonService(ROOT, directory=tmp_path / "runs")
    job = reloaded.create(request, synchronous=True)
    assert len(job["criteria_a"]) == len(job["criteria_b"]) == 1
    assert job["criteria_a"][0]["source"] == rows[0]["source"]
    assert job["input_origin"] == "Uploaded files: subset.json"
    assert job["criteria_a"][0]["original_criterion_id"] == rows[0]["criterion_id"]
    with pytest.raises(ValueError, match="belong to the uploaded"):
        reloaded.create({**request, "trial_b": "NCT00025207"})


def test_multisheet_upload_without_modern_autocrit_headers(tmp_path):
    service = ComparisonService(tmp_path)
    content = workbook_bytes(
        [
            ("Trial A", [["Criteria", "Type"], ["Age >= 18 years", "inclusion"]]),
            ("Trial B", [["Text", "Type"], ["Age >= 21 years", "inclusion"]]),
            ("Summary", [["Metric", "Count"], ["Trials", 2]]),
        ]
    )
    dataset = service.import_files([{"filename": "two-trials.xlsx", "content": content}])
    assert len(dataset["trials"]) == len(dataset["records"]) == 2
    assert {r["upload_sheet"] for r in dataset["records"]} == {"Trial A", "Trial B"}
    assert all(r["review_status"] == "pending" for r in dataset["records"])
    assert dataset["warnings"]


def test_separate_uploads_group_each_side_and_do_not_invent_missing_role(tmp_path):
    service = ComparisonService(tmp_path)
    dataset = service.import_files(
        [
            {"filename": "a.csv", "content": b"Criterion\nAge >= 18 years\n", "side": "A"},
            {"filename": "b.csv", "content": b"Text\nAge >= 21 years\n", "side": "B"},
        ]
    )
    assert len(dataset["trials"]) == 2
    assert all(r["role_unspecified"] for r in dataset["records"])
    assert all(r["criterion_type"] == "unspecified" for r in dataset["records"])
    from trialign.schemas import CriterionRecord

    result = AgentOrchestrator().compare(
        *[CriterionRecord.model_validate(r) for r in dataset["records"]]
    )
    assert result.disposition == "source_review"
    assert result.should_escalate


def test_explicit_header_mapping_and_invalid_batch_are_atomic(tmp_path):
    service = ComparisonService(tmp_path)
    files = [
        {
            "filename": "custom.csv",
            "content": (
                b"Original wording,Protocol name\nAge >= 18 years,Alpha\nAge >= 21 years,Beta\n"
            ),
        }
    ]
    with pytest.raises(ValueError, match="column mapping"):
        service.import_files(files)
    assert not list((tmp_path / "runs/uploads").glob("*.json"))
    dataset = service.import_files(
        files, text_column="Original wording", trial_column="Protocol name", role="inclusion"
    )
    assert len(dataset["trials"]) == 2
    assert not any(r.get("role_unspecified") for r in dataset["records"])
    with pytest.raises(ValueError, match="multiple trial IDs"):
        service.import_files(
            [{**files[0], "side": "A"}],
            text_column="Original wording",
            trial_column="Protocol name",
        )


def test_default_run_processes_more_than_sixteen_candidates_and_reports_explicit_limit(tmp_path):
    service = ComparisonService(tmp_path)
    rows = [age(side, i + 18, criterion_id=f"{side}-{i}") for side in ["A", "B"] for i in range(20)]
    service.catalog.add(rows, origin="test")
    request = {"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "use_llm": False}
    full = service.create(request, synchronous=True)
    assert full["summary"]["compared_pairs"] > 16
    assert full["summary"]["compared_pairs"] == full["summary"]["candidate_pairs"]
    assert full["status"] == "completed"
    partial = service.create({**request, "max_pairs": 2}, synchronous=True)
    assert partial["status"] == "limited"
    assert partial["summary"]["compared_pairs"] == 2
    assert "Pair limit reached" in partial["message"]


def test_stop_comparison_preserves_partial_results_and_stops_next_pair(tmp_path, monkeypatch):
    import threading
    import time

    service = ComparisonService(tmp_path)
    service.catalog.add([age("A"), age("B")], origin="test")
    entered, release = threading.Event(), threading.Event()
    original = AgentOrchestrator.compare

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(AgentOrchestrator, "compare", delayed)
    run = service.create({"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "use_llm": False})
    assert entered.wait(3)
    service.cancel(run["run_id"])
    release.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = service.get(run["run_id"])
        if job["status"] == "cancelled":
            break
        time.sleep(0.01)
    assert job["status"] == "cancelled"
    assert job["model_calls"] == 0


@pytest.mark.parametrize("layout", ["wide_trials", "wide_roles", "reviewed_and_other_trial"])
def test_upload_alternative_workbook_layouts_keep_all_trial_inputs(tmp_path, layout):
    service = ComparisonService(tmp_path)
    if layout == "wide_trials":
        content = workbook_bytes(
            [("Criteria", [["Trial A", "Trial B"], ["Age >= 18 years", "Age >= 21 years"]])]
        )
        files = [{"filename": "comparison.xlsx", "content": content}]
        options = {"role": "inclusion"}
        expected_records = 2
    elif layout == "wide_roles":
        content = workbook_bytes(
            [
                (
                    "Criteria",
                    [
                        ["Inclusion Criteria", "Exclusion Criteria"],
                        ["Age >= 18 years", "Active infection"],
                    ],
                )
            ]
        )
        files = [
            {"filename": "a.xlsx", "content": content, "side": "A"},
            {"filename": "b.xlsx", "content": content, "side": "B"},
        ]
        options = {}
        expected_records = 4
    else:
        header = ["Criterion", "Trial ID", "criterion_id", "Type"]
        content = workbook_bytes(
            [
                ("Human Review", [header, ["Reviewed source", "Alpha", "id-a", "inclusion"]]),
                ("Raw", [header, ["Raw source", "Alpha", "id-a", "inclusion"]]),
                ("Trial B", [header, ["Other source", "Beta", "id-b", "exclusion"]]),
            ]
        )
        files = [{"filename": "reviewed.xlsx", "content": content}]
        options = {}
        expected_records = 2
    dataset = service.import_files(files, **options)
    assert len(dataset["records"]) == expected_records
    assert len(dataset["trials"]) == 2
    if layout == "wide_roles":
        assert {r["criterion_type"] for r in dataset["records"]} == {"inclusion", "exclusion"}
    if layout == "reviewed_and_other_trial":
        assert {r["source"] for r in dataset["records"]} == {"Reviewed source", "Other source"}
        assert any("repeated" in warning for warning in dataset["warnings"])


def test_call_budget_defaults_high_and_can_be_unlimited():
    assert RuntimeBudget().max_model_calls == 200
    unlimited = RuntimeBudget(max_model_calls=None)
    for _ in range(250):
        unlimited.before_model()
    assert unlimited.model_calls == 250
    assert unlimited.remaining_model_calls == float("inf")
    unlimited.cancelled.set()
    with pytest.raises(BudgetExceeded, match="cancelled"):
        unlimited.before_model()


@pytest.mark.parametrize("cap", [200, 500, None])
def test_service_accepts_high_or_unlimited_call_budget(tmp_path, cap):
    service = ComparisonService(tmp_path)
    service.catalog.add([age("A"), age("B", 21)], origin="test")
    job = service.create(
        {"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "max_model_calls": cap, "use_llm": False},
        synchronous=True,
    )
    assert job["max_model_calls"] == cap
    assert job["model_calls"] == 0
    assert job["status"] == "completed"


def test_unlimited_budget_keeps_critic_and_single_revision():
    a, b = contextual()
    provider = ScriptedProvider(
        [
            proposal(a, b),
            proposal(a, b),
            Critique(verdict="revise", issues=["Clarify frame"], rationale="Revise"),
            proposal(a, b),
            Critique(verdict="accept", issues=[], rationale="Supported"),
        ]
    )
    result = AgentOrchestrator(provider).compare(a, b, budget=RuntimeBudget(max_model_calls=None))
    assert len(provider.roles) == 5
    assert result.disposition == "critic_accepted"
    assert sum(e["stage"] == "orchestrator.revision" for e in result.trace) == 1


def test_critic_initial_assessment_is_blind_and_reused_unchanged_on_revision():
    a, b = contextual()
    blind = proposal(a, b)
    blind.rationale = "Independent source-only assessment."
    provider = ScriptedProvider(
        [
            proposal(a, b),
            blind,
            Critique(verdict="revise", issues=["Clarify scope"], rationale="Revise"),
            proposal(a, b),
            Critique(verdict="accept", issues=[], rationale="Supported"),
        ]
    )
    result = AgentOrchestrator(provider).compare(a, b)
    assert result.disposition == "critic_accepted"
    assert provider.roles.count("criterion_critic_assessment") == 1
    blind_payload = provider.payloads[1]
    assert set(blind_payload) == {"A", "B"}
    assert "proposal" not in blind_payload and "critic_feedback" not in blind_payload
    first_review, second_review = provider.payloads[2], provider.payloads[4]
    assert first_review["independent_assessment"] == second_review["independent_assessment"]
    assert first_review["independent_assessment"]["rationale"] == blind.rationale


def test_independent_disagreement_cannot_be_published_as_accepted():
    a, b = contextual()
    blind = proposal(a, b)
    blind.label = RelationshipLabel.AMBIGUOUS
    provider = ScriptedProvider(
        [
            proposal(a, b),
            blind,
            Critique(verdict="accept", issues=[], rationale="Agree"),
        ]
    )
    result = AgentOrchestrator(provider).compare(a, b, budget=RuntimeBudget(max_model_calls=3))
    assert result.label is None and result.should_escalate
    assert "disagree" in " ".join(result.critic.issues)


def test_three_calls_reserved_for_independent_critic_workflow():
    a, b = contextual()
    provider = ScriptedProvider([])
    result = AgentOrchestrator(provider).compare(a, b, budget=RuntimeBudget(max_model_calls=2))
    assert not provider.roles and result.disposition == "not_evaluated"


def test_blind_critic_citations_are_validated_before_review():
    a, b = contextual()
    blind = proposal(a, b)
    blind.evidence[0].quote = "Invented independent quote"
    provider = ScriptedProvider([proposal(a, b), blind])
    result = AgentOrchestrator(provider).compare(a, b)
    assert result.label is None
    assert len(provider.roles) == 2
    assert "independent critic assessment" in result.critic.rationale


def test_critic_requests_use_separately_configured_model():
    from types import SimpleNamespace

    from trialign.agents.provider import OpenAIProvider

    provider = OpenAIProvider(
        {"api_key": "test-key", "model": "judge-model", "critic_model": "critic-model"}
    )
    provider.client.close()
    models = []
    a, b = contextual()

    def create(**request):
        models.append(request["model"])
        return SimpleNamespace(
            model=request["model"],
            usage=None,
            status="completed",
            output_text=proposal(a, b).model_dump_json(),
        )

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    for role in ["criterion_judgment", "criterion_critic_assessment", "criterion_critique"]:
        provider.generate(role, "Test instructions", {}, Judgment)
    assert models == ["judge-model", "critic-model", "critic-model"]


def test_clinical_gate_excludes_shared_parent_false_matches_and_never_reuses_ids():
    from trialign.runtime.service import alignment_candidates

    parent = "Not pregnant or lactating. Total bilirubin below the upper limit of normal."

    def row(side, entity, source, identifier):
        return age(
            side, entity=entity, source=source, parent_statement=parent, criterion_id=identifier
        )

    left = [
        row("A", "Pregnancy Status", "Not pregnant or lactating", "a-p"),
        row("A", "Lactation Status", "Not pregnant or lactating", "a-l"),
        row("A", "Pregnancy Status", "Not pregnant", "a-p-duplicate-concept"),
    ]
    right = [
        row("B", "Pregnancy Status", "Not pregnant or nursing", "b-p"),
        row("B", "Lactation Status", "Not pregnant or nursing", "b-l"),
        row("B", "Total Bilirubin", "Bilirubin below ULN", "b-b"),
    ]
    pairs = alignment_candidates(left, right)
    assert len(pairs) == 2
    assert len({p["a"] for p in pairs}) == len({p["b"] for p in pairs}) == 2
    lookup = {r.criterion_id: r for r in left + right}
    for pair in pairs:
        assert lookup[pair["a"]].entity == lookup[pair["b"]].entity
        assert pair["b"] != "b-b"
    # Parent context changes cannot change selection or ranking.
    changed = [
        r.model_copy(update={"parent_statement": "Entirely different boilerplate"})
        for r in left + right
    ]
    assert alignment_candidates(changed[:3], changed[3:]) == pairs


def test_clinical_gate_aliases_and_unknown_compound_sources():
    from trialign.tools.correspondence import correspondence

    assert correspondence(age("A", entity="Hgb"), age("B", entity="Hemoglobin"))[0]
    assert correspondence(
        age("A", entity="unspecified", source="Age at least 18 years"), age("B", entity="Age")
    )[0]
    assert not correspondence(
        age("A", entity="unspecified", source="Not pregnant or lactating"),
        age("B", entity="Total Bilirubin"),
    )[0]
    assert not correspondence(
        age("A", entity="Pregnancy Status"), age("B", entity="Lactation Status")
    )[0]


def test_assignment_uses_global_score_instead_of_reusing_top_neighbor():
    import numpy as np

    from trialign.runtime.service import alignment_candidates

    class Encoder:
        def encode(self, texts):
            # Cross-similarity matrix [[.9, .8], [.85, .1]].
            return np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.85], [0.8, 0.1]])

    left = [age("A", criterion_id="a1"), age("A", criterion_id="a2")]
    right = [age("B", criterion_id="b1"), age("B", criterion_id="b2")]
    pairs = alignment_candidates(left, right, encoder=Encoder())
    assert {(p["a"], p["b"]) for p in pairs} == {("a1", "b2"), ("a2", "b1")}
    assert alignment_candidates(left, [age("B", entity="Total Bilirubin")]) == []


def test_unrelated_direct_pair_cannot_reach_judge_or_critic():
    provider = ScriptedProvider([])
    result = AgentOrchestrator(provider).compare(
        age("A", entity="Pregnancy Status", source="Not pregnant"),
        age("B", entity="Total Bilirubin", source="Bilirubin below ULN"),
    )
    assert provider.roles == []
    assert result.label is None
    assert result.disposition == "no_counterpart"
    assert result.should_escalate


def test_unassigned_one_to_one_competitor_is_distinct_from_run_limit(tmp_path):
    service = ComparisonService(tmp_path)
    service.catalog.add([age("A"), age("A", 21, criterion_id="extra-a"), age("B")], origin="test")
    job = service.create(
        {"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "use_llm": False}, synchronous=True
    )
    unassigned = [r for r in job["criteria_a"] if r["alignment_status"] == "no_candidate_found"]
    assert len(unassigned) == 1
    assert "assigned to other criteria" in unassigned[0]["alignment_reason"]
    assert not any(r["alignment_status"] == "not_examined" for r in job["criteria_a"])


def test_time_budget_preserves_decision_and_marks_remaining_assignment_unexamined(
    tmp_path, monkeypatch
):
    service = ComparisonService(tmp_path)
    service.catalog.add(
        [age(side, i + 18, criterion_id=f"{side}-{i}") for side in ["A", "B"] for i in range(2)],
        origin="test",
    )
    original = AgentOrchestrator.compare

    def expire_after_first(*args, **kwargs):
        result = original(*args, **kwargs)
        kwargs["budget"].started -= kwargs["budget"].max_seconds + 1
        return result

    monkeypatch.setattr(AgentOrchestrator, "compare", expire_after_first)
    job = service.create(
        {"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "use_llm": False}, synchronous=True
    )
    assert job["status"] == "limited"
    assert len(job["results"]) == 1
    assert "examined 1 of 2 assigned counterparts" in job["message"]
    assert sum(r["alignment_status"] == "not_examined" for r in job["criteria_a"]) == 1
    assert job["results"][0]["label"] == "equivalent"


def test_excel_report_preserves_sources_unexamined_pairs_and_review_history(tmp_path):
    from io import BytesIO

    from openpyxl import load_workbook

    from trialign.runtime.report import comparison_workbook

    service = ComparisonService(tmp_path)
    service.catalog.add(
        [age(side, i + 18, criterion_id=f"{side}-{i}") for side in ["A", "B"] for i in range(2)],
        origin="test",
    )
    job = service.create(
        {"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "max_pairs": 1, "use_llm": False},
        synchronous=True,
    )
    result = job["results"][0]
    job["reviews"][result["pair_id"]] = [
        {
            "reviewer_id": "Reviewer",
            "final_label": "equivalent",
            "rationale": "=Literal source note",
            "created_at": "2026-11-08T00:00:00Z",
            "status": "accepted",
        },
        {
            "reviewer_id": "Reviewer",
            "final_label": "ambiguous",
            "rationale": "Scope needs review",
            "created_at": "2026-11-08T01:00:00Z",
            "status": "ambiguous",
        },
    ]
    wb = load_workbook(BytesIO(comparison_workbook(job)))
    assert wb.sheetnames == [
        "Summary",
        "Pair decisions",
        "Criterion coverage",
        "Review history",
        "API usage",
    ]
    pairs = list(wb["Pair decisions"].values)
    assert len(pairs) == 3
    assert {row[5] for row in pairs[1:]} == {"Not examined", "Evidence-supported proposal"}
    assert next(row for row in pairs[1:] if row[5] != "Not examined")[9] == "Ambiguous"
    assert len(list(wb["Criterion coverage"].values)) == 5
    assert wb["Review history"].max_row == 3
    assert wb["Review history"]["D2"].value == "=Literal source note"
    assert wb["Review history"]["D2"].data_type == "s"
    for ws in wb:
        assert ws.freeze_panes == "A2"
        assert ws.auto_filter.ref


def test_visible_time_budget_default_and_extended_range(tmp_path):
    service = ComparisonService(tmp_path)
    service.catalog.add([age("A"), age("B")], origin="test")
    request = {"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "use_llm": False}
    default = service.create(request, synchronous=True)
    assert default["max_seconds"] == 900
    extended = service.create({**request, "max_seconds": 3600}, synchronous=True)
    assert extended["status"] == "completed"
    assert extended["max_seconds"] == 3600
    with pytest.raises(ValueError, match="between 10 and 3600"):
        service.create({**request, "max_seconds": 3601}, synchronous=True)


def test_default_runs_directory_migrates_history_uploads_and_reviews_once(tmp_path):
    from trialign.runtime.storage import atomic_write, read_run, run_path

    legacy = ComparisonService(tmp_path, directory=tmp_path / "artifacts/demo/runs")
    rows = [age("A").model_dump(mode="json"), age("B").model_dump(mode="json")]
    legacy.import_file("criteria.json", json.dumps(rows).encode())
    uploaded = legacy.import_files(
        [
            {
                "filename": "criteria.csv",
                "content": (
                    b"trial_id,source,type\n"
                    b"Trial A,Age at least 18 years,inclusion\n"
                    b"Trial B,Age at least 21 years,inclusion\n"
                ),
                "side": "",
            }
        ]
    )
    job = legacy.create(
        {
            "trial_a": uploaded["trials"][0]["trial_id"],
            "trial_b": uploaded["trials"][1]["trial_id"],
            "dataset_id": uploaded["dataset_id"],
            "use_llm": False,
        },
        synchronous=True,
    )
    assert len(job["results"]) == 1
    job = legacy.review(
        job["run_id"],
        {
            "pair_id": job["results"][0]["pair_id"],
            "reviewer_id": "Test reviewer",
            "final_label": "ambiguous",
            "rationale": "Test migration preserves this review",
            "status": "ambiguous",
            "expected_revision": job["revision"],
        },
    )
    # Reproduce the previous sibling layout for upload snapshots and imports.
    for folder in ["uploads", "imports"]:
        for source in (legacy.directory / folder).glob("*.json"):
            destination = legacy.directory.parent / folder / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    original = run_path(legacy.directory, job["run_id"]).read_bytes()
    service = ComparisonService(tmp_path)
    assert service.directory == tmp_path / "runs"
    assert service.upload(uploaded["dataset_id"])["records"] == uploaded["records"]
    assert service.get(job["run_id"])["reviews"] == job["reviews"]
    assert len(service.catalog.records) == 2
    assert run_path(legacy.directory, job["run_id"]).read_bytes() == original
    assert not (tmp_path / "uploads").exists()
    assert not (tmp_path / "imports").exists()
    saved = read_run(service.directory, job["run_id"])
    saved["message"] = "Newer saved result"
    atomic_write(run_path(service.directory, job["run_id"]), saved)
    assert ComparisonService(tmp_path).get(job["run_id"])["message"] == "Newer saved result"
    run_path(service.directory, job["run_id"]).unlink()
    assert not ComparisonService(tmp_path).list_runs()
    shutil.rmtree(service.directory)
    assert not ComparisonService(tmp_path).list_runs()


def test_run_records_effective_settings_timing_and_zero_api_cost_without_secrets(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-never-save")
    service = ComparisonService(tmp_path)
    service.catalog.add([age("A"), age("B")], origin="test")
    job = service.create(
        {
            "trial_a": "HARNESS-A",
            "trial_b": "HARNESS-B",
            "use_llm": False,
            "judge_model": "session-judge",
            "critic_model": "session-critic",
            "max_seconds": 1200,
            "max_model_calls": 300,
        },
        synchronous=True,
    )
    settings = job["provider_settings"]
    assert settings["judge_model"] == "session-judge"
    assert settings["critic_model"] == "session-critic"
    assert settings["max_seconds"] == 1200 and settings["max_model_calls"] == 300
    assert settings["retrieval_model"] == "tfidf"
    assert job["metrics"]["elapsed_seconds"] >= job["metrics"]["retrieval_seconds"] >= 0
    assert job["metrics"]["worker_thread_cpu_seconds"] >= 0
    assert job["cost_estimate"]["estimated_api_cost_usd"] == 0
    assert "test-secret-never-save" not in json.dumps(job)
    assert ComparisonService(tmp_path).get(job["run_id"])["provider_settings"] == settings


def test_usage_ledger_prices_judge_critic_and_cached_input_and_retains_failures():
    from types import SimpleNamespace

    from trialign.agents.provider import OpenAIProvider
    from trialign.runtime.metrics import estimate_api_cost, pricing_settings

    provider = OpenAIProvider(
        {"api_key": "fake-key", "model": "judge-model", "critic_model": "critic-model"}
    )
    provider.client.close()
    a, b = contextual()
    responses = [
        SimpleNamespace(
            model="judge-snapshot",
            status="completed",
            output_text=proposal(a, b).model_dump_json(),
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=100,
                input_tokens_details=SimpleNamespace(cached_tokens=200),
            ),
        ),
        SimpleNamespace(
            model="critic-snapshot",
            status="completed",
            output_text=proposal(a, b).model_dump_json(),
            usage=SimpleNamespace(
                input_tokens=500,
                output_tokens=50,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        ),
    ]
    provider.client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: responses.pop(0))
    )
    for role in ["criterion_judgment", "criterion_critic_assessment"]:
        provider.generate(role, "Test", {}, Judgment)
    pricing = pricing_settings(
        {
            "judge": {
                "input_per_million": 2,
                "cached_input_per_million": 1,
                "output_per_million": 10,
            },
            "critic": {"input_per_million": 4, "output_per_million": 20},
        }
    )
    estimate = estimate_api_cost(provider.usage_ledger, pricing, provider.calls)
    assert estimate["estimated_api_cost_usd"] == pytest.approx(0.0058)
    assert [e["resolved_model"] for e in provider.usage_ledger] == [
        "judge-snapshot",
        "critic-snapshot",
    ]

    def failed(**kwargs):
        raise TimeoutError("Do not persist this raw exception body")

    provider.client.responses.create = failed
    with pytest.raises(TimeoutError):
        provider.generate("criterion_judgment", "Test", {}, Judgment)
    estimate = estimate_api_cost(provider.usage_ledger, pricing, provider.calls)
    assert estimate["estimated_api_cost_usd"] is None
    assert estimate["known_api_cost_usd"] == pytest.approx(0.0058)
    assert provider.usage_ledger[-1]["error_type"] == "TimeoutError"
    assert "raw exception body" not in json.dumps(provider.usage_ledger)
    assert (
        estimate_api_cost(provider.usage_ledger[:1], pricing_settings(), 1)[
            "estimated_api_cost_usd"
        ]
        is None
    )


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf"), True])
def test_invalid_cost_rates_are_rejected(rate):
    from trialign.runtime.metrics import pricing_settings

    with pytest.raises(ValueError):
        pricing_settings({"judge": {"input_per_million": rate, "output_per_million": 1}})


def test_readable_filenames_preserve_ids_notes_and_status_transitions(tmp_path):
    from trialign.runtime.storage import atomic_write, read_run, run_path, write_run

    job = {
        "run_id": "run-" + "a" * 32,
        "created_at": "2026-11-08T20:30:45+00:00",
        "trial_a": "NCT01841736",
        "trial_b": "NCT04310007",
        "status": "running",
        "note": "Initial",
        "display_status": "",
    }
    running_path = write_run(tmp_path, job)
    assert (
        running_path.name
        == "run_2026-11-08_15-30-45_EST_NCT01841736_NCT04310007_running_aaaaaaaa.json"
    )
    edited = read_run(tmp_path, job["run_id"])
    edited["note"] = "Presentation example"
    edited["display_status"] = "Reviewed"
    atomic_write(running_path, edited)
    job["status"] = "completed"
    completed = write_run(tmp_path, job)
    assert "_completed_" in completed.name and not running_path.exists()
    assert read_run(tmp_path, job["run_id"])["note"] == "Presentation example"
    assert read_run(tmp_path, job["run_id"])["display_status"] == "Reviewed"
    assert read_run(tmp_path, job["run_id"])["status"] == "completed"
    assert run_path(tmp_path, job["run_id"]) == completed
    # Identification is based on the JSON ID even if a user renames the file.
    renamed = completed.rename(tmp_path / "presentation-example.json")
    assert run_path(tmp_path, job["run_id"]) == renamed


def test_short_filename_id_collisions_never_overwrite_another_run(tmp_path):
    from trialign.runtime.storage import read_run, write_run

    base = {
        "created_at": "2026-11-08T20:30:45+00:00",
        "trial_a": "NCT01841736",
        "trial_b": "NCT04310007",
        "status": "completed",
    }
    first = {**base, "run_id": "run-" + "a" * 32, "note": "First"}
    second = {**base, "run_id": "run-" + "a" * 8 + "b" * 24, "note": "Second"}
    a, b = write_run(tmp_path, first), write_run(tmp_path, second)
    assert a != b
    assert read_run(tmp_path, first["run_id"])["note"] == "First"
    assert read_run(tmp_path, second["run_id"])["note"] == "Second"


def test_existing_run_is_renamed_and_manual_note_is_listed_without_changing_status(tmp_path):
    from trialign.runtime.storage import atomic_write, run_path

    service = ComparisonService(tmp_path)
    service.catalog.add([age("A"), age("B")], origin="test")
    job = service.create(
        {"trial_a": "HARNESS-A", "trial_b": "HARNESS-B", "use_llm": False}, synchronous=True
    )
    path = run_path(service.directory, job["run_id"])
    legacy = path.rename(service.directory / (job["run_id"] + ".json"))
    raw = json.loads(legacy.read_text())
    raw["note"] = "Use for presentation"
    raw["display_status"] = "Ready"
    atomic_write(legacy, raw)
    restarted = ComparisonService(tmp_path)
    listed = restarted.list_runs()[0]
    assert listed["note"] == "Use for presentation" and listed["display_status"] == "Ready"
    assert listed["status"] == "completed"
    assert listed["file_name"].startswith("run_") and not legacy.exists()
    assert restarted.get(job["run_id"])["reviews"] == job["reviews"]
    assert restarted.get(job["run_id"])["cost_estimate"] == job["cost_estimate"]


def test_frozen_partitions_cover_corpus_reservations_and_quarantine():
    from trialign.data import load_parquet_corpus
    from trialign.data.partitions import assign_partitions, load_partition_release, pair_partition

    release = load_partition_release(ROOT)
    assert {k: v["trials"] for k, v in release["counts"].items()} == {
        "train": 68,
        "validation": 15,
        "test": 15,
        "quarantine": 2,
    }
    assert len(release["criterion_assignments"]) == 3537
    assert sum(v["criteria"] for v in release["counts"].values()) == 3537
    assert set(release["quarantined_trial_ids"]) == {"NCT05419375", "NCT07172919"}
    assert all(
        release["assignments"][t] in {"train", "quarantine"}
        for t in release["development_reserved_trial_ids"]
    )
    records = load_parquet_corpus(ROOT / "data/processed/trial100_v1.parquet")
    family = json.loads((ROOT / "data/processed/family_audit_v1/family_mapping.json").read_text())
    inventory = json.loads(
        (ROOT / "data/processed/family_audit_v1/trial_inventory.json").read_text()
    )
    policy = json.loads((ROOT / "configs/evaluation_policy_v1.json").read_text())
    sources_by_partition = {
        part: {
            " ".join(r.source.casefold().split())
            for r in records
            if release["assignments"][r.trial_id] == part
        }
        for part in ["train", "validation", "test"]
    }
    for record in records:
        part = release["assignments"][record.trial_id]
        previous = (
            sources_by_partition["train"]
            if part == "validation"
            else (sources_by_partition["train"] | sources_by_partition["validation"])
        )
        expected = (
            part in {"validation", "test"}
            and " ".join(record.source.casefold().split()) not in previous
        )
        assert release["secondary_exact_source_eligible"][record.criterion_id] == expected

    a, _, _ = assign_partitions(records, family, inventory, policy)
    b, _, _ = assign_partitions(list(reversed(records)), family, inventory, policy)
    assert a == b == release["assignments"]
    with pytest.raises(ValueError, match="explicit"):
        pair_partition("UNKNOWN", "NCT01841736", a)
    assert pair_partition("NCT05419375", "NCT07172919", a) is None
    train = next(t for t, p in a.items() if p == "train")
    test = next(t for t, p in a.items() if p == "test")
    assert pair_partition(train, test, a) is None
    linked = [t for t in a if inventory[t]["cohort"] == "cardiovascular"][:2]
    family["split_group_by_trial"][linked[1]] = family["split_group_by_trial"][linked[0]]
    grouped, _, _ = assign_partitions(records, family, inventory, policy)
    assert grouped[linked[0]] == grouped[linked[1]]


def test_partition_release_rejects_modified_output_and_overwrite(tmp_path):
    from trialign.data.partitions import freeze_partitions

    output = tmp_path / "partition_release"
    freeze_partitions(ROOT, output)
    freeze_partitions(ROOT, output, verify=True)
    with pytest.raises(ValueError, match="already exists"):
        freeze_partitions(ROOT, output)
    path = output / "partitions.json"
    data = json.loads(path.read_text())
    trial = next(t for t, p in data["assignments"].items() if p == "test")
    data["assignments"][trial] = "train"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="output changed"):
        freeze_partitions(ROOT, output, verify=True)


def test_pair_generation_rejects_missing_assignments_and_excludes_quarantine():
    from trialign.pairing.generate import generate_candidate_pairs

    records = [age("A"), age("B")]
    with pytest.raises(ValueError, match="required"):
        generate_candidate_pairs(records, seed=1)
    with pytest.raises(ValueError, match="Missing"):
        generate_candidate_pairs(records, seed=1, split_by_trial={"HARNESS-A": "train"})
    for mapping in [
        {"HARNESS-A": "train", "HARNESS-B": "test"},
        {"HARNESS-A": "quarantine", "HARNESS-B": "quarantine"},
    ]:
        assert generate_candidate_pairs(records, seed=1, split_by_trial=mapping) == []
    mapping = {"HARNESS-A": "train", "HARNESS-B": "train"}
    pairs = generate_candidate_pairs(records, seed=1, split_by_trial=mapping)
    assert len(pairs) == 1 and pairs[0].split == "train"


def test_pair_cli_uses_training_only_and_rejects_unfrozen_assignments(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import trialign.cli as cli
    from trialign.data.partitions import load_partition_release

    release = load_partition_release(ROOT)
    captured = []

    def generate(records, **kwargs):
        captured.extend(records)
        return []

    monkeypatch.setattr(cli, "generate_candidate_pairs", generate)
    output = tmp_path / "controls.parquet"
    result = CliRunner().invoke(cli.app, ["pairs", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert len(captured) == 2665
    assert all(release["assignments"][r.trial_id] == "train" for r in captured)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"assignments": {}}))
    result = CliRunner().invoke(
        cli.app, ["pairs", "--splits", str(wrong), "--output", str(tmp_path / "bad.parquet")]
    )
    assert result.exit_code != 0
    assert not (tmp_path / "bad.parquet").exists()
