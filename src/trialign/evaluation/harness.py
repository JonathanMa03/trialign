"""Repeatable diagnostic harnesses separate from clinical benchmark evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from trialign.agents.orchestrator import AgentOrchestrator
from trialign.runtime.storage import atomic_write
from trialign.schemas import CriterionRecord


def fixture(case_id: str, side: str, fields: dict) -> CriterionRecord:
    return CriterionRecord.model_validate(
        {
            "criterion_id": f"harness-{case_id}-{side}",
            "trial_id": "HARNESS-" + side,
            "parent_statement_id": f"harness-parent-{case_id}-{side}",
            "source": fields["source"],
            "parent_statement": fields.get("parent_statement", fields["source"]),
            "criterion_type": "inclusion",
            "entity": "Age",
            "domain": "person",
            "unit": "years",
            "review_status": "corrected",
            "source_version": "constructed-harness-1",
            **fields,
        }
    )


def run_harness(root: Path, output: Path | None = None) -> dict:
    spec = json.loads((root / "configs/runtime_harness.json").read_text())
    results = []
    for case in spec["cases"]:
        result = AgentOrchestrator().compare(
            fixture(case["id"], "A", case["a"]), fixture(case["id"], "B", case["b"])
        )
        label = result.label.value if result.label else None
        results.append(
            {
                "id": case["id"],
                "passed": label == case["expected_label"]
                and result.should_escalate == case["must_escalate"],
                "expected_label": case["expected_label"],
                "actual_label": label,
                "should_escalate": result.should_escalate,
                "trace": result.trace,
            }
        )
    payload = {
        "version": spec["version"],
        "purpose": spec["purpose"],
        "passed": sum(r["passed"] for r in results),
        "cases": len(results),
        "results": results,
    }
    if output:
        atomic_write(output, payload)
    return payload
