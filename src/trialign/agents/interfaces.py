"""Typed contracts for auditable agent tools."""

from __future__ import annotations

from typing import Any, Protocol

from trialign.schemas import CriterionRecord


class ComparisonTool(Protocol):
    name: str
    version: str

    def compare(self, a: CriterionRecord, b: CriterionRecord) -> dict[str, Any]: ...


class JudgmentProvider(Protocol):
    name: str
    version: str

    def judge(
        self, a: CriterionRecord, b: CriterionRecord, evidence: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


class PairRelationshipModel(Protocol):
    """A prediction tool, not an evidence validator or an autonomous final judge."""

    artifact_digest: str

    def predict_pair(self, a: CriterionRecord, b: CriterionRecord) -> dict[str, Any]: ...
