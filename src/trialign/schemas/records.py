"""Versioned interchange records shared by data, agent, and model modules."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RelationshipLabel(StrEnum):
    EQUIVALENT = "equivalent"
    A_MORE_RESTRICTIVE = "a_more_restrictive"
    B_MORE_RESTRICTIVE = "b_more_restrictive"
    PARTIAL_OVERLAP = "partial_overlap"
    CONTRADICTORY = "contradictory_or_disjoint"
    RELATED_NOT_COMPARABLE = "related_not_comparable"
    UNRELATED = "unrelated"
    AMBIGUOUS = "ambiguous"


class VersionedRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = "0.1.0"


class CriterionRecord(VersionedRecord):
    criterion_id: str
    trial_id: str
    parent_statement_id: str
    source: str
    parent_statement: str
    surrounding_context: str | None = None
    criterion_type: Literal["inclusion", "exclusion"]
    domain: str
    entity: str
    comparator: str | None = None
    value: str | int | float | bool | None = None
    unit: str | None = None
    value_kind: str | None = None
    lower_bound: float | None = None
    lower_inclusive: bool | None = None
    upper_bound: float | None = None
    upper_inclusive: bool | None = None
    negated: bool = False
    temporal: str | None = None
    repetition: str | None = None
    qualifier: str | None = None
    logical_operator: str = "standalone"
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    review_status: str = "pending"
    validation_issues: list[str] = Field(default_factory=list)
    readiness_blockers: list[str] = Field(default_factory=list)
    source_uri: str | None = None
    source_version: str | None = None
    protocol_hash: str | None = None

    @model_validator(mode="after")
    def interval_is_ordered(self) -> "CriterionRecord":
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError("lower_bound cannot exceed upper_bound")
        return self

    @property
    def comparison_text(self) -> str:
        parts = [self.entity]
        if self.comparator:
            parts.extend([self.comparator, str(self.value)])
        elif self.value is not None:
            parts.append(str(self.value))
        if self.unit:
            parts.append(self.unit)
        for modifier in (self.temporal, self.repetition, self.qualifier):
            if modifier:
                parts.append(modifier)
        return " ".join(part for part in parts if part).strip()


class PairCandidate(VersionedRecord):
    pair_id: str
    criterion_a_id: str
    criterion_b_id: str
    split: Literal["train", "validation", "test", "unassigned"] = "unassigned"
    sampling_method: str
    sampling_model: str | None = None
    sampling_score: float | None = None
    clinical_phenomena: list[str] = Field(default_factory=list)
    constructed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentJudgment(VersionedRecord):
    judgment_id: str
    pair_id: str
    proposed_label: RelationshipLabel
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    uncertainty: float = Field(ge=0, le=1)
    should_escalate: bool
    tool_trace_ids: list[str] = Field(default_factory=list)
    model_version: str | None = None
    prompt_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HumanDecision(VersionedRecord):
    decision_id: str
    pair_id: str
    reviewer_id: str
    status: Literal["accepted", "corrected", "rejected", "ambiguous"]
    final_label: RelationshipLabel | None = None
    rationale: str | None = None
    adjudication_of: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def decided_rows_have_label(self) -> "HumanDecision":
        if self.status in {"accepted", "corrected"} and self.final_label is None:
            raise ValueError("accepted or corrected decisions require final_label")
        return self

