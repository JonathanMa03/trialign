"""Strict provider boundaries and serializable runtime decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trialign.schemas import RelationshipLabel


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    side: Literal["A", "B"]
    quote: str


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: RelationshipLabel
    rationale: str
    common_frame: str
    uncertainty: Literal["low", "medium", "high"]
    assumptions: list[str]
    evidence: list[Evidence]


class Critique(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["accept", "revise", "escalate"]
    issues: list[str]
    rationale: str


class PairResult(BaseModel):
    schema_version: str = "runtime-pair-1"
    pair_id: str
    trace_id: str
    criterion_a_id: str
    criterion_b_id: str
    label: RelationshipLabel | None
    rationale: str
    common_frame: str
    uncertainty: str
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    route: str
    disposition: str
    should_escalate: bool
    critic: Critique
    trace: list[dict] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    confidence: float = 0.0  # Compatibility only; UI never presents calibrated probability.
