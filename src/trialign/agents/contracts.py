"""Strict provider boundaries and serializable runtime decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ModelPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposed_label: RelationshipLabel | None
    top_label: RelationshipLabel
    relevance_score: float = Field(ge=-1, le=1)
    class_scores: dict[str, float]
    calibrated: Literal[False] = False
    requires_review: Literal[True] = True
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    serializer: str
    supported_class: bool
    input_truncated: bool = False
    reason: str

    @model_validator(mode="after")
    def valid_scores(self):
        import math

        if (
            set(self.class_scores) != {x.value for x in RelationshipLabel}
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.class_scores.values())
            or abs(sum(self.class_scores.values()) - 1) > 1e-4
        ):
            raise ValueError(
                "Model scores must cover the label map and form a finite normalized distribution."
            )
        if self.proposed_label is not None and (not self.supported_class or self.input_truncated):
            raise ValueError("Unsupported/truncated predictions must abstain.")
        return self


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
    model_prediction: dict | None = None
