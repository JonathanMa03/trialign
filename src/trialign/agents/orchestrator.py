"""Minimal deterministic-first orchestration skeleton."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from trialign.schemas import CriterionRecord, RelationshipLabel
from trialign.tools import compare_structured


@dataclass(frozen=True)
class AgentResult:
    trace_id: str
    label: RelationshipLabel | None
    confidence: float
    should_escalate: bool
    trace: tuple[dict[str, Any], ...]


class AgentOrchestrator:
    """Run exact checks first; future providers plug into unresolved cases."""

    def compare(self, a: CriterionRecord, b: CriterionRecord) -> AgentResult:
        trace_id = "trace-" + hashlib.sha256(
            f"{a.criterion_id}\x1f{b.criterion_id}".encode()
        ).hexdigest()[:20]
        exact = compare_structured(a, b)
        trace = ({
            "tool": "structured_comparison",
            "version": "0.1.0",
            "label": exact.label.value if exact.label else None,
            "confidence": exact.confidence,
            "reasons": list(exact.reasons),
        },)
        resolved = exact.label is not None and not exact.requires_context
        return AgentResult(
            trace_id=trace_id,
            label=exact.label,
            confidence=exact.confidence,
            should_escalate=not resolved,
            trace=trace,
        )

