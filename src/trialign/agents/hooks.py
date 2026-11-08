"""Runtime input, budget, tool and evidence hooks that can veto execution."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from threading import Event

from trialign.agents.contracts import Judgment
from trialign.schemas import CriterionRecord
from trialign.tools.structured import source_text


class BudgetExceeded(RuntimeError):
    pass


class SourceNeedsReview(RuntimeError):
    pass


@dataclass
class RuntimeBudget:
    max_model_calls: int | None = 200
    max_seconds: float = 900
    max_tool_calls: int = 160
    cancelled: Event = field(default_factory=Event)
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    tool_calls: int = 0

    @property
    def remaining_seconds(self):
        return max(0, self.max_seconds - (time.monotonic() - self.started))

    @property
    def remaining_model_calls(self):
        return (
            float("inf")
            if self.max_model_calls is None
            else self.max_model_calls - self.model_calls
        )

    def check(self):
        if self.cancelled.is_set():
            raise BudgetExceeded("Run cancelled")
        if self.remaining_seconds <= 0:
            raise BudgetExceeded("Run time budget reached")

    def before_tool(self):
        self.check()
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded("Tool budget reached")
        self.tool_calls += 1

    def before_model(self):
        self.check()
        if self.max_model_calls is not None and self.model_calls >= self.max_model_calls:
            raise BudgetExceeded("Context-review call budget reached")
        self.model_calls += 1


def validate_inputs(a: CriterionRecord, b: CriterionRecord) -> None:
    if a.criterion_id == b.criterion_id:
        raise ValueError("A criterion cannot be compared with itself.")
    if a.trial_id == b.trial_id:
        raise ValueError("This workflow compares criteria from different trials.")
    if not a.source.strip() or not b.source.strip():
        raise ValueError("Both criteria need source text.")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_evidence(judgment: Judgment, a: CriterionRecord, b: CriterionRecord) -> list[str]:
    issues = []
    if not judgment.rationale.strip() or not judgment.common_frame.strip():
        issues.append("A rationale and explicit common frame are required")
    if {item.side for item in judgment.evidence} != {"A", "B"}:
        issues.append("Citations must cover both criteria")
    for item in judgment.evidence:
        reference = source_text(a if item.side == "A" else b)
        if len(item.quote.strip()) < 3 or normalized(item.quote) not in normalized(reference):
            issues.append(f"Citation {item.side} is not an exact source span")
    return issues
