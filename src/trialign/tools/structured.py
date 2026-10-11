"""Conservative exact checks used by the agent before any LLM judgment."""

from __future__ import annotations

from dataclasses import dataclass

from trialign.schemas import CriterionRecord, RelationshipLabel


@dataclass(frozen=True)
class StructuredComparison:
    label: RelationshipLabel | None
    confidence: float
    reasons: tuple[str, ...]
    requires_context: bool


def _interval(row: CriterionRecord) -> tuple[float | None, bool, float | None, bool] | None:
    lower, upper = row.lower_bound, row.upper_bound
    lower_inclusive = bool(row.lower_inclusive)
    upper_inclusive = bool(row.upper_inclusive)
    if lower is not None or upper is not None:
        return lower, lower_inclusive, upper, upper_inclusive
    if not row.comparator or not isinstance(row.value, (int, float)):
        return None
    if row.comparator == ">":
        return float(row.value), False, None, False
    if row.comparator == ">=":
        return float(row.value), True, None, False
    if row.comparator == "<":
        return None, False, float(row.value), False
    if row.comparator == "<=":
        return None, False, float(row.value), True
    if row.comparator == "=":
        return float(row.value), True, float(row.value), True
    return None


def _disjoint(
    first: tuple[float | None, bool, float | None, bool],
    second: tuple[float | None, bool, float | None, bool],
) -> bool:
    a_low, a_low_i, a_high, a_high_i = first
    b_low, b_low_i, b_high, b_high_i = second
    if a_high is not None and b_low is not None:
        if a_high < b_low or (a_high == b_low and not (a_high_i and b_low_i)):
            return True
    if b_high is not None and a_low is not None:
        if b_high < a_low or (b_high == a_low and not (b_high_i and a_low_i)):
            return True
    return False


def compare_structured(a: CriterionRecord, b: CriterionRecord) -> StructuredComparison:
    """Return only relationships that are safe to infer from normalized fields."""
    if a.entity.casefold().strip() != b.entity.casefold().strip():
        return StructuredComparison(None, 0.0, ("entities differ",), True)
    if (a.unit or "").casefold().strip() != (b.unit or "").casefold().strip():
        return StructuredComparison(None, 0.0, ("units differ or require conversion",), True)
    if a.negated != b.negated:
        return StructuredComparison(
            RelationshipLabel.CONTRADICTORY, 0.95, ("opposite normalized negation",), False
        )
    interval_a, interval_b = _interval(a), _interval(b)
    if interval_a and interval_b:
        if interval_a == interval_b:
            return StructuredComparison(
                RelationshipLabel.EQUIVALENT, 1.0, ("identical normalized intervals",), False
            )
        if _disjoint(interval_a, interval_b):
            return StructuredComparison(
                RelationshipLabel.CONTRADICTORY, 1.0, ("disjoint normalized intervals",), False
            )
        return StructuredComparison(
            RelationshipLabel.PARTIAL_OVERLAP, 0.9, ("overlapping normalized intervals",), True
        )
    if a.value == b.value and a.criterion_type == b.criterion_type:
        return StructuredComparison(
            RelationshipLabel.EQUIVALENT, 0.9, ("identical normalized categorical fields",), True
        )
    return StructuredComparison(None, 0.0, ("no exact rule applies",), True)

