"""Source-gated interval and categorical tools for criterion comparisons."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from trialign.schemas import CriterionRecord, RelationshipLabel

VERSION = "0.2.0"
ALIASES = {
    "age": {"age", "minimum age", "patient age"},
    "hemoglobin": {"hemoglobin", "haemoglobin", "hgb", "hb"},
    "platelets": {"platelets", "platelet count", "platelet"},
    "anc": {"anc", "absolute neutrophil count", "neutrophil count"},
    "creatinine": {"creatinine", "serum creatinine"},
    "lvef": {"lvef", "left ventricular ejection fraction"},
    "ecog": {"ecog", "ecog performance status", "performance status ecog"},
    "alt": {"alt", "alanine aminotransferase", "alanine transaminase"},
    "ast": {"ast", "aspartate aminotransferase", "aspartate transaminase"},
}
CONTEXT = re.compile(
    r"\b(unless|except|if|provided|uncontrolled|within|before|after|prior|history|"
    r"historical|transfusion|gilbert|leukemic|symptomatic|severe|screening|not|no)\b",
    re.I,
)
Interval = tuple[float | None, bool, float | None, bool]


@dataclass(frozen=True)
class StructuredComparison:
    label: RelationshipLabel | None
    confidence: float
    reasons: tuple[str, ...]
    requires_context: bool


def canonical_entity(text: str) -> str:
    value = re.sub(r"\s+", " ", text.casefold().strip())
    for canonical, terms in ALIASES.items():
        if value in terms:
            return canonical
    return value


def source_text(row: CriterionRecord) -> str:
    return "\n".join(
        str(x or "")
        for x in [
            row.source,
            row.parent_statement,
            row.surrounding_context,
            getattr(row, "governing_context", ""),
            getattr(row, "retrieved_context", ""),
        ]
    )


def frame_issues(row: CriterionRecord) -> list[str]:
    issues = list(getattr(row, "source_quality_issues", []) or [])
    if row.criterion_type not in {"inclusion", "exclusion"}:
        issues.append("eligibility role is unspecified")
    if row.review_status not in {"accepted", "corrected"}:
        issues.append("criterion has not been accepted or corrected")
    if row.readiness_blockers:
        issues.append("source has readiness blockers")
    if row.qualifier or row.temporal or row.repetition:
        issues.append("timing, repetition or an exception needs contextual interpretation")
    text = "\n".join(
        [row.source, row.parent_statement, str(getattr(row, "governing_context", "") or "")]
    )
    if CONTEXT.search(text):
        issues.append("source contains a governing context or severity condition")
    if row.logical_operator.casefold() == "or" and row.criterion_type == "inclusion":
        issues.append("inclusion alternative must retain its governing group")
    if row.negated:
        issues.append("internal negation requires source-level interpretation")
    if row.criterion_type == "inclusion" and len(row.source.split()) <= 2:
        issues.append("short inclusion fragment does not establish an independent predicate")
    return list(dict.fromkeys(issues))


def _interval(row: CriterionRecord) -> Interval | None:
    if row.lower_bound is not None or row.upper_bound is not None:
        if row.lower_bound is not None and row.lower_inclusive is None:
            return None
        if row.upper_bound is not None and row.upper_inclusive is None:
            return None
        return (
            row.lower_bound,
            bool(row.lower_inclusive),
            row.upper_bound,
            bool(row.upper_inclusive),
        )
    if isinstance(row.value, bool):
        return None
    try:
        value = float(row.value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return {
        ">": (value, False, None, False),
        ">=": (value, True, None, False),
        "<": (None, False, value, False),
        "<=": (None, False, value, True),
        "=": (value, True, value, True),
        "==": (value, True, value, True),
    }.get(row.comparator or "")


def _supported_by_source(row: CriterionRecord, interval: Interval) -> bool:
    text = row.source.replace("\\", "").replace(",", "").casefold()
    if interval[0] is not None and interval[2] is not None and interval[1] and interval[3]:
        for lower, upper in re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)", text):
            if (float(lower), True, float(upper), True) == interval:
                return True
    patterns = {
        ">=": r"(?:>=|≥|at least)\s*(\d+(?:\.\d+)?)",
        ">": r"(?:(?<![<>=])>(?!=)|above|over|older than)\s*(\d+(?:\.\d+)?)",
        "<=": r"(?:<=|≤|at most)\s*(\d+(?:\.\d+)?)",
        "<": r"(?:(?<![<>=])<(?!=)|below|less than)\s*(\d+(?:\.\d+)?)",
        "=": r"(?:(?<![<>=])=(?!=)|equal to)\s*(\d+(?:\.\d+)?)",
    }
    pattern = patterns.get(row.comparator or "")
    values = [float(n) for n in re.findall(pattern, text)] if pattern else []
    if row.comparator == ">=":
        values += [float(n) for n in re.findall(r"(\d+)\s*years? (?:of age )?or older", text)]
    for value in values:
        expected = {
            ">": (value, False, None, False),
            ">=": (value, True, None, False),
            "<": (None, False, value, False),
            "<=": (None, False, value, True),
            "=": (value, True, value, True),
        }[row.comparator]
        if interval == expected:
            return True
    return False


def _unit(row: CriterionRecord) -> tuple[str, float]:
    value = (row.unit or "").casefold().replace(" ", "").replace("μ", "u").replace("µ", "u")
    value = value.replace("\\", "")
    if value in {"g/l", "g/dl"}:
        return "mass:g/dl", 0.1 if value == "g/l" else 1.0
    if value in {"10^9/l", "x10^9/l", "10^9cells/l"}:
        return "count/ul", 1000.0
    if value in {
        "ul",
        "/ul",
        "cells/ul",
        "/mcl",
        "cells/mcl",
        "mcl",
        "/mm3",
        "cells/mm3",
        "mm3",
        "/mm^3",
        "cells/mm^3",
    }:
        return "count/ul", 1.0
    if value in {"year", "years", "yr", "yrs"}:
        return "years", 1.0
    if value in {"percent", "%"}:
        return "%", 1.0
    if canonical_entity(row.entity) == "ecog" and value in {"", "score", "grade", "points"}:
        return "ecog", 1.0
    return value, 1.0


def _scale(interval: Interval, factor: float) -> Interval:
    lo, li, hi, ui = interval
    return (None if lo is None else lo * factor, li, None if hi is None else hi * factor, ui)


def _complement(interval: Interval) -> list[Interval]:
    lo, li, hi, ui = interval
    result = []
    if lo is not None:
        result.append((None, False, lo, not li))
    if hi is not None:
        result.append((hi, not ui, None, False))
    return result


def _disjoint(a: Interval, b: Interval) -> bool:
    if a[2] is not None and b[0] is not None:
        if a[2] < b[0] or a[2] == b[0] and not (a[3] and b[1]):
            return True
    if b[2] is not None and a[0] is not None:
        if b[2] < a[0] or b[2] == a[0] and not (b[3] and a[1]):
            return True
    return False


def _subset(a: Interval, b: Interval) -> bool:
    lower = b[0] is None or (
        a[0] is not None and (a[0] > b[0] or a[0] == b[0] and (not a[1] or b[1]))
    )
    upper = b[2] is None or (
        a[2] is not None and (a[2] < b[2] or a[2] == b[2] and (not a[3] or b[3]))
    )
    return lower and upper


def _sets_subset(a: list[Interval], b: list[Interval]) -> bool:
    return all(any(_subset(x, y) for y in b) for x in a)


def _allowed(row: CriterionRecord, interval: Interval) -> list[Interval]:
    values = _complement(interval) if row.criterion_type == "exclusion" else [interval]
    if canonical_entity(row.entity) == "ecog":
        # ECOG is discrete, not a continuous score that admits e.g. 1.5.
        scores = []
        for score in range(6):
            point = (float(score), True, float(score), True)
            if any(_subset(point, item) for item in values):
                scores.append(point)
        return scores
    entity = canonical_entity(row.entity)
    if entity in {"age", "hemoglobin", "platelets", "anc", "creatinine", "alt", "ast", "lvef"}:
        restricted = []
        maximum = 100.0 if entity == "lvef" and _unit(row)[0] == "%" else None
        for lo, li, hi, ui in values:
            if lo is None or lo < 0:
                lo, li = 0.0, True
            if maximum is not None and (hi is None or hi > maximum):
                hi, ui = maximum, True
            if hi is not None and (lo > hi or lo == hi and not (li and ui)):
                continue
            restricted.append((lo, li, hi, ui))
        return restricted
    return values


def compare_structured(a: CriterionRecord, b: CriterionRecord) -> StructuredComparison:
    issues = [f"A: {x}" for x in frame_issues(a)] + [f"B: {x}" for x in frame_issues(b)]
    if issues:
        return StructuredComparison(None, 0.0, tuple(issues), True)
    if canonical_entity(a.entity) != canonical_entity(b.entity):
        return StructuredComparison(None, 0.0, ("entities require a supported mapping",), True)
    ua, fa = _unit(a)
    ub, fb = _unit(b)
    if ua != ub:
        return StructuredComparison(
            RelationshipLabel.RELATED_NOT_COMPARABLE,
            0.0,
            ("unit/reference bases differ; no conversion is assumed",),
            True,
        )
    ia, ib = _interval(a), _interval(b)
    if ia is not None and ib is not None:
        if not _supported_by_source(a, ia) or not _supported_by_source(b, ib):
            return StructuredComparison(
                None, 0.0, ("numeric fields lack literal source support",), True
            )
        if any(v is not None and not math.isfinite(v) for i in [ia, ib] for v in [i[0], i[2]]):
            return StructuredComparison(None, 0.0, ("non-finite numeric input",), True)
        aa, bb = _allowed(a, _scale(ia, fa)), _allowed(b, _scale(ib, fb))
        if not aa or not bb:
            return StructuredComparison(
                None, 0.0, ("empty allowed predicate requires source review",), True
            )
        ab, ba = _sets_subset(aa, bb), _sets_subset(bb, aa)
        if ab and ba:
            label = RelationshipLabel.EQUIVALENT
        elif ab:
            label = RelationshipLabel.A_MORE_RESTRICTIVE
        elif ba:
            label = RelationshipLabel.B_MORE_RESTRICTIVE
        elif all(_disjoint(x, y) for x in aa for y in bb):
            label = RelationshipLabel.CONTRADICTORY
        else:
            label = RelationshipLabel.PARTIAL_OVERLAP
        reason = (
            f"Allowed numeric sets in {ua or 'the stated unit'}: A={aa}; B={bb}. "
            "Eligibility role applied once; comparison is limited to these source-gated rules."
        )
        return StructuredComparison(label, 1.0, (reason,), False)

    def normalize(text):
        return re.sub(r"\s+", " ", text.casefold()).strip()

    if (
        normalize(a.source) == normalize(b.source)
        and a.criterion_type == b.criterion_type
        and a.negated == b.negated
    ):
        return StructuredComparison(
            RelationshipLabel.EQUIVALENT,
            1.0,
            ("identical source predicates and eligibility roles",),
            False,
        )
    return StructuredComparison(None, 0.0, ("contextual clinical judgment is required",), True)
