"""Conservative source-unit diagnostics; warnings are not clinical judgments."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass
class SourceAssessment:
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


HEADINGS = {
    "chemotherapy",
    "radiotherapy",
    "prior therapy",
    "surgery",
    "disease characteristics",
    "inclusion criteria",
    "exclusion criteria",
    "eligibility criteria",
}


def assess_source(row):
    result = SourceAssessment()
    source = row.source.strip()
    parent = row.parent_statement.strip()
    context = "\n".join(
        str(v or "")
        for v in [parent, row.surrounding_context, getattr(row, "governing_context", "")]
    )
    result.blockers.extend(getattr(row, "source_quality_issues", []) or [])
    result.blockers.extend(row.readiness_blockers)
    if source.casefold().strip(" .:;") in HEADINGS and source == parent:
        result.blockers.append(
            "heading_only: selected source is a section heading, not an eligibility predicate"
        )
    numeric = (
        row.comparator is not None or row.lower_bound is not None or row.upper_bound is not None
    )
    bare = len(source.split()) <= 5 and source == parent and not numeric
    operative = re.search(
        r"\b(no|not|must|requires?|required|eligible|exclude[ds]?|prior|history|"
        r"consent|pregnant|breastfeeding|lactating)\b|[<>≤≥]",
        source,
        re.I,
    )
    if bare and row.criterion_type == "inclusion" and not operative:
        governing = str(getattr(row, "governing_context", "") or "")
        if source in governing and re.search(
            r"\b(no|not|without|must|requires?)\b", governing, re.I
        ):
            result.warnings.append(
                "governing_header: fragment must be interpreted with its explicit source heading"
            )
        else:
            result.blockers.append(
                "fragment_scope: bare inclusion fragment lacks a source-grounded predicate"
            )
    if row.logical_operator.casefold() == "and" and row.criterion_type == "exclusion":
        if getattr(row, "parent_atom_count", 1) > 1:
            result.blockers.append(
                "joint_exclusion: selected child cannot be complemented without its AND group"
            )
    if row.criterion_type == "inclusion" and row.logical_operator.casefold() == "or":
        result.warnings.append(
            "inclusion_alternative: retain alternatives; selected branch is not universal"
        )
    if re.search(r"\b(unless|except|provided|allowed|also eligible)\b", context, re.I):
        result.warnings.append(
            "exception_or_permission: preserve applicability; permission is not mandatory exposure"
        )
    if bare and re.search(r"\b(no|not|without)\b", context, re.I) and not operative:
        result.warnings.append(
            "inherited_negation: a surrounding negative heading may govern the selected fragment"
        )
    if re.search(r"positivetherapy|Not Pregnant or breastfeeding women,Pregnancy", context, re.I):
        result.blockers.append(
            "malformed_source: reproductive-status sentence has unresolved negation/test scope"
        )
    if (
        "nyha" in context.casefold()
        and "lvef" in context.casefold()
        and ";" in parent
        and not re.search(r"\b(and|or)\b", parent, re.I)
    ):
        result.blockers.append(
            "boolean_unspecified: NYHA/LVEF semicolon does not establish AND versus OR"
        )
    return result
