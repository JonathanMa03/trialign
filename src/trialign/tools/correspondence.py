"""Conservative counterpart gate; this does not determine clinical relationships."""

from __future__ import annotations

import re

from trialign.tools.structured import canonical_entity


class NoClinicalCounterpart(RuntimeError):
    pass


VERSION = "clinical-one-to-one-1"
UNKNOWN = {"", "unspecified", "unknown", "none", "criterion", "eligibility criterion"}
# Correspondence aliases only: these do not authorize exact interval comparisons.
CONCEPTS = {
    "pregnancy": {"pregnancy", "pregnancy status", "pregnant"},
    "lactation": {"lactation", "lactation status", "breastfeeding", "nursing"},
    "total bilirubin": {"total bilirubin", "bilirubin total"},
    "radiotherapy": {"radiotherapy", "radiation therapy", "radiation treatment"},
}
SOURCE_PATTERNS = {
    "pregnancy": r"\bpregnan(?:t|cy)\b",
    "lactation": r"\b(?:lactat\w*|breastfeeding|breast feeding|nursing)\b",
    "total bilirubin": r"\btotal bilirubin\b",
    "age": r"\b(?:age|aged)\b",
    "hemoglobin": r"\b(?:hemoglobin|haemoglobin|hgb)\b",
    "platelets": r"\bplatelets?\b",
    "creatinine": r"\bcreatinine\b",
    "anc": r"\b(?:anc|absolute neutrophil count)\b",
    "ecog": r"\becog\b",
    "lvef": r"\b(?:lvef|left ventricular ejection fraction)\b",
}


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def concept(row):
    entity = normalize(canonical_entity(row.entity))
    if entity not in UNKNOWN:
        for name, aliases in CONCEPTS.items():
            if entity in aliases:
                return name
        return entity
    # Only unambiguous source-local concepts are inferred. Parent prose is excluded.
    matches = [
        name for name, pattern in SOURCE_PATTERNS.items() if re.search(pattern, row.source, re.I)
    ]
    return matches[0] if len(matches) == 1 else None


def correspondence(a, b):
    left, right = concept(a), concept(b)
    if left and right:
        return left == right, f"Clinical concepts: {left} / {right}"
    # Untyped identical wording can be compared, without inventing an entity mapping.
    identical = normalize(a.source) == normalize(b.source) and len(normalize(a.source)) >= 12
    if identical and not left and not right:
        return True, "Identical selected wording; clinical scope still requires review"
    return False, "No supported common clinical concept; leave unassigned"
