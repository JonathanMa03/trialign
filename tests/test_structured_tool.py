from trialign.schemas import CriterionRecord, RelationshipLabel
from trialign.tools import compare_structured


def age(identifier: str, comparator: str, value: int) -> CriterionRecord:
    return CriterionRecord(
        criterion_id=identifier,
        trial_id=f"NCT{identifier[-1] * 8}",
        parent_statement_id=f"statement-{identifier}",
        source=f"Age {comparator} {value}",
        parent_statement=f"Age {comparator} {value}",
        criterion_type="inclusion",
        domain="person",
        entity="Age",
        comparator=comparator,
        value=value,
        unit="years",
    )


def test_disjoint_age_intervals_are_exactly_detected():
    result = compare_structured(age("a1", ">=", 18), age("b2", "<", 18))
    assert result.label is RelationshipLabel.CONTRADICTORY
    assert result.confidence == 1.0
    assert result.requires_context is False

