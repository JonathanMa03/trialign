from trialign.schemas import CriterionRecord, HumanDecision, RelationshipLabel


def criterion(**overrides):
    payload = {
        "criterion_id": "criterion-1",
        "trial_id": "NCT00000001",
        "parent_statement_id": "statement-1",
        "source": "Age >= 18 years",
        "parent_statement": "Age >= 18 years",
        "criterion_type": "inclusion",
        "domain": "person",
        "entity": "Age",
        "comparator": ">=",
        "value": 18,
        "unit": "years",
        "lower_bound": 18,
        "lower_inclusive": True,
    }
    payload.update(overrides)
    return CriterionRecord.model_validate(payload)


def test_comparison_text_is_model_ready():
    assert criterion().comparison_text == "Age >= 18 years"


def test_human_decision_requires_final_label():
    decision = HumanDecision(
        decision_id="decision-1",
        pair_id="pair-1",
        reviewer_id="reviewer-1",
        status="accepted",
        final_label=RelationshipLabel.EQUIVALENT,
    )
    assert decision.final_label is RelationshipLabel.EQUIVALENT

