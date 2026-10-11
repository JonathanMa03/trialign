from trialign.data.splits import assert_no_group_leakage, assign_group_splits
from trialign.pairing import generate_candidate_pairs
from trialign.schemas import CriterionRecord


def make_record(index: int, trial: int, entity: str = "Age") -> CriterionRecord:
    return CriterionRecord(
        criterion_id=f"criterion-{index}",
        trial_id=f"NCT{trial:08d}",
        parent_statement_id=f"statement-{index}",
        source="Age >= 18 years",
        parent_statement="Age >= 18 years",
        criterion_type="inclusion",
        domain="person",
        entity=entity,
        comparator=">=",
        value=18,
        unit="years",
    )


def test_split_and_pair_generation_are_reproducible_and_cross_trial():
    records = [make_record(index, index // 2) for index in range(20)]
    splits = assign_group_splits(records, seed=7)
    assert_no_group_leakage(splits)
    first = generate_candidate_pairs(
        records, seed=7, random_pairs=10, same_entity_pairs=20, split_by_trial=splits
    )
    second = generate_candidate_pairs(
        records, seed=7, random_pairs=10, same_entity_pairs=20, split_by_trial=splits
    )
    assert [pair.pair_id for pair in first] == [pair.pair_id for pair in second]
    by_id = {row.criterion_id: row for row in records}
    assert all(
        by_id[pair.criterion_a_id].trial_id != by_id[pair.criterion_b_id].trial_id
        for pair in first
    )

