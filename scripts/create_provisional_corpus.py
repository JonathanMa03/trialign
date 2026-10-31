"""Create a provisional corpus for integration testing, not clinical evaluation."""

import random
from pathlib import Path

from trialign.config import load_settings
from trialign.data import load_corpus, write_corpus


def main():
    settings = load_settings()
    source = settings.resolve(settings.data.source_workbook)

    records, report = load_corpus(
        source,
        sheet=settings.data.review_sheet,
        fail_on_invalid=False,
    )
    records = sorted(records, key=lambda record: record.criterion_id)

    if not records:
        raise ValueError("No schema-valid records are available.")

    ids = [record.criterion_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate criterion IDs found; resolve before sampling.")

    # Select reproducibly, ensuring every available trial is represented
    # when the number of trials does not exceed the target sample size.
    rng = random.Random(settings.seed)
    by_trial = {}
    for record in records:
        by_trial.setdefault(record.trial_id, []).append(record)

    target = min(200, len(records))
    trial_ids = sorted(by_trial)

    if len(trial_ids) > target:
        trial_ids = rng.sample(trial_ids, target)

    selected = [rng.choice(by_trial[trial_id]) for trial_id in trial_ids]
    selected_ids = {record.criterion_id for record in selected}

    remaining = [
        record for record in records
        if record.criterion_id not in selected_ids
    ]
    selected.extend(rng.sample(remaining, target - len(selected)))
    selected.sort(key=lambda record: record.criterion_id)

    output = settings.resolve(
        Path("data/processed/criteria_provisional.parquet")
    )
    write_corpus(
        selected,
        output,
        metadata={
            "development_only": True,
            "human_review_complete": False,
            "source_workbook": str(source),
            "source_sha256": report.source_sha256,
            "selection_seed": settings.seed,
            "source_valid_rows": report.valid_rows,
            "source_invalid_rows": report.invalid_rows,
            "fewer_than_100_valid_rows": len(records) < 100,
            "purpose": "Integration smoke test; not benchmark results.",
        },
    )

    print(f"Saved {len(selected)} provisional criteria.")
    print(f"Trials represented: {len({r.trial_id for r in selected})}")
    print(f"Schema-invalid source rows excluded: {report.invalid_rows}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()