"""Leakage-resistant group splitting utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from trialign.schemas import CriterionRecord


def assign_group_splits(
    records: Iterable[CriterionRecord],
    *,
    seed: int,
    train: float = 0.70,
    validation: float = 0.15,
    family_by_trial: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign whole trial families to train, validation, or test."""
    record_list = list(records)
    family_by_trial = family_by_trial or {}
    families = sorted({family_by_trial.get(row.trial_id, row.trial_id) for row in record_list})
    rng = np.random.default_rng(seed)
    rng.shuffle(families)
    train_end = round(len(families) * train)
    validation_end = train_end + round(len(families) * validation)
    family_split = {
        family: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, family in enumerate(families)
    }
    return {
        row.trial_id: family_split[family_by_trial.get(row.trial_id, row.trial_id)]
        for row in record_list
    }


def assert_no_group_leakage(
    trial_split: Mapping[str, str], family_by_trial: Mapping[str, str] | None = None
) -> None:
    family_by_trial = family_by_trial or {}
    observed: dict[str, set[str]] = {}
    for trial_id, split in trial_split.items():
        family = family_by_trial.get(trial_id, trial_id)
        observed.setdefault(family, set()).add(split)
    leaked = {family: values for family, values in observed.items() if len(values) > 1}
    if leaked:
        raise ValueError(f"Trial-family leakage detected: {leaked}")

