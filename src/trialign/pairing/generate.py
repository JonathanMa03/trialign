"""Reproducible candidate-pair generation with sampling provenance."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from trialign.schemas import CriterionRecord, PairCandidate


def stable_pair_id(a_id: str, b_id: str) -> str:
    left, right = sorted((a_id, b_id))
    digest = hashlib.sha256(f"{left}\x1f{right}".encode()).hexdigest()[:20]
    return f"pair-{digest}"


def _pair(
    a: CriterionRecord,
    b: CriterionRecord,
    *,
    method: str,
    split_by_trial: Mapping[str, str] | None,
) -> PairCandidate | None:
    if a.criterion_id == b.criterion_id or a.trial_id == b.trial_id:
        return None
    split_a = split_by_trial.get(a.trial_id) if split_by_trial else None
    split_b = split_by_trial.get(b.trial_id) if split_by_trial else None
    if split_a and split_b and split_a != split_b:
        return None
    return PairCandidate(
        pair_id=stable_pair_id(a.criterion_id, b.criterion_id),
        criterion_a_id=a.criterion_id,
        criterion_b_id=b.criterion_id,
        split=split_a or split_b or "unassigned",
        sampling_method=method,
    )


def generate_candidate_pairs(
    records: Iterable[CriterionRecord],
    *,
    seed: int,
    random_pairs: int = 1000,
    same_entity_pairs: int = 2000,
    split_by_trial: Mapping[str, str] | None = None,
) -> list[PairCandidate]:
    """Create random controls and same-entity candidates without split leakage."""
    rows = list(records)
    rng = np.random.default_rng(seed)
    pairs: dict[str, PairCandidate] = {}

    attempts = 0
    max_attempts = max(random_pairs * 50, 1000)
    while sum(pair.sampling_method == "random_control" for pair in pairs.values()) < random_pairs:
        if len(rows) < 2 or attempts >= max_attempts:
            break
        attempts += 1
        indexes = rng.choice(len(rows), size=2, replace=False)
        candidate = _pair(
            rows[int(indexes[0])], rows[int(indexes[1])],
            method="random_control", split_by_trial=split_by_trial,
        )
        if candidate:
            pairs.setdefault(candidate.pair_id, candidate)

    groups: dict[str, list[CriterionRecord]] = defaultdict(list)
    for row in rows:
        key = " ".join(row.entity.casefold().split())
        if key:
            groups[key].append(row)
    entity_candidates: list[PairCandidate] = []
    for group in groups.values():
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                candidate = _pair(
                    group[left], group[right], method="same_entity", split_by_trial=split_by_trial
                )
                if candidate and candidate.pair_id not in pairs:
                    entity_candidates.append(candidate)
    rng.shuffle(entity_candidates)
    for candidate in entity_candidates[:same_entity_pairs]:
        pairs[candidate.pair_id] = candidate
    return list(pairs.values())

