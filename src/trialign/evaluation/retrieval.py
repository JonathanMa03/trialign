"""Small dependency-light retrieval metrics used by every model runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, set[str]],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    queries = [query for query in rankings if relevant.get(query)]
    if not queries:
        return {"queries": 0.0, "mrr": 0.0, **{f"recall@{k}": 0.0 for k in ks}}
    reciprocal_ranks = []
    recalls = {k: [] for k in ks}
    for query in queries:
        ranking = list(rankings[query])
        gold = relevant[query]
        first = next((index for index, item in enumerate(ranking, start=1) if item in gold), None)
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
        for k in ks:
            recalls[k].append(len(set(ranking[:k]) & gold) / len(gold))
    return {
        "queries": float(len(queries)),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        **{f"recall@{k}": sum(values) / len(values) for k, values in recalls.items()},
    }

