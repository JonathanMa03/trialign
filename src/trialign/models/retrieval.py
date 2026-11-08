"""Rank cross-trial candidates using normalized sparse or dense vectors."""

import numpy as np
from scipy import sparse


def retrieve_neighbors(
    records,
    embeddings,
    top_k=20,
    model="tfidf",
    model_revision=None,
    text_field="comparison_text",
):
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix.")
    if embeddings.shape[0] != len(records):
        raise ValueError("Each record must have one embedding.")
    if text_field not in {"comparison_text", "source"}:
        raise ValueError("text_field must be comparison_text or source.")

    neighbors = []

    for i, query in enumerate(records):
        # Score one query at a time, avoiding an all-by-all matrix.
        similarities = embeddings[i:i + 1] @ embeddings.T

        if sparse.issparse(similarities):
            scores = similarities.toarray().ravel()
        else:
            scores = np.asarray(similarities).ravel()

        candidates = [
            j
            for j, candidate in enumerate(records)
            if candidate.criterion_id != query.criterion_id
            and candidate.trial_id != query.trial_id
        ]

        candidates.sort(
            key=lambda j: (-float(scores[j]), records[j].criterion_id)
        )

        for rank, j in enumerate(candidates[:top_k], start=1):
            neighbors.append({
                "query_id": query.criterion_id,
                "candidate_id": records[j].criterion_id,
                "rank": rank,
                "score": float(scores[j]),
                "model": model,
                "model_revision": model_revision,
                "text_field": text_field,
            })

    return neighbors