"""Rank cross-trial candidates using normalized sparse vectors."""

# import numpy as np


def retrieve_neighbors(records, embeddings, top_k=20, model="tfidf"):
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if embeddings.shape[0] != len(records):
        raise ValueError("Each record must have one embedding.")

    neighbors = []

    for i, query in enumerate(records):
        # Calculate one query's scores, avoiding an all-by-all matrix.
        scores = (embeddings[i:i + 1] @ embeddings.T).toarray().ravel()

        candidates = [
            j
            for j, candidate in enumerate(records)
            if candidate.criterion_id != query.criterion_id
            and candidate.trial_id != query.trial_id
        ]

        # Higher scores first; candidate ID breaks ties.
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
                "model_revision": None,
                "text_field": "comparison_text",
            })

    return neighbors