from types import SimpleNamespace

from trialign.models import TfidfEncoder
from trialign.models.retrieval import retrieve_neighbors


def test_cross_trial_ranking_and_ties():
    records = [
        SimpleNamespace(criterion_id="q", trial_id="trial1"),
        SimpleNamespace(criterion_id="same_trial", trial_id="trial1"),
        SimpleNamespace(criterion_id="b", trial_id="trial2"),
        SimpleNamespace(criterion_id="a", trial_id="trial3"),
        SimpleNamespace(criterion_id="c", trial_id="trial4"),
    ]
    texts = [
        "age eighteen",
        "age eighteen",
        "age eighteen",
        "age eighteen",
        "kidney function",
    ]
    encoder = TfidfEncoder().fit(texts)
    embeddings = encoder.encode(texts)

    results = retrieve_neighbors(records, embeddings, top_k=10)
    query_results = [r for r in results if r["query_id"] == "q"]

    assert [r["candidate_id"] for r in query_results] == ["a", "b", "c"]
    assert [r["rank"] for r in query_results] == [1, 2, 3]


def test_empty_corpus():
    from scipy.sparse import csr_matrix

    assert retrieve_neighbors([], csr_matrix((0, 0))) == []