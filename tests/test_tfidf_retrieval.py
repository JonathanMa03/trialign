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


def test_invalid_top_k():
    import pytest
    from scipy.sparse import csr_matrix

    with pytest.raises(ValueError, match="top_k"):
        retrieve_neighbors([], csr_matrix((0, 0)), top_k=0)


def test_embedding_count_must_match_records():
    import pytest
    from scipy.sparse import csr_matrix

    records = [SimpleNamespace(criterion_id="a", trial_id="trial1")]

    with pytest.raises(ValueError, match="one embedding"):
        retrieve_neighbors(records, csr_matrix((0, 2)))


def test_same_trial_has_no_candidates():
    records = [
        SimpleNamespace(criterion_id="a", trial_id="trial1"),
        SimpleNamespace(criterion_id="b", trial_id="trial1"),
    ]
    texts = ["age eighteen", "kidney function"]
    encoder = TfidfEncoder().fit(texts)

    assert retrieve_neighbors(records, encoder.encode(texts)) == []


def test_top_k_limits_results():
    records = [
        SimpleNamespace(criterion_id="a", trial_id="trial1"),
        SimpleNamespace(criterion_id="b", trial_id="trial2"),
        SimpleNamespace(criterion_id="c", trial_id="trial3"),
    ]
    texts = ["age eighteen", "age eighteen", "kidney function"]
    encoder = TfidfEncoder().fit(texts)

    results = retrieve_neighbors(records, encoder.encode(texts), top_k=1)

    assert len(results) == 3
    assert all(row["rank"] == 1 for row in results)
    assert next(
        row["candidate_id"] for row in results if row["query_id"] == "a"
    ) == "b"


def test_dense_and_sparse_retrieval_agree():
    records = [
        SimpleNamespace(criterion_id="a", trial_id="trial1"),
        SimpleNamespace(criterion_id="b", trial_id="trial2"),
        SimpleNamespace(criterion_id="c", trial_id="trial3"),
    ]
    texts = ["age eighteen", "age eighteen", "kidney function"]
    encoder = TfidfEncoder().fit(texts)
    vectors = encoder.encode(texts)

    sparse_results = retrieve_neighbors(records, vectors, top_k=2)
    dense_results = retrieve_neighbors(records, vectors.toarray(), top_k=2)

    assert [
        (r["query_id"], r["candidate_id"], r["rank"])
        for r in sparse_results
    ] == [
        (r["query_id"], r["candidate_id"], r["rank"])
        for r in dense_results
    ]

    import numpy as np

    np.testing.assert_allclose(
        [r["score"] for r in sparse_results],
        [r["score"] for r in dense_results],
        atol=1e-6,
    )


def test_retrieval_preserves_model_metadata():
    import numpy as np

    records = [
        SimpleNamespace(criterion_id="a", trial_id="trial1"),
        SimpleNamespace(criterion_id="b", trial_id="trial2"),
    ]
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    results = retrieve_neighbors(
        records,
        vectors,
        model="synthetic-dense",
        model_revision="test-revision",
        text_field="source",
    )

    assert all(r["model"] == "synthetic-dense" for r in results)
    assert all(r["model_revision"] == "test-revision" for r in results)
    assert all(r["text_field"] == "source" for r in results)