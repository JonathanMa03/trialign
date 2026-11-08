import numpy as np
from scipy import sparse

from trialign.models import TfidfEncoder


def test_sparse_normalized_vectors():
    texts = ["age at least eighteen", "age below eighteen"]
    encoder = TfidfEncoder().fit(texts)
    vectors = encoder.encode(texts)

    assert sparse.issparse(vectors)
    assert vectors.dtype == np.float32
    assert vectors.shape[0] == 2

    lengths = np.sqrt(vectors.multiply(vectors).sum(axis=1))
    np.testing.assert_allclose(np.asarray(lengths), 1.0, atol=1e-6)


def test_save_load_preserves_vectors(tmp_path):
    texts = ["age at least eighteen", "age below eighteen"]
    encoder = TfidfEncoder().fit(texts)
    path = tmp_path / "vectorizer.joblib"

    encoder.save(path)
    restored = TfidfEncoder.load(path)

    np.testing.assert_allclose(
        encoder.encode(texts).toarray(),
        restored.encode(texts).toarray(),
    )


def test_unseen_words_produce_zero_vector():
    encoder = TfidfEncoder().fit(["age eighteen"])
    vector = encoder.encode(["unrecognizedword"])

    assert sparse.issparse(vector)
    assert vector.nnz == 0