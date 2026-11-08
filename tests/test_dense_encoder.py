import numpy as np
import pytest

from trialign.models.dense import DenseEncoder


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.tile([3.0, 4.0], (len(texts), 1))


def test_lazy_loading_and_reuse():
    calls = []

    def factory(name, **kwargs):
        calls.append((name, kwargs))
        return FakeModel()

    encoder = DenseEncoder("fake-model", model_factory=factory)
    assert calls == []

    first = encoder.encode(["age eighteen", "kidney function"])
    encoder.encode(["another criterion"])

    assert len(calls) == 1
    assert first.shape == (2, 2)
    assert first.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(first, axis=1), 1.0, atol=1e-6
    )


def test_empty_input_does_not_load_model():
    def factory(*args, **kwargs):
        raise AssertionError("Model should not load.")

    encoder = DenseEncoder("fake-model", model_factory=factory)

    with pytest.raises(ValueError, match="empty"):
        encoder.encode([])


def test_invalid_device():
    with pytest.raises(ValueError, match="device"):
        DenseEncoder("fake-model", device="invalid")