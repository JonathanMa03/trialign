"""Lazy-loaded sentence embeddings for dense retrieval."""

import numpy as np


class DenseEncoder:
    def __init__(
        self,
        model_name,
        *,
        device="cpu",
        revision=None,
        batch_size=32,
        model_factory=None,
    ):
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Unsupported device.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        self.name = model_name
        self.device = device
        self.revision = revision
        self.batch_size = batch_size
        self._model_factory = model_factory
        self._model = None

    def _load_model(self):
        if self._model is None:
            factory = self._model_factory
            if factory is None:
                from sentence_transformers import SentenceTransformer

                factory = SentenceTransformer

            options = {"revision": self.revision}
            if self.device != "auto":
                options["device"] = self.device

            self._model = factory(self.name, **options)

        return self._model

    def encode(self, texts):
        texts = list(texts)
        if not texts:
            raise ValueError("Cannot encode an empty collection.")

        model = self._load_model()
        vectors = np.asarray(
            model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Model returned an unexpected embedding shape.")
        if not np.isfinite(vectors).all():
            raise ValueError("Model returned non-finite embeddings.")

        # Enforce normalization at our interface boundary.
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(lengths == 0):
            raise ValueError("Model returned a zero embedding.")

        return vectors / lengths
