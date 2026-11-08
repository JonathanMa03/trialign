"""Lazy Sentence Transformers adapter; heavy dependencies load only when used."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class SentenceTransformerEncoder:
    def __init__(self, model_id: str, *, revision: str | None = None, query_prefix: str = ""):
        self.name = model_id
        self.model_id = model_id
        self.revision = revision
        self.query_prefix = query_prefix
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, revision=self.revision)
        return self._model

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEncoder":
        del texts
        self._load()
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        inputs = [self.query_prefix + text for text in texts]
        return np.asarray(
            self._load().encode(inputs, normalize_embeddings=True, show_progress_bar=True),
            dtype=np.float32,
        )

    def save(self, path: str) -> None:
        self._load().save(path)

