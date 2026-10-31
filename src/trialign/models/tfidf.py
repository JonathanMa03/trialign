"""Transparent lexical baseline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class TfidfEncoder:
    name = "tfidf"

    def __init__(self, *, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1):
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=ngram_range, min_df=min_df, sublinear_tf=True
        )
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfEncoder":
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit before encode.")
        matrix = self.vectorizer.transform(texts)
        return normalize(matrix, norm="l2").toarray().astype(np.float32)

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, destination)

