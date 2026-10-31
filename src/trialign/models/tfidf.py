"""Sparse TF-IDF representations of eligibility criteria."""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEncoder:
    name = "tfidf"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            norm="l2",
            dtype=np.float32,
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def encode(self, texts):
        return self.vectorizer.transform(texts)

    def save(self, path):
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path):
        encoder = cls()
        encoder.vectorizer = joblib.load(path)
        return encoder