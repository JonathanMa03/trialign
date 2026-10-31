"""Common embedding interface used by baselines and trained models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> "TextEncoder": ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def save(self, path: str) -> None: ...

