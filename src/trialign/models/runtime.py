"""Pinned runtime encoders, explicit input rendering and content-addressed caches."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from trialign.models.dense import DenseEncoder
from trialign.models.tfidf import TfidfEncoder

BASELINES = {
    "mpnet": (
        "sentence-transformers/all-mpnet-base-v2",
        "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    ),
    "biolord": ("FremyCompany/BioLORD-2023", "167aab527b238a50ca65224e6319215d2ff4fc9f"),
}
BASELINE_TOKEN_LIMITS = {"mpnet": 384, "biolord": 128}
SERIALIZER = "criterion-v1"


def serialize_criterion(row):
    """Source inputs only; never annotation rationales, gold labels or trial IDs."""
    fields = [
        "entity",
        "criterion_type",
        "comparator",
        "value",
        "unit",
        "value_kind",
        "lower_bound",
        "lower_inclusive",
        "upper_bound",
        "upper_inclusive",
        "negated",
        "temporal",
        "qualifier",
        "logical_operator",
        "atom_index",
        "parent_atom_count",
        "reference_type",
        "reference_multiplier",
        "source",
        "parent_statement",
        "surrounding_context",
    ]
    values = []
    for field in fields:
        value = getattr(row, field, None)
        if value is None or value == "":
            continue
        if field == "parent_statement" and value == row.source:
            continue
        values.append(f"{field}: {value}")
    return "\n".join(values)


class EncoderOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: Literal["cpu", "cuda", "mps", "auto"] = "cpu"
    batch_size: int = Field(default=16, ge=1, le=256)
    max_seq_length: int = Field(default=384, ge=16, le=8192)
    min_score: float = Field(default=0, ge=-1, le=1)
    ngram_max: int = Field(default=2, ge=1, le=3)
    max_features: int | None = Field(default=None, ge=100, le=500000)


def configured_artifact(root):
    # Server configuration only: the browser cannot supply an arbitrary checkpoint path.
    from dotenv import dotenv_values

    environment_file = Path(os.environ.get("TRIALIGN_ENV_FILE", root / ".env"))
    local = dotenv_values(environment_file) if environment_file.is_file() else {}
    path = Path(
        os.getenv("TRIALIGN_MODEL_PATH")
        or local.get("TRIALIGN_MODEL_PATH")
        or "artifacts/trained/trialign-v1"
    )
    return path if path.is_absolute() else root / path


def runtime_artifact_permitted(manifest):
    """Draft permission is process-local, set by the backend test launch command."""
    return (
        manifest.get("schema_version") == "trialign-joint-1"
        and manifest.get("synthetic") is False
        and (
            not manifest.get("draft_labels", False)
            or os.environ.get("TRIALIGN_ALLOW_DRAFT_RUNTIME") == "1"
        )
    )


def encoder_capabilities(root):
    dense = (
        importlib.util.find_spec("sentence_transformers") is not None
        and importlib.util.find_spec("torch") is not None
    )
    trained, draft = False, False
    try:
        manifest = json.loads((configured_artifact(root) / "manifest.json").read_text())
        trained = dense and runtime_artifact_permitted(manifest)
        draft = bool(manifest.get("draft_labels"))
    except (OSError, ValueError):
        pass
    return [
        {"id": "tfidf", "label": "TF-IDF", "available": True},
        *[
            {
                "id": k,
                "label": k.upper() if k == "mpnet" else "BioLORD",
                "available": dense,
                "revision": v[1],
                "max_seq_length": BASELINE_TOKEN_LIMITS[k],
                "reason": "" if dense else "Install the models optional dependencies.",
            }
            for k, v in BASELINES.items()
        ],
        {
            "id": "trained",
            "label": "TriAlign trained encoder" + (" — draft pilot (testing)" if draft else ""),
            "available": trained,
            "draft_labels": draft,
            "testing_only": trained and draft,
            "reason": "A registered, verified trained artifact is required.",
        },
    ]


class RuntimeEncoder:
    def __init__(self, root, name, options, *, backend=None):
        self.root, self.name, self.options = root, name, options
        self.lock = threading.RLock()
        self.cache_hit = False
        self.backend = backend
        self.artifact_digest = None
        if name == "trained":
            from trialign.models.joint import TrainedModel

            self.backend = TrainedModel(configured_artifact(root), device=options.device)
            if not runtime_artifact_permitted(self.backend.manifest):
                raise ValueError(
                    "Synthetic artifacts are unavailable. Draft pilots require the "
                    "backend serve --pilot-model test command."
                )
            self.artifact_digest = self.backend.artifact_digest
        elif name not in {"tfidf", *BASELINES}:
            raise ValueError("Unknown retrieval model.")
        self.revision = BASELINES[name][1] if name in BASELINES else self.artifact_digest

    def text(self, record):
        return (
            serialize_criterion(record)
            if self.name == "trained"
            else f"{record.entity} {record.source}"
        )

    def fit(self, texts):
        if self.name == "tfidf":
            self.backend = TfidfEncoder()
            self.backend.vectorizer.set_params(
                ngram_range=(1, self.options.ngram_max), max_features=self.options.max_features
            )
            self.backend.fit(texts)
        return self

    def metadata(self):
        return {
            "id": self.name,
            "model_id": BASELINES.get(self.name, (self.name,))[0],
            "revision": self.revision,
            "artifact_sha256": self.artifact_digest,
            "draft_labels": bool(
                self.name == "trained" and self.backend.manifest.get("draft_labels")
            ),
            "testing_only": bool(
                self.name == "trained" and self.backend.manifest.get("draft_labels")
            ),
            "serializer": SERIALIZER if self.name == "trained" else "entity-source-v1",
            "options": self.options.model_dump(),
            "cache_hit": self.cache_hit,
            "effective_max_seq_length": (
                None
                if self.name == "tfidf"
                else self.backend.config.max_seq_length
                if self.name == "trained"
                else self.options.max_seq_length
            ),
            "effective_device": "cpu"
            if self.name == "tfidf"
            else "embedding_cache"
            if self.cache_hit
            else getattr(self.backend, "effective_device", self.options.device),
            "effective_batch_size": None
            if self.name == "tfidf"
            else self.backend.config.batch_size
            if self.name == "trained"
            else self.options.batch_size,
        }

    def encode(self, texts):
        texts = list(texts)
        if self.name == "tfidf":
            return self.backend.encode(texts)
        key = hashlib.sha256(
            json.dumps(
                {
                    "encoder": self.name,
                    "revision": self.revision,
                    "options": {
                        k: v
                        for k, v in self.options.model_dump().items()
                        if k in {"device", "batch_size", "max_seq_length"}
                    },
                    "texts": texts,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        directory = self.root / "artifacts/model_cache/runtime"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (key + ".npy")
        with self.lock:
            if path.is_file():
                try:
                    vectors = np.load(path, allow_pickle=False)
                    self._validate(vectors, len(texts))
                    self.cache_hit = True
                    return vectors
                except (ValueError, OSError):
                    path.unlink(missing_ok=True)
            if self.backend is None:
                model, revision = BASELINES[self.name]
                self.backend = DenseEncoder(
                    model,
                    revision=revision,
                    device=self.options.device,
                    batch_size=self.options.batch_size,
                    max_seq_length=self.options.max_seq_length,
                    cache_folder=self.root / "artifacts/model_cache/huggingface/hub",
                )
            try:
                vectors = self.backend.encode(texts)
            except ImportError as exc:
                raise ValueError(
                    "Install TriAlign's models dependencies to use dense retrieval."
                ) from exc
            except (OSError, RuntimeError) as exc:
                raise ValueError(
                    f"Dense encoder could not load/run ({type(exc).__name__}); "
                    "check model cache, network and device settings."
                ) from exc
            self._validate(vectors, len(texts))
            temporary = directory / (key + "." + uuid4().hex + ".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
            temporary.replace(path)
            self.cache_hit = False
            return vectors

    @staticmethod
    def _validate(vectors, rows):
        if (
            vectors.ndim != 2
            or vectors.shape[0] != rows
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
        ):
            raise ValueError("Dense embeddings must be finite, aligned and unit normalized.")


class EncoderPool:
    def __init__(self, root):
        self.root, self.items = root, OrderedDict()
        self.lock = threading.Lock()

    def get(self, name, options):
        if name == "tfidf":
            return RuntimeEncoder(self.root, name, options)
        # Ranking thresholds do not alter embeddings or require another loaded model.
        key = name + json.dumps(
            {
                k: v
                for k, v in options.model_dump().items()
                if k in {"device", "batch_size", "max_seq_length"}
            },
            sort_keys=True,
        )
        if name == "trained":
            manifest = configured_artifact(self.root) / "manifest.json"
            if not manifest.is_file():
                raise ValueError(
                    "No trained artifact is registered. Set TRIALIGN_MODEL_PATH after training."
                )
            key += hashlib.sha256(manifest.read_bytes()).hexdigest()
        with self.lock:
            if key not in self.items:
                self.items[key] = RuntimeEncoder(self.root, name, options)
            else:
                self.items[key].options = options
            self.items.move_to_end(key)
            while len(self.items) > 2:
                self.items.popitem(last=False)
            return self.items[key]
