"""Shared criterion encoder and ordered relationship head; torch is optional."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn
from torch.nn import functional as F

from trialign.models.runtime import (
    BASELINE_TOKEN_LIMITS,
    BASELINES,
    SERIALIZER,
    serialize_criterion,
)
from trialign.schemas import RelationshipLabel

LABELS = [label.value for label in RelationshipLabel]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_model: Literal["mpnet", "biolord"] = "mpnet"
    device: Literal["cpu", "auto", "cuda", "mps"] = "cpu"
    max_seq_length: int = Field(default=384, ge=16, le=8192)
    truncation_policy: Literal["error", "allow_recorded"] = "error"
    freeze_encoder: bool = False
    epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=512)
    gradient_accumulation: int = Field(default=1, ge=1, le=128)
    encoder_learning_rate: float = Field(default=2e-5, gt=0, le=1)
    head_learning_rate: float = Field(default=1e-3, gt=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0, le=1)
    warmup_fraction: float = Field(default=0.1, ge=0, lt=1)
    max_grad_norm: float = Field(default=1, gt=0)
    hidden_size: int = Field(default=256, ge=4, le=4096)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    relevance_loss_weight: float = Field(default=1, ge=0)
    relationship_loss_weight: float = Field(default=1, ge=0)
    positive_margin: float = Field(default=1, ge=-1, le=1)
    negative_margin: float = Field(default=0.2, ge=-1, le=1)
    related_topic_negative_weight: float = Field(default=0, ge=0, le=1)
    patience: int = Field(default=3, ge=1, le=1000)
    seed: int = Field(default=20261011, ge=0, le=2147483647)
    deterministic: bool = True
    swap_augmentation: bool = True
    prediction_threshold: float = Field(default=0.6, ge=0, le=1)
    relevance_threshold: float = Field(default=0, ge=-1, le=1)

    @model_validator(mode="after")
    def valid_losses(self):
        if self.base_model == "biolord" and "max_seq_length" not in self.model_fields_set:
            self.max_seq_length = 128
        if self.max_seq_length > BASELINE_TOKEN_LIMITS[self.base_model]:
            raise ValueError("Sequence length exceeds the pinned base model's limit.")
        if not self.relevance_loss_weight and not self.relationship_loss_weight:
            raise ValueError("At least one training loss must be enabled.")
        if self.positive_margin <= self.negative_margin:
            raise ValueError("Positive similarity margin must exceed negative margin.")
        return self


def device_name(requested):
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; select cpu or auto.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; select cpu or auto.")
    return requested


class TinyEncoder(nn.Module):
    """Small trainable encoder ONLY for constructed pipeline checks, never runtime."""

    def __init__(self, dimensions=16, max_seq_length=64):
        super().__init__()
        self.embedding = nn.Embedding(257, dimensions, padding_idx=0)
        self.max_seq_length = max_seq_length
        self.dimensions = dimensions

    def encode_train(self, texts, device):
        tokens = [
            [
                int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % 256 + 1
                for w in re.findall(r"\S+", t)[: self.max_seq_length]
            ]
            or [1]
            for t in texts
        ]
        width = max(map(len, tokens))
        ids = torch.tensor([r + [0] * (width - len(r)) for r in tokens], device=device)
        mask = ids.ne(0).unsqueeze(-1)
        values = self.embedding(ids)
        return (values * mask).sum(1) / mask.sum(1).clamp_min(1)


class JointModel(nn.Module):
    def __init__(self, config, *, encoder=None, synthetic=False, cache_folder=None):
        super().__init__()
        self.config, self.synthetic = config, synthetic
        if encoder is None:
            if synthetic:
                encoder = TinyEncoder(max_seq_length=min(config.max_seq_length, 64))
            else:
                from sentence_transformers import SentenceTransformer

                model, revision = BASELINES[config.base_model]
                encoder = SentenceTransformer(
                    model,
                    revision=revision,
                    device="cpu",
                    cache_folder=str(cache_folder) if cache_folder else None,
                )
                if config.max_seq_length > encoder.max_seq_length:
                    raise ValueError("Configured sequence length exceeds the base encoder limit.")
                encoder.max_seq_length = config.max_seq_length
        self.encoder = encoder
        dimension = encoder.dimensions if synthetic else encoder.get_sentence_embedding_dimension()
        self.head = nn.Sequential(
            nn.LayerNorm(dimension * 5),
            nn.Linear(dimension * 5, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, len(LABELS)),
        )
        if config.freeze_encoder:
            self.encoder.requires_grad_(False)

    def encode_train(self, texts):
        device = next(self.head.parameters()).device
        if self.config.freeze_encoder:
            self.encoder.eval()
        if self.synthetic:
            vectors = self.encoder.encode_train(texts, device)
        else:
            tokens = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in self.encoder.tokenize(texts).items()
            }
            vectors = self.encoder(tokens)["sentence_embedding"]
        return F.normalize(vectors, p=2, dim=1)

    def forward(self, texts_a, texts_b):
        encoded = self.encode_train([*texts_a, *texts_b])
        a, b = encoded[: len(texts_a)], encoded[len(texts_a) :]
        features = torch.cat([a, b, a - b, torch.abs(a - b), a * b], dim=1)
        return (a * b).sum(1), self.head(features)


def read_manifest(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != "trialign-joint-1"
        or manifest.get("serializer") != SERIALIZER
    ):
        raise ValueError("Unsupported model artifact or input serializer.")
    if manifest.get("labels") != LABELS or not manifest.get("files"):
        raise ValueError("Model label map/file manifest is incomplete.")
    for name, expected in manifest["files"].items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory.resolve()) or not file.is_file():
            raise ValueError("Model artifact contains an invalid file path.")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError("Model artifact checksum mismatch: " + name)
    return manifest


class TrainedModel:
    """Both runtime tools share exactly the encoder/head weights in one artifact."""

    def __init__(self, directory, *, device="cpu"):
        from safetensors.torch import load_file

        self.directory = Path(directory)
        self.manifest = read_manifest(directory)
        self.artifact_digest = hashlib.sha256(
            (self.directory / "manifest.json").read_bytes()
        ).hexdigest()
        self.config = TrainingConfig.model_validate(self.manifest["training_config"])
        self.effective_device = device_name(device)
        synthetic = self.manifest["synthetic"]
        if synthetic:
            self.model = JointModel(self.config, synthetic=True)
            self.model.load_state_dict(load_file(str(self.directory / "model.safetensors")))
        else:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(
                str(self.directory / "encoder"), device=self.effective_device, local_files_only=True
            )
            self.model = JointModel(self.config, encoder=encoder)
            self.model.head.load_state_dict(load_file(str(self.directory / "head.safetensors")))
        self.model.to(self.effective_device).eval()

    def encode(self, texts):
        with torch.inference_mode():
            chunks = [
                self.model.encode_train(texts[i : i + self.config.batch_size]).cpu().numpy()
                for i in range(0, len(texts), self.config.batch_size)
            ]
        return np.concatenate(chunks).astype(np.float32)

    def predict_pair(self, a, b):
        texts = [serialize_criterion(a), serialize_criterion(b)]
        truncated = False
        if not self.manifest["synthetic"]:
            tokenizer = self.model.encoder.tokenizer
            truncated = any(
                len(tokenizer(t, truncation=False)["input_ids"]) > self.config.max_seq_length
                for t in texts
            )
        with torch.inference_mode():
            similarity, logits = self.model([texts[0]], [texts[1]])
            probabilities = logits.softmax(1)[0].cpu().tolist()
        index = int(np.argmax(probabilities))
        label = LABELS[index]
        supported = label in self.manifest["trained_relationship_labels"]
        eligible = (
            not truncated
            and supported
            and probabilities[index] >= self.config.prediction_threshold
            and float(similarity[0]) >= self.config.relevance_threshold
        )
        return {
            "proposed_label": label if eligible else None,
            "top_label": label,
            "relevance_score": float(np.clip(float(similarity[0]), -1, 1)),
            "class_scores": dict(zip(LABELS, probabilities, strict=True)),
            "calibrated": False,
            "requires_review": True,
            "artifact_sha256": self.artifact_digest,
            "serializer": SERIALIZER,
            "supported_class": supported,
            "input_truncated": truncated,
            "reason": "Uncalibrated model proposal; source validation "
            "and independent criticism remain required.",
        }
