"""Configuration loading with paths resolved relative to the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]


class DataConfig(BaseModel):
    source_workbook: Path = Path("data/trial100_notreviewed.xlsx")
    review_sheet: str = "Human Review"
    reviewed_statuses: list[str] = Field(default_factory=lambda: ["accepted", "corrected"])
    processed_dir: Path = Path("data/processed")
    artifact_dir: Path = Path("artifacts")


class SplitConfig(BaseModel):
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    group_column: str = "trial_id"


class Settings(BaseModel):
    project: dict[str, Any] = Field(default_factory=dict)
    data: DataConfig = Field(default_factory=DataConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    pairing: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    agent: dict[str, Any] = Field(default_factory=dict)

    @property
    def seed(self) -> int:
        return int(self.project.get("seed", 20261011))

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else ROOT / "configs/default.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(payload)
    total = settings.splits.train + settings.splits.validation + settings.splits.test
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split proportions must sum to 1.0, got {total}.")
    return settings

