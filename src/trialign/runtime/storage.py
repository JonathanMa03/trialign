"""Atomic local job persistence, following Modern AutoCrit's save pattern."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".saving-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_path(directory: Path, run_id: str) -> Path:
    if not re.fullmatch(r"run-[a-f0-9]{32}", run_id):
        raise ValueError("Invalid run identifier.")
    return directory / (run_id + ".json")


def read_run(directory: Path, run_id: str) -> dict:
    path = run_path(directory, run_id)
    if not path.is_file():
        raise KeyError("Unknown run.")
    return json.loads(path.read_text(encoding="utf-8"))


def migrate_legacy_runs(root: Path, directory: Path) -> None:
    """Copy old local history once, preserving originals and newer destinations."""
    marker = directory / ".legacy-migration.json"
    legacy = root / "artifacts/demo"
    permanent_marker = legacy / ".runs-migration.json"
    if permanent_marker.exists():
        return
    if marker.exists():
        atomic_write(permanent_marker, {"completed": True})
        return
    copied = []
    for source, target, pattern in [
        (legacy / "runs", directory, "run-*.json"),
        (legacy / "uploads", directory / "uploads", "upload-*.json"),
        (legacy / "imports", directory / "imports", "import-*.json"),
    ]:
        for path in sorted(source.glob(pattern)):
            destination = target / path.name
            if not destination.exists():
                atomic_write(destination, json.loads(path.read_text()))
                copied.append(str(destination.relative_to(directory)))
    payload = {"source": "artifacts/demo", "copied": copied}
    atomic_write(permanent_marker, payload)
    atomic_write(marker, payload)
