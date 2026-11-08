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
