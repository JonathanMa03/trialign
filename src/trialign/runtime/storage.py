"""Atomic local job persistence, following Modern AutoCrit's save pattern."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


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


def saved_run_files(directory: Path):
    """Discover history by its stable ID, including legacy and renamed filenames."""
    seen = set()

    def modified(path):
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    for path in sorted(directory.glob("*.json"), key=modified, reverse=True):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # A file may be temporarily incomplete while edited manually.
        if not isinstance(job, dict) or not re.fullmatch(
            r"run-[a-f0-9]{32}", str(job.get("run_id", ""))
        ):
            continue
        if job["run_id"] in seen:
            continue
        seen.add(job["run_id"])
        yield path, job


def run_path(directory: Path, run_id: str) -> Path:
    if not re.fullmatch(r"run-[a-f0-9]{32}", run_id):
        raise ValueError("Invalid run identifier.")
    legacy = directory / (run_id + ".json")
    if legacy.is_file():
        return legacy
    for suffix in [run_id[4:12], run_id[4:]]:
        for path in directory.glob(f"run_*_{suffix}.json"):
            try:
                if json.loads(path.read_text()).get("run_id") == run_id:
                    return path
            except (OSError, ValueError, AttributeError):
                continue
    for path, job in saved_run_files(directory):
        if job["run_id"] == run_id:
            return path
    return legacy


def read_run(directory: Path, run_id: str) -> dict:
    path = run_path(directory, run_id)
    if not path.is_file():
        raise KeyError("Unknown run.")
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise KeyError("Unknown run.") from exc


def trial_label(job: dict, side: str) -> str:
    identifier = job["trial_" + side]
    if not identifier.startswith("uploaded-"):
        return identifier
    original = next(
        (
            r.get("original_trial_id")
            for r in job.get("criteria_" + side, [])
            if r.get("original_trial_id")
        ),
        None,
    )
    title = str(original or job.get("title_" + side) or identifier)
    nct = re.search(r"NCT\d{8}", title, re.I)
    return nct.group().upper() if nct else title if len(title) <= 40 else title[:37] + "..."


def readable_run_name(job: dict, *, full_id=False) -> str:
    timestamp = datetime.fromisoformat(job["created_at"]).astimezone(ZoneInfo("America/New_York"))
    stamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S_%Z")

    def safe(value):
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")[:40] or "trial"

    identifier = job["run_id"][4:] if full_id else job["run_id"][4:12]
    return (
        f"run_{stamp}_{safe(trial_label(job, 'a'))}_{safe(trial_label(job, 'b'))}_"
        f"{safe(job['status'])}_{identifier}.json"
    )


def write_run(directory: Path, job: dict) -> Path:
    previous = run_path(directory, job["run_id"])
    if previous.is_file():
        saved = json.loads(previous.read_text())
        # Preserve manual annotations even when an in-memory worker saves progress.
        for field in ["note", "display_status"]:
            job[field] = saved.get(field, "")
    target = directory / readable_run_name(job)
    if target.is_file() and target != previous:
        occupant = json.loads(target.read_text())
        if occupant.get("run_id") != job["run_id"]:
            target = directory / readable_run_name(job, full_id=True)
    atomic_write(target, job)
    if previous != target and previous.is_file():
        previous.unlink()
    return target


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
        (legacy / "runs", directory, "run*.json"),
        (legacy / "uploads", directory / "uploads", "upload-*.json"),
        (legacy / "imports", directory / "imports", "import-*.json"),
    ]:
        for path in sorted(source.glob(pattern)):
            destination = target / path.name
            if not destination.exists():
                payload = json.loads(path.read_text())
                if source == legacy / "runs" and isinstance(payload, dict):
                    identifier = payload.get("run_id", "")
                    if (
                        re.fullmatch(r"run-[a-f0-9]{32}", str(identifier))
                        and run_path(target, identifier).is_file()
                    ):
                        continue
                atomic_write(destination, payload)
                copied.append(str(destination.relative_to(directory)))
    payload = {"source": "artifacts/demo", "copied": copied}
    atomic_write(permanent_marker, payload)
    atomic_write(marker, payload)
