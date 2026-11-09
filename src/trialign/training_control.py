"""Local training progress, cooperative pause and atomic optimizer checkpoints."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


def training_directory(output):
    output = Path(output)
    return output.with_name(output.name + ".training")


def request_pause(output):
    directory = training_directory(output)
    status = json.loads((directory / "progress.json").read_text())
    if status["status"] == "paused":
        return status
    if status["status"] != "running":
        raise ValueError("Training is not running; inspect progress.json before resuming.")
    (directory / "pause.request").touch()
    return {"status": "pause_requested", "training_directory": str(directory)}


class TrainingRun:
    def __init__(self, output, *, resume=False, checkpoint_every_updates=10):
        if checkpoint_every_updates < 1:
            raise ValueError("Checkpoint interval must be positive.")
        self.directory = training_directory(output)
        self.resume = resume
        self.interval = checkpoint_every_updates
        self.pause_requested = False
        self.started = time.perf_counter()
        self.elapsed_before = 0.0
        self.previous_signal = None
        self.lock = None

    def __enter__(self):
        import fcntl

        if self.resume:
            if not (self.directory / "checkpoint.pt").is_file():
                raise ValueError("No resumable checkpoint exists for this output.")
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = (self.directory / "worker.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            self.lock = None
            raise ValueError("Another training process owns this run.") from exc
        (self.directory / "pause.request").unlink(missing_ok=True)
        if threading.current_thread() is threading.main_thread():
            self.previous_signal = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._pause_signal)
        return self

    def _pause_signal(self, *_):
        self.pause_requested = True
        print(
            "\nPause requested; finishing the current optimizer update before saving.", flush=True
        )

    @property
    def should_pause(self):
        return self.pause_requested or (self.directory / "pause.request").exists()

    @property
    def elapsed(self):
        return self.elapsed_before + time.perf_counter() - self.started

    def progress(self, *, status="running", phase, **fields):
        event = {
            "time_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "phase": phase,
            "elapsed_seconds": round(self.elapsed, 2),
            **fields,
        }
        temporary = self.directory / "progress.json.tmp"
        temporary.write_text(json.dumps(event, indent=2) + "\n")
        temporary.replace(self.directory / "progress.json")
        with (self.directory / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        message = (
            f"[{event['time_utc']}] {status}: {phase}"
            + (f" | epoch {fields['epoch']}/{fields['epochs']}" if "epoch" in fields else "")
            + (f" | batch {fields['batch']}/{fields['batches']}" if "batch" in fields else "")
            + (f" | loss {fields['loss']:.5f}" if "loss" in fields else "")
            + (
                f" | update {fields['optimizer_updates']}/{fields['total_optimizer_updates']}"
                if "optimizer_updates" in fields
                else ""
            )
            + f" | elapsed {event['elapsed_seconds']:.1f}s"
            + (
                f" | estimated remaining {fields['eta_seconds']:.1f}s"
                if fields.get("eta_seconds") is not None
                else ""
            )
        )
        print(message, flush=True)
        with (self.directory / "training.log").open("a") as handle:
            handle.write(message + "\n")

    def save(self, state):
        import torch

        self.progress(phase="saving checkpoint")
        state = {**state, "elapsed_seconds": self.elapsed}
        temporary = self.directory / "checkpoint.pt.tmp"
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.directory / "checkpoint.pt")

    def load(self):
        import torch

        state = torch.load(self.directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.elapsed_before = state["elapsed_seconds"]
        return state

    def __exit__(self, kind, error, traceback):
        if self.previous_signal is not None:
            signal.signal(signal.SIGINT, self.previous_signal)
        if kind is not None:
            self.progress(status="failed", phase="error", error_type=kind.__name__)
        if self.lock:
            self.lock.close()
