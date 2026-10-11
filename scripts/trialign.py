#!/usr/bin/env python3
"""Run the source checkout without installing the package first."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trialign.cli import app  # noqa: E402


if __name__ == "__main__":
    app()

