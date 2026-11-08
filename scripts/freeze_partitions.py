"""Freeze or verify exposure-aware family partitions and the relevance contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialign.data.partitions import freeze_partitions  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    release = freeze_partitions(ROOT, args.output, verify=args.verify)
    print(json.dumps(release["counts"], sort_keys=True))
