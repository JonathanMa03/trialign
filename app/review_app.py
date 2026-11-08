"""Launch the local trial-comparison UI from a source checkout."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trialign.web.server import run_browser_app  # noqa: E402

if __name__ == "__main__":
    run_browser_app(ROOT, open_browser=True)
