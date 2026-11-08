"""Launcher for the preflight check:  python scripts/preflight.py [--no-camera]

All the logic lives in src/ghostgrip/preflight.py so it can be tested without a camera.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ghostgrip.preflight import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
