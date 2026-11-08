"""Puts src/ and the repo root (shared contracts.py) on the import path for the Member C
hardware tests (they need no Arduino)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
