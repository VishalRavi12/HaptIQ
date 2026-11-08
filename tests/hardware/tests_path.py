"""Puts src/ on the import path for the Member C hardware tests (they need no Arduino)."""
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[2] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
