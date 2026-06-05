"""Convenience wrapper for the standalone CoCoA-like 2-D Seidel experiment."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "run_cocoa_like_2d_mechanism.py"
    runpy.run_path(str(script), run_name="__main__")
