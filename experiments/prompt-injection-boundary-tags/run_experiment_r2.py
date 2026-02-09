#!/usr/bin/env python3
"""Compatibility wrapper for Round 2B harness."""

from pathlib import Path
import runpy

TARGET = (
    Path(__file__).resolve().parent
    / "rounds"
    / "round2b"
    / "harness"
    / "run_experiment.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
