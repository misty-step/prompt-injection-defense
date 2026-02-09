#!/usr/bin/env python3
"""Compatibility wrapper for Round 1 harness."""

from pathlib import Path
import runpy

TARGET = (
    Path(__file__).resolve().parent
    / "rounds"
    / "round1"
    / "harness"
    / "run_experiment.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
