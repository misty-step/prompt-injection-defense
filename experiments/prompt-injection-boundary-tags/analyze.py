#!/usr/bin/env python3
"""Compatibility wrapper for Round 1 analysis."""

from pathlib import Path
import runpy

TARGET = (
    Path(__file__).resolve().parent
    / "rounds"
    / "round1"
    / "analysis"
    / "analyze.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
