#!/usr/bin/env python3
"""Sanity-check compatibility wrappers and their run-path targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class WrapperSpec:
    wrapper: str
    target: str


WRAPPERS = [
    WrapperSpec(
        wrapper="run_experiment_r2.py",
        target="experiments/prompt-injection-boundary-tags/rounds/round2b/harness/run_experiment.py",
    ),
    WrapperSpec(
        wrapper="analyze_r2.py",
        target="experiments/prompt-injection-boundary-tags/rounds/round2b/analysis/analyze.py",
    ),
    WrapperSpec(
        wrapper="experiments/prompt-injection-boundary-tags/run_experiment.py",
        target="experiments/prompt-injection-boundary-tags/rounds/round1/harness/run_experiment.py",
    ),
    WrapperSpec(
        wrapper="experiments/prompt-injection-boundary-tags/analyze.py",
        target="experiments/prompt-injection-boundary-tags/rounds/round1/analysis/analyze.py",
    ),
    WrapperSpec(
        wrapper="experiments/prompt-injection-boundary-tags/run_experiment_round2.py",
        target="experiments/prompt-injection-boundary-tags/rounds/round2/harness/run_experiment.py",
    ),
    WrapperSpec(
        wrapper="experiments/prompt-injection-boundary-tags/run_experiment_r2.py",
        target="experiments/prompt-injection-boundary-tags/rounds/round2b/harness/run_experiment.py",
    ),
    WrapperSpec(
        wrapper="experiments/prompt-injection-boundary-tags/analyze_r2.py",
        target="experiments/prompt-injection-boundary-tags/rounds/round2b/analysis/analyze.py",
    ),
]


def validate_wrapper(spec: WrapperSpec) -> list[str]:
    errors: list[str] = []

    wrapper_path = REPO_ROOT / spec.wrapper
    target_path = REPO_ROOT / spec.target

    if not wrapper_path.exists():
        errors.append(f"missing wrapper: {spec.wrapper}")
        return errors

    text = wrapper_path.read_text(encoding="utf-8")
    if "TARGET" not in text:
        errors.append(f"wrapper has no TARGET constant: {spec.wrapper}")
    if "runpy.run_path" not in text:
        errors.append(f"wrapper does not call runpy.run_path: {spec.wrapper}")

    if not target_path.exists():
        errors.append(f"missing wrapper target: {spec.target}")

    return errors


def main() -> None:
    failures: list[str] = []
    for spec in WRAPPERS:
        errors = validate_wrapper(spec)
        if errors:
            failures.extend(errors)
            print(f"FAIL {spec.wrapper}")
        else:
            print(f"OK   {spec.wrapper} -> {spec.target}")

    if failures:
        print("\nWrapper sanity check failed:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print(f"\nWrapper sanity check passed ({len(WRAPPERS)} wrappers).")


if __name__ == "__main__":
    main()
