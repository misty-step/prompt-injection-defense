from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ROUNDS_DIR = ROOT / "experiments" / "prompt-injection-boundary-tags" / "rounds"

REQUIRED_PRECHECK_HARNESSES = [
    ROUNDS_DIR / "round2" / "harness" / "run_experiment.py",
    ROUNDS_DIR / "round2b" / "harness" / "run_experiment.py",
    ROUNDS_DIR / "round3" / "harness" / "run_experiment.py",
    ROUNDS_DIR / "round4" / "harness" / "run_experiment.py",
    ROUNDS_DIR / "round5" / "harness" / "run_experiment.py",
    ROUNDS_DIR / "round7" / "harness" / "run_experiment.py",
]


class LivePreflightEnforcementTests(unittest.TestCase):
    def test_known_live_harnesses_must_call_preflight(self) -> None:
        for harness_path in REQUIRED_PRECHECK_HARNESSES:
            text = harness_path.read_text()
            self.assertIn(
                "run_live_preflight(",
                text,
                f"{harness_path} must run mandatory preflight before full trials.",
            )

    def test_any_new_live_harness_must_call_preflight(self) -> None:
        for harness_path in sorted(ROUNDS_DIR.glob("round*/harness/run_experiment.py")):
            text = harness_path.read_text()
            if "--live" not in text:
                continue
            self.assertIn(
                "run_live_preflight(",
                text,
                f"{harness_path} defines --live but does not enforce run_live_preflight().",
            )


if __name__ == "__main__":
    unittest.main()
