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
            # Check for either preflight or dry-run (dry-run is the stronger version)
            has_preflight = "run_live_preflight(" in text
            has_dry_run = "run_live_dry_run(" in text
            self.assertTrue(
                has_preflight or has_dry_run,
                f"{harness_path} must run mandatory preflight/dry-run before full trials.",
            )

    def test_any_new_live_harness_must_call_preflight(self) -> None:
        for harness_path in sorted(ROUNDS_DIR.glob("round*/harness/run_experiment.py")):
            text = harness_path.read_text()
            if "--live" not in text:
                continue
            # Check for either preflight or dry-run (dry-run is the stronger version)
            has_preflight = "run_live_preflight(" in text
            has_dry_run = "run_live_dry_run(" in text
            self.assertTrue(
                has_preflight or has_dry_run,
                f"{harness_path} defines --live but does not enforce run_live_preflight() or run_live_dry_run().",
            )


class LiveDryRunEnforcementTests(unittest.TestCase):
    """Tests that enforce dry-run cannot be skipped for live runs."""

    def test_at_least_one_live_harness_uses_dry_run(self) -> None:
        """At least one live harness must use run_live_dry_run() to validate end-to-end."""
        found_dry_run = False
        for harness_path in sorted(ROUNDS_DIR.glob("round*/harness/run_experiment.py")):
            text = harness_path.read_text()
            if "run_live_dry_run(" in text:
                found_dry_run = True
                break
        self.assertTrue(
            found_dry_run,
            "No live harness uses run_live_dry_run(). "
            "At least one harness must use dry-run to validate end-to-end.",
        )

    def test_any_new_live_harness_must_call_dry_run(self) -> None:
        """Any new live harness added after round8 must call run_live_dry_run()."""
        # Check round9 and beyond
        for harness_path in sorted(ROUNDS_DIR.glob("round9*/harness/run_experiment.py")):
            text = harness_path.read_text()
            if "--live" not in text:
                continue
            self.assertIn(
                "run_live_dry_run(",
                text,
                f"{harness_path} defines --live but does not enforce run_live_dry_run(). "
                "Live runs cannot skip dry-run stage.",
            )


if __name__ == "__main__":
    unittest.main()
