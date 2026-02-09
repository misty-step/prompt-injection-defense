from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tools"
    / "calibrate_round2b_scorer.py"
)

spec = importlib.util.spec_from_file_location("calibrate_round2b_scorer", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from {MODULE_PATH}")
calibrate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = calibrate
spec.loader.exec_module(calibrate)


class CalibrateRound2BScorerTests(unittest.TestCase):
    def test_compute_metrics_counts_and_rates(self) -> None:
        pairs = [
            (1, 1),
            (1, 1),
            (1, 0),
            (0, 1),
            (0, 0),
            (0, 0),
        ]
        metrics = calibrate.compute_metrics(pairs)
        self.assertEqual(metrics.tp, 2)
        self.assertEqual(metrics.fp, 1)
        self.assertEqual(metrics.fn, 1)
        self.assertEqual(metrics.tn, 2)
        self.assertTrue(math.isclose(metrics.precision, 2 / 3, rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics.recall, 2 / 3, rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics.f1, 2 / 3, rel_tol=1e-9))

    def test_parse_tool_calls_handles_invalid_json(self) -> None:
        parsed = calibrate.parse_tool_calls("{not-valid-json")
        self.assertEqual(parsed, [])


if __name__ == "__main__":
    unittest.main()
