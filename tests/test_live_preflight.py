from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "prompt-injection-boundary-tags"
    / "shared"
    / "preflight"
    / "live.py"
)

spec = importlib.util.spec_from_file_location("live_preflight", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from {MODULE_PATH}")
live_preflight = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = live_preflight
spec.loader.exec_module(live_preflight)


class FakeBudget:
    def __init__(self) -> None:
        self.recorded: list[tuple[str, int, int]] = []
        self.stop_for: set[str] = set()

    def before_trial_message(self, model_name: str) -> str | None:
        if model_name in self.stop_for:
            return f"blocked {model_name}"
        return None

    def record_trial(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        self.recorded.append((model_name, input_tokens, output_tokens))
        return 0.001


class LivePreflightTests(unittest.TestCase):
    def test_unique_probe_targets_dedupes(self) -> None:
        targets = [("a", "m1"), ("a", "m1"), ("b", "m2"), ("a", "m1")]
        self.assertEqual(live_preflight.unique_probe_targets(targets), [("a", "m1"), ("b", "m2")])

    def test_non_live_mode_skips_probes(self) -> None:
        called = {"count": 0}

        def probe_call(_target_id: str, _model_name: str):
            called["count"] += 1
            return {"model_id": "x", "input_tokens": 1, "output_tokens": 1}

        results = live_preflight.run_live_preflight(
            mode="simulate",
            targets=[("a", "m1")],
            probe_call=probe_call,
            budget=FakeBudget(),
            fallback_input_tokens=10,
            fallback_output_tokens=5,
        )
        self.assertEqual(results, [])
        self.assertEqual(called["count"], 0)

    def test_success_records_budget_and_uses_fallback_tokens(self) -> None:
        budget = FakeBudget()

        def probe_call(target_id: str, _model_name: str):
            return {"model_id": f"id-{target_id}", "input_tokens": 0, "output_tokens": 0}

        results = live_preflight.run_live_preflight(
            mode="live",
            targets=[("a", "m1"), ("b", "m2")],
            probe_call=probe_call,
            budget=budget,
            fallback_input_tokens=123,
            fallback_output_tokens=45,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(budget.recorded, [("m1", 123, 45), ("m2", 123, 45)])
        self.assertEqual(results[0]["model_id"], "id-a")

    def test_failure_aborts_before_trial_matrix(self) -> None:
        budget = FakeBudget()

        def probe_call(target_id: str, _model_name: str):
            if target_id == "bad":
                raise RuntimeError("model not found")
            return {"model_id": "ok", "input_tokens": 10, "output_tokens": 1}

        with self.assertRaises(SystemExit) as ctx:
            live_preflight.run_live_preflight(
                mode="live",
                targets=[("ok", "m1"), ("bad", "m2")],
                probe_call=probe_call,
                budget=budget,
                fallback_input_tokens=100,
                fallback_output_tokens=20,
            )
        self.assertIn("Live preflight failed", str(ctx.exception))
        self.assertIn("bad", str(ctx.exception))
        self.assertIn("model not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
