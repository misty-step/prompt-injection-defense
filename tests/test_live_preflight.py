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

    def test_success_applies_fallback_tokens_per_side(self) -> None:
        budget = FakeBudget()

        def probe_call(target_id: str, _model_name: str):
            if target_id == "a":
                return {"model_id": f"id-{target_id}", "input_tokens": 0, "output_tokens": 9}
            return {"model_id": f"id-{target_id}", "input_tokens": 8, "output_tokens": 0}

        results = live_preflight.run_live_preflight(
            mode="live",
            targets=[("a", "m1"), ("b", "m2")],
            probe_call=probe_call,
            budget=budget,
            fallback_input_tokens=123,
            fallback_output_tokens=45,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(budget.recorded, [("m1", 123, 9), ("m2", 8, 45)])
        self.assertEqual(results[0]["model_id"], "id-a")

    def test_budget_stop_aborts_preflight(self) -> None:
        budget = FakeBudget()
        budget.stop_for.add("m1")

        def probe_call(_target_id: str, _model_name: str):
            return {"model_id": "ok", "input_tokens": 10, "output_tokens": 1}

        with self.assertRaises(SystemExit) as ctx:
            live_preflight.run_live_preflight(
                mode="live",
                targets=[("a", "m1")],
                probe_call=probe_call,
                budget=budget,
                fallback_input_tokens=100,
                fallback_output_tokens=20,
            )
        self.assertIn("blocked m1", str(ctx.exception))
        self.assertEqual(budget.recorded, [])

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


class ValidateDryRunResultTests(unittest.TestCase):
    def test_valid_result_no_errors(self) -> None:
        result = {
            "model_id": "gpt-5",
            "response_text": "Hello",
            "raw_tool_calls": [{"name": "tool", "arguments": {"arg": "value"}}],
            "input_tokens": 100,
            "output_tokens": 50,
        }
        errors = live_preflight.validate_dry_run_result("target1", result)
        self.assertEqual(errors, [])

    def test_missing_model_id(self) -> None:
        result = {
            "response_text": "Hello",
            "raw_tool_calls": [],
            "input_tokens": 100,
            "output_tokens": 50,
        }
        errors = live_preflight.validate_dry_run_result("target1", result)
        self.assertTrue(any("model_id" in e for e in errors))

    def test_missing_response_text(self) -> None:
        result = {
            "model_id": "gpt-5",
            "raw_tool_calls": [],
            "input_tokens": 100,
            "output_tokens": 50,
        }
        errors = live_preflight.validate_dry_run_result("target1", result)
        self.assertTrue(any("response_text" in e for e in errors))

    def test_invalid_input_tokens(self) -> None:
        result = {
            "model_id": "gpt-5",
            "response_text": "Hello",
            "raw_tool_calls": [],
            "input_tokens": -1,  # Negative is invalid
            "output_tokens": 50,
        }
        errors = live_preflight.validate_dry_run_result("target1", result)
        self.assertTrue(any("input_tokens" in e for e in errors))

    def test_invalid_output_tokens(self) -> None:
        result = {
            "model_id": "gpt-5",
            "response_text": "Hello",
            "raw_tool_calls": [],
            "input_tokens": 100,
            "output_tokens": -1,
        }
        errors = live_preflight.validate_dry_run_result("target1", result)
        self.assertTrue(any("output_tokens" in e for e in errors))

    def test_tool_calls_not_a_list(self) -> None:
        result = {
            "model_id": "gpt-5",
            "response_text": "Hello",
            "raw_tool_calls": "not a list",
            "input_tokens": 100,
            "output_tokens": 50,
        }
        errors = live_preflight.validate_dry_run_result("target1", result)
        self.assertTrue(any("not a list" in e for e in errors))

    def test_tool_call_not_dict(self) -> None:
        result = {
            "model_id": "gpt-5",
            "response_text": "Hello",
            "raw_tool_calls": ["string instead of dict"],
            "input_tokens": 100,
            "output_tokens": 50,
        }
        errors = live_preflight.validate_dry_run_result("target1", result)
        self.assertTrue(any("not a dict" in e for e in errors))

    def test_tool_call_missing_arguments(self) -> None:
        result = {
            "model_id": "gpt-5",
            "response_text": "Hello",
            "raw_tool_calls": [{"name": "tool"}],
            "input_tokens": 100,
            "output_tokens": 50,
        }
        errors = live_preflight.validate_dry_run_result("target1", result)
        self.assertTrue(any("missing arguments" in e for e in errors))

    def test_tool_call_arguments_not_parsed(self) -> None:
        # Arguments should be a dict (parsed JSON), not a raw string
        result = {
            "model_id": "gpt-5",
            "response_text": "Hello",
            "raw_tool_calls": [{"name": "tool", "arguments": "raw string"}],
            "input_tokens": 100,
            "output_tokens": 50,
        }
        errors = live_preflight.validate_dry_run_result("target1", result)
        self.assertTrue(any("not parsed" in e for e in errors))

    def test_skip_tool_parsing_when_disabled(self) -> None:
        # When check_tool_parsing=False, tool validation should be skipped
        result = {
            "model_id": "gpt-5",
            "response_text": "Hello",
            "raw_tool_calls": "invalid",
            "input_tokens": 100,
            "output_tokens": 50,
        }
        errors = live_preflight.validate_dry_run_result(
            "target1", result, check_tool_parsing=False
        )
        self.assertEqual(errors, [])


class RunLiveDryRunTests(unittest.TestCase):
    def test_non_live_mode_skips_dry_run(self) -> None:
        called = {"count": 0}

        def trial_call(_target_id: str, _model_name: str):
            called["count"] += 1
            return {"model_id": "x", "response_text": "ok", "raw_tool_calls": [], "input_tokens": 1, "output_tokens": 1}

        results = live_preflight.run_live_dry_run(
            mode="simulate",
            targets=[("a", "m1")],
            trial_call=trial_call,
            budget=FakeBudget(),
            fallback_input_tokens=10,
            fallback_output_tokens=5,
        )
        self.assertEqual(results, [])
        self.assertEqual(called["count"], 0)

    def test_dry_run_success_records_budget(self) -> None:
        budget = FakeBudget()

        def trial_call(target_id: str, _model_name: str):
            return {
                "model_id": f"id-{target_id}",
                "response_text": "test response",
                "raw_tool_calls": [{"name": "test_tool", "arguments": {"key": "value"}}],
                "input_tokens": 100,
                "output_tokens": 50,
            }

        results = live_preflight.run_live_dry_run(
            mode="live",
            targets=[("a", "m1"), ("b", "m2")],
            trial_call=trial_call,
            budget=budget,
            fallback_input_tokens=123,
            fallback_output_tokens=45,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(budget.recorded, [("m1", 100, 50), ("m2", 100, 50)])
        self.assertEqual(results[0]["model_id"], "id-a")
        self.assertIn("raw_tool_calls", results[0])
        self.assertIn("response_text", results[0])

    def test_dry_run_validates_result(self) -> None:
        budget = FakeBudget()

        def trial_call(_target_id: str, _model_name: str):
            # Missing model_id - should fail validation
            return {
                "response_text": "test response",
                "raw_tool_calls": [],
                "input_tokens": 100,
                "output_tokens": 50,
            }

        with self.assertRaises(SystemExit) as ctx:
            live_preflight.run_live_dry_run(
                mode="live",
                targets=[("a", "m1")],
                trial_call=trial_call,
                budget=budget,
                fallback_input_tokens=100,
                fallback_output_tokens=20,
            )
        self.assertIn("Live dry-run failed", str(ctx.exception))
        self.assertIn("model_id", str(ctx.exception))

    def test_dry_run_fails_fast_on_validation_error(self) -> None:
        budget = FakeBudget()
        call_count = {"count": 0}

        def trial_call(target_id: str, _model_name: str):
            call_count["count"] += 1
            if target_id == "bad":
                # Missing model_id - validation fails
                return {
                    "response_text": "test",
                    "raw_tool_calls": [],
                    "input_tokens": 10,
                    "output_tokens": 5,
                }
            return {
                "model_id": "ok",
                "response_text": "test",
                "raw_tool_calls": [],
                "input_tokens": 10,
                "output_tokens": 5,
            }

        with self.assertRaises(SystemExit) as ctx:
            live_preflight.run_live_dry_run(
                mode="live",
                targets=[("ok", "m1"), ("bad", "m2")],
                trial_call=trial_call,
                budget=budget,
                fallback_input_tokens=100,
                fallback_output_tokens=20,
            )
        self.assertIn("Live dry-run failed", str(ctx.exception))
        self.assertIn("bad", str(ctx.exception))
        # First target should have been called, then fail-fast
        self.assertEqual(call_count["count"], 2)

    def test_dry_run_exception_aborts(self) -> None:
        budget = FakeBudget()

        def trial_call(target_id: str, _model_name: str):
            if target_id == "bad":
                raise RuntimeError("API error")
            return {
                "model_id": "ok",
                "response_text": "test",
                "raw_tool_calls": [],
                "input_tokens": 10,
                "output_tokens": 5,
            }

        with self.assertRaises(SystemExit) as ctx:
            live_preflight.run_live_dry_run(
                mode="live",
                targets=[("ok", "m1"), ("bad", "m2")],
                trial_call=trial_call,
                budget=budget,
                fallback_input_tokens=100,
                fallback_output_tokens=20,
            )
        self.assertIn("Live dry-run failed", str(ctx.exception))
        self.assertIn("API error", str(ctx.exception))

    def test_dry_run_budget_stop_aborts(self) -> None:
        budget = FakeBudget()
        budget.stop_for.add("m1")

        def trial_call(_target_id: str, _model_name: str):
            return {
                "model_id": "ok",
                "response_text": "test",
                "raw_tool_calls": [],
                "input_tokens": 10,
                "output_tokens": 5,
            }

        with self.assertRaises(SystemExit) as ctx:
            live_preflight.run_live_dry_run(
                mode="live",
                targets=[("a", "m1")],
                trial_call=trial_call,
                budget=budget,
                fallback_input_tokens=100,
                fallback_output_tokens=20,
            )
        self.assertIn("blocked m1", str(ctx.exception))
        self.assertEqual(budget.recorded, [])

    def test_dry_run_empty_targets_raises(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            live_preflight.run_live_dry_run(
                mode="live",
                targets=[],
                trial_call=lambda t, m: {},
                budget=FakeBudget(),
                fallback_input_tokens=100,
                fallback_output_tokens=20,
            )
        self.assertIn("at least one trial target", str(ctx.exception))

    def test_dry_run_uses_fallback_tokens(self) -> None:
        budget = FakeBudget()

        def trial_call(_target_id: str, _model_name: str):
            return {
                "model_id": "ok",
                "response_text": "test",
                "raw_tool_calls": [],
                "input_tokens": 0,  # Zero - should use fallback
                "output_tokens": 0,  # Zero - should use fallback
            }

        results = live_preflight.run_live_dry_run(
            mode="live",
            targets=[("a", "m1")],
            trial_call=trial_call,
            budget=budget,
            fallback_input_tokens=200,
            fallback_output_tokens=100,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(budget.recorded, [("m1", 200, 100)])


if __name__ == "__main__":
    unittest.main()
