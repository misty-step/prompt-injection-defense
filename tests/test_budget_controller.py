from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "prompt-injection-boundary-tags"
    / "shared"
    / "budget"
    / "controller.py"
)

spec = importlib.util.spec_from_file_location("budget_controller", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from {MODULE_PATH}")
budget_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = budget_module
spec.loader.exec_module(budget_module)


def fake_estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    model_weight = {
        "a": 1.0,
        "b": 2.0,
    }.get(model_name, 1.0)
    return model_weight * (input_tokens + output_tokens) / 1000.0


class BudgetControllerTests(unittest.TestCase):
    def test_projected_total_cost_sums_trial_plan(self) -> None:
        settings = budget_module.BudgetSettings(
            estimate_input_tokens=100,
            estimate_output_tokens=50,
        )
        controller = budget_module.BudgetController(
            settings=settings,
            estimate_cost_usd=fake_estimate_cost,
            run_id="r",
            timestamp="t",
            mode="simulate",
            planned_trial_models=["a", "a", "b"],
            default_report_path=Path("/tmp/ignored.json"),
        )
        # a: 0.15 each (x2), b: 0.30 (x1)
        self.assertAlmostEqual(controller.projected_total_cost_usd, 0.60, places=8)

    def test_preflight_message_when_over_cap(self) -> None:
        settings = budget_module.BudgetSettings(
            max_cost_usd=0.10,
            estimate_input_tokens=100,
            estimate_output_tokens=50,
        )
        controller = budget_module.BudgetController(
            settings=settings,
            estimate_cost_usd=fake_estimate_cost,
            run_id="r",
            timestamp="t",
            mode="simulate",
            planned_trial_models=["a"],
            default_report_path=Path("/tmp/ignored.json"),
        )
        message = controller.preflight_message()
        self.assertIsNotNone(message)
        self.assertIn("Projected run cost", str(message))

    def test_before_trial_checks_per_trial_and_run_caps(self) -> None:
        settings = budget_module.BudgetSettings(
            max_cost_usd=1.0,
            max_cost_per_trial_usd=0.20,
            guard_input_tokens=100,
            guard_output_tokens=50,
        )
        controller = budget_module.BudgetController(
            settings=settings,
            estimate_cost_usd=fake_estimate_cost,
            run_id="r",
            timestamp="t",
            mode="simulate",
            planned_trial_models=["b"],
            default_report_path=Path("/tmp/ignored.json"),
        )
        # b guard cost = 2 * 0.15 = 0.30, exceeds per-trial cap 0.20
        message = controller.before_trial_message("b")
        self.assertIsNotNone(message)
        self.assertIn("Per-trial guard", str(message))

    def test_record_trial_updates_spend_and_observed_max(self) -> None:
        settings = budget_module.BudgetSettings()
        controller = budget_module.BudgetController(
            settings=settings,
            estimate_cost_usd=fake_estimate_cost,
            run_id="r",
            timestamp="t",
            mode="simulate",
            planned_trial_models=["a"],
            default_report_path=Path("/tmp/ignored.json"),
        )
        controller.record_trial("a", 10, 20)
        controller.record_trial("a", 12, 18)
        self.assertGreater(controller.spent_cost_usd, 0.0)
        self.assertEqual(controller.observed_max_tokens["a"], (12, 20))

    def test_write_report_outputs_json(self) -> None:
        settings = budget_module.BudgetSettings(max_cost_usd=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "budget.json"
            controller = budget_module.BudgetController(
                settings=settings,
                estimate_cost_usd=fake_estimate_cost,
                run_id="r",
                timestamp="t",
                mode="simulate",
                planned_trial_models=["a"],
                default_report_path=path,
            )
            output_path = controller.write_report(completed_trials=0, stop_reason="stopped")
            self.assertIsNotNone(output_path)
            self.assertTrue(path.exists())
            content = path.read_text()
            self.assertIn("\"budget_stop_reason\": \"stopped\"", content)


if __name__ == "__main__":
    unittest.main()
