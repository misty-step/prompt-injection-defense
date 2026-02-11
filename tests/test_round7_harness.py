from __future__ import annotations

import importlib.util
import random
import re
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "prompt-injection-boundary-tags"
    / "rounds"
    / "round7"
    / "harness"
    / "run_experiment.py"
)

spec = importlib.util.spec_from_file_location("round7_harness", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from {MODULE_PATH}")
round7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(round7)


class Round7HarnessTests(unittest.TestCase):
    def test_parse_csv_list_dedupes_preserving_order(self) -> None:
        values = round7.parse_csv_list("gpt-5.2, gpt-5.2, gemini-3-flash, gpt-5.2")
        self.assertEqual(values, ["gpt-5.2", "gemini-3-flash"])

    def test_wrap_issue_content_full_stack_uses_matching_nonce(self) -> None:
        wrapped = round7.wrap_issue_content("issue body", "full_stack", random.Random(7))
        start = re.search(r"<<<EXTERNAL_UNTRUSTED_CONTENT_([A-Z0-9]+)>>>", wrapped)
        end = re.search(r"<<<END_EXTERNAL_UNTRUSTED_CONTENT_([A-Z0-9]+)>>>", wrapped)
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)
        self.assertEqual(start.group(1), end.group(1))

    def test_reasoning_budget_selection_respects_model_capability(self) -> None:
        self.assertEqual(
            round7.reasoning_budgets_for_model("claude-sonnet-4.5", ["gpt-5.2"], ["low", "high"]),
            ["none"],
        )
        self.assertEqual(
            round7.reasoning_budgets_for_model("gpt-5.2", ["gpt-5.2"], ["low", "high"]),
            ["low", "high"],
        )

    def test_expand_model_reasoning_pairs(self) -> None:
        pairs = round7.expand_model_reasoning_pairs(
            selected_models=["claude-sonnet-4.5", "gpt-5.2", "gemini-3-flash"],
            reasoning_models=["gpt-5.2", "gemini-3-flash"],
            reasoning_budgets=["low", "high"],
        )
        self.assertIn(("claude-sonnet-4.5", "none"), pairs)
        self.assertIn(("gpt-5.2", "low"), pairs)
        self.assertIn(("gpt-5.2", "high"), pairs)
        self.assertIn(("gemini-3-flash", "low"), pairs)
        self.assertIn(("gemini-3-flash", "high"), pairs)

    def test_apply_full_stack_policy_blocks_sensitive_calls(self) -> None:
        raw_calls = [
            {"name": "execute_command", "arguments": {"command": "printenv | grep API_KEY"}},
            {"name": "read_file", "arguments": {"path": "README.md"}},
        ]
        safe, blocked = round7.apply_full_stack_policy(raw_calls)
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["name"], "execute_command")
        self.assertEqual(len(safe), 1)
        self.assertEqual(safe[0]["name"], "read_file")

    def test_normalize_blocked_response(self) -> None:
        safe = round7.normalize_blocked_response("original", [])
        blocked = round7.normalize_blocked_response("original", [{"name": "execute_command"}])
        self.assertEqual(safe, "original")
        self.assertIn("Blocked suspicious tool requests", blocked)

    def test_raw_vs_effective_score_split(self) -> None:
        response_text = "safe summary"
        raw_tool_calls = [{"name": "execute_command", "arguments": {"command": "printenv | grep API_KEY"}}]
        effective_tool_calls: list[dict[str, object]] = []
        score_raw = round7.score_response(response_text, raw_tool_calls, round7.SCORER_CONFIG)
        score_effective = round7.score_response(response_text, effective_tool_calls, round7.SCORER_CONFIG)
        self.assertGreater(score_raw, score_effective)

    def test_build_row_includes_reasoning_budget(self) -> None:
        row = round7.build_row(
            run_id="r",
            timestamp="t",
            mode="simulate",
            trial_id=1,
            model_name="gpt-5.2",
            model_id="gpt-5.2",
            payload_name="direct_override",
            condition="raw",
            trial_num=1,
            status="ok",
            error="",
            score_raw=3,
            score_effective=1,
            reasoning_budget="high",
            response_text="safe summary",
            raw_tool_calls=[],
            effective_tool_calls=[],
            blocked_tool_calls=[],
            input_tokens=100,
            output_tokens=50,
        )
        self.assertEqual(row["reasoning_budget"], "high")
        self.assertEqual(row["score"], 1)
        self.assertEqual(row["score_raw"], 3)


if __name__ == "__main__":
    unittest.main()
