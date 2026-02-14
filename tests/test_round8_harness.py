"""Tests for round8 RAG prompt-injection harness deterministic logic."""
from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

EXPERIMENT_DIR = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "prompt-injection-boundary-tags"
)
HARNESS_PATH = EXPERIMENT_DIR / "rounds" / "round8" / "harness" / "run_experiment.py"

spec = importlib.util.spec_from_file_location("round8_harness", HARNESS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from {HARNESS_PATH}")

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)


class TestBuildUserMessage(unittest.TestCase):
    """Verify RAG context template produces correct structure."""

    def test_contains_three_docs(self) -> None:
        rng = random.Random(42)
        msg = harness.build_user_message("PAYLOAD_TEXT", "raw", rng)
        self.assertIn("[Doc 1]", msg)
        self.assertIn("[Doc 2]", msg)
        self.assertIn("[Doc 3]", msg)

    def test_payload_in_doc2_not_issue(self) -> None:
        rng = random.Random(42)
        payload = "INJECT_THIS_PAYLOAD"
        msg = harness.build_user_message(payload, "raw", rng)
        # Payload appears in the retrieved docs section, not the issue section
        issue_section = msg.split("## Retrieved Documentation")[0]
        docs_section = msg.split("## Retrieved Documentation")[1]
        self.assertNotIn(payload, issue_section)
        self.assertIn(payload, docs_section)

    def test_tags_only_wraps_docs(self) -> None:
        rng = random.Random(42)
        msg = harness.build_user_message("test", "tags_only", rng)
        self.assertIn("<<<RETRIEVED_DOCUMENT_1>>>", msg)
        self.assertIn("<<<END_RETRIEVED_DOCUMENT_1>>>", msg)
        self.assertIn("<<<RETRIEVED_DOCUMENT_2>>>", msg)
        self.assertIn("<<<END_RETRIEVED_DOCUMENT_2>>>", msg)

    def test_full_stack_nonce_tags(self) -> None:
        rng = random.Random(42)
        msg = harness.build_user_message("test", "full_stack", rng)
        # Nonce tags should be unique per doc
        self.assertIn("<<<RETRIEVED_DOCUMENT_1_", msg)
        self.assertIn("<<<RETRIEVED_DOCUMENT_2_", msg)
        self.assertIn("<<<RETRIEVED_DOCUMENT_3_", msg)

    def test_raw_no_tags(self) -> None:
        rng = random.Random(42)
        msg = harness.build_user_message("test", "raw", rng)
        self.assertNotIn("<<<RETRIEVED_DOCUMENT", msg)


class TestSimulateTrial(unittest.TestCase):
    """Verify simulation determinism and retrieval uplift."""

    def test_deterministic(self) -> None:
        kwargs = dict(
            model_name="claude-sonnet-4.5",
            payload_name="direct_override",
            condition="raw",
            reasoning_budget="none",
        )
        result1 = harness.simulate_trial(rng=random.Random(42), **kwargs)
        result2 = harness.simulate_trial(rng=random.Random(42), **kwargs)
        self.assertEqual(result1["status"], result2["status"])
        self.assertEqual(result1["response_text"], result2["response_text"])
        self.assertEqual(result1["raw_tool_calls"], result2["raw_tool_calls"])

    def test_full_stack_blocks_suspicious(self) -> None:
        """Full stack should block suspicious tool calls."""
        blocked_count = 0
        for seed in range(200):
            result = harness.simulate_trial(
                rng=random.Random(seed),
                model_name="minimax-m2.1",
                payload_name="gradual_escalation",
                condition="full_stack",
                reasoning_budget="none",
            )
            blocked_count += len(result["blocked_tool_calls"])
        self.assertEqual(result["status"], "ok")
        self.assertGreater(blocked_count, 0, "Expected some tool calls to be blocked in full_stack mode")

    def test_retrieval_uplift_applied(self) -> None:
        """Retrieval channel uplift should increase injection rates vs round7 baseline."""
        self.assertGreater(harness.RETRIEVAL_CHANNEL_UPLIFT, 1.0)

    def test_injection_channel_in_row(self) -> None:
        """Output rows should contain injection_channel field."""
        row = harness.build_row(
            run_id="test",
            timestamp="20260212",
            mode="simulate",
            trial_id=1,
            model_name="claude-sonnet-4.5",
            model_id="claude-sonnet-4-5-20250929",
            payload_name="direct_override",
            condition="raw",
            trial_num=1,
            status="ok",
            error="",
            score_raw=0,
            score_effective=0,
            reasoning_budget="none",
            response_text="test",
            raw_tool_calls=[],
            effective_tool_calls=[],
            blocked_tool_calls=[],
            input_tokens=100,
            output_tokens=50,
        )
        self.assertEqual(row["injection_channel"], "retrieval")


class TestWrapRetrievedDoc(unittest.TestCase):
    def test_instruction_only_no_tags(self) -> None:
        rng = random.Random(42)
        result = harness.wrap_retrieved_doc("content", "instruction_only", rng, 1)
        self.assertNotIn("<<<", result)
        self.assertEqual(result, "content")

    def test_raw_no_tags(self) -> None:
        rng = random.Random(42)
        result = harness.wrap_retrieved_doc("content", "raw", rng, 1)
        self.assertEqual(result, "content")


if __name__ == "__main__":
    unittest.main()
