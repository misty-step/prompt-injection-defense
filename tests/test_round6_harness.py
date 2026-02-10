from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HARNESS_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "prompt-injection-boundary-tags"
    / "rounds"
    / "round6"
    / "harness"
    / "run_experiment.py"
)

harness_spec = importlib.util.spec_from_file_location("round6_harness", HARNESS_PATH)
if harness_spec is None or harness_spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from {HARNESS_PATH}")
round6 = importlib.util.module_from_spec(harness_spec)
harness_spec.loader.exec_module(round6)


class Round6HarnessTests(unittest.TestCase):
    def test_generate_malicious_corpus_expected_count_and_labels(self) -> None:
        corpus = round6.generate_malicious_corpus()
        self.assertEqual(len(corpus), 54)
        self.assertTrue(all(bool(item.get("malicious")) for item in corpus))

    def test_generate_benign_corpus_expected_count_and_labels(self) -> None:
        corpus = round6.generate_benign_corpus()
        self.assertEqual(len(corpus), 36)
        self.assertTrue(all((not bool(item.get("malicious"))) for item in corpus))

    def test_balanced_config_flags_known_malicious_read_file(self) -> None:
        configs = round6.build_policy_configs()
        tool_call = {"name": "read_file", "arguments": {"path": ".env"}}
        self.assertTrue(round6.is_suspicious_tool_call(tool_call, configs["balanced"]))

    def test_balanced_config_allows_known_benign_read_file(self) -> None:
        configs = round6.build_policy_configs()
        tool_call = {"name": "read_file", "arguments": {"path": "README.md"}}
        self.assertFalse(round6.is_suspicious_tool_call(tool_call, configs["balanced"]))

    def test_paranoid_flags_all_tool_calls(self) -> None:
        configs = round6.build_policy_configs()
        paranoid = configs["paranoid"]

        corpus = round6.generate_malicious_corpus() + round6.generate_benign_corpus()
        for example in corpus:
            call = {"name": example["tool_name"], "arguments": example["arguments"]}
            self.assertTrue(round6.is_suspicious_tool_call(call, paranoid))

    def test_csv_output_has_expected_columns(self) -> None:
        latest_path = round6.DATA_DIR / "policy_eval_latest.csv"
        had_latest = latest_path.exists() or latest_path.is_symlink()
        had_latest_symlink = latest_path.is_symlink()
        previous_target = latest_path.resolve() if had_latest_symlink else None
        previous_text = ""
        if had_latest and not had_latest_symlink:
            previous_text = latest_path.read_text()

        expected_columns = [
            "schema_version",
            "run_id",
            "timestamp",
            "config_name",
            "tool_call_id",
            "tool_name",
            "arguments_json",
            "label_malicious",
            "predicted_malicious",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "round6_policy_eval.csv"
            argv = ["run_experiment.py", "--output", str(output_path)]
            try:
                with mock.patch.object(sys, "argv", argv):
                    round6.run()

                with output_path.open() as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, expected_columns)
            finally:
                if latest_path.exists() or latest_path.is_symlink():
                    latest_path.unlink()
                if had_latest:
                    if had_latest_symlink and previous_target is not None:
                        latest_path.symlink_to(previous_target)
                    elif not had_latest_symlink:
                        latest_path.write_text(previous_text)


if __name__ == "__main__":
    unittest.main()

