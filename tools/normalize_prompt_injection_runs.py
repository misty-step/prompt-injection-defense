#!/usr/bin/env python3
"""Normalize historical prompt-injection run CSVs into one canonical schema."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


SCHEMA_VERSION = "prompt_injection_run_v1"


@dataclass(frozen=True)
class SourceSpec:
    round_id: str
    path: Path


CANONICAL_FIELDS = [
    "schema_version",
    "experiment_id",
    "round_id",
    "source_file",
    "source_row",
    "trial_id",
    "model",
    "model_id",
    "provider",
    "payload",
    "condition",
    "trial_num",
    "score",
    "status",
    "error",
    "num_tool_calls",
    "tool_calls_json",
    "response_length",
    "response_preview",
    "is_success",
    "is_error",
]


def infer_provider(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt"):
        return "openai"
    if m.startswith("kimi"):
        return "moonshot"
    return "unknown"


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_round1(raw: Dict[str, str], source_file: str, row_number: int) -> Dict[str, str]:
    status = raw.get("status", "")
    score = parse_int(raw.get("score"), default=-1)
    is_success = int(status == "ok" and score >= 2)
    is_error = int(status != "ok")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "prompt-injection-boundary-tags",
        "round_id": "round1",
        "source_file": source_file,
        "source_row": str(row_number),
        "trial_id": raw.get("trial_id", ""),
        "model": "claude-3-5-haiku-latest",
        "model_id": "claude-3-5-haiku-latest",
        "provider": "anthropic",
        "payload": raw.get("payload", ""),
        "condition": raw.get("condition", ""),
        "trial_num": raw.get("trial_num", ""),
        "score": str(score),
        "status": status,
        "error": "",
        "num_tool_calls": "0",
        "tool_calls_json": "[]",
        "response_length": raw.get("response_length", ""),
        "response_preview": raw.get("response_preview", ""),
        "is_success": str(is_success),
        "is_error": str(is_error),
    }


def normalize_round2(raw: Dict[str, str], source_file: str, row_number: int) -> Dict[str, str]:
    status = raw.get("status", "")
    score = parse_int(raw.get("score"), default=-1)
    is_success = int(status == "ok" and score >= 2)
    is_error = int(status != "ok")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "prompt-injection-boundary-tags",
        "round_id": "round2",
        "source_file": source_file,
        "source_row": str(row_number),
        "trial_id": raw.get("trial_id", ""),
        "model": raw.get("model", ""),
        "model_id": "",
        "provider": raw.get("provider", infer_provider(raw.get("model", ""))),
        "payload": raw.get("payload", ""),
        "condition": raw.get("condition", ""),
        "trial_num": raw.get("trial_num", ""),
        "score": str(score),
        "status": status,
        "error": raw.get("error", ""),
        "num_tool_calls": str(parse_int(raw.get("tool_call_count"), default=0)),
        "tool_calls_json": raw.get("tool_calls_json", "[]"),
        "response_length": raw.get("response_length", ""),
        "response_preview": raw.get("response_preview", ""),
        "is_success": str(is_success),
        "is_error": str(is_error),
    }


def normalize_round2b(raw: Dict[str, str], source_file: str, row_number: int) -> Dict[str, str]:
    status = raw.get("status", "")
    score = parse_int(raw.get("score"), default=-1)
    model = raw.get("model", "")
    is_success = int(status == "ok" and score >= 2)
    is_error = int(status != "ok")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "prompt-injection-boundary-tags",
        "round_id": "round2b",
        "source_file": source_file,
        "source_row": str(row_number),
        "trial_id": raw.get("trial_id", ""),
        "model": model,
        "model_id": raw.get("model_id", ""),
        "provider": infer_provider(model),
        "payload": raw.get("payload", ""),
        "condition": raw.get("condition", ""),
        "trial_num": raw.get("trial_num", ""),
        "score": str(score),
        "status": status,
        "error": "",
        "num_tool_calls": str(parse_int(raw.get("num_tool_calls"), default=0)),
        "tool_calls_json": raw.get("tool_calls", "[]"),
        "response_length": raw.get("response_length", ""),
        "response_preview": raw.get("response_preview", ""),
        "is_success": str(is_success),
        "is_error": str(is_error),
    }


def normalize_round7(raw: Dict[str, str], source_file: str, row_number: int) -> Dict[str, str]:
    """Normalize round7 CSV (cross_model_results schema)."""
    status = raw.get("status", "")
    # Round 7 uses score_effective as the final score
    score = parse_int(raw.get("score_effective"), default=parse_int(raw.get("score"), default=-1))
    model = raw.get("model", "")
    model_id = raw.get("model_id", "")
    is_success = int(status == "ok" and score >= 2)
    is_error = int(status != "ok")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "prompt-injection-boundary-tags",
        "round_id": "round7",
        "source_file": source_file,
        "source_row": str(row_number),
        "trial_id": raw.get("trial_id", ""),
        "model": model,
        "model_id": model_id,
        "provider": infer_provider(model),
        "payload": raw.get("payload", ""),
        "condition": raw.get("condition", ""),
        "trial_num": raw.get("trial_num", ""),
        "score": str(score),
        "status": status,
        "error": raw.get("error", ""),
        "num_tool_calls": str(parse_int(raw.get("num_tool_calls_effective"), default=0)),
        "tool_calls_json": raw.get("tool_calls_effective_json", "[]"),
        "response_length": raw.get("response_length", ""),
        "response_preview": raw.get("response_preview", ""),
        "is_success": str(is_success),
        "is_error": str(is_error),
    }


def normalize_rows(spec: SourceSpec) -> Iterable[Dict[str, str]]:
    with spec.path.open() as handle:
        reader = csv.DictReader(handle)
        for row_number, raw in enumerate(reader, start=1):
            source_file = str(spec.path)
            if spec.round_id == "round1":
                yield normalize_round1(raw, source_file, row_number)
            elif spec.round_id == "round2":
                yield normalize_round2(raw, source_file, row_number)
            elif spec.round_id == "round2b":
                yield normalize_round2b(raw, source_file, row_number)
            elif spec.round_id == "round7":
                yield normalize_round7(raw, source_file, row_number)
            else:
                raise ValueError(f"Unsupported round_id: {spec.round_id}")


def build_default_sources(repo_root: Path) -> List[SourceSpec]:
    base = repo_root / "experiments" / "prompt-injection-boundary-tags" / "rounds"
    sources = [
        SourceSpec("round1", base / "round1" / "data" / "results.csv"),
        SourceSpec("round2", base / "round2" / "data" / "results-round2.csv"),
        SourceSpec("round2b", base / "round2b" / "data" / "results_r2_combined_latest.csv"),
    ]
    # Add round7 if available
    round7_path = base / "round7" / "data" / "cross_model_results_latest.csv"
    if round7_path.exists():
        sources.append(SourceSpec("round7", round7_path))
    return sources


def write_normalized_csv(rows: Iterable[Dict[str, str]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize prompt-injection historical run CSVs into canonical schema."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/prompt-injection-boundary-tags/rounds/canonical/runs_v1.csv"
        ),
        help="Path to write canonical CSV output.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sources = build_default_sources(repo_root)
    missing = [spec.path for spec in sources if not spec.path.exists()]
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing source files:\n{missing_list}")

    all_rows: List[Dict[str, str]] = []
    for source in sources:
        all_rows.extend(normalize_rows(source))

    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    count = write_normalized_csv(all_rows, output_path)
    print(f"Wrote {count} normalized rows to {output_path}")


if __name__ == "__main__":
    main()
