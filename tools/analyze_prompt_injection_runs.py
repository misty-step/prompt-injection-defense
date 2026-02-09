#!/usr/bin/env python3
"""Cross-round analysis for canonical prompt-injection run dataset."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Canonical dataset not found: {path}")
    with path.open() as handle:
        return list(csv.DictReader(handle))


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def summarize_by(rows: Iterable[Dict[str, str]], keys: Tuple[str, ...]) -> Dict[Tuple[str, ...], Dict[str, int]]:
    stats: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        group = tuple(row.get(k, "") for k in keys)
        status = row.get("status", "")
        score = parse_int(row.get("score"), default=-1)
        stats[group]["n_total"] += 1
        if status == "ok":
            stats[group]["n_ok"] += 1
            if score >= 2:
                stats[group]["n_success"] += 1
        else:
            stats[group]["n_error"] += 1
    return stats


def print_table(title: str, keys: Tuple[str, ...], stats: Dict[Tuple[str, ...], Dict[str, int]]) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)
    key_headers = " | ".join(k.upper() for k in keys)
    print(f"{key_headers:<36} {'TOTAL':>7} {'OK':>7} {'ERROR':>7} {'INJ_OK':>8} {'INJ_RATE_OK':>12}")
    print("-" * 92)

    for group in sorted(stats):
        row = stats[group]
        n_total = row.get("n_total", 0)
        n_ok = row.get("n_ok", 0)
        n_error = row.get("n_error", 0)
        n_success = row.get("n_success", 0)
        rate = (n_success / n_ok) if n_ok else 0.0
        key_label = " | ".join(group)
        print(f"{key_label:<36} {n_total:>7} {n_ok:>7} {n_error:>7} {n_success:>8} {format_pct(rate):>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze canonical prompt-injection runs CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/prompt-injection-boundary-tags/rounds/canonical/runs_v1.csv"),
        help="Path to canonical CSV dataset.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_path = args.input
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    rows = load_rows(input_path)
    if not rows:
        raise SystemExit("No rows found in canonical dataset.")

    print(f"Loaded {len(rows)} rows from {input_path}")

    by_round = summarize_by(rows, ("round_id",))
    by_round_condition = summarize_by(rows, ("round_id", "condition"))
    by_round_model = summarize_by(rows, ("round_id", "model"))

    print_table("Summary By Round", ("round_id",), by_round)
    print_table("Summary By Round + Condition", ("round_id", "condition"), by_round_condition)
    print_table("Summary By Round + Model", ("round_id", "model"), by_round_model)


if __name__ == "__main__":
    main()
