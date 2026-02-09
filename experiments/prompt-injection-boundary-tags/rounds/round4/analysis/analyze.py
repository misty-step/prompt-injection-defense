#!/usr/bin/env python3
"""Analyze round4 multi-turn benchmark results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROUND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROUND_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "multiturn_results_latest.csv"


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}")
    with path.open() as handle:
        return list(csv.DictReader(handle))


def summarize(rows: Iterable[Dict[str, str]], keys: Tuple[str, ...]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    stats: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = tuple(row.get(key_name, "") for key_name in keys)
        score = parse_int(row.get("score"), default=-1)
        status = row.get("status", "")

        stats[key]["n"] += 1
        stats[key]["token_in"] += parse_int(row.get("input_tokens"), 0)
        stats[key]["token_out"] += parse_int(row.get("output_tokens"), 0)
        stats[key]["cost_usd"] += parse_float(row.get("estimated_cost_usd"), 0.0)

        if status == "ok":
            stats[key]["ok"] += 1
            stats[key]["score_sum"] += score
            if score >= 2:
                stats[key]["inj"] += 1
        else:
            stats[key]["errors"] += 1

    for values in stats.values():
        ok = values.get("ok", 0.0)
        values["inj_rate"] = (values.get("inj", 0.0) / ok) if ok else 0.0
        values["avg_score"] = (values.get("score_sum", 0.0) / ok) if ok else 0.0
        values["avg_cost"] = (values.get("cost_usd", 0.0) / ok) if ok else 0.0
    return stats


def print_table(title: str, keys: Tuple[str, ...], stats: Dict[Tuple[str, ...], Dict[str, float]]) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)
    print(f"{' | '.join(key.upper() for key in keys):<50} {'N':>6} {'OK':>6} {'ERR':>6} {'AVG':>7} {'INJ%':>8} {'AVG$':>10}")
    print("-" * 112)

    for key in sorted(stats):
        item = stats[key]
        key_text = " | ".join(key)
        print(
            f"{key_text:<50} {int(item['n']):>6} {int(item['ok']):>6} {int(item['errors']):>6} "
            f"{item['avg_score']:>7.2f} {item['inj_rate']:>7.1%} {item['avg_cost']:>10.5f}"
        )


def print_mode_delta(rows: List[Dict[str, str]]) -> None:
    grouped = summarize(rows, ("model", "condition", "attack_mode"))
    pairs = sorted({(model, condition) for model, condition, _mode in grouped})

    print()
    print("=" * 112)
    print("Single-turn vs Multi-turn Injection Delta")
    print("=" * 112)
    print(f"{'Model':<24} {'Condition':<20} {'Single%':>10} {'Multi%':>10} {'Delta':>10}")
    print("-" * 112)

    for model, condition in pairs:
        single = grouped.get((model, condition, "single_turn"), {})
        multi = grouped.get((model, condition, "multi_turn"), {})
        if not single and not multi:
            continue

        single_rate = float(single.get("inj_rate", 0.0))
        multi_rate = float(multi.get("inj_rate", 0.0))
        delta = multi_rate - single_rate

        print(f"{model:<24} {condition:<20} {single_rate:>9.1%} {multi_rate:>9.1%} {delta:>+9.1%}")


def print_defense_degradation_ranking(rows: List[Dict[str, str]]) -> None:
    grouped = summarize(rows, ("model", "condition", "attack_mode"))
    deltas_by_condition: Dict[str, List[float]] = defaultdict(list)

    models = sorted({row.get("model", "") for row in rows})
    conditions = sorted({row.get("condition", "") for row in rows})

    for model in models:
        for condition in conditions:
            single = grouped.get((model, condition, "single_turn"), {})
            multi = grouped.get((model, condition, "multi_turn"), {})
            if not single and not multi:
                continue
            delta = float(multi.get("inj_rate", 0.0)) - float(single.get("inj_rate", 0.0))
            deltas_by_condition[condition].append(delta)

    ranked = []
    for condition, deltas in deltas_by_condition.items():
        if not deltas:
            continue
        ranked.append((condition, sum(deltas) / len(deltas), len(deltas)))

    ranked.sort(key=lambda item: item[1], reverse=True)

    print()
    print("=" * 112)
    print("Defense Degradation Ranking (higher delta = worse under multi-turn)")
    print("=" * 112)
    print(f"{'Condition':<24} {'Avg Delta':>12} {'Models':>8}")
    print("-" * 112)

    for condition, avg_delta, sample_size in ranked:
        print(f"{condition:<24} {avg_delta:>+11.1%} {sample_size:>8}")


def maybe_latest(path_arg: Path) -> Path:
    if path_arg.exists():
        return path_arg
    candidates = sorted(DATA_DIR.glob("multiturn_results_*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    return path_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze round4 CSV benchmark results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV to analyze.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = maybe_latest(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    rows = load_rows(input_path)
    print(f"Loaded {len(rows)} rows from {input_path}")

    print_table("Summary by Attack Mode", ("attack_mode",), summarize(rows, ("attack_mode",)))
    print_table("Summary by Condition + Mode", ("condition", "attack_mode"), summarize(rows, ("condition", "attack_mode")))
    print_table("Summary by Model + Mode", ("model", "attack_mode"), summarize(rows, ("model", "attack_mode")))
    print_mode_delta(rows)
    print_defense_degradation_ranking(rows)


if __name__ == "__main__":
    main()
