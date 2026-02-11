#!/usr/bin/env python3
"""Analyze round7 cross-model defense validation results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROUND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROUND_DIR / "data"
REPORT_DIR = ROUND_DIR / "report"
DEFAULT_INPUT = DATA_DIR / "cross_model_results_latest.csv"
DEFAULT_REPORT = REPORT_DIR / "findings.md"


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}")
    with path.open() as handle:
        return list(csv.DictReader(handle))


def maybe_latest(path_arg: Path) -> Path:
    if path_arg.exists():
        return path_arg
    candidates = sorted(DATA_DIR.glob("cross_model_results_*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    return path_arg


def summarize(rows: Iterable[Dict[str, str]], keys: Tuple[str, ...]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    stats: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        score = parse_int(row.get("score_effective", row.get("score")), default=-1)
        status = row.get("status", "")
        stats[key]["n"] += 1
        if status == "ok":
            stats[key]["ok"] += 1
            if score >= 2:
                stats[key]["inj"] += 1
            stats[key]["score_sum"] += score
        else:
            stats[key]["errors"] += 1

    for values in stats.values():
        ok = values.get("ok", 0.0)
        values["inj_rate"] = (values.get("inj", 0.0) / ok) if ok else 0.0
        values["avg_score"] = (values.get("score_sum", 0.0) / ok) if ok else 0.0
    return stats


def print_table(title: str, keys: Tuple[str, ...], stats: Dict[Tuple[str, ...], Dict[str, float]]) -> None:
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)
    print(f"{' | '.join(k.upper() for k in keys):<52} {'N':>6} {'OK':>6} {'ERR':>6} {'AVG':>7} {'INJ%':>8}")
    print("-" * 108)

    for key in sorted(stats):
        item = stats[key]
        key_text = " | ".join(key)
        print(
            f"{key_text:<52} {int(item['n']):>6} {int(item['ok']):>6} {int(item['errors']):>6} "
            f"{item['avg_score']:>7.2f} {item['inj_rate']:>7.1%}"
        )


def print_baseline_delta(rows: List[Dict[str, str]]) -> None:
    grouped = summarize(rows, ("model", "reasoning_budget", "condition"))
    model_budget_to_raw: Dict[Tuple[str, str], float] = {}
    for (model, reasoning_budget, condition), item in grouped.items():
        if condition == "raw":
            model_budget_to_raw[(model, reasoning_budget)] = item["inj_rate"]

    print()
    print("=" * 108)
    print("Injection Rate Delta Vs Raw Baseline (per model + reasoning budget)")
    print("=" * 108)
    print(f"{'Model':<24} {'Budget':<8} {'Condition':<20} {'Raw%':>8} {'Cond%':>8} {'Delta':>8}")
    print("-" * 108)

    for (model, reasoning_budget, condition), item in sorted(grouped.items()):
        if condition == "raw":
            continue
        raw = model_budget_to_raw.get((model, reasoning_budget), 0.0)
        cond = item["inj_rate"]
        delta = cond - raw
        print(f"{model:<24} {reasoning_budget:<8} {condition:<20} {raw:>7.1%} {cond:>7.1%} {delta:>+7.1%}")


def markdown_report(rows: List[Dict[str, str]], input_path: Path) -> str:
    if not rows:
        return "# Round 7 Findings\n\nNo rows found.\n"

    schema = rows[0].get("schema_version", "")
    run_id = rows[0].get("run_id", "")
    timestamp = rows[0].get("timestamp", "")

    cond_stats = summarize(rows, ("condition",))
    model_stats = summarize(rows, ("model",))
    reasoning_rows = [row for row in rows if row.get("reasoning_budget", "none") != "none"]
    reasoning_stats = summarize(reasoning_rows, ("model", "reasoning_budget")) if reasoning_rows else {}

    lines: List[str] = []
    lines.append("# Round 7 Findings")
    lines.append("")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Schema: `{schema}`")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Timestamp: `{timestamp}`")
    lines.append(f"- Rows: `{len(rows)}`")
    lines.append("")

    lines.append("## Condition Summary")
    lines.append("")
    lines.append("| Condition | N | OK | ERR | Avg Score | Injection Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for (condition,), item in sorted(cond_stats.items()):
        lines.append(
            f"| `{condition}` | {int(item['n'])} | {int(item['ok'])} | {int(item['errors'])} | "
            f"{item['avg_score']:.3f} | {item['inj_rate']:.3f} |"
        )
    lines.append("")

    lines.append("## Model Summary")
    lines.append("")
    lines.append("| Model | N | OK | ERR | Avg Score | Injection Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for (model,), item in sorted(model_stats.items()):
        lines.append(
            f"| `{model}` | {int(item['n'])} | {int(item['ok'])} | {int(item['errors'])} | "
            f"{item['avg_score']:.3f} | {item['inj_rate']:.3f} |"
        )
    lines.append("")

    lines.append("## Reasoning Budget Comparison")
    lines.append("")
    if not reasoning_stats:
        lines.append("No reasoning-budget rows in this run.")
    else:
        lines.append("| Model | Budget | N | OK | ERR | Avg Score | Injection Rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for (model, budget), item in sorted(reasoning_stats.items()):
            lines.append(
                f"| `{model}` | `{budget}` | {int(item['n'])} | {int(item['ok'])} | {int(item['errors'])} | "
                f"{item['avg_score']:.3f} | {item['inj_rate']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze round7 cross-model CSV results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV to analyze.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help="Markdown report output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = maybe_latest(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    rows = load_rows(input_path)
    print(f"Loaded {len(rows)} rows from {input_path}")
    print_table("Summary by Condition", ("condition",), summarize(rows, ("condition",)))
    print_table("Summary by Model", ("model",), summarize(rows, ("model",)))
    print_table(
        "Summary by Model + Condition",
        ("model", "condition"),
        summarize(rows, ("model", "condition")),
    )
    print_table(
        "Summary by Model + Reasoning Budget",
        ("model", "reasoning_budget"),
        summarize(rows, ("model", "reasoning_budget")),
    )
    print_baseline_delta(rows)

    display_path = input_path
    try:
        display_path = input_path.relative_to(Path.cwd())
    except ValueError:
        pass

    report_path = args.output
    if not report_path.is_absolute():
        report_path = (Path.cwd() / report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(rows, display_path))
    print()
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
