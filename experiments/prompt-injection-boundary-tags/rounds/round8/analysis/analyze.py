#!/usr/bin/env python3
"""Analyze round8 RAG prompt-injection benchmark results.

Produces within-round tables and cross-channel comparison with round7.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROUND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROUND_DIR / "data"
REPORT_DIR = ROUND_DIR / "report"
DEFAULT_INPUT = DATA_DIR / "rag_injection_results_latest.csv"
DEFAULT_REPORT = REPORT_DIR / "analysis_tables.md"

ROUND7_DATA_DIR = ROUND_DIR.parent / "round7" / "data"
ROUND7_DEFAULT = ROUND7_DATA_DIR / "cross_model_results_latest.csv"


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


def maybe_latest(path_arg: Path, data_dir: Path, prefix: str) -> Path:
    if path_arg.exists():
        return path_arg
    candidates = sorted(data_dir.glob(f"{prefix}*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    return path_arg


def summarize(
    rows: Iterable[Dict[str, str]], keys: Tuple[str, ...]
) -> Dict[Tuple[str, ...], Dict[str, float]]:
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


def print_table(
    title: str, keys: Tuple[str, ...], stats: Dict[Tuple[str, ...], Dict[str, float]]
) -> None:
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


def print_cross_channel_comparison(
    r8_rows: List[Dict[str, str]], r7_rows: List[Dict[str, str]]
) -> None:
    """Compare retrieval-channel (round8) vs direct-channel (round7) injection rates."""
    r8_stats = summarize(r8_rows, ("model", "condition"))
    r7_stats = summarize(r7_rows, ("model", "condition"))

    all_keys = sorted(set(r8_stats.keys()) | set(r7_stats.keys()))
    if not all_keys:
        return

    print()
    print("=" * 108)
    print("Cross-Channel Comparison: Retrieval (R8) vs Direct (R7)")
    print("=" * 108)
    print(
        f"{'Model':<24} {'Condition':<20} {'R7 Direct%':>12} {'R8 Retrieval%':>14} {'Delta':>8}"
    )
    print("-" * 108)
    for model, condition in all_keys:
        r7_rate = r7_stats.get((model, condition), {}).get("inj_rate", 0.0)
        r8_rate = r8_stats.get((model, condition), {}).get("inj_rate", 0.0)
        delta = r8_rate - r7_rate
        print(
            f"{model:<24} {condition:<20} {r7_rate:>11.1%} {r8_rate:>13.1%} {delta:>+7.1%}"
        )

    # Aggregate by condition only
    r8_cond = summarize(r8_rows, ("condition",))
    r7_cond = summarize(r7_rows, ("condition",))
    cond_keys = sorted(set(r8_cond.keys()) | set(r7_cond.keys()))

    print()
    print("=" * 108)
    print("Cross-Channel Summary by Condition")
    print("=" * 108)
    print(f"{'Condition':<20} {'R7 Direct%':>12} {'R8 Retrieval%':>14} {'Delta':>8}")
    print("-" * 108)
    for (condition,) in cond_keys:
        r7_rate = r7_cond.get((condition,), {}).get("inj_rate", 0.0)
        r8_rate = r8_cond.get((condition,), {}).get("inj_rate", 0.0)
        delta = r8_rate - r7_rate
        print(f"{condition:<20} {r7_rate:>11.1%} {r8_rate:>13.1%} {delta:>+7.1%}")


def markdown_report(
    rows: List[Dict[str, str]],
    input_path: Path,
    r7_rows: List[Dict[str, str]] | None = None,
) -> str:
    if not rows:
        return "# Round 8 Analysis Tables\n\nNo rows found.\n"

    schema = rows[0].get("schema_version", "")
    run_id = rows[0].get("run_id", "")
    timestamp = rows[0].get("timestamp", "")

    cond_stats = summarize(rows, ("condition",))
    model_stats = summarize(rows, ("model",))
    reasoning_rows = [row for row in rows if row.get("reasoning_budget", "none") != "none"]
    reasoning_stats = summarize(reasoning_rows, ("model", "reasoning_budget")) if reasoning_rows else {}

    lines: List[str] = []
    lines.append("# Round 8 Analysis Tables: RAG Prompt-Injection Benchmark")
    lines.append("")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Schema: `{schema}`")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Timestamp: `{timestamp}`")
    lines.append(f"- Rows: `{len(rows)}`")
    lines.append(f"- Injection channel: `retrieval`")
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

    # Cross-channel comparison if round7 data available
    if r7_rows:
        r8_cond = summarize(rows, ("condition",))
        r7_cond = summarize(r7_rows, ("condition",))
        cond_keys = sorted(set(r8_cond.keys()) | set(r7_cond.keys()))

        lines.append("## Cross-Channel Comparison (Retrieval vs Direct)")
        lines.append("")
        lines.append("| Condition | R7 Direct Inj% | R8 Retrieval Inj% | Delta |")
        lines.append("|---|---:|---:|---:|")
        for (condition,) in cond_keys:
            r7_rate = r7_cond.get((condition,), {}).get("inj_rate", 0.0)
            r8_rate = r8_cond.get((condition,), {}).get("inj_rate", 0.0)
            delta = r8_rate - r7_rate
            lines.append(
                f"| `{condition}` | {r7_rate:.3f} | {r8_rate:.3f} | {delta:+.3f} |"
            )
        lines.append("")

        r8_mc = summarize(rows, ("model", "condition"))
        r7_mc = summarize(r7_rows, ("model", "condition"))
        mc_keys = sorted(set(r8_mc.keys()) | set(r7_mc.keys()))

        lines.append("## Cross-Channel by Model + Condition")
        lines.append("")
        lines.append("| Model | Condition | R7 Direct% | R8 Retrieval% | Delta |")
        lines.append("|---|---|---:|---:|---:|")
        for model, condition in mc_keys:
            r7_rate = r7_mc.get((model, condition), {}).get("inj_rate", 0.0)
            r8_rate = r8_mc.get((model, condition), {}).get("inj_rate", 0.0)
            delta = r8_rate - r7_rate
            lines.append(
                f"| `{model}` | `{condition}` | {r7_rate:.3f} | {r8_rate:.3f} | {delta:+.3f} |"
            )
        lines.append("")
    else:
        lines.append("## Cross-Channel Comparison")
        lines.append("")
        lines.append("Round 7 data not available for cross-channel comparison.")
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze round8 RAG injection CSV results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV to analyze.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help="Markdown report output path.")
    parser.add_argument(
        "--round7-input", type=Path, default=ROUND7_DEFAULT,
        help="Round7 CSV for cross-channel comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = maybe_latest(args.input, DATA_DIR, "rag_injection_results_")
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

    # Cross-channel comparison with round7
    r7_input = maybe_latest(args.round7_input, ROUND7_DATA_DIR, "cross_model_results_")
    r7_rows: List[Dict[str, str]] | None = None
    if r7_input.exists():
        r7_rows = load_rows(r7_input)
        print(f"\nLoaded {len(r7_rows)} round7 rows from {r7_input}")
        print_cross_channel_comparison(rows, r7_rows)
    else:
        print(f"\nRound7 data not found at {r7_input}, skipping cross-channel comparison.")

    display_path = input_path
    try:
        display_path = input_path.relative_to(Path.cwd())
    except ValueError:
        pass

    report_path = args.output
    if not report_path.is_absolute():
        report_path = (Path.cwd() / report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(rows, display_path, r7_rows))
    print()
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
