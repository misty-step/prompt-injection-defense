#!/usr/bin/env python3
"""Analyze round6 tool-call policy gate evaluation results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

ROUND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROUND_DIR / "data"
REPORT_DIR = ROUND_DIR / "report"
DEFAULT_INPUT = DATA_DIR / "policy_eval_latest.csv"
DEFAULT_REPORT = REPORT_DIR / "policy_evaluation.md"


def parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}")
    with path.open() as handle:
        return list(csv.DictReader(handle))


def maybe_latest(path_arg: Path) -> Path:
    if path_arg.exists():
        return path_arg
    candidates = sorted(DATA_DIR.glob("policy_eval_*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    return path_arg


def confusion_counts(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for row in rows:
        label = parse_bool(row.get("label_malicious"))
        pred = parse_bool(row.get("predicted_malicious"))
        if label and pred:
            tp += 1
        elif (not label) and pred:
            fp += 1
        elif (not label) and (not pred):
            tn += 1
        elif label and (not pred):
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _metric_div(numer: float, denom: float) -> float:
    return (numer / denom) if denom else 0.0


def metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp = float(counts.get("tp", 0))
    fp = float(counts.get("fp", 0))
    tn = float(counts.get("tn", 0))
    fn = float(counts.get("fn", 0))

    precision = _metric_div(tp, tp + fp)
    recall = _metric_div(tp, tp + fn)
    f1 = _metric_div(2 * precision * recall, precision + recall)
    accuracy = _metric_div(tp + tn, tp + tn + fp + fn)
    fpr = _metric_div(fp, fp + tn)

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "fpr": fpr}


def group_by_key(rows: Iterable[Dict[str, str]], key_name: str) -> DefaultDict[str, List[Dict[str, str]]]:
    grouped: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key_name, "")].append(row)
    return grouped


def print_config_summary(grouped: Dict[str, List[Dict[str, str]]]) -> None:
    print()
    print("=" * 120)
    print("Round 6 Policy Gate Metrics (per config)")
    print("=" * 120)
    print(
        f"{'Config':<12} {'N':>6} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} "
        f"{'Prec':>8} {'Rec':>8} {'F1':>8} {'Acc':>8} {'FPR':>8}"
    )
    print("-" * 120)

    for name in sorted(grouped):
        rows = grouped[name]
        counts = confusion_counts(rows)
        m = metrics_from_counts(counts)
        print(
            f"{name:<12} {len(rows):>6} {counts['tp']:>6} {counts['fp']:>6} {counts['tn']:>6} {counts['fn']:>6} "
            f"{m['precision']:>7.1%} {m['recall']:>7.1%} {m['f1']:>7.1%} {m['accuracy']:>7.1%} {m['fpr']:>7.1%}"
        )


def tool_breakdown(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    grouped: DefaultDict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("config_name", ""), row.get("tool_name", ""))
        grouped[key].append(row)

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, bucket in grouped.items():
        counts = confusion_counts(bucket)
        metrics = metrics_from_counts(counts)
        out[key] = {**metrics, **counts, "n": len(bucket)}
    return out


def print_tool_breakdown(rows: List[Dict[str, str]]) -> None:
    breakdown = tool_breakdown(rows)

    print()
    print("=" * 120)
    print("Per-Tool Breakdown (config x tool_name)")
    print("=" * 120)
    print(
        f"{'Config':<12} {'Tool':<16} {'N':>6} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} "
        f"{'Prec':>8} {'Rec':>8} {'F1':>8} {'FPR':>8}"
    )
    print("-" * 120)

    for (config_name, tool_name) in sorted(breakdown):
        item = breakdown[(config_name, tool_name)]
        print(
            f"{config_name:<12} {tool_name:<16} {int(item['n']):>6} {int(item['tp']):>6} {int(item['fp']):>6} "
            f"{int(item['tn']):>6} {int(item['fn']):>6} {float(item['precision']):>7.1%} "
            f"{float(item['recall']):>7.1%} {float(item['f1']):>7.1%} {float(item['fpr']):>7.1%}"
        )


def markdown_report(rows: List[Dict[str, str]], input_path: Path) -> str:
    if not rows:
        return "# Round 6 Policy Evaluation\n\nNo rows found.\n"

    meta = rows[0]
    schema_version = meta.get("schema_version", "")
    run_id = meta.get("run_id", "")
    timestamp = meta.get("timestamp", "")

    grouped = group_by_key(rows, "config_name")
    lines: List[str] = []
    lines.append("# Round 6 Policy Evaluation")
    lines.append("")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Schema: `{schema_version}`")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Timestamp: `{timestamp}`")
    lines.append(f"- Rows: `{len(rows)}`")
    lines.append("")

    lines.append("## Metrics (Per Config)")
    lines.append("")
    lines.append("| Config | N | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy | FPR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(grouped):
        bucket = grouped[name]
        counts = confusion_counts(bucket)
        m = metrics_from_counts(counts)
        lines.append(
            f"| `{name}` | {len(bucket)} | {counts['tp']} | {counts['fp']} | {counts['tn']} | {counts['fn']} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['accuracy']:.3f} | {m['fpr']:.3f} |"
        )

    lines.append("")
    lines.append("## Per-Tool Breakdown")
    lines.append("")
    lines.append("| Config | Tool | N | TP | FP | TN | FN | Precision | Recall | F1 | FPR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    breakdown = tool_breakdown(rows)
    for (config_name, tool_name) in sorted(breakdown):
        item = breakdown[(config_name, tool_name)]
        lines.append(
            f"| `{config_name}` | `{tool_name}` | {int(item['n'])} | {int(item['tp'])} | {int(item['fp'])} | "
            f"{int(item['tn'])} | {int(item['fn'])} | {float(item['precision']):.3f} | {float(item['recall']):.3f} | "
            f"{float(item['f1']):.3f} | {float(item['fpr']):.3f} |"
        )

    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze round6 policy evaluation CSV results.")
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

    grouped = group_by_key(rows, "config_name")
    print_config_summary(grouped)
    print_tool_breakdown(rows)

    # Use relative path in committed report to avoid leaking local filesystem paths.
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

