#!/usr/bin/env python3
"""Generate Round 2B charts: scorer calibration, injection rates, score distribution."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TrialRow:
    model: str
    payload: str
    condition: str
    trial_num: int
    score: int
    num_tool_calls: int


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while True:
        if (p / "experiments").exists() and (p / "Makefile").exists():
            return p
        if p.parent == p:
            raise RuntimeError("Could not find repo root (expected Makefile + experiments/).")
        p = p.parent


def load_trials(csv_path: Path) -> list[TrialRow]:
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        required = {"model", "payload", "condition", "trial_num", "score", "num_tool_calls"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise RuntimeError(f"Missing columns in {csv_path}: {sorted(missing)}")

        out: list[TrialRow] = []
        for row in r:
            out.append(
                TrialRow(
                    model=row["model"],
                    payload=row["payload"],
                    condition=row["condition"],
                    trial_num=int(row["trial_num"]),
                    score=int(row["score"]),
                    num_tool_calls=int(row["num_tool_calls"]),
                )
            )
        return out


def pct(x: float) -> float:
    return 100.0 * x


def chart_scorer_calibration(out_path: Path) -> None:
    """Grouped bar chart: baseline vs tuned precision/recall/F1."""
    metrics = ["Precision", "Recall", "F1"]
    baseline = [0.811, 1.000, 0.896]
    tuned = [1.000, 1.000, 1.000]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(metrics))
    width = 0.30

    bars_b = ax.bar(
        [xi - width / 2 for xi in x], baseline, width=width,
        label="Baseline (v1)", color="#C44E52", edgecolor="white",
    )
    bars_t = ax.bar(
        [xi + width / 2 for xi in x], tuned, width=width,
        label="Tuned (v2)", color="#4C72B0", edgecolor="white",
    )

    for bars in (bars_b, bars_t):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10,
            )

    ax.set_title("Scorer Calibration: Baseline vs Tuned")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_injection_rate_by_model_condition(
    trials: list[TrialRow], out_path: Path
) -> None:
    """Grouped bar chart: injection rate by model and condition."""
    models = sorted({t.model for t in trials})
    conditions = ["raw", "static_tags", "dynamic_nonce"]
    condition_labels = ["raw", "static tags", "dynamic nonce"]

    rates: dict[tuple[str, str], float] = {}
    for m in models:
        for c in conditions:
            sub = [t for t in trials if t.model == m and t.condition == c]
            if not sub:
                rates[(m, c)] = 0.0
                continue
            rates[(m, c)] = sum(1 for t in sub if t.score >= 2) / len(sub)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(models))
    width = 0.22
    offsets = [-width, 0.0, width]
    colors = ["#444444", "#2E86AB", "#F18F01"]

    for i, c in enumerate(conditions):
        ys = [pct(rates[(m, c)]) for m in models]
        xs = [xi + offsets[i] for xi in x]
        bars = ax.bar(xs, ys, width=width, label=condition_labels[i], color=colors[i])
        for b, y in zip(bars, ys):
            ax.text(
                b.get_x() + b.get_width() / 2, y + 1.0,
                f"{y:.0f}%", ha="center", va="bottom", fontsize=9,
            )

    ax.set_title("Injection Rate by Model and Condition (score >= 2)")
    ax.set_ylabel("Injection rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(0, max(10.0, ax.get_ylim()[1] + 5))
    ax.legend(title="condition", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_score_distribution_by_model(
    trials: list[TrialRow], out_path: Path
) -> None:
    """Stacked bar chart: score 0-3 distribution per model."""
    models = sorted({t.model for t in trials})
    scores = [0, 1, 2, 3]
    score_labels = ["Score 0", "Score 1", "Score 2", "Score 3"]
    colors = ["#55A868", "#C5C56A", "#DD8452", "#C44E52"]

    counts: dict[str, dict[int, int]] = {m: defaultdict(int) for m in models}
    for t in trials:
        counts[t.model][t.score] += 1

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(models))
    bottoms = [0.0] * len(models)

    for s_idx, s in enumerate(scores):
        vals = [counts[m][s] for m in models]
        bars = ax.bar(x, vals, bottom=bottoms, label=score_labels[s_idx], color=colors[s_idx])
        for b, v, bot in zip(bars, vals, bottoms):
            if v > 5:
                ax.text(
                    b.get_x() + b.get_width() / 2, bot + v / 2,
                    str(v), ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold",
                )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_title("Score Distribution by Model")
    ax.set_ylabel("Trial count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.legend(title="score", frameon=False, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    charts_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(charts_dir)

    data_path = (
        repo_root
        / "experiments"
        / "prompt-injection-boundary-tags"
        / "rounds"
        / "round2b"
        / "data"
        / "results_r2_combined_latest.csv"
    )
    trials = load_trials(data_path)

    chart_scorer_calibration(charts_dir / "scorer_calibration_comparison.png")
    chart_injection_rate_by_model_condition(trials, charts_dir / "injection_rate_by_model_condition.png")
    chart_score_distribution_by_model(trials, charts_dir / "score_distribution_by_model.png")

    print(f"Generated 3 charts in {charts_dir}")


if __name__ == "__main__":
    main()
