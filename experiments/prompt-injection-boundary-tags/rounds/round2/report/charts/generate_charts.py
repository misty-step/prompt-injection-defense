#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    tool_calls_json: str


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
        required = {
            "model",
            "payload",
            "condition",
            "trial_num",
            "score",
            "num_tool_calls",
            "tool_calls",
        }
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
                    tool_calls_json=row["tool_calls"],
                )
            )
        return out


def injection_success(row: TrialRow) -> bool:
    return row.score >= 2


def pct(x: float) -> float:
    return 100.0 * x


def safe_label(s: str) -> str:
    return s.replace("_", " ")


def chart_injection_rate_by_condition(trials: Iterable[TrialRow], out_path: Path) -> None:
    models = sorted({t.model for t in trials})
    conditions = ["raw", "static_tags", "dynamic_nonce"]

    rates_by_model_cond: dict[tuple[str, str], float] = {}
    for m in models:
        for c in conditions:
            sub = [t for t in trials if t.model == m and t.condition == c]
            if not sub:
                rates_by_model_cond[(m, c)] = 0.0
                continue
            rates_by_model_cond[(m, c)] = sum(injection_success(t) for t in sub) / len(sub)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(models))
    width = 0.22
    offsets = [-width, 0.0, width]
    colors = ["#444444", "#2E86AB", "#F18F01"]

    for i, c in enumerate(conditions):
        ys = [pct(rates_by_model_cond[(m, c)]) for m in models]
        xs = [xi + offsets[i] for xi in x]
        bars = ax.bar(xs, ys, width=width, label=c, color=colors[i])
        for b, y in zip(bars, ys):
            ax.text(
                b.get_x() + b.get_width() / 2,
                y + 1.0,
                f"{y:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Injection Rate by Condition (score >= 2)")
    ax.set_ylabel("Injection rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(0, max(10.0, ax.get_ylim()[1]))
    ax.legend(title="condition", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_payload_heatmap(trials: Iterable[TrialRow], out_path: Path) -> None:
    conditions = ["raw", "static_tags", "dynamic_nonce"]
    payloads = sorted({t.payload for t in trials})

    # payload x condition injection rate, aggregated across models
    matrix: list[list[float]] = []
    for p in payloads:
        row: list[float] = []
        for c in conditions:
            sub = [t for t in trials if t.payload == p and t.condition == c]
            row.append(pct(sum(injection_success(t) for t in sub) / len(sub)) if sub else 0.0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="magma")
    ax.set_title("Injection Rate Heatmap (Payload x Condition)")
    ax.set_xlabel("condition")
    ax.set_ylabel("payload")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([safe_label(c) for c in conditions])
    ax.set_yticks(range(len(payloads)))
    ax.set_yticklabels([safe_label(p) for p in payloads])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("injection rate (%)")

    # annotate cells for readability at typical sizes
    for i in range(len(payloads)):
        for j in range(len(conditions)):
            val = matrix[i][j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def chart_tool_calls_by_condition(trials: Iterable[TrialRow], out_path: Path) -> None:
    conditions = ["raw", "static_tags", "dynamic_nonce"]
    totals = []
    for c in conditions:
        totals.append(sum(t.num_tool_calls for t in trials if t.condition == c))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#444444", "#2E86AB", "#F18F01"]
    bars = ax.bar([safe_label(c) for c in conditions], totals, color=colors)
    ax.set_title("Malicious Tool Calls by Condition")
    ax.set_ylabel("Tool calls (count)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    for b, v in zip(bars, totals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center", va="bottom", fontsize=10)

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

    chart_injection_rate_by_condition(trials, charts_dir / "injection_rate_by_condition.png")
    chart_payload_heatmap(trials, charts_dir / "payload_heatmap.png")
    chart_tool_calls_by_condition(trials, charts_dir / "tool_calls_by_condition.png")


if __name__ == "__main__":
    main()

