#!/usr/bin/env python3

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")
    return rows


def score_counts_by_condition(
    rows: list[dict[str, str]],
) -> tuple[list[str], dict[str, Counter[int]]]:
    conds = sorted({r["condition"] for r in rows})
    counts: dict[str, Counter[int]] = {c: Counter() for c in conds}
    for r in rows:
        counts[r["condition"]][int(r["score"])] += 1
    return conds, counts


def mean_score_by_condition(
    conds: list[str],
    counts: dict[str, Counter[int]],
) -> dict[str, float]:
    means: dict[str, float] = {}
    for c in conds:
        n = sum(counts[c].values())
        if n == 0:
            means[c] = 0.0
            continue
        means[c] = sum(score * k for score, k in counts[c].items()) / n
    return means


def save_score_distribution(
    *,
    conds: list[str],
    counts: dict[str, Counter[int]],
    out_path: Path,
) -> None:
    scores = [0, 1, 2, 3]
    colors = {
        0: "#2c7bb6",
        1: "#abd9e9",
        2: "#fdae61",
        3: "#d7191c",
    }

    totals = {c: sum(counts[c].values()) for c in conds}
    fractions_by_score = {
        s: [(counts[c].get(s, 0) / totals[c] if totals[c] else 0.0) for c in conds]
        for s in scores
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    bottoms = [0.0 for _ in conds]
    for s in scores:
        vals = fractions_by_score[s]
        ax.bar(
            conds,
            vals,
            bottom=bottoms,
            label=f"score {s}",
            color=colors[s],
            edgecolor="white",
            linewidth=0.8,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]

    ax.set_title("Round 1: Score Distribution by Condition (Claude 3.5 Haiku)")
    ax.set_ylabel("Fraction of trials")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_condition_comparison(
    *,
    conds: list[str],
    means: dict[str, float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [means[c] for c in conds]
    ax.bar(conds, vals, color="#4d4d4d")

    ax.set_title("Round 1: Mean Score by Condition (Claude 3.5 Haiku)")
    ax.set_ylabel("Mean score (0-3)")
    ax.set_ylim(0, max(0.25, max(vals) * 1.25))
    ax.grid(axis="y", alpha=0.25)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = (here / "../../data/results.csv").resolve()
    if not csv_path.exists():
        raise SystemExit(f"missing results csv: {csv_path}")

    rows = read_rows(csv_path)
    conds, counts = score_counts_by_condition(rows)
    means = mean_score_by_condition(conds, counts)

    out_score_dist = (here / "score_distribution.png").resolve()
    out_means = (here / "condition_comparison.png").resolve()

    save_score_distribution(conds=conds, counts=counts, out_path=out_score_dist)
    save_condition_comparison(conds=conds, means=means, out_path=out_means)

    print(f"wrote: {out_score_dist}")
    print(f"wrote: {out_means}")


if __name__ == "__main__":
    main()

