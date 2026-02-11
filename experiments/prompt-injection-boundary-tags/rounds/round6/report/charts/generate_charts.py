#!/usr/bin/env python3
"""Generate Round 6 charts from policy evaluation data."""

import csv
import os
from pathlib import Path

# Optional: use matplotlib if available, otherwise generate SVG directly
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ROUND_DIR = Path(__file__).resolve().parent.parent.parent
def _resolve_data_path() -> Path:
    latest = ROUND_DIR / "data" / "policy_eval_latest.csv"
    if latest.exists():
        return latest
    candidates = sorted(ROUND_DIR.glob("data/policy_eval_*.csv"), reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No policy_eval CSV found in {ROUND_DIR / 'data'}")


DATA_PATH = _resolve_data_path()
CHART_DIR = Path(__file__).resolve().parent


def load_data():
    """Load and aggregate policy evaluation data."""
    configs = {}
    with open(DATA_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row["config_name"]
            if cfg not in configs:
                configs[cfg] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "tools": {}}
            configs[cfg]["tp"] += int(row["true_positive"] == "True")
            configs[cfg]["fp"] += int(row["false_positive"] == "True")
            configs[cfg]["tn"] += int(row["true_negative"] == "True")
            configs[cfg]["fn"] += int(row["false_negative"] == "True")

            tool = row["tool_name"]
            if tool not in configs[cfg]["tools"]:
                configs[cfg]["tools"][tool] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            configs[cfg]["tools"][tool]["tp"] += int(row["true_positive"] == "True")
            configs[cfg]["tools"][tool]["fp"] += int(row["false_positive"] == "True")
            configs[cfg]["tools"][tool]["tn"] += int(row["true_negative"] == "True")
            configs[cfg]["tools"][tool]["fn"] += int(row["false_negative"] == "True")
    return configs


def calc_metrics(d):
    """Calculate precision, recall, F1, FPR from confusion matrix counts."""
    tp, fp, tn, fn = d["tp"], d["fp"], d["tn"], d["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def generate_precision_recall_chart(configs):
    """Bar chart: precision and recall per config."""
    if not HAS_MPL:
        print("matplotlib not available, skipping precision_recall_by_config.png")
        return

    order = ["permissive", "balanced", "strict", "paranoid"]
    metrics = {cfg: calc_metrics(configs[cfg]) for cfg in order}

    x = range(len(order))
    width = 0.35

    _fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar([i - width / 2 for i in x], [metrics[c]["precision"] for c in order],
                   width, label="Precision", color="#2563eb")
    bars2 = ax.bar([i + width / 2 for i in x], [metrics[c]["recall"] for c in order],
                   width, label="Recall", color="#dc2626")

    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall by Policy Configuration")
    ax.set_xticks(list(x))
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "precision_recall_by_config.png", dpi=150)
    plt.close()
    print("Saved precision_recall_by_config.png")


def generate_fpr_chart(configs):
    """Bar chart: false positive rate per config."""
    if not HAS_MPL:
        print("matplotlib not available, skipping fpr_by_config.png")
        return

    order = ["permissive", "balanced", "strict", "paranoid"]
    metrics = {cfg: calc_metrics(configs[cfg]) for cfg in order}

    _fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#22c55e", "#22c55e", "#f59e0b", "#dc2626"]
    bars = ax.bar(order, [metrics[c]["fpr"] for c in order], color=colors)

    ax.set_ylabel("False Positive Rate")
    ax.set_title("False Positive Rate by Policy Configuration")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5)
    ax.text(3.4, 0.07, "5% target", color="gray", fontsize=8)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "fpr_by_config.png", dpi=150)
    plt.close()
    print("Saved fpr_by_config.png")


def generate_tool_heatmap(configs):
    """Heatmap: recall by tool type x config."""
    if not HAS_MPL:
        print("matplotlib not available, skipping tool_recall_heatmap.png")
        return

    order = ["permissive", "balanced", "strict", "paranoid"]
    tools = ["read_file", "execute_command", "send_message"]

    data = []
    for tool in tools:
        row = []
        for cfg in order:
            m = calc_metrics(configs[cfg]["tools"].get(tool, {"tp": 0, "fp": 0, "tn": 0, "fn": 0}))
            row.append(m["recall"])
        data.append(row)

    _fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels(tools)
    ax.set_title("Recall by Tool Type and Policy Configuration")

    for i in range(len(tools)):
        for j in range(len(order)):
            ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center",
                    color="black" if data[i][j] > 0.5 else "white", fontsize=11)

    plt.colorbar(im, ax=ax, label="Recall")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "tool_recall_heatmap.png", dpi=150)
    plt.close()
    print("Saved tool_recall_heatmap.png")


if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH}")
    configs = load_data()
    print(f"Loaded {sum(c['tp'] + c['fp'] + c['tn'] + c['fn'] for c in configs.values())} evaluations")

    generate_precision_recall_chart(configs)
    generate_fpr_chart(configs)
    generate_tool_heatmap(configs)
    print("Done.")
