#!/usr/bin/env python3
"""
Analyze prompt injection defense experiment results.
Reads results/results.csv and produces statistical analysis.
"""

import csv
from collections import defaultdict
from pathlib import Path

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy not available — skipping statistical tests")

ROUND_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROUND_DIR / "data" / "results.csv"


def load_results():
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r["status"] == "ok"]


def analyze():
    results = load_results()
    if not results:
        print("❌ No results found!")
        return

    conditions = ["raw", "static_tags", "dynamic_nonce"]
    payloads = sorted(set(r["payload"] for r in results))

    # ── Summary by condition ────────────────────────────────────────────
    print("=" * 70)
    print("PROMPT INJECTION DEFENSE EXPERIMENT — RESULTS")
    print("=" * 70)
    print()

    print("── Injection Success Rate by Condition ──")
    print(f"{'Condition':<20} {'Trials':>7} {'Avg Score':>10} {'Blocked (0)':>12} {'Ack (1)':>8} {'Partial (2)':>12} {'Full (3)':>9} {'Success%':>10}")
    print("-" * 90)

    condition_data = {}
    for cond in conditions:
        cond_results = [r for r in results if r["condition"] == cond]
        scores = [int(r["score"]) for r in cond_results]
        n = len(scores)
        avg = sum(scores) / n if n else 0
        counts = {i: scores.count(i) for i in range(4)}
        success_rate = (counts.get(2, 0) + counts.get(3, 0)) / n if n else 0
        condition_data[cond] = {
            "n": n, "avg": avg, "counts": counts,
            "success_rate": success_rate, "scores": scores
        }
        print(
            f"{cond:<20} {n:>7} {avg:>10.2f} "
            f"{counts[0]:>12} {counts[1]:>8} {counts[2]:>12} {counts[3]:>9} "
            f"{success_rate:>9.1%}"
        )

    # ── Chi-square test ─────────────────────────────────────────────────
    print()
    print("── Chi-Square Test: Injection Success vs Condition ──")
    print()

    # Build contingency table: rows=conditions, cols=[blocked(0-1), succeeded(2-3)]
    contingency = []
    for cond in conditions:
        scores = condition_data[cond]["scores"]
        blocked = sum(1 for s in scores if s <= 1)
        succeeded = sum(1 for s in scores if s >= 2)
        contingency.append([blocked, succeeded])
        print(f"  {cond:<20}: blocked={blocked}, succeeded={succeeded}")

    print()

    if HAS_SCIPY:
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            print(f"  χ² = {chi2:.4f}")
            print(f"  p-value = {p_value:.6f}")
            print(f"  degrees of freedom = {dof}")
            print(f"  Significant at α=0.05? {'YES ✅' if p_value < 0.05 else 'NO ❌'}")
            print(f"  Significant at α=0.01? {'YES ✅' if p_value < 0.01 else 'NO ❌'}")
            print()
            print("  Expected frequencies:")
            for i, cond in enumerate(conditions):
                print(f"    {cond:<20}: blocked={expected[i][0]:.1f}, succeeded={expected[i][1]:.1f}")
        except Exception as e:
            print(f"  Chi-square test failed: {e}")
            print("  (This can happen if all values are in one column — e.g., zero successes everywhere)")
    else:
        print("  scipy not installed; skipped.")

    # ── Pairwise comparisons ────────────────────────────────────────────
    print()
    print("── Pairwise Fisher's Exact Tests ──")
    print()
    if HAS_SCIPY:
        pairs = [("raw", "static_tags"), ("raw", "dynamic_nonce"), ("static_tags", "dynamic_nonce")]
        for a, b in pairs:
            a_scores = condition_data[a]["scores"]
            b_scores = condition_data[b]["scores"]
            table = [
                [sum(1 for s in a_scores if s <= 1), sum(1 for s in a_scores if s >= 2)],
                [sum(1 for s in b_scores if s <= 1), sum(1 for s in b_scores if s >= 2)],
            ]
            odds, p = stats.fisher_exact(table)
            print(f"  {a} vs {b}: OR={odds:.3f}, p={p:.6f} {'*' if p < 0.05 else ''}")
    else:
        print("  scipy not installed; skipped.")

    # ── Breakdown by payload ────────────────────────────────────────────
    print()
    print("── Injection Success Rate by Payload × Condition ──")
    print(f"{'Payload':<25}", end="")
    for cond in conditions:
        print(f" {cond:>15}", end="")
    print()
    print("-" * 70)

    for payload in payloads:
        print(f"{payload:<25}", end="")
        for cond in conditions:
            pr = [r for r in results if r["payload"] == payload and r["condition"] == cond]
            scores = [int(r["score"]) for r in pr]
            avg = sum(scores) / len(scores) if scores else 0
            succeeded = sum(1 for s in scores if s >= 2)
            total = len(scores)
            print(f" {succeeded}/{total} (avg {avg:.1f})", end="")
        print()

    # ── Effect size (Cramér's V) ────────────────────────────────────────
    print()
    if HAS_SCIPY:
        try:
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            n_total = sum(sum(row) for row in contingency)
            k = min(len(contingency), len(contingency[0]))
            cramers_v = (chi2 / (n_total * (k - 1))) ** 0.5 if n_total * (k - 1) > 0 else 0
            print(f"── Effect Size: Cramér's V = {cramers_v:.4f} ──")
            if cramers_v < 0.1:
                print("   Interpretation: Negligible effect")
            elif cramers_v < 0.3:
                print("   Interpretation: Small effect")
            elif cramers_v < 0.5:
                print("   Interpretation: Medium effect")
            else:
                print("   Interpretation: Large effect")
        except Exception:
            pass

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    analyze()
