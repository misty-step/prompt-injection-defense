#!/usr/bin/env python3
"""
Round 2: Analyze prompt injection defense experiment results.
Reads results_r2/results_r2_combined_latest.csv and produces statistical analysis.

Features:
- Per-model breakdown
- Per-payload breakdown
- Chi-square tests (overall and per-model)
- Fisher's exact pairwise comparisons
- Cramér's V effect size
- Tool call analysis
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy not available — skipping statistical tests")

ROUND_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROUND_DIR / "data"
CSV_PATH = RESULTS_DIR / "results_r2_combined_latest.csv"
CONDITIONS = ["raw", "static_tags", "dynamic_nonce"]


def load_results(path=None):
    p = path or CSV_PATH
    if not p.exists():
        # Try to find most recent combined file
        candidates = sorted(RESULTS_DIR.glob("results_r2_combined_*.csv"), reverse=True)
        if candidates:
            p = candidates[0]
        else:
            print(f"❌ No results found at {p}")
            sys.exit(1)

    print(f"📁 Loading results from {p}")
    with open(p) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if r["status"] == "ok":
                r["score"] = int(r["score"])
                r["num_tool_calls"] = int(r["num_tool_calls"])
                rows.append(r)
    print(f"   Loaded {len(rows)} valid trials")
    return rows


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_subheader(title):
    print(f"\n── {title} {'─'*(74-len(title))}")


def analyze_condition(results, label="Overall"):
    """Analyze results by condition, return condition_data dict."""
    condition_data = {}

    print_subheader(f"Injection Rate by Condition ({label})")
    print(f"{'Condition':<18} {'N':>5} {'Avg':>6} {'S=0':>5} {'S=1':>5} {'S=2':>5} {'S=3':>5} {'Inj%':>7} {'Tool%':>7}")
    print("-" * 80)

    for cond in CONDITIONS:
        cr = [r for r in results if r["condition"] == cond]
        if not cr:
            continue
        scores = [r["score"] for r in cr]
        n = len(scores)
        avg = sum(scores) / n
        counts = {i: scores.count(i) for i in range(4)}
        inj_rate = (counts[2] + counts[3]) / n
        tool_rate = sum(1 for r in cr if r["num_tool_calls"] > 0) / n

        condition_data[cond] = {
            "n": n, "avg": avg, "counts": counts,
            "inj_rate": inj_rate, "tool_rate": tool_rate, "scores": scores
        }

        print(
            f"{cond:<18} {n:>5} {avg:>6.2f} "
            f"{counts[0]:>5} {counts[1]:>5} {counts[2]:>5} {counts[3]:>5} "
            f"{inj_rate:>6.0%} {tool_rate:>6.0%}"
        )

    return condition_data


def chi_square_test(condition_data, label=""):
    """Run chi-square and Fisher's exact tests on condition data."""
    if not HAS_SCIPY:
        return

    prefix = f" ({label})" if label else ""

    print_subheader(f"Chi-Square Test{prefix}")

    # Build contingency table: rows=conditions, cols=[blocked(0-1), succeeded(2-3)]
    contingency = []
    for cond in CONDITIONS:
        if cond not in condition_data:
            continue
        scores = condition_data[cond]["scores"]
        blocked = sum(1 for s in scores if s <= 1)
        succeeded = sum(1 for s in scores if s >= 2)
        contingency.append([blocked, succeeded])
        print(f"  {cond:<18}: blocked={blocked}, succeeded={succeeded}")

    if len(contingency) < 2:
        print("  ⚠️ Not enough conditions with data for chi-square test")
        return

    print()
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        print(f"  χ² = {chi2:.4f}")
        print(f"  p-value = {p_value:.6f}")
        print(f"  df = {dof}")
        print(f"  Significant at α=0.05? {'YES ✅' if p_value < 0.05 else 'NO ❌'}")
        print(f"  Significant at α=0.01? {'YES ✅' if p_value < 0.01 else 'NO ❌'}")

        # Cramér's V
        n_total = sum(sum(row) for row in contingency)
        k = min(len(contingency), len(contingency[0]))
        v = (chi2 / (n_total * (k - 1))) ** 0.5 if n_total * (k - 1) > 0 else 0
        interp = "Negligible" if v < 0.1 else "Small" if v < 0.3 else "Medium" if v < 0.5 else "Large"
        print(f"  Cramér's V = {v:.4f} ({interp} effect)")

    except Exception as e:
        print(f"  Chi-square failed: {e}")
        print("  (Likely all values in one column — zero variance)")

    # Pairwise Fisher's exact
    print_subheader(f"Pairwise Fisher's Exact Tests{prefix}")
    pairs = [("raw", "static_tags"), ("raw", "dynamic_nonce"), ("static_tags", "dynamic_nonce")]
    for a, b in pairs:
        if a not in condition_data or b not in condition_data:
            continue
        a_scores = condition_data[a]["scores"]
        b_scores = condition_data[b]["scores"]
        table = [
            [sum(1 for s in a_scores if s <= 1), sum(1 for s in a_scores if s >= 2)],
            [sum(1 for s in b_scores if s <= 1), sum(1 for s in b_scores if s >= 2)],
        ]
        # Avoid division by zero in odds ratio
        try:
            odds, p = stats.fisher_exact(table)
            sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            print(f"  {a} vs {b}: OR={odds:.3f}, p={p:.6f}{sig}")
        except Exception as e:
            print(f"  {a} vs {b}: Error — {e}")


def payload_breakdown(results):
    """Show injection rate per payload × condition."""
    print_subheader("Injection Rate by Payload × Condition")

    payloads = sorted(set(r["payload"] for r in results))

    print(f"{'Payload':<25}", end="")
    for cond in CONDITIONS:
        print(f" {'':>2}{cond:>15}", end="")
    print(f" {'':>2}{'TOTAL':>8}")
    print("-" * 80)

    for payload in payloads:
        print(f"{payload:<25}", end="")
        for cond in CONDITIONS:
            pr = [r for r in results if r["payload"] == payload and r["condition"] == cond]
            if not pr:
                print(f" {'':>2}{'N/A':>15}", end="")
                continue
            scores = [r["score"] for r in pr]
            succeeded = sum(1 for s in scores if s >= 2)
            total = len(scores)
            avg = sum(scores) / total
            print(f" {'':>2}{succeeded}/{total} avg={avg:.1f}", end="  ")
        # Total across conditions
        all_pr = [r for r in results if r["payload"] == payload]
        all_scores = [r["score"] for r in all_pr]
        all_succ = sum(1 for s in all_scores if s >= 2)
        all_avg = sum(all_scores) / len(all_scores) if all_scores else 0
        print(f" {all_succ}/{len(all_scores)} ({all_avg:.1f})")


def tool_call_analysis(results):
    """Analyze which tools were called and by which payloads."""
    print_subheader("Tool Call Analysis")

    tool_counts = defaultdict(int)
    tool_by_condition = defaultdict(lambda: defaultdict(int))
    tool_by_payload = defaultdict(lambda: defaultdict(int))

    for r in results:
        if r["num_tool_calls"] > 0:
            try:
                tcs = json.loads(r["tool_calls"])
                for tc in tcs:
                    name = tc.get("name", "unknown")
                    tool_counts[name] += 1
                    tool_by_condition[r["condition"]][name] += 1
                    tool_by_payload[r["payload"]][name] += 1
            except json.JSONDecodeError:
                pass

    if not tool_counts:
        print("  No tool calls detected across all trials.")
        return

    print("  Tool call counts:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"    {tool}: {count}")

    print("\n  Tool calls by condition:")
    for cond in CONDITIONS:
        if cond in tool_by_condition:
            tools = tool_by_condition[cond]
            total = sum(tools.values())
            print(f"    {cond}: {total} total — {dict(tools)}")
        else:
            print(f"    {cond}: 0")

    print("\n  Payloads triggering tool calls:")
    for payload, tools in sorted(tool_by_payload.items(), key=lambda x: -sum(x[1].values())):
        total = sum(tools.values())
        print(f"    {payload}: {total} calls — {dict(tools)}")


def model_comparison(results):
    """Compare across models."""
    models = sorted(set(r["model"] for r in results))
    if len(models) <= 1:
        return

    print_subheader("Cross-Model Comparison")
    print(f"{'Model':<22} {'N':>5} {'Avg':>6} {'Inj%':>7} {'Tool%':>7} {'Best Cond':<18} {'Best Inj%':>9}")
    print("-" * 80)

    for model in models:
        mr = [r for r in results if r["model"] == model]
        n = len(mr)
        avg = sum(r["score"] for r in mr) / n
        inj = sum(1 for r in mr if r["score"] >= 2) / n
        tool = sum(1 for r in mr if r["num_tool_calls"] > 0) / n

        # Find best condition for this model
        best_cond = None
        best_inj = 1.0
        for cond in CONDITIONS:
            cr = [r for r in mr if r["condition"] == cond]
            if cr:
                ci = sum(1 for r in cr if r["score"] >= 2) / len(cr)
                if ci < best_inj:
                    best_inj = ci
                    best_cond = cond

        print(f"{model:<22} {n:>5} {avg:>6.2f} {inj:>6.0%} {tool:>6.0%} {best_cond or 'N/A':<18} {best_inj:>8.0%}")


def main():
    results = load_results()
    if not results:
        return

    models = sorted(set(r["model"] for r in results))

    print_header("ROUND 2: Prompt Injection Defense Experiment — Analysis")
    print(f"  Models tested: {', '.join(models)}")
    print(f"  Total valid trials: {len(results)}")
    print(f"  Conditions: {', '.join(CONDITIONS)}")
    print(f"  Payloads: {len(set(r['payload'] for r in results))}")

    # ── Overall analysis ──
    cond_data = analyze_condition(results, "All Models")
    chi_square_test(cond_data, "All Models")

    # ── Per-model analysis ──
    for model in models:
        mr = [r for r in results if r["model"] == model]
        print_header(f"Model: {model} ({len(mr)} trials)")
        cd = analyze_condition(mr, model)
        chi_square_test(cd, model)

    # ── Payload breakdown ──
    print_header("Payload Analysis")
    payload_breakdown(results)

    # Per-model payload breakdown
    for model in models:
        mr = [r for r in results if r["model"] == model]
        print(f"\n  [{model}]")
        payloads = sorted(set(r["payload"] for r in mr))
        for payload in payloads:
            pr = [r for r in mr if r["payload"] == payload]
            raw = [r for r in pr if r["condition"] == "raw"]
            static = [r for r in pr if r["condition"] == "static_tags"]
            dynamic = [r for r in pr if r["condition"] == "dynamic_nonce"]
            raw_inj = sum(1 for r in raw if r["score"] >= 2) / len(raw) if raw else 0
            sta_inj = sum(1 for r in static if r["score"] >= 2) / len(static) if static else 0
            dyn_inj = sum(1 for r in dynamic if r["score"] >= 2) / len(dynamic) if dynamic else 0
            print(f"    {payload:<25} raw={raw_inj:.0%}  static={sta_inj:.0%}  dynamic={dyn_inj:.0%}")

    # ── Tool calls ──
    print_header("Tool Call Analysis")
    tool_call_analysis(results)

    for model in models:
        mr = [r for r in results if r["model"] == model]
        print(f"\n  [{model}]")
        tool_call_analysis(mr)

    # ── Cross-model comparison ──
    print_header("Cross-Model Comparison")
    model_comparison(results)

    print_header("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
