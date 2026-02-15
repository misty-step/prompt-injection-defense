"""Cross-round analysis for canonical prompt-injection run dataset with confidence intervals.

Adds binomial confidence intervals (Wilson score and Clopper-Pearson) for injection rates
per condition/model, plus cross-round baseline comparison reports.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


# ============================================================================
# Confidence Interval Calculations
# ============================================================================

def wilson_score_interval(
    n_success: int, n_total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion.

    The Wilson interval is generally preferred over the Wald interval
    because it works well for extreme proportions and small samples.

    Args:
        n_success: Number of successful trials
        n_total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) in [0, 1]
    """
    if n_total == 0:
        return (0.0, 0.0)

    # Handle edge cases: when p_hat is 0 or 1, Wilson without adjustment
    # gives non-degenerate intervals, which is actually correct behavior.
    # Wilson interval with "plus 4" adjustment is sometimes preferred but
    # we use the standard Wilson interval which handles these correctly.
    p_hat = n_success / n_total

    # Critical value (z) for the confidence level
    # For 95%: z ≈ 1.96
    alpha = 1 - confidence
    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.44 if confidence == 0.85 else math.sqrt(2) * math.erfcinv(alpha)

    z2 = z * z
    n = n_total

    # Wilson score interval formula
    denominator = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denominator
    width = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n) / denominator

    lower = max(0.0, centre - width)
    upper = min(1.0, centre + width)

    return (lower, upper)


def clopper_pearson_interval(
    n_success: int, n_total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate Clopper-Pearson (exact) confidence interval.

    This is the "exact" binomial confidence interval. Uses scipy's
    beta distribution if available, otherwise falls back to Wilson.

    Args:
        n_success: Number of successful trials
        n_total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) in [0, 1]
    """
    if n_total == 0:
        return (0.0, 0.0)

    alpha = 1 - confidence

    try:
        from scipy import stats

        # Use scipy's beta distribution for exact Clopper-Pearson
        # Lower bound: beta distribution quantile with alpha/2
        # Upper bound: beta distribution quantile with 1 - alpha/2
        if n_success == 0:
            lower = 0.0
        else:
            lower = stats.beta.ppf(alpha / 2, n_success, n_total - n_success + 1)

        if n_success == n_total:
            upper = 1.0
        else:
            upper = stats.beta.ppf(1 - alpha / 2, n_success + 1, n_total - n_success)

        return (float(lower), float(upper))
    except ImportError:
        # Fallback to Wilson if scipy not available
        return wilson_score_interval(n_success, n_total, confidence)


# ============================================================================
# CI-Aware Statistics Container
# ============================================================================

@dataclass
class CIStats:
    """Statistics with confidence intervals for a group."""

    n_total: int = 0
    n_ok: int = 0
    n_error: int = 0
    n_success: int = 0  # injection successes (score >= 2)
    ci_method: str = "wilson"
    ci_confidence: float = 0.95

    @property
    def inj_rate(self) -> float:
        """Injection rate among successful trials."""
        return (self.n_success / self.n_ok) if self.n_ok else 0.0

    @property
    def inj_rate_ci(self) -> Tuple[float, float]:
        """Confidence interval for injection rate."""
        if self.ci_method == "wilson":
            return wilson_score_interval(self.n_success, self.n_ok, self.ci_confidence)
        else:
            return clopper_pearson_interval(self.n_success, self.n_ok, self.ci_confidence)

    @property
    def error_rate(self) -> float:
        """Error rate among all trials."""
        return (self.n_error / self.n_total) if self.n_total else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate (ok status) among all trials."""
        return (self.n_ok / self.n_total) if self.n_total else 0.0

    def format_inj_rate_with_ci(self) -> str:
        """Format injection rate with CI for display."""
        rate = self.inj_rate
        lower, upper = self.inj_rate_ci

        if self.n_ok == 0:
            return "N/A (no ok trials)"

        rate_str = f"{rate * 100:.1f}%"
        ci_str = f"[{lower * 100:.1f}%, {upper * 100:.1f}%]"

        if self.n_ok < 10:
            return f"{rate_str} {ci_str} (n={self.n_ok}, low-N warning)"
        return f"{rate_str} {ci_str}"


# ============================================================================
# Data Loading
# ============================================================================

def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Canonical dataset not found: {path}")
    with path.open() as handle:
        return list(csv.DictReader(handle))


# ============================================================================
# Analysis Functions with CI
# ============================================================================

def summarize_by_with_ci(
    rows: Iterable[Dict[str, str]],
    keys: Tuple[str, ...],
    ci_method: str = "wilson",
    ci_confidence: float = 0.95,
) -> Dict[Tuple[str, ...], CIStats]:
    """Summarize rows by grouping keys, with confidence intervals."""
    stats: Dict[Tuple[str, ...], CIStats] = defaultdict(
        lambda: CIStats(ci_method=ci_method, ci_confidence=ci_confidence)
    )

    for row in rows:
        group = tuple(row.get(k, "") for k in keys)
        status = row.get("status", "")
        score = parse_int(row.get("score"), default=-1)

        stats[group].n_total += 1
        if status == "ok":
            stats[group].n_ok += 1
            if score >= 2:
                stats[group].n_success += 1
        else:
            stats[group].n_error += 1

    return dict(stats)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_pct_ci(lower: float, upper: float) -> str:
    return f"[{lower * 100:.1f}%, {upper * 100:.1f}%]"


def print_table_with_ci(
    title: str,
    keys: Tuple[str, ...],
    stats: Dict[Tuple[str, ...], CIStats],
    show_ci: bool = True,
) -> None:
    """Print formatted table with confidence intervals."""
    print()
    print("=" * 110)
    print(title)
    if show_ci:
        print("(Injection rates shown with 95% Wilson score confidence intervals)")
    print("=" * 110)

    key_headers = " | ".join(k.upper() for k in keys)
    if show_ci:
        print(
            f"{key_headers:<40} {'TOTAL':>8} {'OK':>8} {'ERROR':>8} "
            f"{'INJ_OK':>10} {'INJ_RATE':>20} {'CI_95%':>25}"
        )
    else:
        print(
            f"{key_headers:<40} {'TOTAL':>8} {'OK':>8} {'ERROR':>8} "
            f"{'INJ_OK':>10} {'INJ_RATE':>12}"
        )
    print("-" * 110)

    for group in sorted(stats):
        row = stats[group]
        n_total = row.n_total
        n_ok = row.n_ok
        n_error = row.n_error
        n_success = row.n_success
        rate = row.inj_rate
        lower, upper = row.inj_rate_ci
        key_label = " | ".join(group)

        if show_ci:
            rate_str = format_pct(rate)
            ci_str = format_pct_ci(lower, upper)
            print(
                f"{key_label:<40} {n_total:>8} {n_ok:>8} {n_error:>8} "
                f"{n_success:>10} {rate_str:>20} {ci_str:>25}"
            )
        else:
            print(
                f"{key_label:<40} {n_total:>8} {n_ok:>8} {n_error:>8} "
                f"{n_success:>10} {format_pct(rate):>12}"
            )


# ============================================================================
# Cross-Round Baseline Comparison
# ============================================================================

@dataclass
class RoundBaseline:
    """Baseline metrics for a single round."""

    round_id: str
    raw_stats: Optional[CIStats] = None
    tags_only_stats: Optional[CIStats] = None
    instruction_only_stats: Optional[CIStats] = None
    instruction_tags_stats: Optional[CIStats] = None
    full_stack_stats: Optional[CIStats] = None

    @property
    def defense_ordering(self) -> List[Tuple[str, Optional[CIStats]]]:
        """Return defense conditions in canonical order."""
        return [
            ("raw", self.raw_stats),
            ("tags_only", self.tags_only_stats),
            ("instruction_only", self.instruction_only_stats),
            ("instruction_tags", self.instruction_tags_stats),
            ("full_stack", self.full_stack_stats),
        ]

    def format_defense_stability(self) -> str:
        """Format defense ordering stability report."""
        lines = [f"  {self.round_id} defense ordering (by inj rate):"]

        # Sort conditions by injection rate
        sorted_defenses = []
        for name, stats in self.defense_ordering:
            if stats and stats.n_ok > 0:
                sorted_defenses.append((name, stats.inj_rate, stats))

        if not sorted_defenses:
            return f"  {self.round_id}: No data"

        sorted_defenses.sort(key=lambda x: x[1], reverse=True)

        for i, (name, rate, stats) in enumerate(sorted_defenses, 1):
            lines.append(f"    {i}. {name:<20} {rate * 100:>6.1f}%  (n={stats.n_ok})")

        return "\n".join(lines)

    def format_delta_vs_baseline(self, baseline: "RoundBaseline") -> str:
        """Format delta comparison to another round's baseline."""
        lines = [f"  {baseline.round_id} → {self.round_id} deltas:"]

        for name, stats in self.defense_ordering:
            baseline_stats = getattr(baseline, f"{name}_stats", None)
            if stats and baseline_stats and stats.n_ok > 0 and baseline_stats.n_ok > 0:
                delta = stats.inj_rate - baseline_stats.inj_rate
                delta_str = f"{delta * 100:+.1f}%"
                lines.append(f"    {name:<20} {baseline_stats.inj_rate * 100:>6.1f}% → {stats.inj_rate * 100:>6.1f}% = {delta_str}")

        return "\n".join(lines)


def baseline_compare_rounds(
    round_stats: Dict[str, Dict[str, CIStats]], baseline_round_id: str = "round3"
) -> List[RoundBaseline]:
    """Build round baselines and compute defense ordering stability.

    Args:
        round_stats: Dict mapping round_id -> condition -> CIStats
        baseline_round_id: The round to use as comparison baseline

    Returns:
        List of RoundBaseline objects
    """
    baselines = []

    for round_id, conditions in sorted(round_stats.items()):
        rb = RoundBaseline(round_id=round_id)
        rb.raw_stats = conditions.get("raw")
        rb.tags_only_stats = conditions.get("tags_only")
        rb.instruction_only_stats = conditions.get("instruction_only")
        rb.instruction_tags_stats = conditions.get("instruction_tags")
        rb.full_stack_stats = conditions.get("full_stack")
        baselines.append(rb)

    return baselines


def print_cross_round_comparison(
    baselines: List[RoundBaseline], baseline_round_id: str = "round3"
) -> None:
    """Print cross-round baseline comparison report."""
    print()
    print("=" * 80)
    print("CROSS-ROUND BASELINE COMPARISON")
    print(f"(Using {baseline_round_id} as reference baseline)")
    print("=" * 80)

    # Defense ordering for each round
    print("\n--- Defense Ordering Stability ---")
    for rb in baselines:
        print(rb.format_defense_stability())

    # Delta comparison to baseline
    baseline_rb = next((rb for rb in baselines if rb.round_id == baseline_round_id), None)
    if baseline_rb:
        print(f"\n--- Delta vs {baseline_round_id} (lower = better defense) ---")
        for rb in baselines:
            if rb.round_id != baseline_round_id:
                print(rb.format_delta_vs_baseline(baseline_rb))


# ============================================================================
# Main Entry Point
# ============================================================================


def format_rate_with_ci(n_success: int, n_total: int, method: str = "wilson") -> str:
    """Public API: format rate with CI for external use."""
    if n_total == 0:
        return "N/A"
    rate = n_success / n_total
    if method == "wilson":
        lower, upper = wilson_score_interval(n_success, n_total)
    else:
        lower, upper = clopper_pearson_interval(n_success, n_total)

    if n_total < 10:
        return f"{rate * 100:.1f}% [{lower * 100:.1f}%, {upper * 100:.1f}%] (n={n_total}, low-N)"
    return f"{rate * 100:.1f}% [{lower * 100:.1f}%, {upper * 100:.1f}%]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze canonical prompt-injection runs CSV with confidence intervals."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/prompt-injection-boundary-tags/rounds/canonical/runs_v1.csv"),
        help="Path to canonical CSV dataset.",
    )
    parser.add_argument(
        "--ci-method",
        choices=["wilson", "clopper-pearson"],
        default="wilson",
        help="Confidence interval method (default: wilson).",
    )
    parser.add_argument(
        "--ci-confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95).",
    )
    parser.add_argument(
        "--baseline-round",
        type=str,
        default="round3",
        help="Round to use as baseline for cross-round comparison (default: round3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write cross-round comparison report markdown.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_path = args.input
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    rows = load_rows(input_path)
    if not rows:
        raise SystemExit("No rows found in canonical dataset.")

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Using CI method: {args.ci_method} at {args.ci_confidence * 100:.0f}% confidence")

    # Standard tables with CI
    by_round = summarize_by_with_ci(
        rows, ("round_id",), ci_method=args.ci_method, ci_confidence=args.ci_confidence
    )
    by_round_condition = summarize_by_with_ci(
        rows,
        ("round_id", "condition"),
        ci_method=args.ci_method,
        ci_confidence=args.ci_confidence,
    )
    by_round_model = summarize_by_with_ci(
        rows,
        ("round_id", "model"),
        ci_method=args.ci_method,
        ci_confidence=args.ci_confidence,
    )

    print_table_with_ci("Summary By Round", ("round_id",), by_round)
    print_table_with_ci(
        "Summary By Round + Condition", ("round_id", "condition"), by_round_condition
    )
    print_table_with_ci("Summary By Round + Model", ("round_id", "model"), by_round_model)

    # Cross-round baseline comparison
    # Build nested structure: round_id -> condition -> CIStats
    round_stats_nested: Dict[str, Dict[str, CIStats]] = defaultdict(dict)
    for keys, stats in by_round_condition.items():
        round_id, condition = keys
        round_stats_nested[round_id][condition] = stats

    baselines = baseline_compare_rounds(round_stats_nested, baseline_round_id=args.baseline_round)
    print_cross_round_comparison(baselines, baseline_round_id=args.baseline_round)

    # Optional markdown output
    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            f.write("# Cross-Round Injection Rate Analysis\n\n")
            f.write(f"Analysis generated from {len(rows)} canonical run records.\n\n")
            f.write(f"**CI Method:** {args.ci_method} at {args.ci_confidence * 100:.0f}% confidence\n")
            f.write(f"**Baseline Round:** {args.baseline_round}\n\n")

            f.write("## By Round + Condition\n\n")
            f.write("| Round | Condition | Total | OK | Errors | Inj OK | Inj Rate | 95% CI |\n")
            f.write("|-------|-----------|-------|-----|--------|--------|----------|--------|\n")

            for keys in sorted(by_round_condition):
                round_id, condition = keys
                stats = by_round_condition[keys]
                lower, upper = stats.inj_rate_ci
                ci_str = f"[{lower * 100:.1f}%, {upper * 100:.1f}%]"
                f.write(
                    f"| {round_id} | {condition} | {stats.n_total} | "
                    f"{stats.n_ok} | {stats.n_error} | {stats.n_success} | "
                    f"{stats.inj_rate * 100:.1f}% | {ci_str} |\n"
                )

            f.write("\n## Defense Ordering by Round\n\n")
            for rb in baselines:
                f.write(f"### {rb.round_id}\n\n")
                f.write("| Rank | Defense | Inj Rate | n (OK trials) |\n")
                f.write("|------|---------|----------|---------------|\n")

                sorted_defenses = []
                for name, stats in rb.defense_ordering:
                    if stats and stats.n_ok > 0:
                        sorted_defenses.append((name, stats.inj_rate, stats.n_ok))

                sorted_defenses.sort(key=lambda x: x[1], reverse=True)
                for rank, (name, rate, n_ok) in enumerate(sorted_defenses, 1):
                    f.write(f"| {rank} | {name} | {rate * 100:.1f}% | {n_ok} |\n")
                f.write("\n")

            f.write(f"## Delta vs {args.baseline_round} Baseline\n\n")
            baseline_rb = next(
                (rb for rb in baselines if rb.round_id == args.baseline_round), None
            )
            if baseline_rb:
                for rb in baselines:
                    if rb.round_id != args.baseline_round:
                        f.write(f"### {rb.round_id} vs {args.baseline_round}\n\n")
                        f.write("| Defense | Baseline | Current | Delta |\n")
                        f.write("|---------|----------|---------|-------|\n")

                        for name, stats in rb.defense_ordering:
                            baseline_stats = getattr(baseline_rb, f"{name}_stats", None)
                            if stats and baseline_stats and stats.n_ok > 0 and baseline_stats.n_ok > 0:
                                delta = stats.inj_rate - baseline_stats.inj_rate
                                f.write(
                                    f"| {name} | {baseline_stats.inj_rate * 100:.1f}% | "
                                    f"{stats.inj_rate * 100:.1f}% | {delta * 100:+.1f}% |\n"
                                )
                        f.write("\n")

        print(f"\nWrote cross-round comparison report to {output_path}")


if __name__ == "__main__":
    main()