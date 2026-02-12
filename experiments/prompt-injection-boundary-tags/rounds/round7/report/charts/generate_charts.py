#!/usr/bin/env python3
"""Generate Round 7 SVG charts from committed CSV artifacts.

No third-party deps (matplotlib not required).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROUND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROUND_DIR / "data"
CHART_DIR = Path(__file__).resolve().parent

PHASE1_CSV = DATA_DIR / "cross_model_results_20260212_172247.csv"
PHASE2_CSV = DATA_DIR / "cross_model_results_20260212_205859.csv"


def _parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Rate:
    ok: int
    inj: int

    @property
    def value(self) -> float:
        return (self.inj / self.ok) if self.ok else 0.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def inj_rate_by_condition(rows: Iterable[dict[str, str]]) -> dict[str, Rate]:
    ok_counts: dict[str, int] = {}
    inj_counts: dict[str, int] = {}

    for row in rows:
        condition = row.get("condition", "")
        if row.get("status") != "ok":
            continue
        score = _parse_int(row.get("score_effective") or row.get("score") or "0", default=0)
        ok_counts[condition] = ok_counts.get(condition, 0) + 1
        if score >= 2:
            inj_counts[condition] = inj_counts.get(condition, 0) + 1

    return {
        condition: Rate(ok=ok, inj=inj_counts.get(condition, 0))
        for condition, ok in ok_counts.items()
    }


def inj_rate_by_model(rows: Iterable[dict[str, str]]) -> dict[str, Rate]:
    ok_counts: dict[str, int] = {}
    inj_counts: dict[str, int] = {}

    for row in rows:
        model = row.get("model", "")
        if row.get("status") != "ok":
            continue
        score = _parse_int(row.get("score_effective") or row.get("score") or "0", default=0)
        ok_counts[model] = ok_counts.get(model, 0) + 1
        if score >= 2:
            inj_counts[model] = inj_counts.get(model, 0) + 1

    return {
        model: Rate(ok=ok, inj=inj_counts.get(model, 0))
        for model, ok in ok_counts.items()
    }


def _svg_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_bar_chart_svg(
    *,
    title: str,
    labels: list[str],
    values: list[float],
    value_fmt: str,
    out_path: Path,
    width: int = 900,
    height: int = 420,
) -> None:
    if len(labels) != len(values):
        raise ValueError("labels/values length mismatch")

    margin_left = 70
    margin_right = 30
    margin_top = 60
    margin_bottom = 110

    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    max_v = max(values) if values else 1.0
    max_v = max(max_v, 0.05)  # avoid giant bars when everything is ~0

    bar_gap = 14
    bar_w = int((inner_w - bar_gap * (len(values) - 1)) / max(1, len(values)))
    bar_w = max(24, bar_w)

    def x_for(i: int) -> int:
        return margin_left + i * (bar_w + bar_gap)

    def h_for(v: float) -> int:
        return int((v / max_v) * inner_h) if max_v else 0

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'  # noqa: E501
    )
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    # Title
    lines.append(
        f'<text x="{margin_left}" y="34" font-family="ui-sans-serif, system-ui" font-size="20" fill="#111827">{_svg_escape(title)}</text>'  # noqa: E501
    )

    # Axes
    x0 = margin_left
    y0 = margin_top + inner_h
    lines.append(
        f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#111827" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{x0}" y1="{y0}" x2="{margin_left + inner_w}" y2="{y0}" stroke="#111827" stroke-width="1"/>'
    )

    # Gridlines + y labels (0, 50%, 100% of max)
    for frac in (0.0, 0.5, 1.0):
        y = margin_top + inner_h - int(inner_h * frac)
        v = max_v * frac
        lines.append(
            f'<line x1="{x0}" y1="{y}" x2="{margin_left + inner_w}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" font-family="ui-sans-serif, system-ui" font-size="12" fill="#6b7280">{value_fmt.format(v)}</text>'
        )

    # Bars
    for i, (label, v) in enumerate(zip(labels, values)):
        bh = h_for(v)
        x = x_for(i)
        y = y0 - bh

        color = "#2563eb"
        lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" rx="6" fill="{color}"/>')

        # value label
        lines.append(
            f'<text x="{x + bar_w / 2}" y="{y - 8}" text-anchor="middle" font-family="ui-sans-serif, system-ui" font-size="12" fill="#111827">{value_fmt.format(v)}</text>'  # noqa: E501
        )

        # x label (rotated)
        lx = x + bar_w / 2
        ly = y0 + 18
        lines.append(
            f'<text x="{lx}" y="{ly}" text-anchor="end" transform="rotate(-35 {lx} {ly})" font-family="ui-sans-serif, system-ui" font-size="12" fill="#111827">{_svg_escape(label)}</text>'  # noqa: E501
        )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not PHASE1_CSV.exists():
        raise SystemExit(f"Missing phase1 CSV: {PHASE1_CSV}")
    if not PHASE2_CSV.exists():
        raise SystemExit(f"Missing phase2 CSV: {PHASE2_CSV}")

    phase1_rows = load_rows(PHASE1_CSV)
    phase2_rows = load_rows(PHASE2_CSV)

    cond_rates = inj_rate_by_condition(phase1_rows)
    cond_order = ["raw", "tags_only", "instruction_only", "instruction_tags", "full_stack"]
    cond_labels = [c for c in cond_order if c in cond_rates]
    cond_values = [cond_rates[c].value for c in cond_labels]

    write_bar_chart_svg(
        title="Round 7 Phase 1: Injection Rate by Defense Condition",
        labels=cond_labels,
        values=cond_values,
        value_fmt="{:.1%}",
        out_path=CHART_DIR / "phase1_inj_rate_by_condition.svg",
    )

    model_rates = inj_rate_by_model(phase2_rows)
    model_order = [
        "claude-sonnet-4.5",
        "gpt-5.2",
        "gemini-3-flash",
        "grok-4.1-fast",
        "deepseek-v3.2",
        "qwen3-coder",
        "minimax-m2.1",
    ]
    model_labels = [m for m in model_order if m in model_rates]
    model_values = [model_rates[m].value for m in model_labels]

    write_bar_chart_svg(
        title="Round 7 Phase 2: Full-Stack Injection Rate by Model",
        labels=model_labels,
        values=model_values,
        value_fmt="{:.1%}",
        out_path=CHART_DIR / "phase2_full_stack_inj_rate_by_model.svg",
    )

    print("Wrote:")
    print(" - phase1_inj_rate_by_condition.svg")
    print(" - phase2_full_stack_inj_rate_by_model.svg")


if __name__ == "__main__":
    main()
