from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class BudgetSettings:
    max_cost_usd: float = 0.0
    max_cost_per_trial_usd: float = 0.0
    estimate_input_tokens: int = 1350
    estimate_output_tokens: int = 350
    guard_input_tokens: int = 1800
    guard_output_tokens: int = 520
    mode: str = "hard"
    report_path: Path | None = None


def add_budget_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.0,
        help="Hard/soft run budget cap in USD. 0 disables cap enforcement.",
    )
    parser.add_argument(
        "--max-cost-per-trial-usd",
        type=float,
        default=0.0,
        help="Hard/soft per-trial guard cap in USD. 0 disables per-trial cap.",
    )
    parser.add_argument(
        "--budget-estimate-input-tokens",
        type=int,
        default=1350,
        help="Estimated input tokens per trial used for preflight cost projection.",
    )
    parser.add_argument(
        "--budget-estimate-output-tokens",
        type=int,
        default=350,
        help="Estimated output tokens per trial used for preflight cost projection.",
    )
    parser.add_argument(
        "--budget-guard-input-tokens",
        type=int,
        default=1800,
        help="Conservative input-token guard used before each trial to avoid budget overshoot.",
    )
    parser.add_argument(
        "--budget-guard-output-tokens",
        type=int,
        default=520,
        help="Conservative output-token guard used before each trial to avoid budget overshoot.",
    )
    parser.add_argument(
        "--budget-mode",
        choices=["hard", "warn"],
        default="hard",
        help="hard: stop run when cap would be exceeded; warn: log but continue.",
    )
    parser.add_argument(
        "--budget-report",
        type=Path,
        default=None,
        help="Optional JSON output path for budget/usage summary.",
    )


def settings_from_args(args: Any) -> BudgetSettings:
    settings = BudgetSettings(
        max_cost_usd=float(getattr(args, "max_cost_usd", 0.0)),
        max_cost_per_trial_usd=float(getattr(args, "max_cost_per_trial_usd", 0.0)),
        estimate_input_tokens=int(getattr(args, "budget_estimate_input_tokens", 1350)),
        estimate_output_tokens=int(getattr(args, "budget_estimate_output_tokens", 350)),
        guard_input_tokens=int(getattr(args, "budget_guard_input_tokens", 1800)),
        guard_output_tokens=int(getattr(args, "budget_guard_output_tokens", 520)),
        mode=str(getattr(args, "budget_mode", "hard")),
        report_path=getattr(args, "budget_report", None),
    )
    validate_budget_settings(settings)
    return settings


def validate_budget_settings(settings: BudgetSettings) -> None:
    if settings.max_cost_usd < 0:
        raise SystemExit("--max-cost-usd must be >= 0")
    if settings.max_cost_per_trial_usd < 0:
        raise SystemExit("--max-cost-per-trial-usd must be >= 0")
    if settings.estimate_input_tokens <= 0 or settings.estimate_output_tokens <= 0:
        raise SystemExit("Budget estimate tokens must be > 0")
    if settings.guard_input_tokens <= 0 or settings.guard_output_tokens <= 0:
        raise SystemExit("Budget guard tokens must be > 0")
    if settings.mode not in {"hard", "warn"}:
        raise SystemExit("Budget mode must be one of: hard, warn")


@dataclass
class BudgetController:
    settings: BudgetSettings
    estimate_cost_usd: Callable[[str, int, int], float]
    run_id: str
    timestamp: str
    mode: str
    planned_trial_models: Sequence[str]
    default_report_path: Path
    spent_cost_usd: float = 0.0
    per_model_spend_usd: Dict[str, float] = field(default_factory=dict)
    observed_max_tokens: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_names = sorted(set(self.planned_trial_models))
        self.per_model_spend_usd = {model_name: 0.0 for model_name in model_names}
        self.observed_max_tokens = {model_name: (0, 0) for model_name in model_names}

    @property
    def planned_trials(self) -> int:
        return len(self.planned_trial_models)

    @property
    def projected_total_cost_usd(self) -> float:
        return sum(
            self.estimate_cost_usd(
                model_name,
                self.settings.estimate_input_tokens,
                self.settings.estimate_output_tokens,
            )
            for model_name in self.planned_trial_models
        )

    def preflight_message(self) -> str | None:
        if self.settings.max_cost_usd <= 0:
            return None
        projected = self.projected_total_cost_usd
        if projected <= self.settings.max_cost_usd:
            return None
        return (
            f"Projected run cost ${projected:.2f} exceeds --max-cost-usd "
            f"${self.settings.max_cost_usd:.2f}."
        )

    def guard_cost_for_model(self, model_name: str) -> float:
        observed_input, observed_output = self.observed_max_tokens.get(model_name, (0, 0))
        guard_input = max(self.settings.guard_input_tokens, observed_input)
        guard_output = max(self.settings.guard_output_tokens, observed_output)
        return self.estimate_cost_usd(model_name, guard_input, guard_output)

    def before_trial_message(self, model_name: str) -> str | None:
        next_guard_cost = self.guard_cost_for_model(model_name)

        if (
            self.settings.max_cost_per_trial_usd > 0
            and next_guard_cost > self.settings.max_cost_per_trial_usd
        ):
            return (
                f"Per-trial guard ${next_guard_cost:.4f} for {model_name} exceeds "
                f"--max-cost-per-trial-usd ${self.settings.max_cost_per_trial_usd:.4f}."
            )

        if (
            self.settings.max_cost_usd > 0
            and self.spent_cost_usd + next_guard_cost > self.settings.max_cost_usd
        ):
            return (
                f"Next trial guard would exceed run cap: spent=${self.spent_cost_usd:.2f}, "
                f"next_guard=${next_guard_cost:.4f}, cap=${self.settings.max_cost_usd:.2f}."
            )
        return None

    def budget_progress_text(self) -> str:
        if self.settings.max_cost_usd > 0:
            return f" budget=${self.spent_cost_usd:.2f}/${self.settings.max_cost_usd:.2f}"
        return f" budget=${self.spent_cost_usd:.2f}"

    def record_trial(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        trial_cost_usd = self.estimate_cost_usd(model_name, input_tokens, output_tokens)
        self.spent_cost_usd += trial_cost_usd
        self.per_model_spend_usd[model_name] = (
            self.per_model_spend_usd.get(model_name, 0.0) + trial_cost_usd
        )
        observed_input, observed_output = self.observed_max_tokens.get(model_name, (0, 0))
        self.observed_max_tokens[model_name] = (
            max(observed_input, input_tokens),
            max(observed_output, output_tokens),
        )
        return trial_cost_usd

    def needs_reporting(self) -> bool:
        return (
            self.settings.report_path is not None
            or self.settings.max_cost_usd > 0
            or self.settings.max_cost_per_trial_usd > 0
        )

    def report_dict(self, completed_trials: int, stop_reason: str = "") -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "planned_trials": self.planned_trials,
            "completed_trials": int(completed_trials),
            "remaining_trials": int(max(0, self.planned_trials - completed_trials)),
            "projected_total_cost_usd": round(self.projected_total_cost_usd, 8),
            "spent_cost_usd": round(self.spent_cost_usd, 8),
            "max_cost_usd": float(self.settings.max_cost_usd),
            "max_cost_per_trial_usd": float(self.settings.max_cost_per_trial_usd),
            "budget_mode": self.settings.mode,
            "budget_stop_reason": stop_reason,
            "estimate_input_tokens": int(self.settings.estimate_input_tokens),
            "estimate_output_tokens": int(self.settings.estimate_output_tokens),
            "guard_input_tokens": int(self.settings.guard_input_tokens),
            "guard_output_tokens": int(self.settings.guard_output_tokens),
            "per_model_spend_usd": {
                model_name: round(cost, 8)
                for model_name, cost in sorted(self.per_model_spend_usd.items())
            },
        }

    def write_report(self, completed_trials: int, stop_reason: str = "") -> Path | None:
        if not self.needs_reporting():
            return None
        path = self.settings.report_path or self.default_report_path
        output_path = path if path.is_absolute() else (Path.cwd() / path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.report_dict(completed_trials, stop_reason), indent=2) + "\n")
        return output_path

