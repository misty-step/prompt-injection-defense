from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple


ProbeTarget = Tuple[str, str]


def unique_probe_targets(targets: Sequence[ProbeTarget]) -> List[ProbeTarget]:
    seen: set[ProbeTarget] = set()
    unique: List[ProbeTarget] = []
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        unique.append(target)
    return unique


def run_live_preflight(
    *,
    mode: str,
    targets: Sequence[ProbeTarget],
    probe_call: Callable[[str, str], Dict[str, Any]],
    budget: Any | None,
    fallback_input_tokens: int,
    fallback_output_tokens: int,
) -> List[Dict[str, Any]]:
    if mode != "live":
        return []

    unique_targets = unique_probe_targets(targets)
    if not unique_targets:
        raise SystemExit("Live preflight requires at least one probe target.")

    results: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    for target_id, model_name in unique_targets:
        if budget is not None:
            before_message = budget.before_trial_message(model_name)
            if before_message:
                settings = getattr(budget, "settings", None)
                budget_mode = str(getattr(settings, "mode", "hard") or "hard")
                if budget_mode != "warn":
                    raise SystemExit(
                        f"Preflight budget stop before {target_id} ({model_name}): {before_message}"
                    )

        try:
            probe_result = probe_call(target_id, model_name)
            input_tokens = int(probe_result.get("input_tokens", 0) or 0)
            output_tokens = int(probe_result.get("output_tokens", 0) or 0)
            if input_tokens <= 0 and fallback_input_tokens > 0:
                input_tokens = fallback_input_tokens
            if output_tokens <= 0 and fallback_output_tokens > 0:
                output_tokens = fallback_output_tokens

            trial_cost_usd = 0.0
            if budget is not None:
                trial_cost_usd = float(budget.record_trial(model_name, input_tokens, output_tokens))

            results.append(
                {
                    "target_id": target_id,
                    "model_name": model_name,
                    "model_id": str(probe_result.get("model_id", "")),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "trial_cost_usd": trial_cost_usd,
                }
            )
        except Exception as err:  # pragma: no cover - runtime-only provider failures
            failures.append((target_id, str(err)))

    if failures:
        details = "\n".join(f"- {target_id}: {error}" for target_id, error in failures)
        raise SystemExit(
            "Live preflight failed. Aborting before full trial run.\n"
            f"{details}"
        )

    return results
