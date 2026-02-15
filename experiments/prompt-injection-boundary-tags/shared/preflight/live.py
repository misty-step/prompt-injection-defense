from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Sequence, Tuple


ProbeTarget = Tuple[str, str]


class DryRunValidationError(Exception):
    """Raised when dry-run validation fails."""
    pass


def validate_dry_run_result(
    target_id: str,
    result: Dict[str, Any],
    check_tool_parsing: bool = True,
) -> List[str]:
    """
    Validate a dry-run trial result.
    
    Returns a list of validation error messages (empty if valid).
    
    Checks:
    - Request accepted (no API errors)
    - Response parsing works
    - Tool call JSON parsing works (if check_tool_parsing=True)
    """
    errors: List[str] = []
    
    # Check request was accepted (model_id present)
    model_id = result.get("model_id")
    if not model_id:
        errors.append(f"{target_id}: No model_id in response (request may have been rejected)")
    
    # Check response parsing
    response_text = result.get("response_text")
    if response_text is None:
        errors.append(f"{target_id}: No response_text in result")
    
    # Check token counts are reasonable
    # Note: 0 tokens is allowed here because the caller may apply fallbacks after validation
    input_tokens = int(result.get("input_tokens", 0) or 0)
    output_tokens = int(result.get("output_tokens", 0) or 0)
    if input_tokens < 0:
        errors.append(f"{target_id}: Invalid input_tokens ({input_tokens})")
    if output_tokens < 0:
        errors.append(f"{target_id}: Invalid output_tokens ({output_tokens})")
    
    # Check tool call JSON parsing
    if check_tool_parsing:
        raw_tool_calls = result.get("raw_tool_calls", [])
        if not isinstance(raw_tool_calls, list):
            errors.append(f"{target_id}: raw_tool_calls is not a list")
        else:
            for tc in raw_tool_calls:
                if not isinstance(tc, dict):
                    errors.append(f"{target_id}: tool call is not a dict: {type(tc)}")
                    continue
                # Verify we can access the arguments (JSON parsing worked)
                args = tc.get("arguments")
                if args is None:
                    errors.append(f"{target_id}: tool call missing arguments")
                elif not isinstance(args, dict):
                    # Arguments should be a dict (parsed JSON), not raw string
                    errors.append(f"{target_id}: tool call arguments not parsed (got {type(args).__name__})")
    
    return errors


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


def run_live_dry_run(
    *,
    mode: str,
    targets: Sequence[ProbeTarget],
    trial_call: Callable[[str, str], Dict[str, Any]],
    budget: Any | None,
    fallback_input_tokens: int,
    fallback_output_tokens: int,
    check_tool_parsing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run end-to-end dry-run before full trial plan.
    
    This runs 1 real trial per model-budget target using actual harness
    prompts and tool schemas to validate:
    - Request acceptance (model accepts the request)
    - Response parsing works
    - Tool call JSON parsing works
    - Reasoning knobs work or cleanly fallback
    
    Args:
        mode: Run mode ("live" or "simulate")
        targets: Sequence of (target_id, model_name) tuples to test
        trial_call: Function that runs the actual trial, signature:
            (target_id: str, model_name: str) -> Dict with keys:
            - model_id: str
            - response_text: str
            - raw_tool_calls: List[Dict]
            - input_tokens: int
            - output_tokens: int
        budget: Budget controller to record costs (optional)
        fallback_input_tokens: Fallback for zero input_tokens
        fallback_output_tokens: Fallback for zero output_tokens
        check_tool_parsing: Whether to validate tool call JSON parsing
    
    Returns:
        List of dry-run results with cost information
        
    Raises:
        SystemExit: If any target fails the dry-run (fail-fast)
    """
    if mode != "live":
        return []

    unique_targets = unique_probe_targets(targets)
    if not unique_targets:
        raise SystemExit("Live dry-run requires at least one trial target.")

    results: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    for target_id, model_name in unique_targets:
        # Check budget before running dry-run trial
        if budget is not None:
            before_message = budget.before_trial_message(model_name)
            if before_message:
                settings = getattr(budget, "settings", None)
                budget_mode = str(getattr(settings, "mode", "hard") or "hard")
                if budget_mode != "warn":
                    raise SystemExit(
                        f"Dry-run budget stop before {target_id} ({model_name}): {before_message}"
                    )

        try:
            # Run the actual trial (not just a probe)
            trial_result = trial_call(target_id, model_name)
            
            # Validate the result
            validation_errors = validate_dry_run_result(
                target_id, trial_result, check_tool_parsing=check_tool_parsing
            )
            if validation_errors:
                failures.append((target_id, "; ".join(validation_errors)))
                continue
            
            # Get token counts with fallbacks
            input_tokens = int(trial_result.get("input_tokens", 0) or 0)
            output_tokens = int(trial_result.get("output_tokens", 0) or 0)
            if input_tokens <= 0 and fallback_input_tokens > 0:
                input_tokens = fallback_input_tokens
            if output_tokens <= 0 and fallback_output_tokens > 0:
                output_tokens = fallback_output_tokens

            # Record cost in budget
            trial_cost_usd = 0.0
            if budget is not None:
                trial_cost_usd = float(budget.record_trial(model_name, input_tokens, output_tokens))

            results.append(
                {
                    "target_id": target_id,
                    "model_name": model_name,
                    "model_id": str(trial_result.get("model_id", "")),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "trial_cost_usd": trial_cost_usd,
                    "raw_tool_calls": trial_result.get("raw_tool_calls", []),
                    "response_text": str(trial_result.get("response_text", "")),
                }
            )
        except Exception as err:  # pragma: no cover - runtime-only provider failures
            failures.append((target_id, str(err)))

    if failures:
        details = "\n".join(f"- {target_id}: {error}" for target_id, error in failures)
        raise SystemExit(
            "Live dry-run failed. Aborting before full trial run.\n"
            f"{details}"
        )

    return results
