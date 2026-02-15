"""Output filtering for secret detection and redaction.

This module provides a defense layer that scans model output text
for potential secret leaks and redacts them before the response
is returned to the user or logged.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


def redact_secrets(text: str, config: Mapping[str, Any]) -> tuple[str, bool]:
    """Scan text for secrets and redact them.
    
    Args:
        text: The output text to scan.
        config: Scorer configuration with secret_literals and secret_regexes.
        
    Returns:
        Tuple of (redacted_text, was_modified). If no secrets found,
        returns (original_text, False). If secrets found, returns
        (redacted_text, True) with secrets replaced with [REDACTED].
    """
    if not text:
        return text, False
        
    redacted = text
    was_modified = False
    
    # Redact literal secrets
    for literal in config.get("secret_literals", []):
        literal_str = str(literal)
        if literal_str in redacted:
            redacted = redacted.replace(literal_str, "[REDACTED]")
            was_modified = True
            
    # Redact regex patterns
    for pattern in config.get("secret_regexes", []):
        pattern_str = str(pattern)
        try:
            regex = re.compile(pattern_str, re.IGNORECASE)
            if regex.search(redacted):
                redacted = regex.sub("[REDACTED]", redacted)
                was_modified = True
        except re.error:
            # Skip invalid regex patterns
            continue
            
    return redacted, was_modified


def contains_secret_leak(text: str, config: Mapping[str, Any]) -> bool:
    """Check if text contains potential secret leaks.
    
    Args:
        text: The output text to check.
        config: Scorer configuration with secret patterns.
        
    Returns:
        True if secrets are detected, False otherwise.
    """
    if not text:
        return False
        
    text_lower = text.lower()
    
    # Check literal secrets
    for literal in config.get("secret_literals", []):
        if str(literal).lower() in text_lower:
            return True
            
    # Check regex patterns
    for pattern in config.get("secret_regexes", []):
        try:
            if re.search(str(pattern), text_lower):
                return True
        except re.error:
            continue
            
    return False


def apply_output_filter(
    response_text: str,
    tool_calls: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Apply output filtering to response text and tool calls.
    
    This is the main entry point for the full_stack_plus defense condition.
    It scans both the response text and tool call arguments for secrets
    and redacts them.
    
    Args:
        response_text: The model's response text.
        tool_calls: List of tool calls from the model.
        config: Scorer configuration.
        
    Returns:
        Tuple of (filtered_text, filtered_tool_calls, filter_metadata).
        filter_metadata contains details about what was filtered.
    """
    metadata: dict[str, Any] = {
        "text_redacted": False,
        "tool_calls_redacted": [],
        "secrets_detected": False,
    }
    
    # Filter response text
    filtered_text, text_redacted = redact_secrets(response_text, config)
    if text_redacted:
        metadata["text_redacted"] = True
        metadata["secrets_detected"] = True
        
    # Filter tool call arguments
    filtered_tool_calls: list[dict[str, Any]] = []
    for i, call in enumerate(tool_calls):
        filtered_call = dict(call)
        arguments = call.get("arguments", {})
        
        if isinstance(arguments, dict):
            filtered_args: dict[str, Any] = {}
            for key, value in arguments.items():
                if isinstance(value, str):
                    filtered_value, was_redacted = redact_secrets(value, config)
                    filtered_args[key] = filtered_value
                    if was_redacted:
                        metadata["secrets_detected"] = True
                        metadata["tool_calls_redacted"].append({
                            "index": i,
                            "tool": call.get("name", "unknown"),
                            "argument": key,
                        })
                else:
                    filtered_args[key] = value
            filtered_call["arguments"] = filtered_args
            
        filtered_tool_calls.append(filtered_call)
        
    return filtered_text, filtered_tool_calls, metadata
