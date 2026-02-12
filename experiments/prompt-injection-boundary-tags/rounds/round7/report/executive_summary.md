# Round 7 Executive Summary

**Date:** 2026-02-12

## Key Finding

Across 9 model families (11 model-budget targets) spanning Anthropic, OpenAI, Google, xAI, and OpenRouter, defense effectiveness ranks consistently:

`raw` (worst) > `tags_only` > `instruction_only` > `instruction_tags` > `full_stack` (best).

Phase 1 (all conditions, trials=1, N=660):

- `raw`: 18.9% injection
- `tags_only`: 9.1%
- `instruction_only`: 5.3%
- `instruction_tags`: 2.3%
- `full_stack`: 0.0%

Phase 2 (full_stack only, trials=5, N=540):

- Overall injection: 0.2% (1/540)
- Concentrated in `minimax-m2.1`: 1.7% (1/60)

## Business Implication

“Boundary tags only” is not a safe default. It helps on average but can worsen outcomes on some models. A layered configuration (`full_stack`) is the only profile that consistently drives effective injection near zero across providers.

Residual risk remains: a model can leak secrets in plain text even while refusing, without any tool calls. If outputs can reach untrusted channels, you need an output secret-scan/redaction layer.

## Recommended Default Defense

- Minimum: `instruction_tags`
- Default: `full_stack` (security instruction + nonce boundary tags + tool-call policy filtering)
- For near-zero leak tolerance: `full_stack` + output secret scanning/redaction

## Cost / Operability

- Phase 1 cost: $1.956 (hard cap enforced)
- Phase 2 cost: $1.794 (hard cap enforced)
- Mandatory live preflight probes model ids before trials; budgets enforce ceilings and write JSON spend reports.
