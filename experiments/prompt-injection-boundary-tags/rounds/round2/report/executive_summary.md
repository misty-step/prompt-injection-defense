# Round 2 Executive Summary: Prompt Injection Boundary Tags

**Finding (headline):** Boundary tags reduced injection success from **31%** (raw) to **10%** (static tags) and **9%** (dynamic nonce tags) in a realistic tool-using agent (χ²=24.22, p=0.000006).

## What We Tested
- A realistic agent that summarizes GitHub issues and can call tools.
- 3 models: Claude Haiku 3.5, GPT-4o, Kimi K2.5.
- 12 prompt-injection payload categories.
- 3 conditions:
  - `raw` (no tags)
  - `static_tags` (fixed untrusted-content boundary tags)
  - `dynamic_nonce` (randomized nonce boundary tags)
- 324 total trials (12 payloads × 3 conditions × 3 trials × 3 models).

## Results (Key Metrics)

| Condition | Injection rate | Tool calls (count) |
|---|---:|---:|
| raw | 31.5% | 32 |
| static_tags | 10.2% | 5 |
| dynamic_nonce | 9.3% | 7 |

Per-model vulnerability (raw condition):
- GPT-4o: **47%** (highest)
- Haiku 3.5: 25%
- Kimi K2.5: 22%

Most dangerous payload:
- `gradual_escalation`: **56%** injection overall; **100%** against GPT-4o across all conditions.

## Recommendation
Ship boundary tags, but treat them as one layer:
- wrap all untrusted content in explicit boundary tags
- deny-by-default tool policy (allowlists + argument validation)
- high-friction confirmation for dangerous tools (`execute_command`, outbound messaging)
- tool-call logging + anomaly alerts

## Limitation (Read Before Over-Generalizing)
- Small repetitions (3 per cell); tail risk may be under-sampled.
- Single task and tool set; real deployments vary.
- Tags reduce risk, not eliminate it; worst-case payloads still win on some models.

