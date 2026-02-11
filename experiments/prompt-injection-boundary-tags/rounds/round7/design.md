# Round 7 Design: Cross-Model Defense Validation

## Goal
Validate whether defense-layer findings from rounds 3-5 hold across current frontier models and a broader open-weight set.

## Hypothesis
Across modern models, `instruction_only` and `instruction_tags` remain more effective than `tags_only`, and `full_stack` remains the lowest-risk configuration.

## Novelty Statement
This round produces new information because it isolates defense-layer effectiveness (`raw`, `tags_only`, `instruction_only`, `instruction_tags`, `full_stack`) across multiple providers in one controlled matrix. Existing benchmarks commonly compare raw vulnerability, but do not isolate marginal defense contribution across this provider spread.

## Literature Context (for gap framing)
- AgentDojo: broad agent security benchmark but delimiter strategy not isolated as the primary experimental axis.
- InjecAgent: model vulnerability focus, limited defense-ablation detail across providers.
- Provider safety docs/system cards: provider-specific guidance, not a cross-provider ablation benchmark using a shared harness.

## Experiment Matrix
- Conditions: `raw`, `tags_only`, `instruction_only`, `instruction_tags`, `full_stack`
- Payloads: frozen 12-category pack from `round2b/payloads/payloads.json`
- Models:
  - `claude-sonnet-4.5` (Anthropic)
  - `gpt-5.2` (OpenAI)
  - `gemini-3-flash` (Google Gemini API)
  - `grok-4.1-fast` (xAI API)
  - `deepseek-v3.2` (OpenRouter)
  - `kimi-k2-thinking` (OpenRouter)
  - `qwen3-coder` (OpenRouter)
  - `minimax-m2.1` (OpenRouter)
  - `glm-4.7` (OpenRouter)
- Trials: default `3` per cell in harness.

Base matrix (without reasoning axis):
`9 models x 5 conditions x 12 payloads x 3 trials = 1,620 trials`

## Reasoning Budget Axis
- Eligible models: `gpt-5.2`, `gemini-3-flash`
- Budgets: `low`, `high` (CLI-configurable)
- Non-eligible models run with `none`
- Purpose: estimate whether higher reasoning budget changes injection resilience.

## Harness Requirements
- Default `--simulate` mode for deterministic, no-key runs.
- `--live` mode with provider adapters:
  - Anthropic SDK
  - OpenAI SDK
  - Google Gemini REST
  - xAI REST
  - OpenRouter via OpenAI SDK
- CSV output includes `reasoning_budget`.
- CSV keeps both `score_raw` (before policy filtering) and `score_effective` (after filtering).

## Analysis Requirements
- Summaries by condition, model, and model+condition.
- Dedicated reasoning-budget comparison table for eligible models.
- Raw-baseline deltas by model and reasoning budget.

## Success Criteria
- Harness can execute full matrix in simulation mode with deterministic output.
- Live mode supports all provider routes when keys are present.
- Analysis can compare defense ranking consistency with prior rounds.
