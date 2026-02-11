# Rounds Index

| Round | Purpose | Canonical data | Notes |
|---|---|---|---|
| `round1` | Baseline test with explicit anti-injection system instruction | `round1/data/results.csv` | 72 trials, single model (Haiku). |
| `round2` | Alternate realistic harness with different condition design | `round2/data/results-round2.csv` | 432 trials, includes GPT-5.2 API failure cases. |
| `round2b` | Canonical realistic harness and analysis | `round2b/data/results_r2_combined_latest.csv` | 324 successful trials across 3 models. |
| `round3` | Defense-ablation harness and analysis | `round3/data/ablation_results_latest.csv` | Five-condition ablation matrix (`raw`, `tags_only`, `instruction_only`, `instruction_tags`, `full_stack`). |
| `round4` | Single-turn vs multi-turn context-poisoning benchmark | `round4/data/multiturn_results_latest.csv` | 3-turn escalation protocol (`benign -> subtle -> explicit`) over top-4 payload categories. |
| `round5` | Security vs utility tradeoff benchmark | `round5/data/tradeoff_results_latest.csv` | Dual-metric frontier: injection rate vs benign-task utility. |
| `round6` | Tool-call policy gate precision/recall eval | `round6/data/policy_eval_latest.csv` | Offline labeled corpus across 4 scorer policy configs. |
| `round7` | Cross-model defense validation benchmark | `round7/data/cross_model_results_latest.csv` | 9-model defense-ablation matrix with optional reasoning-budget axis. |
| `canonical` | Schema-normalized cross-round dataset | `canonical/runs_v1.csv` | Unified `prompt_injection_run_v1` format for aggregate analysis. |

## Invariants

- Treat each `data/` directory as immutable run artifacts.
- Keep one script pair per active round (`harness/run_experiment.py`, `analysis/analyze.py`).
- Preserve old harnesses for reproducibility; create new rounds instead of mutating historical logic.
- Use `tools/normalize_prompt_injection_runs.py` to regenerate `canonical/runs_v1.csv`.

## Common Budget Controls

Rounds with live model calls expose shared budget flags:
- `--max-cost-usd`
- `--max-cost-per-trial-usd`
- `--budget-mode hard|warn`
- `--budget-report <path>`
- `--budget-estimate-input-tokens`, `--budget-estimate-output-tokens`
- `--budget-guard-input-tokens`, `--budget-guard-output-tokens`

Current coverage: `round3`, `round4`, `round5`, `round7`.
