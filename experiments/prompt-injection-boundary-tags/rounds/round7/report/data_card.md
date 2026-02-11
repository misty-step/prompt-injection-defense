# Round 7 Data Card (Draft)

## Dataset
- File pattern: `round7/data/cross_model_results_*.csv`
- Latest pointer: `round7/data/cross_model_results_latest.csv`
- Schema: `round7_cross_model_v1`

## Key Fields
- `model`, `model_id`, `provider`
- `payload`, `condition`, `trial_num`
- `reasoning_budget`
- `score_raw`, `score_effective`, `score`, `status`, `error`
- `tool_calls_raw_json`, `tool_calls_effective_json`

## Collection Method
- Deterministic simulation by default (`--simulate`).
- Optional live provider calls with API keys (`--live`).

## Limitations
- Live availability depends on provider/model access at run time.
- Simulation risk multipliers are heuristic, not observed measurements.
