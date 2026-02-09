# Canonical Run Schema

This repository uses one canonical CSV schema for cross-round analysis of the prompt-injection experiment family.

- Schema id: `prompt_injection_run_v1`
- Canonical dataset path: `experiments/prompt-injection-boundary-tags/rounds/canonical/runs_v1.csv`
- Normalizer: `tools/normalize_prompt_injection_runs.py`
- Cross-round analyzer: `tools/analyze_prompt_injection_runs.py`

## Fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Canonical schema version id. |
| `experiment_id` | string | Experiment family id (`prompt-injection-boundary-tags`). |
| `round_id` | string | Source round (`round1`, `round2`, `round2b`). |
| `source_file` | string | Original CSV path used for normalization. |
| `source_row` | int | 1-based row number in source CSV. |
| `trial_id` | string | Trial id from source data. |
| `model` | string | Model name used in the trial. |
| `model_id` | string | Provider model id when available. |
| `provider` | string | Provider (`anthropic`, `openai`, `moonshot`, `unknown`). |
| `payload` | string | Payload category. |
| `condition` | string | Defense condition for the trial. |
| `trial_num` | int/string | Repeat index within payload+condition. |
| `score` | int | Injection outcome score (0-3, negative for some failed runs). |
| `status` | string | Trial execution status. |
| `error` | string | Error details when status is non-OK. |
| `num_tool_calls` | int | Number of tool calls made by model. |
| `tool_calls_json` | string | Raw serialized tool call payloads. |
| `response_length` | int/string | Response length from source run. |
| `response_preview` | string | Truncated response preview. |
| `is_success` | int | Derived flag: `1` if `status=ok` and `score>=2`, else `0`. |
| `is_error` | int | Derived flag: `1` if `status!=ok`, else `0`. |

## Usage

```bash
python3 tools/normalize_prompt_injection_runs.py
python3 tools/analyze_prompt_injection_runs.py
```

These commands work without manual row/column remapping across rounds.
