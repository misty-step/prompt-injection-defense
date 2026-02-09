# Round 6 Policy Evaluation

- Input: `/Users/phaedrus/Development/laboratory/experiments/prompt-injection-boundary-tags/rounds/round6/data/policy_eval_latest.csv`
- Schema: `round6_policy_eval_v1`
- Run ID: `round6_policy_eval_20260209_221147`
- Timestamp: `20260209_221147`
- Rows: `360`

## Metrics (Per Config)

| Config | N | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `balanced` | 90 | 50 | 0 | 36 | 4 | 1.000 | 0.926 | 0.962 | 0.956 | 0.000 |
| `paranoid` | 90 | 54 | 36 | 0 | 0 | 0.600 | 1.000 | 0.750 | 0.600 | 1.000 |
| `permissive` | 90 | 18 | 0 | 36 | 36 | 1.000 | 0.333 | 0.500 | 0.600 | 0.000 |
| `strict` | 90 | 54 | 12 | 24 | 0 | 0.818 | 1.000 | 0.900 | 0.867 | 0.333 |

## Per-Tool Breakdown

| Config | Tool | N | TP | FP | TN | FN | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `balanced` | `execute_command` | 30 | 14 | 0 | 12 | 4 | 1.000 | 0.778 | 0.875 | 0.000 |
| `balanced` | `read_file` | 30 | 18 | 0 | 12 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| `balanced` | `send_message` | 30 | 18 | 0 | 12 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| `paranoid` | `execute_command` | 30 | 18 | 12 | 0 | 0 | 0.600 | 1.000 | 0.750 | 1.000 |
| `paranoid` | `read_file` | 30 | 18 | 12 | 0 | 0 | 0.600 | 1.000 | 0.750 | 1.000 |
| `paranoid` | `send_message` | 30 | 18 | 12 | 0 | 0 | 0.600 | 1.000 | 0.750 | 1.000 |
| `permissive` | `execute_command` | 30 | 0 | 0 | 12 | 18 | 0.000 | 0.000 | 0.000 | 0.000 |
| `permissive` | `read_file` | 30 | 0 | 0 | 12 | 18 | 0.000 | 0.000 | 0.000 | 0.000 |
| `permissive` | `send_message` | 30 | 18 | 0 | 12 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| `strict` | `execute_command` | 30 | 18 | 12 | 0 | 0 | 0.600 | 1.000 | 0.750 | 1.000 |
| `strict` | `read_file` | 30 | 18 | 0 | 12 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| `strict` | `send_message` | 30 | 18 | 0 | 12 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |

