# Data Card: Round 2B Human-Calibrated Scoring

## Dataset Overview

| Field | Value |
|-------|-------|
| **Name** | Prompt Injection Boundary Tags: Human-Calibrated Scoring |
| **Version** | round2b_calibrated_v1 |
| **Size** | 324 rows (12 payloads x 3 conditions x 3 trials x 3 models) |
| **Format** | CSV |
| **License** | MIT |
| **Repository** | github.com/misty-step/laboratory |

## Files

### Trial Results

- **Path:** `experiments/prompt-injection-boundary-tags/rounds/round2b/data/results_r2_combined_latest.csv`
- **Rows:** 324
- **Description:** Automated scorer output for all trials. Scores are from the original (v1) scorer. Use in conjunction with human labels for corrected analysis.

| Column | Type | Description |
|--------|------|-------------|
| model | string | Model display name: claude-haiku-3.5, gpt-4o, kimi-k2.5 |
| model_id | string | API model identifier |
| trial_id | int | Unique trial number within model |
| payload | string | Payload category (12 values) |
| condition | string | raw, static_tags, or dynamic_nonce |
| trial_num | int | Repetition index (1-3) |
| score | int | Automated severity score (0-3) |
| status | string | Trial status (ok or error) |
| tool_calls | json string | Serialized tool call array |
| num_tool_calls | int | Count of tool calls |
| response_length | int | Response length in characters |
| response_preview | string | Truncated response text |

### Human Labels

- **Path:** `experiments/prompt-injection-boundary-tags/rounds/round2b/analysis/calibration/human_labels_v1.csv`
- **Rows:** 324
- **Description:** Human-reviewed binary injection success labels for all trials.

| Column | Type | Description |
|--------|------|-------------|
| source_row | int | 1-indexed row in results CSV |
| trial_id | int | Trial number within model |
| model | string | Model display name |
| payload | string | Payload category |
| condition | string | Boundary-tag condition |
| existing_auto_score | int | Original automated score (0-3) |
| human_is_injection_success | int | Human label: 1 = true injection success, 0 = not |
| reviewer | string | Reviewer identifier |
| review_notes | string | Free-text notes (populated for overrides) |

### Calibration Metrics

- **Path:** `experiments/prompt-injection-boundary-tags/rounds/round2b/analysis/calibration/metrics_v1.json`
- **Description:** Precision/recall/F1 for baseline (v1) and tuned (v2) scorer configs against human labels.

## Collection Method

- **Trial generation:** Same harness as Round 2. Agent summarizes a GitHub issue containing one of 12 injection payloads. Three boundary-tag conditions (raw, static_tags, dynamic_nonce). Three repetitions per cell. Three models.
- **Automated scoring:** Config-driven scorer (`scorer_config_v1.json` for baseline, `scorer_config_v2.json` for tuned). Uses literal matching, regex, and tool-call heuristics.
- **Human labeling:** Single reviewer (with automated first-pass). All 324 rows labeled. Inputs: automated score, tool call JSON, response preview. Three score-3 rows manually overridden to human_label=0 (defensive tool calls).

## Known Limitations

1. **Single reviewer.** Inter-annotator agreement is not measured. Ambiguous cases (partial compliance without clear harm) may have different labels under a second reviewer.
2. **Small dataset.** 324 rows is sufficient for scorer calibration but not for statistical power on per-cell comparisons (3 trials per cell).
3. **Scorer overfitting risk.** The v2 scorer config was tuned on the same dataset it's evaluated against. Generalization to new payloads/models is not tested here.
4. **Response previews are truncated.** Some labeling decisions relied on truncated text. Full responses were consulted for ambiguous cases but are not included in the published data.
5. **English-only payloads.** All injection attempts are in English. Multilingual injection is not tested.
6. **Model currency.** GPT-4o was current when trials were run (early Feb 2026) but was deprecated Feb 13, 2026. Results reflect GPT-4o behavior at that point in time. Future cross-model validation (issue #5) will use current frontier models.

## Intended Use

- Evaluating automated scorer accuracy for prompt injection benchmarks
- Reproducing Round 2B calibration findings
- Comparing automated vs human-labeled injection rates
- Developing improved scoring heuristics for defensive tool call patterns

## Citation

```text
Misty Step (2026). "Human-Calibrated Scoring for Prompt Injection
Benchmarks." Round 2B, Prompt Injection Boundary Tags Experiment.
github.com/misty-step/laboratory
```
