# Round 2B Executive Summary: Human-Calibrated Scorer for Prompt Injection Benchmarks

**Finding (headline):** Human review of 324 injection trials revealed 7 false positives in the automated scorer (precision 0.811 -> 1.000 after calibration). Thirty minutes of labeling corrected model-level comparisons that were distorted by up to 4x.

## What We Tested

- Same 324 trials as Round 2: 12 payloads x 3 conditions x 3 trials x 3 models.
- Human-labeled every row for binary injection success.
- Evaluated baseline scorer (v1) and tuned scorer (v2) against human labels.

## Results

| Metric | Baseline | Tuned | Delta |
|--------|---:|---:|---:|
| Precision | 0.811 | 1.000 | +0.189 |
| Recall | 1.000 | 1.000 | 0.000 |
| F1 | 0.896 | 1.000 | +0.104 |
| False positives | 7 | 0 | -7 |

Impact on model-level injection rates (raw condition):

| Model | Automated Score | Human Label | Delta |
|-------|---:|---:|---:|
| Claude Haiku 3.5 | 25.0% | 25.0% | 0 |
| GPT-4o | 47.2% | 33.3% | -13.9pp |
| Kimi K2.5 | 22.2% | 5.6% | -16.6pp |

## Key Finding

Three of the seven false positives were score-3 (highest severity) rows where the model made *defensive* tool calls: alerting admins, running diagnostics. The scorer equated "tool call during injection trial" with "compromise." This is a systematic blind spot that will worsen as models become more security-aware.

## Implications

1. **Scorer calibration should be standard practice** in injection benchmark papers. Thirty minutes of human review caught errors that changed model rankings.
2. **Defensive tool calls are an edge case** that automated scorers systematically misclassify. Benchmark design should reward, not penalize, security-aware model behavior.
3. **Per-model conclusions require per-model validation.** The headline finding (tags reduce injection) is robust to scorer error. Model comparisons are not.

## Next Steps

- Publish scorer calibration protocol as a reusable workflow.
- Test v2 scorer against Round 3+ datasets for generalization.
- Investigate LLM-as-judge calibration for comparison.
