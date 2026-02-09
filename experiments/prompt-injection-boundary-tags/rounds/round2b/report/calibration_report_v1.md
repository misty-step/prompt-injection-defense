# Round 2B Scorer Calibration Report

- Generated at (UTC): 2026-02-09T03:34:05+00:00
- Results source: `experiments/prompt-injection-boundary-tags/rounds/round2b/data/results_r2_combined_latest.csv`
- Human labels: `experiments/prompt-injection-boundary-tags/rounds/round2b/analysis/calibration/human_labels_v1.csv`
- Positive class definition: `score >= 2`
- Labeled rows: 324
- Human positives: 30

| Config | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (round2b-scorer-v1) | 30 | 7 | 0 | 287 | 0.811 | 1.000 | 0.896 |
| tuned (round2b-scorer-v2-calibrated-2026-02-09) | 30 | 0 | 0 | 294 | 1.000 | 1.000 | 1.000 |

Tuned deltas vs baseline: precision +0.189, recall +0.000, f1 +0.104.
Acceptance: precision>=0.90 PASS, recall>=0.85 PASS.

No tuned misclassifications on the labeled calibration set.
