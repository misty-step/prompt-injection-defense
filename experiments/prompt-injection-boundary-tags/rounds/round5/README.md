# Round 5: Security vs Utility Tradeoff

This round measures injection resistance and benign-task utility under the same defense settings.

## Run

Simulation mode (default):

```bash
python3 harness/run_experiment.py
```

Live mode:

```bash
python3 harness/run_experiment.py --live
```

Budget controls:

```bash
python3 harness/run_experiment.py --live --max-cost-usd 5 --budget-mode hard --budget-report data/budget_report.json
```

## Analyze

```bash
python3 analysis/analyze.py
```
