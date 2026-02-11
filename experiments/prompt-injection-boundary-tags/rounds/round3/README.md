# Round 3: Defense Ablation Matrix

This round implements the defense-ablation experiment tracked in issue #1.

## Conditions

- `raw`
- `tags_only`
- `instruction_only`
- `instruction_tags`
- `full_stack`

## Run

Simulation mode (default, safe for local validation):

```bash
python3 harness/run_experiment.py
```

Live mode (uses provider APIs):

```bash
python3 harness/run_experiment.py --live
```

Useful flags:

```bash
python3 harness/run_experiment.py --trials 5 --payload-limit 12 --models "claude-sonnet-4,claude-haiku-3.5,gpt-4o,kimi-k2.5"
```

Budget controls:

```bash
python3 harness/run_experiment.py --live --max-cost-usd 5 --budget-mode hard --budget-report data/budget_report.json
```

## Analyze

```bash
python3 analysis/analyze.py
```

Outputs are written to `data/` (`ablation_results_<timestamp>.csv` and `ablation_results_latest.csv`).
