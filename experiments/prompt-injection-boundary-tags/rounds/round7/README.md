# Round 7: Cross-Model Defense Validation

## Run (simulation default)
```bash
python3 experiments/prompt-injection-boundary-tags/rounds/round7/harness/run_experiment.py
```

## Run live
```bash
python3 experiments/prompt-injection-boundary-tags/rounds/round7/harness/run_experiment.py --live
```

## Budget Controls
Hard-cap a full run:
```bash
python3 experiments/prompt-injection-boundary-tags/rounds/round7/harness/run_experiment.py \
  --live \
  --max-cost-usd 5.00 \
  --budget-mode hard
```

Hard-cap per-trial guard and write budget JSON:
```bash
python3 experiments/prompt-injection-boundary-tags/rounds/round7/harness/run_experiment.py \
  --live \
  --max-cost-per-trial-usd 0.02 \
  --budget-report experiments/prompt-injection-boundary-tags/rounds/round7/data/budget_report.json
```

Expected env vars by provider:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)
- `XAI_API_KEY`
- `OPENROUTER_API_KEY`

## Analyze
```bash
python3 experiments/prompt-injection-boundary-tags/rounds/round7/analysis/analyze.py
```
