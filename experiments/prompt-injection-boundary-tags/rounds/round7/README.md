# Round 7: Cross-Model Defense Validation

## Run (simulation default)
```bash
python3 experiments/prompt-injection-boundary-tags/rounds/round7/harness/run_experiment.py
```

## Run live
```bash
python3 experiments/prompt-injection-boundary-tags/rounds/round7/harness/run_experiment.py --live
```

Expected env vars by provider:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `XAI_API_KEY`
- `OPENROUTER_API_KEY`

## Analyze
```bash
python3 experiments/prompt-injection-boundary-tags/rounds/round7/analysis/analyze.py
```
