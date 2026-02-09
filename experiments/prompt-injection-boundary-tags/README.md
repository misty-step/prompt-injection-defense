# Prompt Injection Boundary Tags

This experiment family measures how boundary-tagging and related defenses change prompt-injection success rates in agent-like workflows.

## Canonical Layout

```text
rounds/
  round1/   # baseline harness + paper
  round2/   # alternate realistic harness variant
  round2b/  # canonical realistic harness + analysis
  round3/   # design for next iteration
shared/
  wrapper/  # untrusted CLI output wrapper
```

## Canonical Commands

```bash
# Round 1
python3 rounds/round1/harness/run_experiment.py
python3 rounds/round1/analysis/analyze.py

# Round 2
python3 rounds/round2/harness/run_experiment.py

# Round 2B (recommended baseline)
python3 rounds/round2b/harness/run_experiment.py
python3 rounds/round2b/analysis/analyze.py

# Cross-round canonical analysis
python3 ../../tools/normalize_prompt_injection_runs.py
python3 ../../tools/analyze_prompt_injection_runs.py
```

## Data Policy

- Treat files in each round's `data/` directory as immutable run artifacts.
- Add new run outputs with timestamped names.
- Keep one `*_latest.csv` per active round for convenience, but never replace historical files.

## Backward Compatibility

Legacy entrypoints remain in this folder (`run_experiment.py`, `run_experiment_round2.py`, `run_experiment_r2.py`, `analyze.py`, `analyze_r2.py`) and forward to canonical round scripts.
