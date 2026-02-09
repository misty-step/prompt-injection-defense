# Canonical Dataset

This directory stores normalized, cross-round datasets.

- `runs_v1.csv`: normalized historical runs from `round1`, `round2`, and `round2b`
- Schema contract: `docs/RUN_SCHEMA.md`

Regenerate:

```bash
python3 tools/normalize_prompt_injection_runs.py
python3 tools/analyze_prompt_injection_runs.py
```
