# Laboratory

Misty Step's computational laboratory for reproducible software-engineering experiments.

## Philosophy

Every claim should be testable. Every run should be reproducible. Every result should be preserved with context.

**Observe -> Hypothesize -> Test -> Document -> Share**

## Repository Structure

```text
laboratory/
├── experiments/
│   └── prompt-injection-boundary-tags/
│       ├── rounds/
│       │   ├── round1/   # baseline (single-model, 72 trials)
│       │   ├── round2/   # alternate harness (432 trials)
│       │   ├── round2b/  # realistic harness + analysis (324 trials)
│       │   ├── round3/   # defense-ablation matrix
│       │   └── round4/   # single-turn vs multi-turn benchmark
│       │   ├── round5/   # security vs utility tradeoff
│       │   ├── round6/   # tool-call policy gate eval
│       │   └── round7/   # cross-model defense validation
│       └── shared/       # reusable assets (e.g., wrappers)
├── templates/            # new experiment skeletons
├── tools/                # shared utilities
└── papers/               # finalized publications
```

Each round has its own `design.md`, `harness/`, `analysis/` (when present), `data/`, and `report/`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Run canonical Round 2B workflow:

```bash
python3 experiments/prompt-injection-boundary-tags/rounds/round2b/harness/run_experiment.py
python3 experiments/prompt-injection-boundary-tags/rounds/round2b/analysis/analyze.py
python3 tools/calibrate_round2b_scorer.py
```

Normalize and analyze cross-round historical data with one schema:

```bash
python3 tools/normalize_prompt_injection_runs.py
python3 tools/analyze_prompt_injection_runs.py
```

Back-compat entrypoints remain available at repo root:

```bash
python3 run_experiment_r2.py
python3 analyze_r2.py
```

## CI Smoke Checks

GitHub Actions workflow: `.github/workflows/ci.yml`

It runs on each pull request and each push to `master`:

- `make check` (compile check for harnesses, analyzers, wrappers, tools)
- `make check-wrappers` (compat wrapper run-path sanity)
- `make test` (unit tests)
- `make smoke-analyze` (analysis smoke run on committed datasets)

Run the same gate locally:

```bash
make ci-smoke
```

To block merges on failures, set branch protection to require the `ci-smoke` workflow check.

## Experiments

| Experiment | Status | Summary |
|---|---|---|
| [prompt-injection-boundary-tags](experiments/prompt-injection-boundary-tags/) | R1-R7 implemented | Tests boundary-tagging, defense layering, multi-turn escalation risk, policy filtering, and cross-model defense validation. |

## Contributing

Open GitHub issues using the experiment or bug templates. Use labels to separate research backlog from codebase defects.

## License

MIT
