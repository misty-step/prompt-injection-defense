# Repository Guidelines

## Project Structure & Module Organization
- `experiments/` is the core. The active program is `experiments/prompt-injection-boundary-tags/`.
- Rounds are isolated in `experiments/prompt-injection-boundary-tags/rounds/round*/`.
- Each round owns `design.md`, `harness/`, `analysis/` (if needed), `data/`, and `report/`.
- `shared/` stores reusable assets across rounds (for example `shared/wrapper/untrusted-exec.sh`).
- Root scripts `run_experiment_r2.py` and `analyze_r2.py` are compatibility wrappers; prefer canonical round paths.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create and activate local env.
- `python3 -m pip install -e .`: install dependencies from `pyproject.toml`.
- `make run-r2b`: run canonical Round 2B harness.
- `make analyze-r2b`: analyze canonical Round 2B results.
- `make normalize-runs`: normalize historical round datasets into canonical schema.
- `make analyze-runs`: run cross-round analysis on canonical dataset.
- `make check`: compile-check active harnesses and wrappers.

## Coding Style & Naming Conventions
- Python 3, 4-space indentation, PEP 8 defaults.
- Use `snake_case` for functions/variables; `UPPER_SNAKE_CASE` for constants.
- Keep modules focused: scoring, payload generation, provider calls, and analysis separated.
- New rounds go under `rounds/roundN/` and keep file names explicit (`design.md`, `run_experiment.py`, `analyze.py`).

## Testing Guidelines
- No full automated suite yet; run syntax checks and at least one analysis command per change.
- For deterministic logic (scoring/classification), add `pytest` tests in `tests/test_<module>.py`.
- Target strong patch confidence (around 80%+ on new logic), not global percentage chasing.

## Commit & Pull Request Guidelines
- Use concise typed commits aligned with history: `docs:`, `experiment:`, `data:`, `refactor:`, `fix:`.
- Keep commits single-purpose (methodology, code, or data).
- PRs must include: hypothesis/goal, what changed, commands run, data output paths, and linked issue.

## Issue Taxonomy & Labels
- Use `kind:experiment` for research hypotheses and trial plans.
- Use `kind:bug` for implementation defects in harnesses, scoring, tooling, or docs.
- Add `area:*` and `priority:*` labels to keep backlog triage cheap.

## Security & Configuration Tips
- Use env vars only: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `MOONSHOT_AI_API_KEY`.
- Never commit real secrets; test fixtures must remain fake.
- `.env`, `.venv/`, `venv/`, and local cache artifacts must stay untracked.
