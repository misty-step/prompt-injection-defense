# OpenCode Agent Model Evaluation

This experiment compares how different LLMs behave as **coding agents** when driven via **OpenCode**. The goal is to detect models that:

- actually edit files and commit changes, vs
- get stuck in read/think loops (analysis-paralysis) and never write code.

## What It Does

For each model:

1. Creates a temporary Git repo with a tiny Go module containing a TODO.
2. Runs:

```bash
opencode run -m MODEL --format json \
  "Add a function called Add(a, b int) int that returns a+b. Write a test for it. Commit your changes."
```

3. Records:

- duration (seconds)
- tokens used (best-effort extraction from OpenCode JSON output)
- whether any files changed (committed or left dirty)
- whether any new commits were created
- (bonus) whether `go test ./...` passes

Results are written to `results/<model_slug>.json` (model ID with unsafe characters replaced).

## Prereqs

- `opencode` installed and configured (e.g., OpenRouter credentials via env vars)
- `git`
- `go`
- `python3`

## Usage

Run a single model:

```bash
cd experiments/opencode-agent-models
./run-test.sh openrouter/moonshotai/kimi-k2.5
```

Run all models listed in `models.txt`:

```bash
cd experiments/opencode-agent-models
while IFS= read -r model; do
  [[ -z "$model" || "$model" == \#* ]] && continue
  ./run-test.sh "$model"
done < models.txt
```

Compare results:

```bash
cd experiments/opencode-agent-models
./analyze.sh
```

## Notes

- This is intentionally minimal. It measures whether a model **acts like an agent** (writes, tests, commits), not deep code quality.
- OpenCode JSON schemas may differ by version; token extraction is best-effort and stored alongside raw output for post-hoc parsing.

