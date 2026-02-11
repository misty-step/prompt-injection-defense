# Round 4: Multi-turn Context Poisoning Benchmark

This round implements issue #3: compare injection resistance in single-turn vs multi-turn conversations.

## Protocol

- `single_turn`: one request containing explicit exploit payload.
- `multi_turn`: three-turn escalation:
  1. benign issue-summary request
  2. subtle context escalation
  3. explicit exploit payload

## Defenses

- `raw`
- `tags_only`
- `instruction_only`
- `instruction_tags`
- `full_stack`

## Payload Set

Default payloads are top historical performers from prior rounds:

- `gradual_escalation`
- `helpful_framing`
- `indirect_extraction`
- `multi_step`

## Run

Simulation mode (default):

```bash
python3 harness/run_experiment.py
```

Live mode (provider APIs):

```bash
python3 harness/run_experiment.py --live
```

Common flags:

```bash
python3 harness/run_experiment.py \
  --trials 3 \
  --attack-modes single_turn,multi_turn \
  --payloads gradual_escalation,helpful_framing,indirect_extraction,multi_step \
  --models claude-sonnet-4,claude-haiku-3.5,gpt-4o,kimi-k2.5
```

Budget controls:

```bash
python3 harness/run_experiment.py --live --max-cost-usd 5 --budget-mode hard --budget-report data/budget_report.json
```

## Analyze

```bash
python3 analysis/analyze.py
```

Outputs land in `data/`:

- `multiturn_results_<timestamp>.csv`
- `multiturn_results_latest.csv`
