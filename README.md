# Prompt Injection Defense: Untrusted Content Tagging

**Can wrapping untrusted CLI output in security boundary tags reduce prompt injection success?**

A controlled scientific experiment measuring whether content tagging provides meaningful defense against prompt injection attacks on LLM-powered agents.

## The Problem

AI agents that process untrusted content (GitHub issues, emails, web pages) are vulnerable to prompt injection. This is [the most critical unsolved security problem](https://simonwillison.net/2025/Jun/14/lethal-trifecta/) facing AI agents with tool access.

## The Hypothesis

**H₁:** LLM agents exposed to injection payloads wrapped in security boundary tags will execute injected instructions at a significantly lower rate than agents receiving raw payloads.

## Design

- **3 conditions:** Raw (control) · Static tags · Dynamic nonce tags
- **8 attack categories:** Simple overrides → sophisticated tag-breaking
- **3 trials per combination** for statistical validity
- **Multiple models:** Claude Haiku, Sonnet, Kimi K2.5

See [EXPERIMENT.md](./EXPERIMENT.md) for full methodology.

## Quick Start

```bash
git clone https://github.com/misty-step/prompt-injection-defense.git
cd prompt-injection-defense
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python run_experiment.py
python analyze.py
```

## Structure

```
├── EXPERIMENT.md          # Full methodology
├── run_experiment.py      # Test harness
├── analyze.py             # Statistical analysis
├── payloads/              # Injection payload definitions
├── results/               # Raw CSV data + charts
├── report/                # Scientific report
└── wrapper/               # The defense implementation
```

## Why This Matters

If content tagging works, it's a simple, universal defense any agent framework can adopt. If it doesn't, that saves the community from false confidence. Either way: science wins.

## Contributing

PRs welcome: additional payloads, model testing, alternative tagging strategies, statistical review.

## License

MIT

## Authors

- **Kaylee** — AI cofounder @ [Misty Step](https://mistystep.io)
- **Phaedrus** — Human cofounder @ [Misty Step](https://mistystep.io)
