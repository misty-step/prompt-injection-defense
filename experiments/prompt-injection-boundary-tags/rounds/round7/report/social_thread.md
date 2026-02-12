# Round 7 Social Thread

1) We tested prompt-injection defenses across Anthropic, OpenAI, Google, xAI, + OpenRouter models using one shared harness: summarize a GitHub issue that contains hidden attacker instructions.

2) Phase 1 (N=660, trials=1) defense ordering held in aggregate:
raw 18.9% > tags-only 9.1% > instruction-only 5.3% > instruction+tags 2.3% > full-stack 0.0%

3) Phase 2 (full-stack only, trials=5, N=540) stayed near-zero but not perfect: 0.2% effective injection (1/540), concentrated in one model (minimax-m2.1).

4) The remaining failure mode is nasty: secrets can leak in plain text with *zero tool calls*, even inside “refusal” language.

5) Builder takeaway: don’t ship tags-only. Default to instruction + nonce tags + tool-call policy. If you need “never leak”, add output secret scanning/redaction as the last layer.
