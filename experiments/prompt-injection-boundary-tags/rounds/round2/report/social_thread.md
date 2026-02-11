Post 1
Prompt injection in tool-using agents is not theoretical. We ran 324 trials across 3 models. Wrapping untrusted content in boundary tags cut injection success from 31% to 10% (p=0.000006).

Post 2
Conditions:
raw (no tags): 31.5% injection
static tags: 10.2%
dynamic nonce tags: 9.3%
Also: malicious tool calls dropped from 32 (raw) to 5 (static) / 7 (dynamic).

Post 3
Model spread was big:
GPT-4o: 47% raw injection (most vulnerable)
Claude Haiku 3.5: 25% raw, 3% with nonce tags (most resistant)
Kimi K2.5: 22% raw
Same agent prompt, same tools.

Post 4
Deadliest payload: gradual_escalation.
56% injection overall.
Against GPT-4o it hit 100% across raw/static/dynamic. Tags help, but “task-shaped” escalation can still win.

Post 5
Recommendation:
1) boundary-tag every untrusted input
2) deny-by-default tool policy (allowlist + arg validation)
3) confirm high-impact tools (exec, outbound comms)
Round 2 report + charts are in the repo: experiments/prompt-injection-boundary-tags/rounds/round2/report/

