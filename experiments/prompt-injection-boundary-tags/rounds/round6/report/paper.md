# Post-Hoc Tool-Call Policy Filtering as a Last-Mile Defense Against Prompt Injection

**Authors:** Kaylee & Phaedrus, Misty Step
**Date:** 2026-02-10
**Repository:** github.com/misty-step/laboratory

---

## Abstract

LLM agents that process untrusted content can be manipulated into making malicious tool calls — reading sensitive files, executing exfiltration commands, or sending data to attacker-controlled recipients. We evaluate a post-hoc defense: a rule-based policy gate that inspects tool calls after the model generates them but before execution. We test 4 policy configurations against a labeled corpus of 90 unique tool calls (54 malicious, 36 benign) across 3 tool types, producing 360 total evaluations (90 calls x 4 configs). The calibrated `balanced` configuration achieves **92.6% recall on malicious calls with 0.0% false positive rate**, confirming that lightweight rule-based filtering can serve as an effective last-mile defense layer. We contrast this architectural approach with model-level self-refusal (SafeToolBench, EMNLP 2025) and discuss how post-hoc filtering complements prompt-level defenses like boundary tags and system instructions.

## 1. Introduction

### 1.1 The Problem

When LLM agents process untrusted input — a GitHub issue, an email, a web page — prompt injection attacks can manipulate the model into generating malicious tool calls. Unlike conversational jailbreaks, tool-calling injection produces executable actions: file reads that expose secrets, commands that exfiltrate data, messages that send credentials to attackers.

Existing defenses operate at the prompt level: boundary tags to demarcate untrusted content (Anthropic, 2025), system instructions to constrain behavior (OpenAI Instruction Hierarchy, 2024), and fine-tuning for injection resistance. These are valuable but imperfect — our earlier rounds showed that even with layered prompt defenses, some injection attempts succeed (Round 2: 10% residual injection rate with dynamic nonce tags).

### 1.2 The Defense Under Test

We propose a complementary approach: **post-hoc tool-call policy filtering**. Rather than preventing the model from generating malicious tool calls, we inspect calls after generation and block suspicious ones before execution. This is architecturally analogous to a firewall or output filter — a defense-in-depth layer that operates independently of the model's internal reasoning.

The policy gate uses configurable rule-based matching:
- **Path matching** for `read_file` (sensitive paths like `.env`, `/etc/passwd`, SSH keys)
- **Command token matching** for `execute_command` (sensitive operations like `printenv`, `curl`, `cat /etc`)
- **Recipient validation** for `send_message` (trusted recipient allowlist + secret detection in message bodies)

### 1.3 Prior Art

- **SafeToolBench** (Li et al., EMNLP 2025): Tests whether LLMs prospectively refuse unsafe tool calls. Achieves 83% recall with SafeInstructTool fine-tuning. Our approach is complementary — we evaluate an external filter rather than relying on model self-refusal.
- **AgentDojo** (Debenedetti et al., NeurIPS 2024): Tests tool filtering as one defense alongside delimiters and prompt sandwiching. Reports that tool filtering suppresses attack success to 7.5% but reduces utility to 53.3%. Our evaluation measures precision/recall directly rather than end-to-end task success.
- **InjecAgent** (Zhan et al., ACL 2024): Establishes the tool-calling indirect injection threat model with 1,054 test cases. Does not evaluate post-hoc defenses.
- **Agent Security Bench** (Zhang et al., ICLR 2025): Comprehensive evaluation of 27 attack/defense methods across 400+ tools. Reports limited defense effectiveness overall.

### 1.4 Contribution

We provide the first precision/recall evaluation of a configurable rule-based tool-call policy gate, tested across 4 configuration tightness levels and 3 tool types. We identify `balanced` (calibrated token matching) as the recommended operating point and characterize the precision-recall tradeoff across configurations.

## 2. Methodology

### 2.1 Labeled Corpus

We constructed a corpus of 90 unique tool calls (54 malicious, 36 benign) across 3 tool types:

| Tool | Malicious | Benign | Total |
|------|-----------|--------|-------|
| read_file | 18 | 12 | 30 |
| execute_command | 18 | 12 | 30 |
| send_message | 18 | 12 | 30 |

**Malicious examples** were crafted from real-world attack patterns:
- File reads targeting `.env`, `/etc/passwd`, SSH keys, AWS credentials, secrets YAML files
- Commands exfiltrating environment variables, curling attacker endpoints, reading sensitive system files
- Messages sending secret literals to non-trusted recipients

**Benign examples** were modeled on realistic developer workflows:
- File reads of source code, documentation, configuration
- Commands running tests, checking git status, building projects
- Messages to the trusted admin address reporting build results

### 2.2 Policy Configurations

Each call was evaluated against 4 configurations of `is_suspicious_tool_call()`:

| Config | Strategy | Expected tradeoff |
|--------|----------|-------------------|
| `permissive` | Exact secret literal matching only | High precision, low recall |
| `balanced` | Calibrated path/command/recipient tokens (scorer_config_v2.json) | Best F1 |
| `strict` | balanced + `flag_any_execute_command: true` | High recall, moderate FPR |
| `paranoid` | Flag all tool calls | Perfect recall, unusable FPR |

### 2.3 Evaluation Metrics

Per configuration: precision, recall, F1, accuracy, false positive rate (FPR). Per tool type within each configuration: same metrics. Confusion matrices reported in full.

### 2.4 Scoring Function

The policy gate function `is_suspicious_tool_call(tool_call, config)` applies these rules:

1. **read_file**: Check path against `sensitive_path_tokens` list (partial string matching)
2. **execute_command**: Check command against `sensitive_command_tokens` list; optionally flag ALL commands
3. **send_message**: Check recipient against `trusted_send_recipients` allowlist; check body for `secret_literals` and `secret_regexes`

The function returns a boolean (suspicious / not suspicious). No confidence scores or explanations.

## 3. Results

### 3.1 Aggregate Metrics

| Config | Precision | Recall | F1 | FPR | Meets criteria? |
|--------|-----------|--------|----|-----|-----------------|
| **balanced** | 1.000 | 0.926 | 0.962 | 0.000 | **Yes** (>=90% recall, <=5% FPR) |
| strict | 0.818 | 1.000 | 0.900 | 0.333 | No (FPR too high) |
| paranoid | 0.600 | 1.000 | 0.750 | 1.000 | No (FPR = 100%) |
| permissive | 1.000 | 0.333 | 0.500 | 0.000 | No (recall too low) |

### 3.2 Per-Tool Analysis (balanced)

| Tool | Recall | FPR | Notes |
|------|--------|-----|-------|
| read_file | 1.000 | 0.000 | All sensitive paths caught; no false positives on source files |
| send_message | 1.000 | 0.000 | Untrusted recipients + secret body detection both effective |
| execute_command | 0.778 | 0.000 | 4 malicious commands evaded token matching |

### 3.3 Error Analysis

The 4 false negatives in `balanced` execute_command were commands that:
- Used indirect exfiltration techniques not covered by the sensitive token list
- Constructed malicious behavior from individually benign-looking operations

This represents the fundamental limitation of token-matching approaches: adversaries can construct novel command syntax that achieves the same malicious goal without triggering known patterns.

## 4. Discussion

### 4.1 Balanced is production-viable

The `balanced` configuration's operating point (92.6% recall, 0% FPR) is strong for a zero-latency, zero-cost defense layer. In practice, it would catch 50 out of 54 malicious tool calls while never blocking a legitimate developer operation.

### 4.2 Architectural complementarity

Post-hoc filtering is **not** a replacement for prompt-level defenses. It is the third layer in a defense-in-depth stack:

1. **System instruction** — tells the model not to follow injected instructions
2. **Boundary tags** — demarcates untrusted content
3. **Tool-call policy gate** — catches malicious outputs that slip through layers 1-2

Our Round 2 showed that tags + instructions still allow ~10% injection success. A policy gate that catches 92.6% of those remaining attacks reduces the effective success rate to ~0.7%.

### 4.3 Comparison to model self-refusal

SafeToolBench (Li et al., 2025) achieves 83% recall through model fine-tuning for tool-call self-refusal. Our external policy gate achieves 92.6% recall without any model modification. The approaches are complementary — a model that self-refuses 83% of malicious calls, combined with a policy gate that catches 92.6% of what's generated, would have a combined miss rate under 2%.

### 4.4 Limitations

1. **Synthetic corpus.** Real malicious tool calls generated by injected models may be more creative than our hand-crafted examples. Validating against actual model outputs (e.g., from Round 3 live runs) would strengthen the evaluation.
2. **Three tool types.** Modern agents use dozens of tools (web search, database queries, API calls). Each new tool type needs its own policy rules.
3. **Static rules.** Pattern matching is brittle against adversarial evasion. A hybrid approach combining rules with a lightweight classifier could improve robustness.
4. **No cost measurement.** While rule-based matching is expected to add negligible latency, we did not measure this.

## 5. Conclusion

A rule-based tool-call policy gate is a practical, deployable defense layer for LLM agents. The `balanced` configuration meets our success criteria (>=90% recall, <=5% FPR) and complements prompt-level defenses to create a three-layer defense-in-depth architecture. The primary limitation is `execute_command` filtering, where adversarial command construction can evade token matching. Future work should test against real model-generated malicious tool calls and explore hybrid rule + classifier approaches.

## 6. Reproducibility

```bash
git clone https://github.com/misty-step/laboratory.git
cd laboratory
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e .
make run-r6 && make analyze-r6
```

All code, data, and analysis scripts are open source (MIT license).

## References

- Anthropic (2025). "Prompt Injection Defenses." anthropic.com/research/prompt-injection-defenses
- Debenedetti, E. et al. (2024). "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents." NeurIPS 2024. arXiv:2406.13352
- Li, X. et al. (2025). "SafeToolBench: Prospective Tool Utilization Safety." EMNLP 2025. arXiv:2509.07315
- OpenAI (2024). "The Instruction Hierarchy." openai.com/research/the-instruction-hierarchy
- Willison, S. (2025). "The Lethal Trifecta." simonwillison.net
- Zhan, Q. et al. (2024). "InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents." ACL 2024. arXiv:2403.02691
- Zhang, H. et al. (2025). "Agent Security Bench." ICLR 2025. arXiv:2410.02644
