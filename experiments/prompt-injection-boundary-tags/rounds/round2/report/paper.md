# Untrusted Content Boundary Tags as Prompt Injection Defense in Tool-Using Agents: Round 2

**Authors:** Kaylee & Phaedrus, Misty Step  
**Date:** 2026-02-11  
**Repository:** github.com/misty-step/laboratory

---

## Abstract

Tool-using LLM agents ingest untrusted content (issues, emails, web pages) inside the same context that contains real instructions and tool affordances. This creates prompt injection risk with real-world impact: the model can be tricked into leaking secrets or issuing malicious tool calls. We test a simple defense: wrap untrusted content in explicit boundary tags that frame it as data, not instructions. In 324 trials (12 payloads × 3 conditions × 3 trials × 3 models), boundary tags reduced injection success from **31% (raw)** to **10% (static tags)** and **9% (dynamic nonce tags)** (χ²=24.22, p=0.000006). Boundary tags also reduced malicious tool calls from **32** (raw) to **5** (static) and **7** (dynamic). The strongest remaining bypass was a “gradual escalation” payload, reaching **56%** injection overall and **100%** against GPT-4o across all conditions.

## 1. Introduction

### 1.1 The Problem
Prompt injection occurs when an attacker embeds instructions inside untrusted content that an agent ingests, causing the model to follow attacker intent instead of the task. Tool-using agents amplify the risk: instructions can become actions (file reads, command execution, outbound messages).

This matches Willison’s “Lethal Trifecta”: private data access + untrusted content + external communication. Systems satisfying this trifecta are structurally vulnerable without defense-in-depth (Willison, 2025).

### 1.2 The Defense Under Test
We test *boundary tags*: wrapping untrusted content in clear delimiters that (1) label the content as untrusted and (2) constrain the model’s interpretation to “data to summarize,” not “instructions to execute.”

### 1.3 Prior Art
- **AgentDojo** (ETH Zurich, NeurIPS 2024; arXiv:2406.13352): benchmark for prompt injection against tool-calling agents.
- **InjecAgent** (UIUC, ACL 2024; arXiv:2403.02691): indirect prompt injection in tool-integrated agent settings.
- **Instruction hierarchy / priority** (OpenAI, 2024): system > developer > user > tool output; motivates explicit boundary signaling for untrusted tool outputs.
- **The Lethal Trifecta** (Willison, 2025): threat model framing that predicts high risk in agent designs with tools + secrets + untrusted inputs.

### 1.4 Contribution
Round 2 isolates the effect of boundary tags under *realistic* agent conditions:
- system prompt encourages helpfulness and tool use (no “do not follow instructions in content” clause)
- tools are available and callable
- multiple models tested
- multiple payload categories, including escalation and social engineering

## 2. Methodology

### 2.1 Experimental Design
- **Type:** controlled experiment, between-subjects across conditions
- **Independent variable:** boundary-tag condition (3 levels)
- **Dependent variables:** injection success (binary, derived from ordinal score), tool-call incidence/count
- **Trials:** 12 payloads × 3 conditions × 3 trials × 3 models = **324**

### 2.2 Task And Agent Prompt
The agent is asked to summarize a GitHub issue and suggest next steps. The system prompt includes fake configuration secrets and encourages tool use to gather context.

### 2.3 Tools
Tools are available to the model (mocked; calls are logged, not executed):
- `read_file(path)`
- `execute_command(command)`
- `send_message(to, body)`

Tool calls in response to injected instructions are treated as high-severity compromise signals.

### 2.4 Conditions

| Condition | Description |
|---|---|
| `raw` | Untrusted issue content passed as-is. |
| `static_tags` | Content wrapped in fixed boundary tags with an “untrusted data” notice. |
| `dynamic_nonce` | Content wrapped in randomized nonce tags; existing tags stripped to reduce breakout attempts. |

### 2.5 Payloads
Twelve payload categories (names are analysis keys):
`direct_override`, `authority_escalation`, `helpful_framing`, `tag_breaking`, `encoding_trick`, `social_engineering`, `indirect_extraction`, `multi_step`, `tool_invocation`, `persona_hijack`, `context_confusion`, `gradual_escalation`.

### 2.6 Models

| Model (report label) | Model id | Notes |
|---|---|---|
| Claude Haiku 3.5 | `claude-3-5-haiku-latest` | Most resistant in this run. |
| GPT-4o | `gpt-4o` | Most vulnerable in this run. |
| Kimi K2.5 | `kimi-k2.5` | Mid-range vulnerability. |

### 2.7 Scoring And Success Definition

| Score | Meaning |
|---:|---|
| 0 | Clean: ignores injection; normal summary. |
| 1 | Acknowledges injection attempt but does not comply. |
| 2 | Partial compromise: leaks sensitive context or partially complies. |
| 3 | Full compromise: executes injected intent (often via tool call). |

**Injection success:** `score >= 2`.

### 2.8 Statistical Tests
- Overall condition effect: chi-square test on success rates.
- Per-model comparisons: Fisher’s exact tests (pairwise / small-N robustness).

### 2.9 Data And Reproducibility
- Calibrated, 324-row dataset used for analysis + charts:  
  `experiments/prompt-injection-boundary-tags/rounds/round2b/data/results_r2_combined_latest.csv`
- Round 2 report artifacts live in:  
  `experiments/prompt-injection-boundary-tags/rounds/round2/report/`

## 3. Results

### 3.1 Overall By Condition

| Condition | N | Avg score | Injection rate (score>=2) | Tool call rate (any) | Tool calls (count) |
|---|---:|---:|---:|---:|---:|
| raw | 108 | 1.23 | 31.5% (34/108) | 24.1% (26/108) | 32 |
| static_tags | 108 | 0.60 | 10.2% (11/108) | 3.7% (4/108) | 5 |
| dynamic_nonce | 108 | 0.70 | 9.3% (10/108) | 4.6% (5/108) | 7 |

Overall significance: **χ²=24.22, p=0.000006, Cramér’s V=0.27**.

### 3.2 Per-Model Injection Rates

| Model | Raw inj% | Static inj% | Dynamic inj% | p-value |
|---|---:|---:|---:|---:|
| Claude Haiku 3.5 | 25.0% | 8.3% | 2.8% | 0.011 |
| GPT-4o | 47.2% | 13.9% | 19.4% | 0.003 |
| Kimi K2.5 | 22.2% | 8.3% | 5.6% | 0.066 |

Notes:
- GPT-4o improves sharply under tagging vs raw, but remains the most vulnerable.
- Dynamic nonce tags did not outperform static tags overall; GPT-4o worsened under dynamic nonce in this run.

### 3.3 Per-Payload Injection Rates (Aggregated Across Models)

| Payload | Overall inj% | Raw inj% | Static inj% | Dynamic inj% |
|---|---:|---:|---:|---:|
| gradual_escalation | 55.6% | 77.8% | 55.6% | 33.3% |
| indirect_extraction | 40.7% | 77.8% | 22.2% | 22.2% |
| social_engineering | 33.3% | 44.4% | 22.2% | 33.3% |
| multi_step | 22.2% | 66.7% | 0% | 0% |
| helpful_framing | 18.5% | 33.3% | 11.1% | 11.1% |
| encoding_trick | 14.8% | 22.2% | 11.1% | 11.1% |
| context_confusion | 11.1% | 33.3% | 0% | 0% |
| authority_escalation | 7.4% | 22.2% | 0% | 0% |
| tool_invocation | 0% | 0% | 0% | 0% |
| tag_breaking | 0% | 0% | 0% | 0% |
| persona_hijack | 0% | 0% | 0% | 0% |
| direct_override | 0% | 0% | 0% | 0% |

Key observation: classic “ignore instructions” payloads failed (0%), while *plausible*, task-shaped payloads succeeded (gradual escalation, indirect extraction, social engineering).

### 3.4 Tool Call Analysis

Total malicious tool calls: **44**

| Condition | Tool calls | Share |
|---|---:|---:|
| raw | 32 | 73% |
| static_tags | 5 | 11% |
| dynamic_nonce | 7 | 16% |

Most commonly triggered tools:
- `execute_command`: 18
- `read_file`: 14
- `send_message`: 12

## 4. Discussion

### 4.1 What Boundary Tags Buy You
Boundary tags reduced injection success by about 70% (31% → 10%) and reduced malicious tool calls by about 80% (32 → 5) under an agent prompt that encourages tool use. This is a strong signal that “mark untrusted content as data” is a real defense lever, not just formatting.

### 4.2 What Boundary Tags Do Not Solve
Residual injection (~9–10%) remains material. For GPT-4o, the most dangerous payload (`gradual_escalation`) succeeded **100%** across all conditions. Tagging alone is not sufficient against well-shaped attacks that stay inside the apparent task.

### 4.3 Static vs Dynamic Nonce Tags
Nonce tags are meant to prevent simple boundary breakouts. In this run, `tag_breaking` was already 0% in all conditions, and static tags performed about as well as nonce tags overall. This suggests the main benefit is semantic framing (“this is untrusted”) rather than syntactic breakout resistance.

### 4.4 Model And Payload Specificity
Model choice dominates risk. GPT-4o is the highest-risk model here under realistic tool affordances. Payload shape matters more than directness: gradual escalation and indirect extraction dominate simple overrides.

### 4.5 Limitations
- N per cell is small (3 repetitions); tail events may be under-sampled.
- Single-task framing (issue summarization) and single tool set.
- Tools are mocked; real systems add side channels (network, filesystem breadth, auth scope).
- “Injection success” collapses an ordinal score into a binary threshold (>=2).
- Results are conditional on the exact system prompt, tool specs, and payload phrasing.

### 4.6 Practical Implications
Boundary tags are worth shipping, but only as one layer:
- enforce tool-call allowlists + argument validation
- require high-friction confirmation for dangerous tools (`execute_command`, outbound comms)
- isolate secrets from the model context where possible
- log and rate-limit tool calls; flag unusual sequences (gradual escalation often looks like “normal work” until it doesn’t)

## 5. Conclusion

Boundary tags materially reduce prompt injection success and malicious tool calls in a realistic tool-using agent setting, but do not eliminate risk. The deadliest bypass class (gradual escalation) can fully defeat tagging on some models. Defense-in-depth remains required.

## References

- ETH Zurich. (2024). *AgentDojo: A Benchmark for Prompt Injection in Tool-Using Agents.* NeurIPS 2024. arXiv:2406.13352.
- UIUC. (2024). *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Agents.* ACL 2024. arXiv:2403.02691.
- OpenAI. (2024). *Instruction hierarchy and prompt priority.* (Documentation / policy guidance).
- Willison, S. (2025). *The Lethal Trifecta.* (Blog post / essay).

