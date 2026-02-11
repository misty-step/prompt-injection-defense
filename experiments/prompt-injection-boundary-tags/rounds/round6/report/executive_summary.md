# Round 6: Tool-Call Policy Gate — Executive Summary

## Finding

A rule-based filter that inspects LLM tool calls before execution catches **93% of malicious calls with zero false positives** using calibrated pattern matching.

## What we tested

When AI agents process untrusted content (emails, GitHub issues, web pages), prompt injection attacks can trick the model into making dangerous tool calls — reading secret files, running exfiltration commands, or sending data to attackers. We tested whether a simple pre-execution filter can catch these calls before they cause harm.

## Result

| Configuration | Catches | Blocks legitimate work? |
|--------------|---------|------------------------|
| **Balanced (recommended)** | **93% of attacks** | **No (0% false positives)** |
| Strict | 100% of attacks | Yes (blocks 1 in 3 legitimate commands) |

## Recommendation

Deploy the `balanced` policy gate as a third defense layer alongside system instructions and boundary tags. Combined, these three layers reduce effective injection success to under 1%.

## Cost

Zero runtime cost (rule-based pattern matching). Configuration maintained in a single JSON file.

## Limitation

Pattern matching can be evaded by creative command construction. 4 out of 54 malicious commands were missed. Consider hybrid approaches (rules + classifier) for high-security deployments.
