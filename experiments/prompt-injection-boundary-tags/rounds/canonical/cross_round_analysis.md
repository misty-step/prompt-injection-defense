# Cross-Round Injection Rate Analysis

Analysis generated from 1368 canonical run records.

**CI Method:** wilson at 95% confidence
**Baseline Round:** round3

## By Round + Condition

| Round | Condition | Total | OK | Errors | Inj OK | Inj Rate | 95% CI |
|-------|-----------|-------|-----|--------|--------|----------|--------|
| round1 | dynamic_nonce | 24 | 24 | 0 | 0 | 0.0% | [0.0%, 13.8%] |
| round1 | raw | 24 | 24 | 0 | 0 | 0.0% | [0.0%, 13.8%] |
| round1 | static_tags | 24 | 24 | 0 | 0 | 0.0% | [0.0%, 13.8%] |
| round2 | instructed | 144 | 108 | 36 | 25 | 23.1% | [16.2%, 31.9%] |
| round2 | raw | 144 | 108 | 36 | 57 | 52.8% | [43.4%, 61.9%] |
| round2 | tagged | 144 | 108 | 36 | 54 | 50.0% | [40.7%, 59.3%] |
| round2b | dynamic_nonce | 108 | 108 | 0 | 10 | 9.3% | [5.1%, 16.2%] |
| round2b | raw | 108 | 108 | 0 | 34 | 31.5% | [23.5%, 40.7%] |
| round2b | static_tags | 108 | 108 | 0 | 11 | 10.2% | [5.8%, 17.3%] |
| round7 | full_stack | 540 | 540 | 0 | 1 | 0.2% | [0.0%, 1.0%] |

## Defense Ordering by Round

### round1

| Rank | Defense | Inj Rate | n (OK trials) |
|------|---------|----------|---------------|
| 1 | raw | 0.0% | 24 |

### round2

| Rank | Defense | Inj Rate | n (OK trials) |
|------|---------|----------|---------------|
| 1 | raw | 52.8% | 108 |

### round2b

| Rank | Defense | Inj Rate | n (OK trials) |
|------|---------|----------|---------------|
| 1 | raw | 31.5% | 108 |

### round7

| Rank | Defense | Inj Rate | n (OK trials) |
|------|---------|----------|---------------|
| 1 | full_stack | 0.2% | 540 |

## Delta vs round3 Baseline

