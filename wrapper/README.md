# Untrusted Content Wrapper

Tags CLI output as untrusted, using randomized boundary markers with per-invocation nonces.

```bash
./untrusted-exec.sh gh issue view 44 --repo owner/repo
```

Anti-breakout: strips existing tags, random nonce makes closing tag unpredictable.
