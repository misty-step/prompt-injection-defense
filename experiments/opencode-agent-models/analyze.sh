#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

RESULTS_DIR="$RESULTS_DIR" python3 - <<'PY'
import glob
import json
import os
from pathlib import Path

results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
paths = sorted(glob.glob(str(results_dir / "*.json")))

if not paths:
    print(f"No results found in {results_dir}/. Run ./run-test.sh first.")
    raise SystemExit(0)

def fmt_bool(v):
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return ""

rows = []
for p in paths:
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:
        rows.append(
            {
                "Model": Path(p).stem,
                "Dur(s)": "",
                "Tokens": "",
                "Files": "",
                "Commits": "",
                "GoTest": "",
                "Exit": "",
                "File": os.path.basename(p),
                "_err": str(e),
            }
        )
        continue

    duration = data.get("duration_sec")
    tokens = data.get("tokens_used")
    files_written = data.get("files_written")
    commits_made = data.get("commits_made")
    go_test = (data.get("go_test") or {})
    go_pass = go_test.get("pass")
    exit_code = data.get("opencode_exit_code")

    rows.append(
        {
            "Model": data.get("model_id") or data.get("model_slug") or Path(p).stem,
            "Dur(s)": f"{duration:.1f}" if isinstance(duration, (int, float)) else "",
            "Tokens": str(tokens) if isinstance(tokens, int) else "",
            "Files": fmt_bool(files_written),
            "Commits": fmt_bool(commits_made),
            "GoTest": fmt_bool(go_pass),
            "Exit": str(exit_code) if isinstance(exit_code, int) else "",
            "File": os.path.basename(p),
        }
    )

headers = ["Model", "Dur(s)", "Tokens", "Files", "Commits", "GoTest", "Exit", "File"]

widths = {h: len(h) for h in headers}
for r in rows:
    for h in headers:
        widths[h] = max(widths[h], len(str(r.get(h, ""))))

def md_row(values):
    return "| " + " | ".join(str(values[h]).ljust(widths[h]) for h in headers) + " |"

print(md_row({h: h for h in headers}))
print("| " + " | ".join(("-" * widths[h]) for h in headers) + " |")
for r in rows:
    print(md_row(r))
PY
