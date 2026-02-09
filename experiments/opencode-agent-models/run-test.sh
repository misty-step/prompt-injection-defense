#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run-test.sh MODEL_ID

Example:
  ./run-test.sh openrouter/moonshotai/kimi-k2.5

Environment:
  KEEP_TMP=1                 Keep the temp workdir for debugging (default: 0)
  OPENCODE_TIMEOUT_SEC=600   If `timeout`/`gtimeout` exists, kill opencode after this many seconds
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_ID="${1:-}"
if [[ -z "$MODEL_ID" ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR"

MODEL_SLUG="$(printf '%s' "$MODEL_ID" | sed -E 's/[^a-zA-Z0-9._-]+/_/g')"
RESULT_PATH="${RESULTS_DIR}/${MODEL_SLUG}.json"

KEEP_TMP="${KEEP_TMP:-0}"

mktemp_dir() {
  mktemp -d 2>/dev/null || mktemp -d -t opencode-model-eval
}

TMP_ROOT="$(mktemp_dir)"
cleanup() {
  if [[ "$KEEP_TMP" == "1" ]]; then
    echo "KEEP_TMP=1 set; temp dir preserved at: $TMP_ROOT" >&2
    return 0
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

PROJ_DIR="${TMP_ROOT}/proj"
mkdir -p "$PROJ_DIR"
cd "$PROJ_DIR"

git init -q
git config user.email "opencode-experiment@example.com"
git config user.name "OpenCode Experiment"

cat > go.mod <<'EOF'
module example.com/opencodeeval

go 1.20
EOF

cat > main.go <<'EOF'
package main

import "fmt"

// TODO: Implement Add(a, b int) int and add tests.
func main() {
	fmt.Println("opencode agent eval")
}
EOF

git add go.mod main.go
git commit -q -m "chore: initial skeleton"
BASE_COMMIT="$(git rev-parse HEAD)"

PROMPT="Add a function called Add(a, b int) int that returns a+b. Write a test for it. Commit your changes."

OPENCODE_STDOUT="${TMP_ROOT}/opencode.stdout"
OPENCODE_STDERR="${TMP_ROOT}/opencode.stderr"
GO_TEST_OUT="${TMP_ROOT}/go_test.txt"

START_EPOCH="$(python3 -c 'import time; print(time.time())')"

set +e
if [[ -n "${OPENCODE_TIMEOUT_SEC:-}" ]]; then
  if command -v timeout >/dev/null 2>&1; then
    timeout "${OPENCODE_TIMEOUT_SEC}" opencode run -m "$MODEL_ID" --format json "$PROMPT" >"$OPENCODE_STDOUT" 2>"$OPENCODE_STDERR"
    OPENCODE_EXIT=$?
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "${OPENCODE_TIMEOUT_SEC}" opencode run -m "$MODEL_ID" --format json "$PROMPT" >"$OPENCODE_STDOUT" 2>"$OPENCODE_STDERR"
    OPENCODE_EXIT=$?
  else
    echo "OPENCODE_TIMEOUT_SEC set but timeout/gtimeout not found; running without timeout" >&2
    opencode run -m "$MODEL_ID" --format json "$PROMPT" >"$OPENCODE_STDOUT" 2>"$OPENCODE_STDERR"
    OPENCODE_EXIT=$?
  fi
else
  opencode run -m "$MODEL_ID" --format json "$PROMPT" >"$OPENCODE_STDOUT" 2>"$OPENCODE_STDERR"
  OPENCODE_EXIT=$?
fi
set -e

END_EPOCH="$(python3 -c 'import time; print(time.time())')"

GO_TEST_RAN=0
GO_TEST_EXIT=""
if command -v go >/dev/null 2>&1; then
  GO_TEST_RAN=1
  set +e
  go test ./... >"$GO_TEST_OUT" 2>&1
  GO_TEST_EXIT=$?
  set -e
else
  echo "go not found in PATH; skipping go test" >"$GO_TEST_OUT"
fi

HEAD_COMMIT="$(git rev-parse HEAD)"
WORKTREE_FILES="$(
  git ls-files -m -o --exclude-standard || true
)"

COMMITTED_FILES=""
if [[ "$HEAD_COMMIT" != "$BASE_COMMIT" ]]; then
  COMMITTED_FILES="$(
    git diff --name-only "$BASE_COMMIT" "$HEAD_COMMIT" || true
  )"
fi

RESULT_PATH="$RESULT_PATH" \
MODEL_ID="$MODEL_ID" \
MODEL_SLUG="$MODEL_SLUG" \
BASE_COMMIT="$BASE_COMMIT" \
HEAD_COMMIT="$HEAD_COMMIT" \
WORKTREE_FILES="$WORKTREE_FILES" \
COMMITTED_FILES="$COMMITTED_FILES" \
OPENCODE_EXIT="$OPENCODE_EXIT" \
OPENCODE_STDOUT="$OPENCODE_STDOUT" \
OPENCODE_STDERR="$OPENCODE_STDERR" \
START_EPOCH="$START_EPOCH" \
END_EPOCH="$END_EPOCH" \
GO_TEST_RAN="$GO_TEST_RAN" \
GO_TEST_EXIT="$GO_TEST_EXIT" \
GO_TEST_OUT="$GO_TEST_OUT" \
python3 - <<'PY'
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

result_path = Path(os.environ["RESULT_PATH"])

model_id = os.environ["MODEL_ID"]
model_slug = os.environ["MODEL_SLUG"]
base_commit = os.environ["BASE_COMMIT"]
head_commit = os.environ["HEAD_COMMIT"]

opencode_exit = int(os.environ["OPENCODE_EXIT"])
opencode_stdout_path = Path(os.environ["OPENCODE_STDOUT"])
opencode_stderr_path = Path(os.environ["OPENCODE_STDERR"])
opencode_stdout = opencode_stdout_path.read_text(encoding="utf-8", errors="replace") if opencode_stdout_path.exists() else ""
opencode_stderr = opencode_stderr_path.read_text(encoding="utf-8", errors="replace") if opencode_stderr_path.exists() else ""

start_epoch = float(os.environ["START_EPOCH"])
end_epoch = float(os.environ["END_EPOCH"])
duration_sec = max(0.0, end_epoch - start_epoch)

go_test_ran = os.environ.get("GO_TEST_RAN") == "1"
go_test_exit = os.environ.get("GO_TEST_EXIT", "")
go_test_output_path = Path(os.environ["GO_TEST_OUT"])
go_test_output = go_test_output_path.read_text(encoding="utf-8", errors="replace") if go_test_output_path.exists() else ""
go_test_exit_code: Optional[int]
try:
    go_test_exit_code = int(go_test_exit) if go_test_exit != "" else None
except Exception:
    go_test_exit_code = None

worktree_files = [p for p in (os.environ.get("WORKTREE_FILES") or "").splitlines() if p.strip()]
committed_files = [p for p in (os.environ.get("COMMITTED_FILES") or "").splitlines() if p.strip()]

def parse_json_or_jsonl(raw: str) -> Tuple[Optional[Any], List[Any]]:
    s = raw.strip()
    if not s:
        return None, []
    try:
        return json.loads(s), []
    except Exception:
        pass
    objs: List[Any] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except Exception:
            continue
    return None, objs

def walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from walk(it)

def usage_total_from_dict(d: Dict[str, Any]) -> Optional[int]:
    def to_int(v: Any) -> Optional[int]:
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        return None

    # Common schemas
    total = to_int(d.get("total_tokens"))
    if total is not None:
        return total
    prompt = to_int(d.get("prompt_tokens"))
    comp = to_int(d.get("completion_tokens"))
    if prompt is not None and comp is not None:
        return prompt + comp
    inp = to_int(d.get("input_tokens"))
    out = to_int(d.get("output_tokens"))
    if inp is not None and out is not None:
        return inp + out
    # Variants
    inp = to_int(d.get("inputTokens"))
    out = to_int(d.get("outputTokens"))
    if inp is not None and out is not None:
        return inp + out
    return None

def extract_usage(raw: str) -> Dict[str, Any]:
    parsed_obj, jsonl_objs = parse_json_or_jsonl(raw)
    events: List[Any] = []
    if parsed_obj is not None:
        events = [parsed_obj]
    else:
        events = jsonl_objs

    usage_entries: List[Dict[str, Any]] = []
    totals: List[int] = []

    for ev in events:
        for d in walk(ev):
            if not isinstance(d, dict):
                continue
            if "usage" in d and isinstance(d["usage"], dict):
                usage_entries.append(d["usage"])
                t = usage_total_from_dict(d["usage"])
                if t is not None:
                    totals.append(t)
            # Some tools inline usage-like dicts without a "usage" key.
            elif any(k in d for k in ("total_tokens", "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")):
                usage_entries.append(d)
                t = usage_total_from_dict(d)
                if t is not None:
                    totals.append(t)

    return {
        "parsed_as": "json" if parsed_obj is not None else ("jsonl" if jsonl_objs else "unknown"),
        "entries_found": len(usage_entries),
        "tokens_total_sum": sum(totals) if totals else None,
        "tokens_total_max": max(totals) if totals else None,
    }

usage = extract_usage(opencode_stdout)
tokens_used = usage.get("tokens_total_sum")

def scan_go_project(root: Path) -> Dict[str, Any]:
    go_files = list(root.rglob("*.go"))
    has_test_file = any(p.name.endswith("_test.go") for p in go_files)
    add_re = re.compile(r"(?m)^func\\s+Add\\s*\\(\\s*a\\s*,\\s*b\\s+int\\s*\\)\\s*int\\s*\\{")
    add_found = False
    for p in go_files:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if add_re.search(txt):
            add_found = True
            break
    return {
        "go_files": [str(p.relative_to(root)) for p in go_files],
        "has_test_file": has_test_file,
        "add_function_found": add_found,
    }

proj_scan = scan_go_project(Path("."))

commits_made = head_commit != base_commit
worktree_dirty = bool(worktree_files)
files_written = commits_made or worktree_dirty

data = {
    "schema_version": 1,
    "model_id": model_id,
    "model_slug": model_slug,
    "started_at_utc": dt.datetime.fromtimestamp(start_epoch, tz=dt.timezone.utc).isoformat(),
    "ended_at_utc": dt.datetime.fromtimestamp(end_epoch, tz=dt.timezone.utc).isoformat(),
    "duration_sec": duration_sec,
    "tokens_used": tokens_used if isinstance(tokens_used, int) else None,
    "files_written": files_written,
    "commits_made": commits_made,
    "opencode_exit_code": opencode_exit,
    "usage": usage,
    "git": {
        "base_commit": base_commit,
        "head_commit": head_commit,
        "committed_files": committed_files,
        "worktree_files": worktree_files,
        "worktree_dirty": worktree_dirty,
    },
    "project": proj_scan,
    "go_test": {
        "ran": go_test_ran,
        "exit_code": go_test_exit_code,
        "pass": (go_test_exit_code == 0) if go_test_ran and go_test_exit_code is not None else None,
        "output": go_test_output,
    },
    # Preserve raw OpenCode output for offline parsing/version differences.
    "opencode_output": {
        "stdout": opencode_stdout,
        "stderr": opencode_stderr,
    },
}

result_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

python3 - <<PY
import json
from pathlib import Path
data = json.loads(Path("$RESULT_PATH").read_text(encoding="utf-8"))
print(f"Wrote: {Path('$RESULT_PATH').relative_to(Path('$SCRIPT_DIR'))}")
print(f"  duration_sec: {data.get('duration_sec'):.1f}")
print(f"  tokens_used: {data.get('tokens_used')}")
print(f"  files_written: {data.get('files_written')}")
print(f"  commits_made: {data.get('commits_made')}")
print(f"  go_test_pass: {((data.get('go_test') or {}).get('pass'))}")
print(f"  add_function_found: {((data.get('project') or {}).get('add_function_found'))}")
print(f"  has_test_file: {((data.get('project') or {}).get('has_test_file'))}")
print(f"  opencode_exit_code: {data.get('opencode_exit_code')}")
PY
