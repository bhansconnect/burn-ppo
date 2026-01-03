#!/bin/bash
set -euo pipefail

# Read JSON input from stdin
INPUT=$(cat)

# Skip checks if in plan mode
permission_mode=$(echo "$INPUT" | jq -r '.permission_mode // "default"')
if [ "$permission_mode" = "plan" ]; then
  echo "Skipping checks (plan mode)" >&2
  exit 0
fi

# Skip checks if Claude just asked a question
transcript_path=$(echo "$INPUT" | jq -r '.transcript_path // ""')
if [ -n "$transcript_path" ] && [ -f "$transcript_path" ]; then
    # Get the name of the last tool_use in the transcript
    last_tool=$(grep '"tool_use"' "$transcript_path" | tail -1 | jq -r '.name // empty' 2>/dev/null)
    if [ "$last_tool" = "AskUserQuestion" ]; then
        echo "Skipping checks (asked question)" >&2
        exit 0
    fi
fi

# Retry counter to prevent infinite loops while ensuring all checks run
MAX_RETRIES=10
COUNTER_FILE="/tmp/claude-stop-check-$(echo "$PWD" | md5 | cut -c1-8)"

# Read current retry count
retry_count=0
if [ -f "$COUNTER_FILE" ]; then
    # Check if file is stale (older than 30 minutes)
    file_age=$(( $(date +%s) - $(stat -f %m "$COUNTER_FILE" 2>/dev/null || stat -c %Y "$COUNTER_FILE") ))
    if [ "$file_age" -gt 1800 ]; then
        rm -f "$COUNTER_FILE"
    else
        retry_count=$(cat "$COUNTER_FILE")
    fi
fi

# Increment and save
retry_count=$((retry_count + 1))
echo "$retry_count" > "$COUNTER_FILE"

# Safety valve: after max retries, warn and allow stop
if [ "$retry_count" -gt "$MAX_RETRIES" ]; then
    echo "Error: Exceeded $MAX_RETRIES retry attempts. Allowing stop despite possible failures." >&2
    echo "Tell Claude to continue if it should keep trying."
    rm -f "$COUNTER_FILE"
    exit 1
fi

# 1. Run cargo fmt --check
echo "Running cargo fmt..." >&2
if ! fmt_output=$(cargo fmt -- --check 2>&1); then
    echo "## Formatting Issues

Code is not properly formatted:

\`\`\`
$fmt_output
\`\`\`

**To fix:** Run \`cargo fmt\` to auto-format the code.
" >&2
    exit 2
fi

# 2. Run cargo check --all-targets
echo "Running cargo check..." >&2
if ! check_output=$(cargo check --all-targets 2>&1); then
    echo "## Cargo Check Failed

The code has compilation errors that must be fixed:

\`\`\`
$check_output
\`\`\`

**To fix:** Read the error messages above. Common issues:
- Missing imports: add \`use\` statements
- Type mismatches: check function signatures
- Unused variables: prefix with \`_\` or remove
- Missing fields: ensure all struct fields are provided
" >&2
    exit 2
fi

# 3. Run cargo clippy
echo "Running cargo clippy..." >&2
if ! clippy_output=$(cargo clippy --all-targets -- -D warnings 2>&1); then
    echo "## Clippy Warnings

Clippy found issues that should be fixed:

\`\`\`
$clippy_output
\`\`\`

**To fix:** Follow the suggested code changes from clippy. Run \`cargo clippy --fix\` to auto-fix some issues.
" >&2
    exit 2
fi

# 4. Run tests (prefer nextest, fallback to cargo test)
echo "Running tests..." >&2
if command -v cargo-nextest &> /dev/null; then
    test_cmd="cargo nextest run"
else
    test_cmd="cargo test"
fi

if ! test_output=$($test_cmd 2>&1); then
    echo "## Tests Failed

Some tests are failing:

\`\`\`
$test_output
\`\`\`

**To fix:**
- Read the failing test names and their assertions
- Check the expected vs actual values
- Run the specific failing test to iterate quickly
" >&2
    exit 2
fi

# 5. Run code coverage (use nextest if available)
echo "Running coverage..." >&2
if command -v cargo-nextest &> /dev/null; then
    cov_cmd="cargo llvm-cov nextest --fail-under-regions 80"
else
    cov_cmd="cargo llvm-cov --fail-under-regions 80"
fi

if ! cov_output=$($cov_cmd 2>&1); then
    echo "## Code Coverage Below 80%

Coverage is below the required 80% region coverage threshold:

\`\`\`
$cov_output
\`\`\`

**To fix:**
- Add tests for uncovered code paths
- Focus on critical business logic and error handling
- Run \`cargo llvm-cov --json | jq\` to see detailed coverage data
- Region coverage counts individual code regions (branches, expressions), not just lines
" >&2
    exit 2
fi

# All checks passed - clean up counter
rm -f "$COUNTER_FILE"
exit 0
