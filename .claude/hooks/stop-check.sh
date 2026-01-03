#!/bin/bash
set -euo pipefail

# Parse input from Claude
input=$(cat)
stop_hook_active=$(echo "$input" | jq -r '.stop_hook_active // false')

# Prevent infinite loops - if we're already in a continuing loop, allow stop
if [ "$stop_hook_active" = "true" ]; then
    exit 0
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

# All checks passed
exit 0
