#!/bin/bash
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

# Require approval for protected files
if [[ "$file_path" == *"Cargo.toml" ]] || [[ "$file_path" == *".claude/hooks/protected-files.sh" ]] || [[ "$file_path" == *".claude/hooks/stop-check.sh" ]] || [[ "$file_path" == *".claude/settings.json" ]]; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"Protected file requires manual approval"}}'
  exit 0
fi

# Allow other edits
exit 0
