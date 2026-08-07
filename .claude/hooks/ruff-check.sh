#!/usr/bin/env bash
# PostToolUse hook: lint an edited Python file with the uv project that owns it.
# Autofixes what ruff can fix, then exits 2 so the findings it cannot fix are
# fed back to the caller instead of passing silently.
set -uo pipefail

file=$(jq -r '.tool_response.filePath // .tool_input.file_path // empty')
[[ $file == *.py ]] || exit 0
[[ -f $file ]] || exit 0

# The owning uv project is the nearest ancestor holding pyproject.toml: the
# block repos keep it at the repo root or one level down under software/.
project=$(dirname "$file")
while [[ $project != / && ! -f $project/pyproject.toml ]]; do
  project=$(dirname "$project")
done
[[ -f $project/pyproject.toml ]] || exit 0

output=$(uv run --project "$project" ruff check --fix "$file" 2>&1) && exit 0
printf '%s\n' "$output"
exit 2
