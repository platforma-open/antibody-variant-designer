#!/usr/bin/env bash
# Regenerate `src/requirements-antifold.txt` from pyproject.toml, the single
# source of truth.
#
# It lands inside `src/` — which `stage-build.sh` copies into the artifact root
# — because the python artifact schema takes a `requirements` path relative to
# that root, and `uv` is not a toolset the package builder accepts — `pip` is
# the only one (`pythonToolsets = ["pip"]`).
#
# --no-deps: top-level pins only. The runenv supplies transitive deps for the
# offline (--no-index) install, so a requirements file must not pin a full
# closure.
set -euo pipefail
cd "$(dirname "$0")/.."

# --python-version pins resolution to the runenv's interpreter rather than
# whatever the developer happens to be running. torch 2.2 ships no wheel past
# cp312, so compiling on a newer local python fails outright.
PY_VERSION=3.12

# `--extra antifold`, because `[project].dependencies` is empty here on purpose
# — see the comment on it in pyproject.toml. The name keeps the `-antifold`
# suffix so the file stays distinct from the developability one that
# `stage-build.sh` copies alongside it.
uv pip compile pyproject.toml --no-deps --no-annotate --extra antifold \
  --python-version "$PY_VERSION" \
  --custom-compile-command "pnpm deps:export" -o src/requirements-antifold.txt
