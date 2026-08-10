#!/usr/bin/env bash
# Regenerate the two requirements files inside the artifact root from
# pyproject.toml, the single source of truth.
#
# They live inside `developability/` rather than beside pyproject.toml
# because the python artifact schema takes a `requirements` path relative to
# the artifact root, and `uv` is not a toolset the package builder accepts —
# `pip` is the only one (`pythonToolsets = ["pip"]`). So pyproject stays
# authoritative for humans and uv, and these two files are the build inputs
# pl-pkg consumes.
#
# --no-deps: top-level pins only. The runenv supplies transitive deps for the
# offline (--no-index) install, so a requirements file must not pin a full
# closure.
set -euo pipefail
cd "$(dirname "$0")/.."

# --python-version pins resolution to the runenv's interpreter rather than
# whatever the developer happens to be running. torch 2.2 ships no wheel
# past cp312, so compiling on a newer local python fails outright.
PY_VERSION=3.12

# One file per extra, and `[project].dependencies` is empty so neither set
# inherits the other's packages. uv has no `--only-extra`, so an empty base
# is what keeps the two artifacts disjoint.
uv pip compile pyproject.toml --no-deps --no-annotate --extra light \
  --python-version "$PY_VERSION" \
  --custom-compile-command "pnpm deps:export" -o developability/requirements.txt

uv pip compile pyproject.toml --no-deps --no-annotate --extra antifold \
  --python-version "$PY_VERSION" \
  --custom-compile-command "pnpm deps:export" -o developability/requirements-antifold.txt
