#!/usr/bin/env bash
# Regenerate `src/requirements.txt` from pyproject.toml, the single source of
# truth.
#
# It lands inside the artifact root rather than beside pyproject.toml because
# the python artifact schema takes a `requirements` path relative to that root,
# and `uv` is not a toolset the package builder accepts — `pip` is the only one
# (`pythonToolsets = ["pip"]`). So pyproject stays authoritative for humans and
# uv, and this file is the build input pl-pkg consumes.
#
# --no-deps: top-level pins only. The runenv supplies transitive deps for the
# offline (--no-index) install, so a requirements file must not pin a full
# closure.
set -euo pipefail
cd "$(dirname "$0")/.."

# --python-version pins resolution to the runenv's interpreter rather than
# whatever the developer happens to be running.
PY_VERSION=3.12

# No `--extra`: this package's whole runtime set is `[project].dependencies`.
# The wrapper package exports its own file, from its own pyproject.
uv pip compile pyproject.toml --no-deps --no-annotate \
  --python-version "$PY_VERSION" \
  --custom-compile-command "pnpm deps:export" -o src/requirements.txt
