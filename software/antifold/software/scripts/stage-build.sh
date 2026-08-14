#!/usr/bin/env bash
# Assemble `build/`, the one directory both of this package's artifacts point
# at — the python `root` and the docker `context`.
#
# It exists because `antifold.py` imports four modules that belong to the
# developability package (`batch`, `residue_store`, `roster`,
# `tolerance_store`, and `skip_store` through `batch`), while the python
# artifact schema takes a single `root` directory and offers no second root,
# no include list, and no artifact-to-artifact dependency. One directory must
# therefore hold source from both packages, and a copy is the only way to
# build it.
#
# The whole sibling source tree is copied rather than a named list of five
# modules: a list drifts the moment `antifold.py` imports one more module, and
# the light source is a few tens of KiB against a torch install of about 2 GiB.
#
# `dependencies` in package.json (not `devDependencies`) plus turbo's
# `dependsOn: ["^build"]` is what guarantees the sibling exists before this
# runs.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVELOPABILITY_SRC=../../developability/software/src

rm -rf build
mkdir -p build
cp -R "$DEVELOPABILITY_SRC"/. build/
# The light requirements file arrives with that copy and is never installed
# here — this package names `requirements-antifold.txt`. Removing it keeps the
# archive from carrying a dependency list that does not describe it.
rm -f build/requirements.txt
cp -R src/. build/
# A local test run leaves `__pycache__` in both source trees, and a plain copy
# would ship it. Bytecode compiled against the developer's interpreter has no
# business in the archive.
find build -type d -name __pycache__ -prune -exec rm -rf {} +
