#!/usr/bin/env bash
# Assemble `build/`, the one directory both of this package's artifacts point
# at — the python `root` and the docker `context`.
#
# It exists because `read_tolerance.py` imports the developability package's
# `engine`, while the python artifact schema takes a single `root` directory
# and offers no second root, no include list, and no artifact-to-artifact
# dependency. One directory must therefore hold source from both packages, and
# a copy is the only way to build it.
#
# `engine` is copied whole, and it is the only thing copied from the sibling:
# the shared package is a directory rather than a list of module names, so it
# cannot drift when `read_tolerance.py` reaches for one more module, and the
# sibling's own entrypoints stay out of an archive that never runs them.
#
# `dependencies` in package.json (not `devDependencies`) plus turbo's
# `dependsOn: ["^build"]` is what guarantees the sibling exists before this
# runs.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVELOPABILITY_SRC=../../developability/software/src

rm -rf build
mkdir -p build
cp -R "$DEVELOPABILITY_SRC/engine" build/
cp -R src/. build/
# A local test run leaves `__pycache__` in both source trees, and a plain copy
# would ship it. Bytecode compiled against the developer's interpreter has no
# business in the archive.
find build -type d -name __pycache__ -prune -exec rm -rf {} +
