"""The declared entrypoints and the files that have a `main` must be the
same set, in both software packages.

Three layers spell each step's name — `package.json`'s `block-software`
entrypoint key, the tengo lib that imports it, and the python file the `cmd`
runs — and nothing typechecks across those boundaries. A key renamed without
its file, a file renamed without its key, or a `main` grown on a module that
no entrypoint runs all pass lint and pass every other test here, then fail
in the workflow.

This is also what keeps a flat source root readable: with the sets equal, a
filename answers "is this an entrypoint" on its own, and no module has to
say so in prose.

The antifold package's artifact root is `build/`, assembled by
`stage-build.sh` from two source trees and absent until it runs, so the file
check resolves against `src/` — the tree the entrypoint is authored in.
"""

import json
from pathlib import Path

import pytest

_SOFTWARE_DIR = Path(__file__).resolve().parents[3]

# One row per software package: where its manifest is, and the tree its
# entrypoints are authored in.
PACKAGES = [
    ("developability", _SOFTWARE_DIR / "developability" / "software"),
    ("antifold", _SOFTWARE_DIR / "antifold" / "software"),
]

MAIN_GUARD = 'if __name__ == "__main__":'


def _declared_entrypoints(package_dir: Path) -> dict[str, set[str]]:
    """Each entrypoint key mapped to the python filenames its variants run.
    Every variant of one key (`binary`, `docker`) must name the same file, so
    the value is a set the caller asserts is a singleton."""
    manifest = json.loads((package_dir / "package.json").read_text())
    entrypoints = manifest["block-software"]["entrypoints"]
    return {
        key: {Path(variant["cmd"][-1]).name for variant in variants.values()}
        for key, variants in entrypoints.items()
    }


def _files_with_a_main(source_root: Path) -> set[str]:
    """Top-level modules only — the vendored tree carries its own upstream
    CLI, which no entrypoint of ours runs."""
    return {path.name for path in source_root.glob("*.py") if MAIN_GUARD in path.read_text()}


@pytest.mark.parametrize(("name", "package_dir"), PACKAGES, ids=[p[0] for p in PACKAGES])
class TestEntrypointParity:
    def test_every_variant_of_one_key_runs_the_same_file(self, name, package_dir):
        for key, filenames in _declared_entrypoints(package_dir).items():
            assert len(filenames) == 1, f"{name}: {key} runs more than one file: {filenames}"

    def test_the_filename_is_the_key_with_underscores(self, name, package_dir):
        for key, filenames in _declared_entrypoints(package_dir).items():
            expected = f"{key.replace('-', '_')}.py"
            assert filenames == {expected}, (
                f"{name}: entrypoint {key} runs {filenames}, expected {expected} — "
                "the key and the file it runs must spell the step the same way"
            )

    def test_the_declared_files_are_exactly_the_files_with_a_main(self, name, package_dir):
        declared = {
            filename
            for names in _declared_entrypoints(package_dir).values()
            for filename in names
        }
        assert _files_with_a_main(package_dir / "src") == declared, (
            f"{name}: a module with a `main` that no entrypoint runs, or an entrypoint "
            "naming a file that has none"
        )
