"""Shared pytest fixtures for the Antibody Variant Designer test suite.

Module discovery (script source + `tests/` helpers) is wired via
`pyproject.toml` `[tool.pytest.ini_options].pythonpath`, so no manual
`sys.path` mutation is needed here."""

import json
from pathlib import Path

import pytest

from engine import parent_clonotypes


class StagedBatch:
    """One run's workdir, laid out the way `main.tpl.tengo` stages it: PDB blobs
    under `pdbs/`, and a directory per boundary artifact.

    Every batch CLI test builds its input through this, so a test says which
    antibodies are in the run and nothing about paths."""

    def __init__(self, root: Path):
        self.root = root
        self.pdb_dir = root / "pdbs"
        self.pdb_dir.mkdir(parents=True, exist_ok=True)

    def add(self, clonotype_key: str, pdb_text: str = ""):
        """Stage one antibody and return its parent clonotype. The staged filename
        is the clonotype key, which is how every entrypoint recovers them."""
        entry = parent_clonotypes.ParentClonotype(clonotype_key=clonotype_key)
        (self.pdb_dir / entry.filename).write_text(pdb_text)
        return entry

    def dir(self, name: str) -> str:
        """A named directory inside the workdir, created on demand — used
        for both a step's output dir and any input dir a test pre-populates."""
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def path(self, name: str) -> str:
        """A file path inside the workdir, for the dataset-wide TSVs and the
        per-step rejection files."""
        return str(self.root / name)

    def definitions(self, liabilities: list[dict]) -> str:
        """Stage `definitions.json` in the taxonomy package's own document
        shape and return its path. The package writes an object, not the flat
        list the detectors iterate, so a fixture that wrote the list would
        pass while the shipped file fails."""
        path = self.root / "definitions.json"
        path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "liabilities": liabilities,
                    "fixabilityWeights": {},
                }
            )
        )
        return str(path)


@pytest.fixture
def batch(tmp_path):
    return StagedBatch(tmp_path)
