"""Shared pytest fixtures for the Antibody Variant Designer test suite.

Module discovery (script source + `tests/` helpers) is wired via
`pyproject.toml` `[tool.pytest.ini_options].pythonpath`, so no manual
`sys.path` mutation is needed here."""

import json
from pathlib import Path

import pytest

import pdb_index_store


class StagedBatch:
    """One run's workdir, laid out the way `main.tpl.tengo` stages it: PDB
    blobs under `pdbs/`, a `pdb_index.tsv` pdb_index beside them, and a
    directory per boundary artifact.

    Every batch CLI test builds its input through this, so a test says which
    antibodies are in the run and nothing about paths."""

    def __init__(self, root: Path):
        self.root = root
        self.pdb_dir = root / "pdbs"
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[pdb_index_store.Entry] = []

    def add(self, clonotype_key: str, pdb_text: str = "", stem: str | None = None):
        """Stage one antibody and return its pdb_index entry. `stem` defaults
        to the clonotype key, which keeps simple tests readable; pass it
        explicitly when the key is not filename-safe."""
        stem = stem or clonotype_key
        filename = f"{stem}.pdb"
        (self.pdb_dir / filename).write_text(pdb_text)
        entry = pdb_index_store.Entry(clonotype_key=clonotype_key, filename=filename)
        self.entries.append(entry)
        return entry

    def add_without_blob(self, clonotype_key: str, stem: str | None = None):
        """Index an antibody whose PDB never arrived — the upstream block
        leaves no ResourceMap entry at all for a failed clonotype, so this
        is a real input shape, not a corrupted one."""
        stem = stem or clonotype_key
        entry = pdb_index_store.Entry(clonotype_key=clonotype_key, filename=f"{stem}.pdb")
        self.entries.append(entry)
        return entry

    @property
    def index(self) -> str:
        """The pdb_index path, rewritten from the current entries on each read
        so a test may add antibodies in any order before running a CLI."""
        path = self.root / "pdb_index.tsv"
        pdb_index_store.write_index(str(path), self.entries)
        return str(path)

    def dir(self, name: str) -> str:
        """A named directory inside the workdir, created on demand — used
        for both a step's output dir and any input dir a test pre-populates."""
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def path(self, name: str) -> str:
        """A file path inside the workdir, for the dataset-wide TSVs and the
        per-step skip files."""
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
