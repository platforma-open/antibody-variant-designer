"""Read/write for pdb_index.tsv, the index batch entrypoints loop over.

The workflow writes one row per clonotype (once per run, sorted by key). This is
the only mapping of clonotype key to filename. Clonotype keys are JSON arrays
(illegal as filenames), so per-clonotype artifacts are named after the staged
PDB's stem instead.
"""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ["clonotypeKey", "filename"]


@dataclass(frozen=True)
class Entry:
    """One pdb_index row: which clonotype, and the bare filename of its staged
    PDB. `stem` is what every per-clonotype artifact is named after."""

    clonotype_key: str
    filename: str

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


def write_index(path: str, entries: list[Entry]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    for entry in entries:
        writer.writerow([entry.clonotype_key, entry.filename])
    Path(path).write_text(buf.getvalue())


def read_index(path: str) -> list[Entry]:
    """In file order, which the workflow writes sorted by key — every step
    iterates in this order so a batch's exec input stays canonical."""
    with Path(path).open(newline="") as fh:
        return [
            Entry(clonotype_key=row["clonotypeKey"], filename=row["filename"])
            for row in csv.DictReader(fh, delimiter="\t")
        ]
