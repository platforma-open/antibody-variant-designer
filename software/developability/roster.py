"""Read/write for `pdb_index.tsv`, the roster every batch entrypoint loops
over.

Two columns, `clonotypeKey` and `filename`, written once per run by the
workflow in sorted-key order. It is the only place a clonotype key and a
filename are related, which is why every step reads it: a clonotype key is
a JSON-encoded array and is not a legal filename, so no step may ever
derive a path from a key directly. Each per-clonotype artifact is named
after the staged PDB's `stem` instead — the filename with its extension
removed.
"""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ["clonotypeKey", "filename"]


@dataclass(frozen=True)
class Entry:
    """One roster row: which clonotype, and the bare filename of its staged
    PDB. `stem` is what every per-clonotype artifact is named after."""

    clonotype_key: str
    filename: str

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


def write_roster(path: str, entries: list[Entry]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    for entry in entries:
        writer.writerow([entry.clonotype_key, entry.filename])
    Path(path).write_text(buf.getvalue())


def read_roster(path: str) -> list[Entry]:
    """In file order, which the workflow writes sorted by key — every step
    iterates in this order so a batch's exec input stays canonical."""
    with Path(path).open(newline="") as fh:
        return [
            Entry(clonotype_key=row["clonotypeKey"], filename=row["filename"])
            for row in csv.DictReader(fh, delimiter="\t")
        ]
