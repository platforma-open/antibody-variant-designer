"""Read/write for the tolerance keyed artifact, written by `read_tolerance.py`: one flat TSV
carrying every parent, a leading `clonotypeKey` column grouping each parent's rows.

One row per position AntiFold scored. `imgt` is the IMGT label, matching
`residue_index.Residue.imgt`. `perplexity` is entropy in bits, `2^H₂(p)`, in the range `[1,20]`.
The twenty amino-acid columns hold log-probabilities, never the raw
logits `save_flag=False` returns. A later step therefore never has to
remember which base it is comparing against.
"""

import csv
from pathlib import Path

from engine import keyed_artifact

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
COLUMNS = [keyed_artifact.KEY_COLUMN, "chain", "imgt", "perplexity", *AMINO_ACIDS]
_ROW_COLUMNS = COLUMNS[1:]


class ToleranceWriter:
    """The one tolerance artifact, header written once, rows appended per parent."""

    def __init__(self, path: str) -> None:
        self._path = path
        with Path(path).open("w", newline="") as fh:
            csv.writer(fh, delimiter="\t", lineterminator="\n").writerow(COLUMNS)

    def write(self, clonotype_key: str, rows: list[dict]) -> None:
        with Path(self._path).open("a", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            for row in rows:
                writer.writerow([clonotype_key, *(row[c] for c in _ROW_COLUMNS)])

    def close(self) -> None:
        pass

    def __enter__(self) -> "ToleranceWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_tolerance(path: str) -> keyed_artifact.KeyedReader:
    return keyed_artifact.read_keyed_tsv(path, _ROW_COLUMNS)


def lookup_from_payload(payload: object) -> dict[tuple[str, str], dict]:
    """Keyed `(chain, imgt)`, the same pair `residue_index.index_residues`
    assigns. A later step looks a row up with the residue it already
    holds. There is no second join format to remember."""
    lookup: dict[tuple[str, str], dict] = {}
    for row in payload:
        key = (row["chain"], row["imgt"])
        lookup[key] = {
            "perplexity": float(row["perplexity"]),
            "logProbs": {aa: float(row[aa]) for aa in AMINO_ACIDS},
        }
    return lookup
