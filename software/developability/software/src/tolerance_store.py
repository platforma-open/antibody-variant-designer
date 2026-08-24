"""Read/write for `tolerance.tsv`, written by `read_tolerance.py`.

One row per position AntiFold scored. `posins` is the IMGT label, matching
`residue_store.Residue.imgt`. `perplexity` is entropy in bits, `2^H₂(p)`, in the range `[1,20]`.
The twenty amino-acid columns hold log-probabilities, never the raw
logits `save_flag=False` returns. A later step therefore never has to
remember which base it is comparing against.
"""

import csv
import io
from pathlib import Path

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
COLUMNS = ["chain", "posins", "perplexity", *AMINO_ACIDS]


def write_tolerance_tsv(path: str, rows: list[dict]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow([row[c] for c in COLUMNS])
    Path(path).write_text(buf.getvalue())


def read_tolerance_tsv(path: str) -> dict[tuple[str, str], dict]:
    """Keyed `(chain, posins)`, the same pair `structure.index_residues`
    assigns. A later step looks a row up with the residue it already
    holds. There is no second join format to remember."""
    lookup: dict[tuple[str, str], dict] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (row["chain"], row["posins"])
            lookup[key] = {
                "perplexity": float(row["perplexity"]),
                "logProbs": {aa: float(row[aa]) for aa in AMINO_ACIDS},
            }
    return lookup
