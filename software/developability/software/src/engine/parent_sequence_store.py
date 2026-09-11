"""Appends one parent's unedited V-domain sequence to the run's one parent-sequence file.

The string every `variantSequence` is a substitution of, in the same rendering order, so a
reader can set the two side by side and see only the designed positions differ.

One row per parent rather than a column on every variant row: the sequence is a fact about
the parent, and repeating a 480-character field on each of its variants would grow the
design step's peak by more than the whole re-rank budget allows for a variant.

Mirrors `humanness_store.py`'s append-per-parent file.
"""

import csv
import io
from pathlib import Path

TSV_COLUMNS = [
    "clonotypeKey",
    "parentSequence",
]


def write_parent_sequences_header(path: str) -> None:
    """Starts the run's one dataset-wide file with the header row alone.

    Called once, before the batch loop, so an empty run still leaves a header-only file
    rather than no file."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def append_parent_sequences_tsv(path: str, clonotype_key: str, parent_sequence: str) -> None:
    """Appends this parent's one row.

    Appends rather than returning a row to collect, for the reason `humanness_store.py`'s
    `append_humanness_tsv` gives: one file holds the whole dataset and the pipeline streams
    one antibody at a time."""
    row_buffer = io.StringIO()
    writer = csv.writer(row_buffer, delimiter="\t", lineterminator="\n")
    writer.writerow([clonotype_key, parent_sequence])
    with Path(path).open("a") as out_file:
        out_file.write(row_buffer.getvalue())
