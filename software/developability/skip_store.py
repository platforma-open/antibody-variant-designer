"""Read/write for a step's skip TSV: `clonotypeKey`, `reason`.

One row per clonotype the step **attempted**, with an empty reason when it
passed. A clonotype whose reason a predecessor already named is absent
rather than empty, so summing the five files counts it once — the reduce in
`main.tpl.tengo` takes the first non-empty reason in step order, and a
second row for an already-skipped clonotype would double-count it.

This is what keeps failure attribution per clonotype now that one exec
covers the whole dataset: the failure unit stays the reporting unit, moved
from a file per invocation to a row per antibody.
"""

import csv
import io
from pathlib import Path

COLUMNS = ["clonotypeKey", "reason"]


def write_skips(path: str, rows: list[tuple[str, str]]) -> None:
    """`rows` is `(clonotype_key, reason)`, reason `""` on pass."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    for clonotype_key, reason in rows:
        writer.writerow([clonotype_key, reason])
    Path(path).write_text(buf.getvalue())


def read_skips(path: str) -> list[tuple[str, str]]:
    with Path(path).open(newline="") as fh:
        return [
            (row["clonotypeKey"], row["reason"])
            for row in csv.DictReader(fh, delimiter="\t")
        ]
