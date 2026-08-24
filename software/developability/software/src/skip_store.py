"""Skip log: one row per clonotype the step attempted. The reason is empty on pass.

A clonotype that an earlier step already skipped gets no row here. The reduce in
`main.tpl.tengo` takes the first non-empty reason in step order. A second row for
that clonotype would count it twice.
"""

import csv
import io
from pathlib import Path

# `detail` holds the caught exception message on a `backend-failed` row, because
# the exception differs from row to row. Every other reason is a named condition,
# and the operator docs explain each one in full. A copy here would drift.
COLUMNS = ["clonotypeKey", "reason", "detail"]


def write_skips(path: str, rows: list[tuple[str, str, str]]) -> None:
    """`rows` is `(clonotype_key, reason, detail)`, reason and detail `""` on pass."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    for clonotype_key, reason, detail in rows:
        writer.writerow([clonotype_key, reason, detail])
    Path(path).write_text(buf.getvalue())


def read_skips(path: str) -> list[tuple[str, str, str]]:
    with Path(path).open(newline="") as fh:
        return [
            (row["clonotypeKey"], row["reason"], row["detail"])
            for row in csv.DictReader(fh, delimiter="\t")
        ]
