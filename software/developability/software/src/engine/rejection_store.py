"""Rejection log: one row per (clonotype, objective) pair a step attempted. The reason is empty
on pass.

A pair that an earlier step already rejected gets no row here. The reduce in `main.tpl.tengo` takes
the first non-empty reason per (clonotype, objective) pair, in step order. A second row for that
pair would count it twice.
"""

import csv
import io
from pathlib import Path

# `detail` holds the measurement behind the reason where the step has one: the caught
# exception message on a `backend-failed` row, the failed check on a design-gate row.
#
# `rejectedType` says what was lost. `PARENT_REJECTED` means the parent shipped no variant at
# all. `VARIANT_REJECTED` means it shipped some, and one objective's own variant was the thing
# turned away — a row that would otherwise read as a parent this run never designed.
PARENT_REJECTED = "parent"
VARIANT_REJECTED = "variant"

COLUMNS = ["clonotypeKey", "reason", "detail", "rejectedType", "objective"]


def write_rejections(path: str, rows: list[tuple[str, str, str, str, str]]) -> None:
    """`rows` is `(clonotype_key, reason, detail, rejected_type, objective)`, reason and detail
    `""` on a pass."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    for clonotype_key, reason, detail, rejected_type, objective in rows:
        writer.writerow([clonotype_key, reason, detail, rejected_type, objective])
    Path(path).write_text(buf.getvalue())


def read_rejections(path: str) -> list[tuple[str, str, str, str, str]]:
    with Path(path).open(newline="") as fh:
        return [
            (
                row["clonotypeKey"],
                row["reason"],
                row["detail"],
                row.get("rejectedType") or PARENT_REJECTED,
                row.get("objective") or "",
            )
            for row in csv.DictReader(fh, delimiter="\t")
        ]
