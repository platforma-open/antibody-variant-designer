"""Read/write for a step's skip TSV: `clonotypeKey`, `reason`, `detail`.

One row per clonotype the step **attempted**, with an empty reason when it
passed. A clonotype whose reason a predecessor already named is absent
rather than empty, so summing the five files counts it once — the reduce in
`main.tpl.tengo` takes the first non-empty reason in step order, and a
second row for an already-skipped clonotype would double-count it.

This is what keeps failure attribution per clonotype now that one exec
covers the whole dataset: the failure unit stays the reporting unit, moved
from a file per invocation to a row per antibody.

`detail` is free text, empty for every reason except `backend-failed`: the
six other reasons are named, deterministic conditions the operator-facing
docs already explain in full, so a per-row repeat of that explanation would
only drift from it. `backend-failed` is `batch.run`'s catch-all for an
exception `process_one` raised, and which exception varies row to row — the
caught message is the only thing that says which.
"""

import csv
import io
from pathlib import Path

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
