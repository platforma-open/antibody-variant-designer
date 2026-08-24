"""The loop each of the three entrypoints runs its per-antibody step under.

The loop is sequential: it releases one antibody's state before the next starts. This holds the
process at one antibody's memory footprint, plus the tolerance step's model. Each step's Tengo
recipe requests RAM for exactly that footprint, so accumulating state across iterations would
force it to request more.
"""

import sys
from collections.abc import Callable

import pdb_index_store
import skip_store


def run(
    entries: list[pdb_index_store.Entry],
    process_one: Callable[[pdb_index_store.Entry], str | None],
    out_skip: str,
    error_reason: str | None = None,
) -> int:
    """`error_reason`, when set, turns a `process_one` exception into a skip row for that
    antibody's clonotype, instead of raising. Only the tolerance step passes it, because AntiFold
    exits 0 on its own swallowed exceptions.

    `process_one` returning `None` means an earlier step already named this antibody's skip
    reason. The loop then writes no row, so the skip files count the antibody once."""
    rows: list[tuple[str, str, str]] = []
    for entry in entries:
        try:
            reason = process_one(entry)
        except Exception as exc:
            if error_reason is None:
                raise
            print(f"{entry.clonotype_key}: {exc}", file=sys.stderr)
            rows.append((entry.clonotype_key, error_reason, str(exc)))
            continue
        if reason is not None:
            rows.append((entry.clonotype_key, reason, ""))
    skip_store.write_skips(out_skip, rows)
    return 0
