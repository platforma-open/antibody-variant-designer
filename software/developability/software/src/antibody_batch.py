"""The loop each of the three entrypoints runs its per-antibody function
under.

One exec now covers the whole dataset, so this is where per-clonotype
attribution lives: each iteration contributes at most one skip row, and the
row names the antibody. The loop is deliberately **sequential** — an
iteration's state is released before the next one starts, and rows are
appended as they are produced rather than accumulated — because that is
what holds the peak footprint at one antibody (plus, for the tolerance
step, the model). Each step's Tengo recipe asks for exactly that shape — a
fixed base plus one antibody — so a change that started accumulating across
iterations would have to raise the RAM those recipes request.

A `process_one` returning `None` means the antibody's inputs are absent
because an earlier step already named its reason. No row is written for it,
so summing the three skip files counts it once rather than twice.
"""

import sys
from collections.abc import Callable

import pdb_index
import skip_store


def run(
    entries: list[pdb_index.Entry],
    process_one: Callable[[pdb_index.Entry], str | None],
    out_skip: str,
    error_reason: str | None = None,
) -> int:
    """Loop `entries`, collect one skip row per attempted antibody, write
    the skip TSV.

    `error_reason` set makes an exception from `process_one` recoverable:
    it is reported against its own clonotype and the loop continues with
    the rest. Only the tolerance step passes it, because AntiFold swallows
    its own exceptions and exits 0 — inside this loop is the one place a
    fault can still be named after the antibody that caused it. Every other
    step leaves exceptions propagating, so a bug fails the exec loudly
    instead of degrading into a dataset of skip rows."""
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
