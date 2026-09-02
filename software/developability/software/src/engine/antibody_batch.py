"""The loop each of the three entrypoints runs its per-antibody step under.

The loop is sequential: it releases one antibody's state before the next starts. This holds the
process at one antibody's memory footprint, plus the tolerance step's model. Each step's Tengo
recipe requests RAM for exactly that footprint, so accumulating state across iterations would
force it to request more.
"""

import sys
from collections.abc import Callable

from engine import parent_clonotypes, rejection_store


def process_every_parent(
    parents: list[parent_clonotypes.ParentClonotype],
    process_one: Callable[[parent_clonotypes.ParentClonotype], tuple[str, str, str] | None],
    out_rejected: str,
    error_reason: str | None = None,
) -> int:
    """`error_reason`, when set, turns a `process_one` exception into a rejection row for that
    antibody's clonotype, instead of raising. Only the tolerance step passes it, because AntiFold
    exits 0 on its own swallowed exceptions.

    `process_one` returns `(reason, detail, rejected_type)`, reason and detail `""` on a pass.
    `detail` carries the measurement behind the reason where a step has one, and
    `rejected_type` is one of `rejection_store`'s two values. Every step writes all three, so no
    caller has to remember which reasons come with a detail.

    `process_one` returning `None` means an earlier step already named this antibody's rejection
    reason. The loop then writes no row, so the rejection files count the antibody once."""
    rejection_rows: list[tuple[str, str, str, str]] = []
    for parent in parents:
        try:
            outcome = process_one(parent)
        except Exception as exc:
            if error_reason is None:
                raise
            print(f"{parent.clonotype_key}: {exc}", file=sys.stderr)
            rejection_rows.append(
                (parent.clonotype_key, error_reason, str(exc), rejection_store.PARENT_REJECTED)
            )
            continue
        if outcome is not None:
            reason, detail, rejected_type = outcome
            rejection_rows.append((parent.clonotype_key, reason, detail, rejected_type))
    rejection_store.write_rejections(out_rejected, rejection_rows)
    return 0
