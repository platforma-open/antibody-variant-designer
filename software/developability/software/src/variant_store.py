"""Read/write for `variants.tsv`, written by `ranking.py` — the block's
final artifact, the one file downstream imports as a PFrame.

One dataset-wide file for the whole run, so every row carries its axis
values as columns: `clonotypeKey` for the parent and `variantKey` for the
variant. `xsv.importFile` builds each axis from a column, and there is
nowhere else a per-row axis value could come from.

**The run carries no column here.** Run identity is a spec domain the
workflow applies to the emitted columns after this process has run, so no
block id reaches it — which is what lets two blocks over one dataset share
this exec instead of computing the same variants twice.

`variantKey` is a per-parent ordinal — `v01`, `v02`, … in the rank order
`ranking.rank_variants` fixed, zero-padded to two digits. The key therefore
depends on that per-parent order: a settings change that reorders one
parent's variants renumbers them, which is the CSV's contract, not an
accident — the CSV is a per-run artifact, never a stable cross-run
identifier.

`rank` itself is written twice. `append_variants_tsv` first writes each
parent's own local position — the value `variantKey` is fixed from, and the
only one available while the batch loop is still mid-run. `rewrite_global_rank`
then reads the finished file back and overwrites `rank` with one ordinal
across every parent's survivors, by the same order `ranking.rank_variants`
already sorts by. `variantKey` is not touched by that second pass — it stays
the per-parent identifier `append_variants_tsv` already fixed
(`088-decision-rank-becomes-a-global-ordinal-via-a-second-pass`).

`parentRank` carries the value `rank` used to hold before that decision: the
per-parent-only position, fixed once by `ranking.rank_variants` and never
touched again — the Variants page's own column for "best pick of this
antibody", now that `rank` answers "best pick of the whole run" instead.
"""

import csv
import io
from dataclasses import dataclass, replace
from pathlib import Path

OBJECTIVE = "liability"

TSV_COLUMNS = [
    "clonotypeKey",
    "variantKey",
    "rank",
    "parentRank",
    "objective",
    "chain",
    "addressedTarget",
    "changedPositions",
    "variantSequence",
    "structuralTolerance",
    "worstConfidence",
    "bindingRisk",
    "lowConfidenceWarning",
    "status",
]


@dataclass(frozen=True)
class Variant:
    rank: int
    parent_rank: int
    chain: str  # "H" for a VHH, "H,L" for a paired Fv
    addressed_target: str
    changed_positions: str
    variant_sequence: str
    structural_tolerance: float
    worst_confidence_angstroms: float | None
    binding_risk: str  # "Low" | "Medium" | "High"
    low_confidence_warning: bool
    status: str


def _tsv_value(value) -> str:
    return "" if value is None else str(value)


def _low_confidence_warning_str(variant: Variant) -> str:
    return "yes" if variant.low_confidence_warning else "no"


def variant_key(parent_rank: int) -> str:
    """The per-parent ordinal variant axis value — `v01`, `v02`, … in that
    parent's own local rank order. No hash, no `blockId`: uniqueness across
    parents already comes from pairing this with `clonotypeKey`, and run
    identity is a domain on the emitted axis rather than an ingredient
    folded into this one."""
    return f"v{parent_rank:02d}"


def write_variants_header(path: str) -> None:
    """Start the run's one dataset-wide file, before the batch loop, so an
    empty pdb_index still leaves a header-only TSV."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def _row(clonotype_key: str, variant_key_str: str, v: Variant) -> list:
    """One TSV row, in `TSV_COLUMNS` order. Shared by `append_variants_tsv`
    and `rewrite_global_rank`, which write the same columns from two
    different moments — mid-run with the local rank, and once more with the
    global one — but never the variant key or `parent_rank`, which only the
    first ever computes."""
    return [
        clonotype_key,
        variant_key_str,
        v.rank,
        v.parent_rank,
        OBJECTIVE,
        v.chain,
        v.addressed_target,
        v.changed_positions,
        v.variant_sequence,
        v.structural_tolerance,
        _tsv_value(v.worst_confidence_angstroms),
        v.binding_risk,
        _low_confidence_warning_str(v),
        v.status,
    ]


def append_variants_tsv(path: str, clonotype_key: str, variants: list[Variant]) -> None:
    """Append one parent's ranked variants, computing each row's
    `variantKey` here from `parent_rank` — this is the one place a variant's
    local rank and its parent are both in hand at once. `rewrite_global_rank`
    overwrites `rank` afterwards; it never touches `variantKey` or
    `parent_rank`."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    for v in variants:
        writer.writerow(_row(clonotype_key, variant_key(v.parent_rank), v))
    with Path(path).open("a") as fh:
        fh.write(buf.getvalue())


def read_variants_tsv(path: str) -> list[tuple[str, str, Variant]]:
    """`(clonotype_key, variant_key, variant)` per row, in file order.
    `objective` round-trips through the file but not through `Variant` — it
    is the fixed constant `OBJECTIVE`."""
    variants = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            variants.append(
                (
                    row["clonotypeKey"],
                    row["variantKey"],
                    Variant(
                        rank=int(row["rank"]),
                        parent_rank=int(row["parentRank"]),
                        chain=row["chain"],
                        addressed_target=row["addressedTarget"],
                        changed_positions=row["changedPositions"],
                        variant_sequence=row["variantSequence"],
                        structural_tolerance=float(row["structuralTolerance"]),
                        worst_confidence_angstroms=(
                            float(row["worstConfidence"]) if row["worstConfidence"] else None
                        ),
                        binding_risk=row["bindingRisk"],
                        low_confidence_warning=row["lowConfidenceWarning"] == "yes",
                        status=row["status"],
                    ),
                )
            )
    return variants


def rewrite_global_rank(path: str) -> None:
    """Read every surviving variant back, sort once across the whole run by
    the same key `ranking.rank_variants` already sorts by inside one
    parent — structural tolerance descending, then `changed_positions` —
    with `(clonotypeKey, variantKey)` added so two parents' tied variants
    still order deterministically, then overwrite `rank` 1..N over the
    file. `variantKey` and `parent_rank` are untouched: both already carry
    the per-parent position `append_variants_tsv` fixed, and `_row` never
    recomputes either.

    Called once, after the whole batch loop finishes writing this file —
    not from inside it — so the peak memory this holds is the run's
    already-filtered survivor set, never the per-antibody working state the
    loop itself is bounded by (`088-decision-rank-becomes-a-global-ordinal-via-a-second-pass`)."""
    rows = read_variants_tsv(path)
    if not rows:
        return
    rows.sort(
        key=lambda row: (-row[2].structural_tolerance, row[2].changed_positions, row[0], row[1])
    )

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    for global_rank, (clonotype_key, variant_key_str, v) in enumerate(rows, start=1):
        writer.writerow(_row(clonotype_key, variant_key_str, replace(v, rank=global_rank)))
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n" + buf.getvalue())
