"""Read/write for `variants.tsv`, written by `ranking.py` — the block's
final artifact, the one file downstream imports as a PFrame.

One dataset-wide file for the whole run, so every row carries the three
axis values as columns: `clonotypeKey` for the parent, `variantKey` for the
variant, and `blockId` for the run. `xsv.importFile` builds each axis from
a column, and there is nowhere else a per-row axis value could come from.

`variantKey` is a per-parent ordinal — `v01`, `v02`, … in the rank order
`ranking.rank_variants` fixed, zero-padded to two digits. The key therefore
depends on rank: a settings change that reorders one parent's variants
renumbers them, which is the CSV's contract, not an accident — the CSV is a
per-run artifact, never a stable cross-run identifier.
"""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

OBJECTIVE = "liability"

TSV_COLUMNS = [
    "clonotypeKey",
    "variantKey",
    "blockId",
    "rank",
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


def variant_key(rank: int) -> str:
    """The per-parent ordinal variant axis value — `v01`, `v02`, … in rank
    order. No hash, no `blockId`: uniqueness across parents already comes
    from pairing this with `clonotypeKey`, and `pl7.app/blockId` is its own
    axis rather than an ingredient folded into this one."""
    return f"v{rank:02d}"


def write_variants_header(path: str) -> None:
    """Start the run's one dataset-wide file, before the batch loop, so an
    empty roster still leaves a header-only TSV."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def append_variants_tsv(
    path: str, clonotype_key: str, block_id: str, variants: list[Variant]
) -> None:
    """Append one parent's ranked variants, computing each row's
    `variantKey` here from its rank — this is the one place a variant's
    rank and its parent are both in hand at once."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    for v in variants:
        writer.writerow(
            [
                clonotype_key,
                variant_key(v.rank),
                block_id,
                v.rank,
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
        )
    with Path(path).open("a") as fh:
        fh.write(buf.getvalue())


def read_variants_tsv(path: str) -> list[tuple[str, str, Variant]]:
    """`(clonotype_key, variant_key, variant)` per row, in file order.
    `blockId` and `objective` round-trip through the file but not through
    `Variant` — `blockId` is the caller's own argument to
    `append_variants_tsv`, not a per-variant property, and `objective` is
    the fixed constant `OBJECTIVE`."""
    variants = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            variants.append(
                (
                    row["clonotypeKey"],
                    row["variantKey"],
                    Variant(
                        rank=int(row["rank"]),
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
