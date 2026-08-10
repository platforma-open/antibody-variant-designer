"""Read/write for `variants.tsv`, written by `ranking.py` — the block's
final artifact, the one file downstream imports as a PFrame.

One dataset-wide file for the whole run, so every row carries the two axis
values as columns: `clonotypeKey` for the parent and `variantKey` for the
variant. `xsv.importFile` builds each axis from a column, and there is
nowhere else a per-row axis value could come from.

`variantKey` is content-addressed — `hash(clonotypeKey + blockId +
changedPositions)`. Hashing the parent and the block alone would collide
across one parent's variants, since they share both; `changedPositions` is
the ingredient that distinguishes them, and it is already the canonical,
order-fixed rendering of the edits. Re-running the block on the same
antibody with the same settings therefore reproduces the same key.
"""

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

TSV_COLUMNS = [
    "clonotypeKey",
    "variantKey",
    "rank",
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


def variant_key(clonotype_key: str, block_id: str, changed_positions: str) -> str:
    """The content-addressed variant axis value. Truncated to 16 hex
    characters: long enough that a collision across one run's variants is
    not a practical concern, short enough to stay readable in a table cell
    and in the CSV's `variantId`."""
    digest = hashlib.sha256(
        "\x00".join([clonotype_key, block_id, changed_positions]).encode()
    )
    return digest.hexdigest()[:16]


def write_variants_header(path: str) -> None:
    """Start the run's one dataset-wide file, before the batch loop, so an
    empty roster still leaves a header-only TSV."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def append_variants_tsv(
    path: str, clonotype_key: str, block_id: str, variants: list[Variant]
) -> None:
    """Append one parent's ranked variants, computing each row's
    `variantKey` here — this is the one place all three hash ingredients
    are in hand at once."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    for v in variants:
        writer.writerow(
            [
                clonotype_key,
                variant_key(clonotype_key, block_id, v.changed_positions),
                v.rank,
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
    """`(clonotype_key, variant_key, variant)` per row, in file order."""
    variants = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            variants.append(
                (
                    row["clonotypeKey"],
                    row["variantKey"],
                    Variant(
                        rank=int(row["rank"]),
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
