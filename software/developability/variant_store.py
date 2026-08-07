"""Read/write for `variants.tsv`, written by `ranking.py` — the block's
final artifact, the one file downstream imports as a PFrame.

One row per ranked variant: its rank within the parent it was built for,
the liability it addresses, the fixed-spelling changed-positions string,
the designed sequence, the two carried-forward metrics, the binding-risk
band and the low-confidence warning, and the constant status string every
row carries. Nothing here is keyed by parent — one file already holds
exactly one parent's variants, the same convention every other boundary
file in this package follows.
"""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

TSV_COLUMNS = [
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


def write_variants_tsv(path: str, variants: list[Variant]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(TSV_COLUMNS)
    for v in variants:
        writer.writerow(
            [
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
    Path(path).write_text(buf.getvalue())


def read_variants_tsv(path: str) -> list[Variant]:
    variants = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            variants.append(
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
                )
            )
    return variants
