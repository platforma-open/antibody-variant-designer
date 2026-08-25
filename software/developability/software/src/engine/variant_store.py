"""Reads and writes variants.tsv, the block's final artifact.

Every row carries axis values: clonotypeKey for the parent, variantKey for the
variant. xsv.importFile builds each axis from a column.
"""

import csv
import io
from dataclasses import dataclass, replace
from pathlib import Path

# This file carries no run or block id column.
# The workflow attaches run identity to these columns downstream, once this process finishes.
# That lets two blocks over one dataset share this exec, instead of
# computing the same variants twice.
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0

# The fixed domain each re-rank term normalises over before weighting. A
# per-run min/max would make one variant's rank depend on which other
# variants happen to share its file.
STRUCTURAL_TOLERANCE_DOMAIN = (1.0, 20.0)
HUMANNESS_DOMAIN = (0.0, 100.0)

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
    "humannessScore",
    "worstConfidence",
    "bindingRisk",
    "lowConfidenceWarning",
    "status",
]


@dataclass(frozen=True)
class Variant:
    rank: int  # Global ordinal across the whole run, written by rewrite_global_rank.
    parent_rank: int  # Local rank inside one parent — shown as "best pick of this antibody"
    chain: str  # "H" for a VHH, "H,L" for a paired Fv
    addressed_target: str
    changed_positions: str
    variant_sequence: str
    structural_tolerance: float
    humanness_score: float | None
    worst_confidence_angstroms: float | None
    binding_risk: str  # "Low" | "Medium" | "High"
    low_confidence_warning: bool
    status: str


def _tsv_value(value) -> str:
    return "" if value is None else str(value)


def _low_confidence_warning_str(variant: Variant) -> str:
    return "yes" if variant.low_confidence_warning else "no"


def variant_key(parent_rank: int) -> str:
    """Returns zero-padded v01, v02, … from parent_rank.

    A settings change that changes the order of a parent's variants renumbers
    this value. variantKey identifies
    rows only within one run. Pairing it with clonotypeKey ensures uniqueness
    across parents."""
    return f"v{parent_rank:02d}"


def write_variants_header(path: str) -> None:
    """Writes the header for the run's one dataset-wide file.

    Call this before the batch loop starts. Then an empty `pdb_index`
    still leaves a header-only TSV."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def _row(clonotype_key: str, variant_key_str: str, objective: str, v: Variant) -> list:
    """Returns one TSV row, in `TSV_COLUMNS` order.

    `append_variants_tsv` and `rewrite_global_rank` both call this to write
    the same columns, at two different moments:
    - mid-run, with the local `rank`
    - once more, with the global `rank`

    Only `append_variants_tsv` computes `variantKey` and `parent_rank`.
    `rewrite_global_rank` passes through the values `v` already carries."""
    return [
        clonotype_key,
        variant_key_str,
        v.rank,
        v.parent_rank,
        objective,
        v.chain,
        v.addressed_target,
        v.changed_positions,
        v.variant_sequence,
        v.structural_tolerance,
        _tsv_value(v.humanness_score),
        _tsv_value(v.worst_confidence_angstroms),
        v.binding_risk,
        _low_confidence_warning_str(v),
        v.status,
    ]


def append_variants_tsv(
    path: str, clonotype_key: str, objective: str, variants: list[Variant]
) -> None:
    """Appends one parent's ranked variants to path.

    variantKey is computed here from parent_rank — the only place both the
    variant's rank and its parent clonotype are available. Later,
    rewrite_global_rank overwrites rank with a dataset-wide ordinal without
    touching variantKey or parent_rank."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    for v in variants:
        writer.writerow(_row(clonotype_key, variant_key(v.parent_rank), objective, v))
    with Path(path).open("a") as fh:
        fh.write(buf.getvalue())


def read_variants_tsv(path: str) -> list[tuple[str, str, str, Variant]]:
    """`(clonotype_key, variant_key, objective, variant)` per row, in file
    order. `objective` round-trips through the file but not through
    `Variant` — it is the objective that produced the row, carried
    alongside it rather than on it."""
    variants = []
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            variants.append(
                (
                    row["clonotypeKey"],
                    row["variantKey"],
                    row["objective"],
                    Variant(
                        rank=int(row["rank"]),
                        parent_rank=int(row["parentRank"]),
                        chain=row["chain"],
                        addressed_target=row["addressedTarget"],
                        changed_positions=row["changedPositions"],
                        variant_sequence=row["variantSequence"],
                        structural_tolerance=float(row["structuralTolerance"]),
                        humanness_score=(
                            float(row["humannessScore"]) if row["humannessScore"] else None
                        ),
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


def _normalize(value: float, domain: tuple[float, float]) -> float:
    lo, hi = domain
    return (value - lo) / (hi - lo)


def _rerank_score(v: Variant, alpha: float, beta: float) -> float:
    """Returns `alpha * norm(structural tolerance) + beta * norm(humanness)`.

    Each term is rescaled to `[0, 1]` over its own fixed domain first.
    `alpha = beta = 1.0` then weighs the two terms equally. A variant with
    no humanness number contributes nothing to the second term."""
    structural = _normalize(v.structural_tolerance, STRUCTURAL_TOLERANCE_DOMAIN)
    humanness = 0.0 if v.humanness_score is None else _normalize(v.humanness_score, HUMANNESS_DOMAIN)
    return alpha * structural + beta * humanness


def rewrite_global_rank(path: str, alpha: float, beta: float) -> None:
    """Overwrites rank in path as one dataset-wide ordinal, 1..N.

    Reads all surviving variants, sorts by the normalised, weighted re-rank
    score (descending) then changed_positions, breaking ties by clonotypeKey
    and variantKey for determinism.

    variantKey and parent_rank stay unchanged. The batch loop streams each
    antibody's rows without accumulating them, never holding the whole run at
    once. This function instead reads the already-written survivors back,
    once the loop's own per-antibody working state is gone."""
    rows = read_variants_tsv(path)
    if not rows:
        return
    rows.sort(
        key=lambda row: (-_rerank_score(row[3], alpha, beta), row[3].changed_positions, row[0], row[1])
    )

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    for global_rank, (clonotype_key, variant_key_str, objective, v) in enumerate(rows, start=1):
        writer.writerow(
            _row(clonotype_key, variant_key_str, objective, replace(v, rank=global_rank))
        )
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n" + buf.getvalue())
