"""Reads and writes variants.tsv, the block's final artifact.

Every row carries axis values: clonotypeKey for the parent, variantKey for the
variant. xsv.importFile builds each axis from a column.
"""

import csv
import io
from dataclasses import replace
from pathlib import Path

from engine import variant_ranking

# This file carries no run or block id column.
# The workflow attaches run identity to these columns downstream, once this process finishes.
# That lets two blocks over one dataset share this exec, instead of
# computing the same variants twice.
TSV_COLUMNS = [
    "clonotypeKey",
    "variantKey",
    "rank",
    "parentRank",
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
    "developabilityScore",
]


def _tsv_value(value) -> str:
    return "" if value is None else str(value)


def _low_confidence_warning_str(variant: variant_ranking.Variant) -> str:
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

    Call this before the batch loop starts. Then an empty run
    still leaves a header-only TSV."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def _row(clonotype_key: str, variant_key_str: str, v: variant_ranking.Variant) -> list:
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
        v.developability_score,
    ]


def append_variants_tsv(
    path: str, clonotype_key: str, variants: list[variant_ranking.Variant]
) -> None:
    """Appends one parent's ranked variants to path.

    variantKey is computed here from parent_rank — the only place both the
    variant's rank and its parent clonotype are available. Later,
    rewrite_global_rank overwrites rank with a dataset-wide ordinal without
    touching variantKey or parent_rank."""
    row_buffer = io.StringIO()
    writer = csv.writer(row_buffer, delimiter="\t", lineterminator="\n")
    for v in variants:
        writer.writerow(_row(clonotype_key, variant_key(v.parent_rank), v))
    with Path(path).open("a") as out_file:
        out_file.write(row_buffer.getvalue())


def read_variants_tsv(path: str) -> list[tuple[str, str, variant_ranking.Variant]]:
    """`(clonotype_key, variant_key, variant)` per row, in file order."""
    variants = []
    with Path(path).open(newline="") as out_file:
        for row in csv.DictReader(out_file, delimiter="\t"):
            variants.append(
                (
                    row["clonotypeKey"],
                    row["variantKey"],
                    variant_ranking.Variant(
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
                        developability_score=float(row["developabilityScore"]),
                    ),
                )
            )
    return variants


def rewrite_global_rank(
    path: str,
    rerank_structural_weight: float,
    rerank_humanness_weight: float,
) -> None:
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
        key=lambda row: (
            -variant_ranking.rerank_score(
                row[2], rerank_structural_weight, rerank_humanness_weight
            ),
            row[2].changed_positions,
            row[0],
            row[1],
        )
    )

    row_buffer = io.StringIO()
    writer = csv.writer(row_buffer, delimiter="\t", lineterminator="\n")
    for global_rank, (clonotype_key, variant_key_str, v) in enumerate(rows, start=1):
        writer.writerow(_row(clonotype_key, variant_key_str, replace(v, rank=global_rank)))
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n" + row_buffer.getvalue())
