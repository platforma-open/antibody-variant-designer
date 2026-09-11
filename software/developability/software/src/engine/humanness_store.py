"""Appends one parent's row to the run's one humanness file: the verdict and summary
`humanness_objective.summarize_humanness` reduces to, beside the parent's own per-chain
baseline score, measured once regardless of what the reduction finds.

Mirrors `liability_store.py`'s append-per-parent file, one row per parent.
"""

import csv
import io
from pathlib import Path

from engine import humanness_objective, residue_index, variant_candidates

TSV_COLUMNS = [
    "clonotypeKey",
    "humannessVerdict",
    "humannessSummary",
    "heavyHumannessScore",
]

def _tsv_value(value) -> str:
    return "" if value is None else str(value)


def write_humanness_header(path: str) -> None:
    """Starts the run's one dataset-wide file with the header row alone.

    Called once, before the batch loop, so an empty run still leaves a header-only
    file rather than no file."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def append_humanness_tsv(
    path: str,
    clonotype_key: str,
    targets: list[residue_index.Residue] | None,
    cleared: list[variant_candidates.Candidate],
    humanness_parent_scores: dict[str, float | None],
) -> None:
    """Appends this parent's one row, reducing through summarize_humanness.

    humanness_parent_scores is the parent's own baseline, keyed by chain role. Only the heavy
    chain's score is emitted: it is the chain every antibody format in scope carries, and the
    one the row is keyed on. The column is filled in every run mode, unlike the verdict and
    summary beside it, and is empty only when the gate could not score the chain.

    Appends rather than returning a row to collect, for the reason `liability_store.py`'s
    `append_liabilities_tsv` gives: one file holds the whole dataset and the pipeline
    streams one antibody at a time."""
    summary = humanness_objective.summarize_humanness(targets, cleared)
    row_buffer = io.StringIO()
    writer = csv.writer(row_buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(
        [
            _tsv_value(clonotype_key),
            summary.verdict,
            summary.summary,
            _tsv_value(humanness_parent_scores.get("H")),
        ]
    )
    with Path(path).open("a") as out_file:
        out_file.write(row_buffer.getvalue())
