"""Reduces one parent's considered framework positions to a coarse verdict and one joined
summary line, and appends its row to the run's one humanness file.

Mirrors `liability_store.py`'s verdict/summary pair: same two literals for "nothing found",
same comma-joined summary shape, same append-per-parent file. The one addition is a third
state — targets of `None` — because here "the run never looked" and "it looked and found
nothing" are both real and must not collapse into one string.
"""

import csv
import io
from pathlib import Path

from engine import liability_store, residue_store, variant_candidates

TSV_COLUMNS = [
    "clonotypeKey",
    "humannessVerdict",
    "humannessSummary",
]

AMINO_SEP = ", "


def _tsv_value(value) -> str:
    return "" if value is None else str(value)


def write_humanness_header(path: str) -> None:
    """Starts the run's one dataset-wide file with the header row alone.

    Called once, before the batch loop, so an empty pdb_index still leaves a header-only
    file rather than no file."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def summarize_humanness(
    targets: list[residue_store.Residue] | None,
    cleared: list[variant_candidates.Candidate],
) -> liability_store.ParentSummary:
    """Build coarse verdict and summary for a parent's considered framework positions.

    targets is the humanization objective's selected positions for this parent, in its own
    order; `None` when the objective did not run for this parent at all. cleared is the
    candidates that passed the Humanness gate for that parent.

    ("",  "")            targets is None      — the objective did not look
    ("none", "None")     targets is empty     — it looked and left everything alone
    ("present", <line>)  otherwise            — one entry per target, in target order

    A target reads "humanised" when some cleared candidate edits its position, "declined"
    otherwise. The per-parent variant cap runs after the gate, so a candidate the cap later
    dropped still counts here — this reduction never sees the cap's decision."""
    if targets is None:
        return liability_store.ParentSummary(verdict="", summary="")
    if not targets:
        return liability_store.ParentSummary(verdict="none", summary="None")

    edited = {(edit.chain, edit.imgt) for candidate in cleared for edit in candidate.edits}
    entries = [
        f"{target.wild_type}@{target.chain}{target.imgt} "
        f"({'humanised' if target.join_key in edited else 'declined'})"
        for target in targets
    ]
    return liability_store.ParentSummary(verdict="present", summary=AMINO_SEP.join(entries))


def append_humanness_tsv(
    path: str,
    clonotype_key: str,
    targets: list[residue_store.Residue] | None,
    cleared: list[variant_candidates.Candidate],
) -> None:
    """Appends this parent's one row, reducing through summarize_humanness.

    Appends rather than returning a row to collect, for the reason `liability_store.py`'s
    `append_liabilities_tsv` gives: one file holds the whole dataset and the pipeline
    streams one antibody at a time."""
    parent_summary = summarize_humanness(targets, cleared)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow([_tsv_value(clonotype_key), parent_summary.verdict, parent_summary.summary])
    with Path(path).open("a") as fh:
        fh.write(buf.getvalue())
