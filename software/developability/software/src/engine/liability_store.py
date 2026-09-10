"""Read/write helpers for the two files `index_and_scan.py` writes: `triaged.json` and
`liabilities.tsv`.

A `Triaged` site round-trips as full `residue_index.Residue` rows.
`variant_candidates.py` needs each edited residue's chain, IMGT label and wild
type to build an edit. Rejoining bare offsets onto `residues.json` would
read the index a second time.
"""

import csv
import io
import json
from pathlib import Path

from engine import developability_score, liability_triage, residue_store

TSV_COLUMNS = [
    "clonotypeKey",
    "verdict",
    "summary",
    "developabilityScore",
]


def _tsv_value(value) -> str:
    return "" if value is None else str(value)


def write_triaged(path: str, triaged_list: list[liability_triage.Triaged]) -> None:
    """Takes only the actionable subset: rows with verdict `"exposed"`.
    The caller filters; this function writes whatever list it receives."""
    rows = [
        {
            "definitionId": t.definition_id,
            "liabilityType": t.liability_type,
            "riskLevel": t.risk_level,
            "fixability": t.fixability,
            "site": [residue_store.residue_to_json(r) for r in t.site],
            "verdict": t.verdict,
            "lowConfidence": t.low_confidence,
            "confidenceAngstroms": t.confidence_angstroms,
            "rsasa": t.rsasa,
        }
        for t in triaged_list
    ]
    Path(path).write_text(json.dumps(rows))


def read_triaged(path: str) -> list[liability_triage.Triaged]:
    rows = json.loads(Path(path).read_text())
    return [
        liability_triage.Triaged(
            definition_id=row["definitionId"],
            liability_type=row["liabilityType"],
            risk_level=row["riskLevel"],
            fixability=row["fixability"],
            site=[residue_store.residue_from_json(r) for r in row["site"]],
            verdict=row["verdict"],
            low_confidence=row["lowConfidence"],
            confidence_angstroms=row["confidenceAngstroms"],
            rsasa=row["rsasa"],
        )
        for row in rows
    ]


def write_liabilities_header(path: str) -> None:
    """Start the run's one dataset-wide file. Call this once, before the batch loop begins.
    An empty run then still produces a header-only TSV, not zero files."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def append_liabilities_tsv(
    path: str,
    clonotype_key: str,
    triaged_list: list[liability_triage.Triaged],
    fixability_weights: dict[str, float],
) -> None:
    """Append one row: coarse verdict, summary of every triaged liability, and the burden
    they add up to.

    The summary includes buried and fixability-declined sites — the Parents page has no
    other source for them. This function appends (not returns) because one file holds the
    whole dataset and the pipeline streams one antibody at a time.

    The score covers that same whole list, buried and declined sites included: it measures
    what the parent carries, not what this run chose to act on, so a variant scored after a
    repair is compared against everything the parent started with."""
    summary = liability_triage.summarize_liabilities(triaged_list)
    row_buffer = io.StringIO()
    writer = csv.writer(row_buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(
        [
            _tsv_value(clonotype_key),
            summary.verdict,
            summary.summary,
            developability_score.score(triaged_list, fixability_weights),
        ]
    )
    with Path(path).open("a") as out_file:
        out_file.write(row_buffer.getvalue())
