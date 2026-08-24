"""Read/write helpers for the two files `index_and_scan.py` writes: `triaged.json` and
`liabilities.tsv`.

A `Triaged` site round-trips as full `residue_store.Residue` rows.
`variant_candidates.py` needs each edited residue's chain, IMGT label and wild
type to build an edit. Rejoining bare offsets onto `residues.json` would
read the index a second time.
"""

import csv
import io
import json
from pathlib import Path

from engine import liability_triage, residue_store

TSV_COLUMNS = [
    "clonotypeKey",
    "verdict",
    "summary",
]


def _tsv_value(value) -> str:
    return "" if value is None else str(value)


def liability_key(triaged: liability_triage.Triaged) -> str:
    """`<liabilityType>@<chain><imgtLabel>` built from the site's first residue, the span start.
    Two liabilities of the same type can never share a key within one parent."""
    start = triaged.site[0]
    return f"{triaged.liability_type}@{start.chain}{start.imgt}"


def write_triaged(path: str, triaged_list: list[liability_triage.Triaged]) -> None:
    """Takes only the actionable subset: rows with verdict `"exposed"`.
    The caller filters; this function writes whatever list it receives."""
    rows = [
        {
            "definitionId": t.definition_id,
            "liabilityType": t.liability_type,
            "riskLevel": t.risk_level,
            "fixability": t.fixability,
            "site": [r.to_json() for r in t.site],
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
            site=[residue_store.Residue.from_json(r) for r in row["site"]],
            verdict=row["verdict"],
            low_confidence=row["lowConfidence"],
            confidence_angstroms=row["confidenceAngstroms"],
            rsasa=row["rsasa"],
        )
        for row in rows
    ]


def write_liabilities_header(path: str) -> None:
    """Start the run's one dataset-wide file. Call this once, before the batch loop begins.
    An empty pdb_index then still produces a header-only TSV, not zero files."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def summarize_liabilities(triaged_list: list[liability_triage.Triaged]) -> tuple[str, str]:
    """Build coarse verdict ("present" or "none") and summary line for a parent's triaged
    liabilities.

    summary lists each liability as <type>@<chain><imgtLabel> (<verdict>), declined ones
    included. fixability-declined sites append their fixability class — the only place this
    decline reason survives after the per-liability columns are dropped.
    """
    if not triaged_list:
        return "none", "None"
    parts = []
    for t in triaged_list:
        entry = f"{liability_key(t)} ({t.verdict}"
        if t.verdict == "fixability-declined":
            entry += f": {t.fixability}"
        entry += ")"
        parts.append(entry)
    return "present", ", ".join(parts)


def append_liabilities_tsv(
    path: str, clonotype_key: str, triaged_list: list[liability_triage.Triaged]
) -> None:
    """Append one row: coarse verdict plus summary of every triaged liability.

    The summary includes buried and fixability-declined sites — the Parents page has no
    other source for them. This function appends (not returns) because one file holds the
    whole dataset and the pipeline streams one antibody at a time."""
    verdict, summary = summarize_liabilities(triaged_list)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow([_tsv_value(clonotype_key), verdict, summary])
    with Path(path).open("a") as fh:
        fh.write(buf.getvalue())
