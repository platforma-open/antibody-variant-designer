"""Read/write helpers for the two files `scan.py` writes.

`triaged.json` carries only the liabilities `candidates.py` may act on —
verdict `"exposed"` — since that is the exact `actionable` set this file
carries across unchanged. `liabilities.tsv` carries every triaged
liability, including the ones triage declined, because the Parents page
has no other source for them.

Each `Triaged` site round-trips as full `residue_store.Residue` rows rather
than bare offsets: `candidates.py` needs each edited residue's chain, IMGT
label and wild type to build an edit, and re-deriving that by re-joining
offsets back onto `residues.json` would be a second index read for data
this file already carries once triage has run.
"""

import csv
import io
import json
from pathlib import Path

import residue_store
import triage

TSV_COLUMNS = [
    "liabilityKey",
    "liabilityType",
    "verdict",
    "region",
    "rsasa",
    "lowConfidence",
    "fixability",
]


def _tsv_value(value) -> str:
    return "" if value is None else str(value)


def liability_key(triaged: triage.Triaged) -> str:
    """`<liabilityType>@<chain><imgtLabel>` from the site's first residue —
    the span start, so two liabilities of one type can never share a key
    within a parent."""
    start = triaged.site[0]
    return f"{triaged.liability_type}@{start.chain}{start.imgt}"


def _low_confidence_str(triaged: triage.Triaged) -> str:
    return "yes" if triaged.low_confidence else "no"


def write_triaged(path: str, triaged_list: list[triage.Triaged]) -> None:
    """Only the actionable subset — verdict `"exposed"` — belongs here.
    The caller decides which rows qualify; this function does not filter."""
    rows = [
        {
            "definitionId": t.definition_id,
            "liabilityType": t.liability_type,
            "riskLevel": t.risk_level,
            "fixability": t.fixability,
            "site": [r.to_json() for r in t.site],
            "verdict": t.verdict,
            "lowConfidence": t.low_confidence,
            "rsasa": t.rsasa,
        }
        for t in triaged_list
    ]
    Path(path).write_text(json.dumps(rows))


def read_triaged(path: str) -> list[triage.Triaged]:
    rows = json.loads(Path(path).read_text())
    return [
        triage.Triaged(
            definition_id=row["definitionId"],
            liability_type=row["liabilityType"],
            risk_level=row["riskLevel"],
            fixability=row["fixability"],
            site=[residue_store.Residue.from_json(r) for r in row["site"]],
            verdict=row["verdict"],
            low_confidence=row["lowConfidence"],
            rsasa=row["rsasa"],
        )
        for row in rows
    ]


def write_liabilities_tsv(path: str, triaged_list: list[triage.Triaged]) -> None:
    """Every triaged liability, whatever its verdict — the Parents page's
    only source for a `buried` or `fixability-declined` row."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(TSV_COLUMNS)
    for t in triaged_list:
        writer.writerow(
            [
                _tsv_value(liability_key(t)),
                _tsv_value(t.liability_type),
                _tsv_value(t.verdict),
                _tsv_value(t.site[0].region),
                _tsv_value(t.rsasa),
                _tsv_value(_low_confidence_str(t)),
                _tsv_value(t.fixability),
            ]
        )
    Path(path).write_text(buf.getvalue())
