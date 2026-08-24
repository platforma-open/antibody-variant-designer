"""Read/write helpers for the two files `index_and_scan.py` writes.

`triaged.json` carries only the liabilities `candidates.py` may act on —
verdict `"exposed"` — since that is the exact `actionable` set this file
carries across unchanged. `liabilities.tsv` carries one row per parent, a
coarse verdict plus a summary of every triaged liability including the ones
triage declined, because the Parents page has no other source for them
(`082-decision-the-liabilities-group-drops-to-one-axis`).

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
    "clonotypeKey",
    "verdict",
    "summary",
]


def _tsv_value(value) -> str:
    return "" if value is None else str(value)


def liability_key(triaged: triage.Triaged) -> str:
    """`<liabilityType>@<chain><imgtLabel>` from the site's first residue —
    the span start, so two liabilities of one type can never share a key
    within a parent."""
    start = triaged.site[0]
    return f"{triaged.liability_type}@{start.chain}{start.imgt}"


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
            "confidenceAngstroms": t.confidence_angstroms,
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
            confidence_angstroms=row["confidenceAngstroms"],
            rsasa=row["rsasa"],
        )
        for row in rows
    ]


def write_liabilities_header(path: str) -> None:
    """Start the run's one dataset-wide file. Called once, before the batch
    loop, so an empty pdb_index still leaves a header-only TSV rather than no
    file at all."""
    Path(path).write_text("\t".join(TSV_COLUMNS) + "\n")


def summarize_liabilities(triaged_list: list[triage.Triaged]) -> tuple[str, str]:
    """A coarse verdict plus one joined summary line for a parent's triaged
    liabilities, mirroring the sibling block's
    `_create_sequence_liabilities_summary_str`
    (`antibody-sequence-liabilities/liabilities-calc-script/src/main.py:133-226`).

    `verdict` is `"present"` when the parent carries at least one triaged
    liability, `"none"` when it triaged clean. `summary` lists every one,
    declined ones included, each written `<liabilityType>@<chain><imgtLabel>
    (<verdict>)` — the fixability class is appended for a
    `fixability-declined` site, since that is the only place its decline
    reason survives once the per-liability columns are gone.
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
    path: str, clonotype_key: str, triaged_list: list[triage.Triaged]
) -> None:
    """Append one row for this antibody — a coarse verdict plus a summary of
    every triaged liability, whatever its verdict, since the Parents page has
    no other source for a `buried` or `fixability-declined` site.

    Appends rather than returning a row to collect: one file now holds the
    whole dataset, and accumulating every antibody's row before a single
    write is what the sequential-streaming constraint forbids."""
    verdict, summary = summarize_liabilities(triaged_list)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow([_tsv_value(clonotype_key), verdict, summary])
    with Path(path).open("a") as fh:
        fh.write(buf.getvalue())
