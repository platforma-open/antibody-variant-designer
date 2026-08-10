"""Residue indexing, total liability scan and non-destructive triage — the
`index-and-scan` entrypoint, which writes the residue index, one
`triaged.json` per antibody and the dataset-wide `liabilities.tsv`.

Four phases, one exec, because they are one deployment unit and nothing
between them is read by anything else. `structure.py` indexes; `exposure.py`,
`motifs.py`, `cysteine.py` and `triage.py` do the rest. They stay separate
modules — only the exec is one. A verdict needs exposure and detection at
once, so no boundary inside the scan could attribute a skip anyway.

**One reason per antibody.** The index phase short-circuits the scan phase,
so a structure that cannot be indexed is named by the index phase alone. And
every output file is written only for the antibodies that got that far: a
skipped antibody leaves no `residues.json` and no `triaged.json`, which is
what lets a later step read absence as "already named". An empty placeholder
would make that check a no-op and the antibody would be counted twice.

`--out-triaged-dir` carries only the liabilities this run may act on —
verdict `"exposed"` — the exact set the re-scan gate reads.
`--out-liabilities` carries every triaged liability, including the ones
triage declined, because the Parents page has no other source for a `buried`
or `fixability-declined` row. It is appended for every antibody that reached
triage at all, so an antibody with zero variants still has data there.
"""

import argparse
import json
import sys
from pathlib import Path

import batch
import cysteine
import exposure
import liability_store
import motifs
import residue_store
import roster
import structure
import triage

DEFAULT_RSASA_BURIED_CUTOFF = 0.075
DEFAULT_FR_CONFIDENCE_THRESHOLD = 4.0
DEFAULT_CDR_CONFIDENCE_THRESHOLD = 6.0
DEFAULT_ACT_ON_FIXABILITY = "fixable,easily_fixable"


def _load_confidence_sidecar(path: str | None) -> list[dict]:
    """upstream's `pl7.app/structure/confidence/perResidue` JSON, staged for
    this one antibody: a list of `{"pos", "chain", "errorAngstroms"}`
    records. Absent or unreadable reads as no sidecar at all, not an error —
    the PDB B-factor column is the second source precisely for this case."""
    if path is None or not Path(path).is_file():
        return []
    try:
        records = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return records if isinstance(records, list) else []


def _confidence_lookup(
    residues: list[residue_store.Residue], sidecar_path: str | None
) -> dict[tuple[str, str], float | None]:
    """Per-residue confidence, JSON sidecar first, the PDB B-factor column
    as the second source — the only available check on index alignment,
    since ImmuneBuilder writes the same error array to both. A residue the
    sidecar does not mention falls through to its own B-factor; a residue
    with neither stays unmeasured, read as `None` by `triage.py`, never as
    high-confidence."""
    from_sidecar: dict[tuple[str, str], float] = {}
    for record in _load_confidence_sidecar(sidecar_path):
        if not isinstance(record, dict):
            continue
        pos, chain, err = record.get("pos"), record.get("chain"), record.get("errorAngstroms")
        if pos is None or chain is None or err is None:
            continue
        try:
            from_sidecar[(str(chain), str(pos))] = float(err)
        except (TypeError, ValueError):
            continue

    lookup: dict[tuple[str, str], float | None] = {}
    for residue in residues:
        key = (residue.chain, residue.imgt)
        lookup[key] = from_sidecar.get(key, residue.b_factor)
    return lookup


def process_one(
    pdb_path: str,
    out_residues: str,
    out_triaged: str,
    taxonomy: list[dict],
    confidence_sidecar: str | None,
    rsasa_buried_cutoff: float,
    fr_confidence_threshold: float,
    cdr_confidence_threshold: float,
    act_on_fixability: list[str],
) -> tuple[str, list[triage.Triaged]]:
    """Index, then scan and triage, one antibody. Returns **one** skip reason
    (`""` on pass) and every triaged liability, whatever its verdict — the
    caller appends those to the run's one `liabilities.tsv`, because the
    Parents page needs the declined ones too.

    The index phase short-circuits the scan phase: an antibody that cannot be
    indexed carries the index phase's reason and nothing else, so a
    non-IMGT structure is named once rather than reported again as having no
    surviving liability."""
    index_reason = structure.index_one(pdb_path, out_residues)
    if index_reason:
        return index_reason, []

    residues = residue_store.read_residues(out_residues)

    rsasa_lookup = exposure.annotate(residues, pdb_path)
    confidence_lookup = _confidence_lookup(residues, confidence_sidecar)

    detected = motifs.detect_all(residues, taxonomy) + cysteine.detect_all(residues, taxonomy)
    triaged = triage.verdict_for(
        detected,
        rsasa_lookup,
        confidence_lookup,
        rsasa_buried_cutoff,
        fr_confidence_threshold=fr_confidence_threshold,
        cdr_confidence_threshold=cdr_confidence_threshold,
        act_on_fixability=act_on_fixability,
    )
    actionable = [t for t in triaged if triage.generates_for(t)]

    liability_store.write_triaged(out_triaged, actionable)
    return ("" if actionable else "no-liability-survived-triage"), triaged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan every staged antibody's liabilities and triage each hit, "
        "non-destructively."
    )
    parser.add_argument(
        "--pdb-dir",
        required=True,
        help="the staged blobs — the index phase parses them and freesasa reads them again",
    )
    parser.add_argument("--pdb-index", required=True, help="the roster")
    parser.add_argument("--out-residues-dir", required=True)
    parser.add_argument(
        "--definitions", required=True, help="taxonomy JSON from the shared package"
    )
    parser.add_argument(
        "--confidence-dir",
        default=None,
        help="optional upstream sidecars; the PDB B-factor column is the second source",
    )
    parser.add_argument("--out-triaged-dir", required=True)
    parser.add_argument("--out-liabilities", required=True)
    parser.add_argument("--out-skip", required=True)
    parser.add_argument("--rsasa-buried-cutoff", type=float, default=DEFAULT_RSASA_BURIED_CUTOFF)
    parser.add_argument(
        "--fr-confidence-gating-threshold", type=float, default=DEFAULT_FR_CONFIDENCE_THRESHOLD
    )
    parser.add_argument(
        "--cdr-confidence-gating-threshold", type=float, default=DEFAULT_CDR_CONFIDENCE_THRESHOLD
    )
    parser.add_argument("--act-on-fixability", default=DEFAULT_ACT_ON_FIXABILITY)
    args = parser.parse_args(argv)

    taxonomy = json.loads(Path(args.definitions).read_text())
    act_on_fixability = [v for v in args.act_on_fixability.split(",") if v]

    pdb_dir = Path(args.pdb_dir)
    confidence_dir = Path(args.confidence_dir) if args.confidence_dir else None
    residues_dir = Path(args.out_residues_dir)
    residues_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_triaged_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    liability_store.write_liabilities_header(args.out_liabilities)

    def one(entry: roster.Entry) -> str | None:
        pdb_path = pdb_dir / entry.filename
        # A clonotype the upstream block failed for has no staged blob at
        # all — never a null one — so absence is this step's own named
        # reason rather than an error.
        if not pdb_path.is_file():
            return "no-structure"
        sidecar = None
        if confidence_dir is not None:
            candidate = confidence_dir / f"{entry.stem}.json"
            sidecar = str(candidate) if candidate.is_file() else None

        reason, triaged = process_one(
            str(pdb_path),
            str(residues_dir / f"{entry.stem}.json"),
            str(out_dir / f"{entry.stem}.json"),
            taxonomy,
            sidecar,
            args.rsasa_buried_cutoff,
            args.fr_confidence_gating_threshold,
            args.cdr_confidence_gating_threshold,
            act_on_fixability,
        )
        # Appended even when triage declined everything, so an antibody
        # that produces no variant still has Parents-page data.
        liability_store.append_liabilities_tsv(args.out_liabilities, entry.clonotype_key, triaged)
        return reason

    return batch.run(roster.read_roster(args.pdb_index), one, args.out_skip)


if __name__ == "__main__":
    sys.exit(main())
