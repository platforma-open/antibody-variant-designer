"""Total liability scan and non-destructive triage, the entrypoint that
writes `triaged.json` and `liabilities.tsv`.

Fused on purpose: exposure, detection and triage all read the same residue
table, and a verdict needs exposure and detection at once, so no earlier
boundary inside this step can attribute a skip. `exposure.py`, `motifs.py`,
`cysteine.py` and `triage.py` stay separate modules; only the step itself is
one exec.

Writes two artifacts. `--out-triaged` carries only the liabilities this run
may act on — verdict `"exposed"` — the exact set `candidates.py` reads.
`--out-liabilities` carries every triaged liability,
including the ones triage declined, because the Parents page has no other
source for a `buried` or `fixability-declined` row. Both are written even
when nothing survives triage, so an antibody with zero variants still has
data for the Parents page.
"""

import argparse
import json
import sys
from pathlib import Path

import cysteine
import exposure
import liability_store
import motifs
import residue_store
import triage

DEFAULT_RSASA_BURIED_CUTOFF = 0.075
DEFAULT_FR_CONFIDENCE_THRESHOLD = 4.0
DEFAULT_CDR_CONFIDENCE_THRESHOLD = 6.0
DEFAULT_ACT_ON_FIXABILITY = "fixable,easily_fixable"


def _write_skip(path: str, reason: str) -> None:
    Path(path).write_text(reason)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan every liability and triage each hit, non-destructively."
    )
    parser.add_argument("--pdb", required=True, help="the same staged PDB blob structure.py read")
    parser.add_argument("--residues", required=True, help="structure.py's output")
    parser.add_argument(
        "--clonotype-key", required=True, help="the parent this invocation is for"
    )
    parser.add_argument(
        "--definitions", required=True, help="taxonomy JSON from the shared package"
    )
    parser.add_argument(
        "--per-residue-confidence",
        default=None,
        help="optional upstream sidecar; the PDB B-factor column is the second source",
    )
    parser.add_argument("--out-triaged", required=True)
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

    residues = residue_store.read_residues(args.residues)
    taxonomy = json.loads(Path(args.definitions).read_text())
    act_on_fixability = [v for v in args.act_on_fixability.split(",") if v]

    rsasa_lookup = exposure.annotate(residues, args.pdb)
    confidence_lookup = _confidence_lookup(residues, args.per_residue_confidence)

    detected = motifs.detect_all(residues, taxonomy) + cysteine.detect_all(residues, taxonomy)
    triaged = triage.verdict_for(
        detected,
        rsasa_lookup,
        confidence_lookup,
        args.rsasa_buried_cutoff,
        fr_confidence_threshold=args.fr_confidence_gating_threshold,
        cdr_confidence_threshold=args.cdr_confidence_gating_threshold,
        act_on_fixability=act_on_fixability,
    )
    actionable = [t for t in triaged if triage.generates_for(t)]

    liability_store.write_triaged(args.out_triaged, actionable)
    liability_store.write_liabilities_tsv(args.out_liabilities, triaged)
    _write_skip(args.out_skip, "" if actionable else "no-liability-survived-triage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
