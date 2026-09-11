"""The `index-and-scan` entrypoint: indexes residues, scans them for
liabilities, and triages each hit.

Writes the residues and triaged keyed artifacts, and the
dataset-wide `liabilities.tsv`.

The four phases run in one exec because a verdict needs exposure and
detection together. No boundary inside the scan could name a rejection on
its own. Each antibody therefore gets exactly one rejection reason.
"""

import argparse
import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from engine import (
    antibody_batch,
    design_objective,
    keyed_artifact,
    liability_objective,
    liability_store,
    liability_triage,
    parent_clonotypes,
    rejection_store,
    residue_exposure,
    residue_index,
    residue_store,
    run_mode,
    taxonomy_store,
)

DEFAULT_RSASA_BURIED_CUTOFF = 0.075
DEFAULT_FR_CONFIDENCE_THRESHOLD = 4.0
DEFAULT_CDR_CONFIDENCE_THRESHOLD = 6.0
DEFAULT_ACT_ON_FIXABILITY = "fixable,easily_fixable"

# The two reasons meaning `process_one` reached the scan phase at all.
# Every other reason comes from `index_one` below and leaves
# `triaged` as `[]` too — from the short-circuit, not a clean scan.
# Only these two are safe to summarize into `liabilities.tsv`.
_REACHED_TRIAGE_REASONS = ("", "no-liability-survived-triage")


def _iter_clonotype_keyed_tsv(path: Path) -> Iterator[tuple[str, str]]:
    """Yield `(clonotype_key, raw_value)` for a dataset-wide TSV exported by
    `main.tpl.tengo` with the clonotype key in the first column and the value
    in the second. Skips rows with an empty value. Shared by the confidence
    and clonotype-filter loaders below, since both TSVs take that same
    two-column shape."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []
        if len(fields) < 2:
            return
        key_col, val_col = fields[0], fields[1]
        for row in reader:
            raw = row.get(val_col)
            if raw is None or raw == "":
                continue
            yield row[key_col], raw


def _load_confidence_tsv(path: str | None) -> dict[str, list[dict]]:
    """Per-clonotype `pl7.app/structure/confidence/perResidue` records,
    read from the one dataset-wide TSV `main.tpl.tengo` exports, not from
    a directory of per-stem files.

    A clonotype falls through to its own B-factor column when it is
    missing here, or when its value fails to parse as a JSON list. That
    column is the second source `_confidence_lookup` reads."""
    result: dict[str, list[dict]] = {}
    if path is None or not Path(path).is_file():
        return result
    for key, raw in _iter_clonotype_keyed_tsv(Path(path)):
        try:
            records = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(records, list):
            result[key] = records
    return result


def _load_clonotype_filter(path: str | None) -> set[str] | None:
    """The optional Lead Selection subset, exported as one dataset-wide TSV.

    `None` means no filter was picked, so every parent clonotype gets
    processed. A picked filter yields a keep set, possibly empty. A
    clonotype whose value is falsy is treated as not selected — the same
    grammar the sibling block's `--clonotype-filter` uses."""
    if path is None or not Path(path).is_file():
        return None
    keep: set[str] = set()
    for key, raw in _iter_clonotype_keyed_tsv(Path(path)):
        value = raw.strip().lower()
        if value in {"0", "false", "no", "null"}:
            continue
        try:
            if float(value) == 0.0:
                continue
        except ValueError:
            pass
        keep.add(key)
    return keep


def _confidence_lookup(
    residues: list[residue_index.Residue], confidence_records: list[dict] | None
) -> dict[tuple[str, str], float | None]:
    """Per-residue confidence: sidecar records first, the PDB B-factor
    column second.

    ImmuneBuilder writes the same error array to both, so a residue's
    B-factor also works as a check on index alignment. A residue the
    sidecar omits falls through to its own B-factor. A residue with
    neither stays unmeasured, reported as `None`. `liability_triage.py` never
    reads a `None` confidence as high."""
    from_sidecar: dict[tuple[str, str], float] = {}
    for record in confidence_records or []:
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


def index_one(
    pdb_path: str, residues_writer: keyed_artifact.KeyedWriter, clonotype_key: str
) -> tuple[str, list[residue_index.Residue]]:
    """Index one antibody and return its rejection reason (empty on success) alongside its
    residues — `[]` on a rejection.

    A rejected antibody gets no row in the residues artifact. A missing entry indicates an
    earlier phase rejected this antibody; an empty one would break that pattern
    (double-counting)."""
    parsed = residue_store.read_pdb(pdb_path)
    if not parsed.chain_order:
        return "no-structure", []

    if not residue_index.is_imgt_numbered(parsed):
        return "structure-not-imgt", []

    if residue_index.multi_domain_chain(parsed) is not None:
        return "structure-multi-domain-chain", []

    residues = residue_index.index_residues(parsed)
    # The index keeps every residue, including a constant-domain, antigen,
    # or second-arm residue, so exposure and AntiFold still see it. The
    # rejection fires only when nothing at all is researchable. A parent of pure
    # constant region therefore never reaches the scan phase as a silent
    # empty scan.
    if not any(r.in_scope for r in residues):
        return "no-researchable-residue", []

    residue_store.write_residues(residues_writer, clonotype_key, residues)
    # Read back through the same store rather than from a path — there is no longer a
    # per-parent file to open, only the shared writer's own conversion.
    reread = residue_store.residues_from_payload(
        [residue_store.residue_to_json(r) for r in residues]
    )
    return "", reread


def process_one(
    pdb_path: str,
    residues_writer: keyed_artifact.KeyedWriter,
    triaged_writer: keyed_artifact.KeyedWriter,
    clonotype_key: str,
    taxonomy: list[dict],
    objective: design_objective.Objective,
    confidence_records: list[dict] | None,
    rsasa_buried_cutoff: float,
    fr_confidence_threshold: float,
    cdr_confidence_threshold: float,
    act_on_fixability: list[str],
) -> tuple[str, list[liability_triage.Triaged]]:
    """Index, scan, and triage one antibody.

    Returns rejection reason (empty on pass) and every triaged liability (the
    Parents page needs declined ones too). The triaged artifact holds only "exposed"
    verdicts — the set the re-scan gate reads.

    Index phase short-circuits scan; a non-indexable antibody carries only
    the index reason.

    A rejected antibody gets no row in the residues or triaged artifact. A missing entry
    indicates "already rejected"; an empty one would double-count."""
    index_reason, residues = index_one(pdb_path, residues_writer, clonotype_key)
    if index_reason:
        return index_reason, []

    rsasa_lookup = residue_exposure.annotate(residues, pdb_path)
    confidence_lookup = _confidence_lookup(residues, confidence_records)

    detected = objective.select_target_positions(residues, taxonomy)
    triaged = liability_triage.verdict_for(
        detected,
        rsasa_lookup,
        confidence_lookup,
        rsasa_buried_cutoff,
        fr_confidence_threshold=fr_confidence_threshold,
        cdr_confidence_threshold=cdr_confidence_threshold,
        act_on_fixability=act_on_fixability,
    )
    actionable = [t for t in triaged if liability_triage.generates_for(t)]

    liability_store.write_triaged(triaged_writer, clonotype_key, actionable)
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
    parser.add_argument("--out-residues", required=True)
    parser.add_argument(
        "--definitions", required=True, help="taxonomy JSON from the shared package"
    )
    parser.add_argument(
        "--per-residue-confidence",
        default=None,
        dest="per_residue_confidence",
        help="optional dataset-wide TSV exported from upstream's per-residue confidence "
        "column; the PDB B-factor column is the second source",
    )
    parser.add_argument(
        "--clonotype-filter",
        default=None,
        dest="clonotype_filter",
        help="optional dataset-wide TSV naming the subset to process; omitted means the "
        "every parent clonotype",
    )
    parser.add_argument("--out-triaged", required=True)
    parser.add_argument("--out-liabilities", required=True)
    parser.add_argument("--out-rejected", required=True)
    parser.add_argument("--rsasa-buried-cutoff", type=float, default=DEFAULT_RSASA_BURIED_CUTOFF)
    parser.add_argument(
        "--fr-confidence-threshold", type=float, default=DEFAULT_FR_CONFIDENCE_THRESHOLD
    )
    parser.add_argument(
        "--cdr-confidence-threshold", type=float, default=DEFAULT_CDR_CONFIDENCE_THRESHOLD
    )
    parser.add_argument("--act-on-fixability", default=DEFAULT_ACT_ON_FIXABILITY)
    args = parser.parse_args(argv)

    taxonomy = taxonomy_store.read_taxonomy(args.definitions)
    fixability_weights = taxonomy_store.read_fixability_weights(args.definitions)
    act_on_fixability = [v for v in args.act_on_fixability.split(",") if v]

    pdb_dir = Path(args.pdb_dir)
    confidence_by_clonotype = _load_confidence_tsv(args.per_residue_confidence)
    keep_clonotypes = _load_clonotype_filter(args.clonotype_filter)
    liability_store.write_liabilities_header(args.out_liabilities)

    with (
        keyed_artifact.KeyedWriter(args.out_residues) as residues_writer,
        keyed_artifact.KeyedWriter(args.out_triaged) as triaged_writer,
    ):

        def one(
            entry: parent_clonotypes.ParentClonotype,
        ) -> list[tuple[str, str, str, str]] | None:
            # A picked filter narrows which parents this step attempts. A
            # filtered-out clonotype was never in scope, so it gets no rejection
            # row at all.
            if keep_clonotypes is not None and entry.clonotype_key not in keep_clonotypes:
                return None
            pdb_path = pdb_dir / entry.filename

            reason, triaged = process_one(
                str(pdb_path),
                residues_writer,
                triaged_writer,
                entry.clonotype_key,
                taxonomy,
                liability_objective.OBJECTIVE,
                confidence_by_clonotype.get(entry.clonotype_key),
                args.rsasa_buried_cutoff,
                args.fr_confidence_threshold,
                args.cdr_confidence_threshold,
                act_on_fixability,
            )
            # Appended for every antibody that reached triage, so a
            # zero-variant antibody still gets Parents-page data. An antibody
            # that failed at the index phase never scanned, so appending its
            # absence here would read as a false "none" verdict.
            if reason in _REACHED_TRIAGE_REASONS:
                liability_store.append_liabilities_tsv(
                    args.out_liabilities, entry.clonotype_key, triaged, fixability_weights
                )
            # The design step, not this one, decides whether a parent with no triaged
            # liability still ships a variant — it owns this reason's row, and only when
            # `mode` actually runs the liability objective (a `humanization`-only run never
            # reaches this reason with a row worth writing).
            if reason == "no-liability-survived-triage":
                return []
            return [(reason, "", rejection_store.PARENT_REJECTED, run_mode.LIABILITY)]

        parents = parent_clonotypes.scan_parent_clonotypes(
            args.pdb_dir, parent_clonotypes.PDB_SUFFIX
        )
        return antibody_batch.process_every_parent(parents, one, args.out_rejected)


if __name__ == "__main__":
    sys.exit(main())
