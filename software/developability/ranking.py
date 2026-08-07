"""Ranking, binding-risk banding and sequence rendering, the entrypoint
that writes `variants.tsv` — the block's final artifact.

There is no paratope model and no binding-affinity prediction anywhere in
this package, so `binding_risk` is a band, not a score: a CDR edit at a
position AntiFold tolerates poorly, or at a site the structure prediction
itself was unsure about, is `High`; a framework edit with neither problem
is `Low`; everything else is `Medium`. `--low-tolerance-floor` is the
perplexity below which a position counts as poorly tolerated for this
banding — the same number, on the same scale, `candidate_store.Candidate.
tolerance` already carries.

`--epistasis-rescore-top-k` exists because the per-candidate tolerance is
the worst single edited position, which says nothing about a multi-edit
candidate's edits interacting with each other. Re-scoring every candidate
this way would cost a call this package has no model to back, so instead
only the leading `--epistasis-rescore-top-k` candidates (by the initial
band-then-tolerance order) are re-sorted among themselves, penalizing each
extra edit past the first — a deliberately blunt proxy for an interaction
effect this package cannot actually compute. The penalty only reorders
that leading window; a candidate outside it never moves, and no
candidate's reported `structuralTolerance` changes because of it.

This entrypoint reads `residues.json` in addition to `candidates.json`,
which the boundary-file table does not list — a candidate's edits are not
enough to render `variantSequence` on their own; the wild-type residues at
every unedited position are needed too, and `residues.json` is the only
place they still are.
"""

import argparse
import sys
from pathlib import Path

import candidate_store
import residue_store
import variant_store

DEFAULT_VARIANTS_PER_PARENT = 10
DEFAULT_LOW_TOLERANCE_FLOOR = 3.0
DEFAULT_EPISTASIS_RESCORE_TOP_K = 20

STATUS = "unvalidated-hypothesis"

# Perplexity units subtracted per edit beyond a candidate's first, when
# re-scoring the leading window — see the module docstring.
EPISTASIS_EDIT_PENALTY = 1.0

_BAND_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def binding_risk(candidate: candidate_store.Candidate, low_tolerance_floor: float) -> str:
    """`Low` / `Medium` / `High` from the candidate's region, its
    structural tolerance against `low_tolerance_floor`, and its
    low-confidence warning — see the module docstring for why this is a
    band and not a number."""
    is_cdr = candidate.region is not None and candidate.region.startswith("CDR")
    is_low_tolerance = candidate.tolerance < low_tolerance_floor
    if is_cdr:
        return "High" if (is_low_tolerance or candidate.low_confidence) else "Medium"
    return "Medium" if (is_low_tolerance and candidate.low_confidence) else "Low"


def _epistasis_adjusted_tolerance(candidate: candidate_store.Candidate) -> float:
    return candidate.tolerance - EPISTASIS_EDIT_PENALTY * (len(candidate.edits) - 1)


def build_variant_sequence(
    residues: list[residue_store.Residue], edits: tuple[candidate_store.Edit, ...]
) -> str:
    """The full researched sequence with `edits` applied — every in-scope
    residue, ordered `chain_role` then `chain` then `offset` (H before L,
    matching the convention every rendered antibody sequence in this
    package follows), each replaced by its edit's `to` when one exists at
    its `(chain, offset)`, its own wild type otherwise."""
    edit_by_key = {(e.chain, e.offset): e.to for e in edits}
    in_scope = sorted(
        (r for r in residues if r.in_scope),
        key=lambda r: (r.chain_role, r.chain, r.offset),
    )
    return "".join(edit_by_key.get((r.chain, r.offset), r.wild_type) for r in in_scope)


def rank_variants(
    candidates: list[candidate_store.Candidate],
    residues: list[residue_store.Residue],
    variants_per_parent: int,
    low_tolerance_floor: float,
    epistasis_rescore_top_k: int,
) -> list[variant_store.Variant]:
    """Band every candidate, order by band then by tolerance (best first),
    re-sort the leading `epistasis_rescore_top_k` window by the edit-count
    penalty, then keep the top `variants_per_parent`. Rank restarts at 1
    here, because one call already covers exactly one parent's
    candidates."""
    scored = [(c, binding_risk(c, low_tolerance_floor)) for c in candidates]
    scored.sort(
        key=lambda pair: (_BAND_ORDER[pair[1]], -pair[0].tolerance, pair[0].changed_positions)
    )

    window, rest = scored[:epistasis_rescore_top_k], scored[epistasis_rescore_top_k:]
    window.sort(
        key=lambda pair: (
            _BAND_ORDER[pair[1]],
            -_epistasis_adjusted_tolerance(pair[0]),
            pair[0].changed_positions,
        )
    )

    variants = []
    for rank, (candidate, band) in enumerate((window + rest)[:variants_per_parent], start=1):
        variants.append(
            variant_store.Variant(
                rank=rank,
                addressed_target=candidate.addressed_target,
                changed_positions=candidate.changed_positions,
                variant_sequence=build_variant_sequence(residues, candidate.edits),
                structural_tolerance=candidate.tolerance,
                worst_confidence_angstroms=candidate.worst_confidence_angstroms,
                binding_risk=band,
                low_confidence_warning=candidate.low_confidence,
                status=STATUS,
            )
        )
    return variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank cleared candidates, band their binding risk, and render each sequence."
    )
    parser.add_argument("--candidates", required=True, help="candidates.py's --out-candidates")
    parser.add_argument("--residues", required=True, help="structure.py's output")
    parser.add_argument("--out-variants", required=True)
    parser.add_argument(
        "--variants-per-parent", type=int, default=DEFAULT_VARIANTS_PER_PARENT
    )
    parser.add_argument(
        "--low-tolerance-floor", type=float, default=DEFAULT_LOW_TOLERANCE_FLOOR
    )
    parser.add_argument(
        "--epistasis-rescore-top-k", type=int, default=DEFAULT_EPISTASIS_RESCORE_TOP_K
    )
    args = parser.parse_args(argv)

    for path, producer in (
        (args.candidates, "candidates.py's --out-candidates"),
        (args.residues, "structure.py's output"),
    ):
        if not Path(path).is_file():
            raise SystemExit(f"{path} does not exist — expected {producer}")

    candidates = candidate_store.read_candidates(args.candidates)
    residues = residue_store.read_residues(args.residues)

    variants = rank_variants(
        candidates,
        residues,
        args.variants_per_parent,
        args.low_tolerance_floor,
        args.epistasis_rescore_top_k,
    )

    variant_store.write_variants_tsv(args.out_variants, variants)
    return 0


if __name__ == "__main__":
    sys.exit(main())
