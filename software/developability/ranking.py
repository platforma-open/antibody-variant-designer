"""Ranking, binding-risk banding and sequence rendering.

A module, not an entrypoint: `variants.py` calls this straight after
`candidates.py` in one exec. Nothing here filters — every candidate reaching
this module already cleared the re-scan gate — so this is a presentation
pass, and its own thresholds only order and truncate.

There is no paratope model and no binding-affinity prediction anywhere in
this package, so `binding_risk` is a band, not a score: a CDR edit at a
low-tolerance position, or at a site the structure prediction itself was
unsure about, is `High`; a framework edit with either problem is `Medium`;
everything else is `Low`. A position counts as low-tolerance for `Candidate.
tolerance`'s own antibody when its AntiFold perplexity sits in the bottom
third of that antibody's own tolerance table, or at/below `low_tolerance_
floor` — either arm alone is enough, and `_low_tolerance_positions` computes
both over the whole table so filtering candidates first never shrinks the
third.

The rank key is a candidate's own structural tolerance (best first, ties
broken by `changed_positions` for determinism) — the binding-risk band is
reported alongside every variant, but it is not what orders them.

`epistasis_rescore_top_k` exists because the per-candidate tolerance is
the mean over its edited positions, which says nothing about a multi-edit
candidate's edits interacting with each other. Re-scoring every candidate
properly would need a model this package does not have, so instead only the
leading `epistasis_rescore_top_k` candidates (by the initial tolerance
order) are re-sorted among themselves, penalizing each extra edit past the
first — a deliberately blunt proxy for an interaction effect this package
cannot actually compute. The penalty only reorders that leading window; a
candidate outside it never moves, and no candidate's reported structural
tolerance changes because of it.

Rendering a variant's sequence needs the residue index as well as the
edits: the wild-type residues at every unedited position are what the edits
are applied over, and the index is the only place they are. The same index
also derives whether the antibody is a VHH — no residue carries the `L`
role — the same test `antifold.pick_chains` already makes, so this module
never carries that as a boundary field.
"""

import math

import candidates
import residue_store
import variant_store

DEFAULT_VARIANTS_PER_PARENT = 10
DEFAULT_LOW_TOLERANCE_FLOOR = 3.0
DEFAULT_EPISTASIS_RESCORE_TOP_K = 20

STATUS = "unvalidated-hypothesis"

# Perplexity units subtracted per edit beyond a candidate's first, when
# re-scoring the leading window — see the module docstring.
EPISTASIS_EDIT_PENALTY = 1.0


def _low_tolerance_positions(tolerance_lookup: dict, floor: float) -> set:
    """The `(chain, imgt)` keys counting as low-tolerance for THIS
    antibody: the bottom third of its own perplexity distribution, plus
    everything at or below `floor`. The population is the whole tolerance
    table — `rank_variants` already receives one parent's candidates only,
    so this is that parent's full table, not the candidate set filtering
    would shrink."""
    perplexities = sorted(row["perplexity"] for row in tolerance_lookup.values())
    n = len(perplexities)
    threshold = perplexities[max(0, math.ceil(n / 3) - 1)] if n else None
    return {
        key
        for key, row in tolerance_lookup.items()
        if row["perplexity"] <= floor or (threshold is not None and row["perplexity"] <= threshold)
    }


def binding_risk(candidate: candidates.Candidate, low_tolerance_positions: set) -> str:
    """`Low` / `Medium` / `High` from the candidate's region, whether any
    of its edited positions is low-tolerance, and its low-confidence
    warning — see the module docstring for why this is a band and not a
    number."""
    is_cdr = candidate.region is not None and candidate.region.startswith("CDR")
    is_low_tolerance = any((e.chain, e.imgt) in low_tolerance_positions for e in candidate.edits)
    if is_cdr:
        return "High" if (is_low_tolerance or candidate.low_confidence) else "Medium"
    return "Medium" if is_low_tolerance else "Low"


def _epistasis_adjusted_tolerance(candidate: candidates.Candidate) -> float:
    return candidate.tolerance - EPISTASIS_EDIT_PENALTY * (len(candidate.edits) - 1)


def build_variant_sequence(
    residues: list[residue_store.Residue], edits: tuple[candidates.Edit, ...]
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
    candidate_list: list[candidates.Candidate],
    residues: list[residue_store.Residue],
    tolerance_lookup: dict,
    variants_per_parent: int,
    low_tolerance_floor: float,
    epistasis_rescore_top_k: int,
) -> list[variant_store.Variant]:
    """Order by structural tolerance (best first, ties broken by
    `changed_positions` for determinism), re-sort the leading
    `epistasis_rescore_top_k` window by the edit-count penalty, then keep
    the top `variants_per_parent`. Rank restarts at 1 here, because one
    call already covers exactly one parent's candidates. Band every
    candidate for reporting, but the band decides no order here."""
    low_tolerance_positions = _low_tolerance_positions(tolerance_lookup, low_tolerance_floor)
    is_vhh = not any(r.chain_role == "L" for r in residues)
    chain = "H" if is_vhh else "H,L"

    scored = [(c, binding_risk(c, low_tolerance_positions)) for c in candidate_list]
    scored.sort(key=lambda pair: (-pair[0].tolerance, pair[0].changed_positions))

    window, rest = scored[:epistasis_rescore_top_k], scored[epistasis_rescore_top_k:]
    window.sort(
        key=lambda pair: (
            -_epistasis_adjusted_tolerance(pair[0]),
            pair[0].changed_positions,
        )
    )

    variants = []
    for rank, (candidate, band) in enumerate((window + rest)[:variants_per_parent], start=1):
        variants.append(
            variant_store.Variant(
                rank=rank,
                chain=chain,
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
