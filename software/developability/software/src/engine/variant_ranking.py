"""Ranks candidates by structural tolerance, bands each one's binding
risk, and renders each variant's sequence.

`build_variants.py` calls this module right after `variant_candidates.py`, in
the same process.

Every candidate reaching this module already passed the re-scan gate in
`variant_candidates.py`. This module orders and truncates. It filters nothing.
"""

import math

from engine import residue_store, variant_candidates, variant_store

DEFAULT_VARIANTS_PER_PARENT = 10
DEFAULT_LOW_TOLERANCE_FLOOR = 3.0
DEFAULT_EPISTASIS_RESCORE_TOP_K = 20

STATUS = "unvalidated-hypothesis"

# Perplexity units subtracted per edit beyond a candidate's first edit.
EPISTASIS_EDIT_PENALTY = 1.0


def _low_tolerance_positions(tolerance_lookup: dict, floor: float) -> set:
    """Find low-tolerance positions in a tolerance table.

    tolerance_lookup must be the whole table, never a filtered candidate set —
    filtering would shrink the bottom third and shift the threshold.

    Perplexity is entropy in bits of AntiFold's amino-acid distribution at
    each position. Higher perplexity means more substitutions fit there.

    A position qualifies as low-tolerance when its perplexity falls in the
    bottom third or at/below floor."""
    perplexities = sorted(row["perplexity"] for row in tolerance_lookup.values())
    n = len(perplexities)
    threshold = perplexities[max(0, math.ceil(n / 3) - 1)] if n else None
    return {
        key
        for key, row in tolerance_lookup.items()
        if row["perplexity"] <= floor or (threshold is not None and row["perplexity"] <= threshold)
    }


def binding_risk(candidate: variant_candidates.Candidate, low_tolerance_positions: set) -> str:
    """Bands `candidate`'s binding risk from its region, its edited
    positions' tolerance, and its low-confidence flag.

    This package has no paratope model and no binding-affinity predictor.
    The band is a heuristic proxy for risk, not a computed score."""
    is_cdr = candidate.region is not None and candidate.region.startswith("CDR")
    is_low_tolerance = any((e.chain, e.imgt) in low_tolerance_positions for e in candidate.edits)
    if is_cdr:
        return "High" if (is_low_tolerance or candidate.low_confidence) else "Medium"
    return "Medium" if is_low_tolerance else "Low"


def _epistasis_adjusted_tolerance(candidate: variant_candidates.Candidate) -> float:
    """Lowers the tolerance score of a multi-edit candidate.

    `candidate.tolerance` is the mean perplexity of the edited positions. A mean
    cannot show how two edits change each other. Biologists call this epistasis.
    This package has no epistasis model. The lower score is a guess.

    The lower score changes the order inside `rank_variants`'s top window only.
    The stored `tolerance` keeps its value."""
    return candidate.tolerance - EPISTASIS_EDIT_PENALTY * (len(candidate.edits) - 1)


def build_variant_sequence(
    residues: list[residue_store.Residue], edits: tuple[variant_candidates.Edit, ...]
) -> str:
    """Returns edits applied to every in-scope residue in rendering order.

    residues carries wild-type values for all unedited positions. Residues
    render in chain_role, chain, offset order (H before L). Every antibody
    sequence in this package uses this order.

    Each position takes its edit's to value when one exists at (chain, offset),
    otherwise its wild-type."""
    edit_by_key = {(e.chain, e.offset): e.to for e in edits}
    in_scope = sorted(
        (r for r in residues if r.in_scope),
        key=lambda r: (r.chain_role, r.chain, r.offset),
    )
    return "".join(edit_by_key.get((r.chain, r.offset), r.wild_type) for r in in_scope)


def rank_variants(
    candidate_list: list[variant_candidates.Candidate],
    residues: list[residue_store.Residue],
    tolerance_lookup: dict,
    variants_per_parent: int,
    low_tolerance_floor: float,
    epistasis_rescore_top_k: int,
) -> list[variant_store.Variant]:
    """Rank one parent's candidates and return the top variants.

    Sort by structural tolerance (best first), breaking ties by
    changed_positions. Re-sort the leading K candidates by edit-count penalty,
    then keep the top N.

    Binding risk is reported but does not decide the order.

    Variant.rank holds the local (parent-wise) position here.
    variant_store.rewrite_global_rank overwrites rank later with a
    dataset-wide ordinal (this call cannot compute it — not all parents' data
    is available yet). Variant.parent_rank stays unchanged and is used for the
    Variants page's "best pick" column."""
    low_tolerance_positions = _low_tolerance_positions(tolerance_lookup, low_tolerance_floor)
    # No residue carries the `L` role in a VHH.
    # `read_tolerance.pick_chains` derives VHH the same way.
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
                parent_rank=rank,
                chain=chain,
                addressed_target=candidate.addressed_target,
                changed_positions=candidate.changed_positions,
                variant_sequence=build_variant_sequence(residues, candidate.edits),
                structural_tolerance=candidate.tolerance,
                worst_confidence_angstroms=candidate.worst_confidence_angstroms,
                binding_risk=band,
                low_confidence_warning=candidate.low_confidence,
                humanness_score=candidate.humanness_score,
                status=STATUS,
            )
        )
    return variants
