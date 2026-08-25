"""Builds candidate substitutions and checks each one against its
objective's goal — the last filter between a design target and a
shipped variant.

`build_variants.py` runs this module and then `variant_ranking.py` in one exec.
Ranking discards nothing and reads nothing this module did not just
produce, so nothing needs to cross a process boundary between the two.
"""

import itertools
from dataclasses import dataclass, replace

from engine import design_objective, humanness_objective

DEFAULT_MAX_EDITS_PER_VARIANT = 5
DEFAULT_CANDIDATE_RESIDUES_PER_POSITION = 3
DEFAULT_W_STRUCT = 1.0
DEFAULT_W_OBJ = 1.0


@dataclass(frozen=True)
class Edit:
    """One substitution: `wild_type` at `(chain, offset)` becomes `to`.

    Carries `imgt` alongside `offset` for the same reason
    `residue_store` does. The display label and the join key are
    different things. A reader needing either finds it here, with no
    second lookup."""

    chain: str
    offset: int
    imgt: str
    wild_type: str
    to: str


@dataclass(frozen=True)
class Candidate:
    """One re-scan-cleared substitution set, still keyed to the liability it was built to
    address. Handed straight to `variant_ranking.py` in the same exec, so it is never serialized."""

    target_definition_id: str
    edits: tuple[Edit, ...]
    # The mean AntiFold perplexity over the edited positions — the entropy, in bits, of
    # AntiFold's amino-acid distribution at each position, a property of the positions rather
    # than of the substituted amino acid, unlike the per-amino-acid log-probability
    # `_top_substitutions` ranks by. Computed by `design_objective.mean_tolerance`.
    tolerance: float
    # region, low_confidence and worst_confidence_angstroms ride forward unchanged from the
    # design_objective.DesignTarget this candidate was built from; this module computes none
    # of them.
    region: str | None
    low_confidence: bool
    worst_confidence_angstroms: float | None
    # addressed_target and changed_positions are this module's own output: a human-readable label
    # for the targeted liability, and the fixed-spelling rendering of the edits.
    addressed_target: str
    changed_positions: str
    # humanness_score keeps the humanization objective's own score separate from
    # `tolerance`.
    humanness_score: float | None = None


def _combined_scores(
    residue,
    log_probs: dict[str, float],
    prior: dict | None,
    w_struct: float,
    w_obj: float,
) -> dict[str, float]:
    """The structural log-probability row and the objective's position
    prior at this residue, each scaled by its own weight, then summed.

    An uncovered position, or an objective with no prior at all, takes
    only the `w_struct` scaling. The weight then means the same thing on
    every path."""
    prior_row = None if prior is None else prior.get((residue.chain, residue.imgt))
    if prior_row is None:
        return {aa: w_struct * value for aa, value in log_probs.items()}
    return {
        aa: w_struct * value + w_obj * prior_row.get(aa, 0.0)
        for aa, value in log_probs.items()
    }


def _top_substitutions(
    residue,
    tolerance_lookup: dict,
    k: int,
    prior: dict | None,
    w_struct: float,
    w_obj: float,
) -> list[str]:
    """Up to `k` amino acids at `residue`'s position, ranked by the
    combined score, wild type excluded.

    Substituting a position to its own residue changes nothing, so that
    substitution never becomes a candidate."""
    row = tolerance_lookup.get((residue.chain, residue.imgt))
    if row is None:
        return []
    scores = _combined_scores(residue, row["logProbs"], prior, w_struct, w_obj)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [aa for aa, _ in ranked if aa != residue.wild_type][:k]


def _changed_positions(edits: tuple) -> str:
    """The fixed CSV-contract spelling: `<chain>:<wt><imgtLabel><mut>`,
    comma-separated, one entry per edit in site order."""
    return ", ".join(f"{e.chain}:{e.wild_type}{e.imgt}{e.to}" for e in edits)


def _addressed_target(definition_id: str, site: tuple, taxonomy_by_id: dict) -> str:
    """A human-readable label for the liability this candidate was built
    to clear: the taxonomy's own name plus where it sits. Neither
    `liability_triage.Triaged` nor `Candidate` carries a display string of its
    own."""
    definition = taxonomy_by_id.get(definition_id, {})
    name = definition.get("name") or definition_id
    start = site[0]
    if start.region:
        return f"{name} @ {start.region} {start.chain}:{start.imgt}"
    return f"{name} @ {start.chain}:{start.imgt}"


def _humanization_addressed_target(site: tuple) -> str:
    """A human-readable label for a humanization candidate: the fixed label plus which chain
    it targets. A humanization target has no taxonomy entry to name instead."""
    return f"{humanness_objective.HUMANIZATION_LABEL} @ {site[0].chain}"


def _cleared_candidate(
    target: design_objective.DesignTarget,
    combo: tuple,
    taxonomy: list[dict],
    tolerance_lookup: dict,
    objective: design_objective.Objective,
    is_humanness_objective: bool,
    addressed_target: str,
) -> Candidate | None:
    """Scores one substitution combo against `objective`'s goal check and, on a pass, builds
    the `Candidate` every field of `target` rides forward onto — `None` on a fail."""
    site = target.site
    mutated_site = [
        replace(residue, wild_type=to_aa) for residue, to_aa in zip(site, combo, strict=True)
    ]
    check = objective.score_candidate(mutated_site, taxonomy, tolerance_lookup)
    if not check.meets_goal:
        return None

    tolerance = design_objective.mean_tolerance(mutated_site, tolerance_lookup)
    humanness_score = check.score if is_humanness_objective else None

    edits = tuple(
        Edit(
            chain=residue.chain,
            offset=residue.offset,
            imgt=residue.imgt,
            wild_type=residue.wild_type,
            to=to_aa,
        )
        for residue, to_aa in zip(site, combo, strict=True)
    )
    return Candidate(
        target_definition_id=target.definition_id,
        edits=edits,
        tolerance=tolerance,
        region=target.region,
        low_confidence=target.is_low_confidence,
        worst_confidence_angstroms=target.confidence_angstroms,
        addressed_target=addressed_target,
        changed_positions=_changed_positions(edits),
        humanness_score=humanness_score,
    )


def build_candidates(
    targets: list[design_objective.DesignTarget],
    tolerance_lookup: dict,
    residues: list,
    taxonomy: list[dict],
    objective: design_objective.Objective,
    is_humanness_objective: bool,
    max_edits_per_variant: int,
    candidate_residues_per_position: int,
    w_struct: float,
    w_obj: float,
) -> list[Candidate]:
    """One candidate per target whose objective goal check passed.

    A liability target (`definition_id` set) enumerates the product of every position's
    top substitutions, capped by `max_edits_per_variant`. A humanization target
    (`definition_id` is `None`) never enumerates a product and is never capped: it builds
    one candidate substituting every selected position to its own single top-scoring residue.

    Either target is skipped before the objective runs when one of its positions has no
    admissible substitution.

    `is_humanness_objective` comes from `run_mode.objectives_for`'s `(name, builder)` pair.
    `objective` carries no name.

    `score_candidate` sees only the candidate's site, never the residue index or the PDB."""
    taxonomy_by_id = {d["id"]: d for d in taxonomy}
    prior = (
        None if objective.position_prior is None else objective.position_prior(residues, targets)
    )
    candidates: list[Candidate] = []
    for target in targets:
        is_liability = target.definition_id is not None
        if is_liability and len(target.site) > max_edits_per_variant:
            continue

        per_position_options = [
            _top_substitutions(
                residue,
                tolerance_lookup,
                candidate_residues_per_position,
                prior,
                w_struct,
                w_obj,
            )
            for residue in target.site
        ]
        if any(len(options) == 0 for options in per_position_options):
            continue

        if is_liability:
            addressed_target = _addressed_target(target.definition_id, target.site, taxonomy_by_id)
            combos = itertools.product(*per_position_options)
        else:
            addressed_target = _humanization_addressed_target(target.site)
            combos = [tuple(options[0] for options in per_position_options)]

        for combo in combos:
            candidate = _cleared_candidate(
                target, combo, taxonomy, tolerance_lookup, objective, is_humanness_objective,
                addressed_target,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates
