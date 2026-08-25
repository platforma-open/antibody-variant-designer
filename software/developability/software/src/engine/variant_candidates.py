"""Builds candidate substitutions and checks each one against its
objective's goal — the last filter between a triaged liability and a
shipped variant.

`build_variants.py` runs this module and then `variant_ranking.py` in one exec.
Ranking discards nothing and reads nothing this module did not just
produce, so nothing needs to cross a process boundary between the two.
"""

import itertools
from dataclasses import dataclass, replace

from engine import design_objective

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
    # liability_triage.Triaged this candidate was built from; this module computes none of them.
    region: str | None
    low_confidence: bool
    worst_confidence_angstroms: float | None
    # addressed_target and changed_positions are this module's own output: a human-readable label
    # for the targeted liability, and the fixed-spelling rendering of the edits.
    addressed_target: str
    changed_positions: str
    # humanness_score keeps the humanization objective's own score separate from
    # `tolerance`, which every objective computes the same way.
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


_RISK_LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _risk_level_order(triaged) -> int:
    """Sort key from `_RISK_LEVEL_ORDER`, the order `build_candidates`
    iterates liabilities in.

    A higher-risk liability's candidates are therefore built, and reach
    ranking, before a lower-risk one's, within one parent."""
    return _RISK_LEVEL_ORDER[triaged.risk_level]


def _addressed_target(definition_id: str, site: list, taxonomy_by_id: dict) -> str:
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


def build_candidates(
    triaged_list: list,
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
    """One candidate per liability whose objective goal check passed. Every position in the site
    is substituted together, so a candidate's edit count equals the site's length.

    A site longer than `max_edits_per_variant`, or holding a position with no admissible
    substitution, is skipped before the objective runs.

    `is_humanness_objective` comes from `run_mode.objectives_for`'s `(name, builder)` pair.
    `objective` carries no name.

    `score_candidate` sees only the candidate's site, never the residue index or the PDB."""
    taxonomy_by_id = {d["id"]: d for d in taxonomy}
    prior = (
        None
        if objective.position_prior is None
        else objective.position_prior(residues, triaged_list)
    )
    candidates: list[Candidate] = []
    for triaged in sorted(triaged_list, key=_risk_level_order):
        site = triaged.site
        if len(site) > max_edits_per_variant:
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
            for residue in site
        ]
        if any(len(options) == 0 for options in per_position_options):
            continue

        addressed_target = _addressed_target(triaged.definition_id, site, taxonomy_by_id)

        for combo in itertools.product(*per_position_options):
            mutated_site = [
                replace(residue, wild_type=to_aa)
                for residue, to_aa in zip(site, combo, strict=True)
            ]
            check = objective.score_candidate(mutated_site, taxonomy, tolerance_lookup)
            if not check.meets_goal:
                continue

            # tolerance is one field with one meaning for every objective; see its doc
            # comment above.
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
            candidates.append(
                Candidate(
                    target_definition_id=triaged.definition_id,
                    edits=edits,
                    tolerance=tolerance,
                    region=site[0].region,
                    low_confidence=triaged.low_confidence,
                    worst_confidence_angstroms=triaged.confidence_angstroms,
                    addressed_target=addressed_target,
                    changed_positions=_changed_positions(edits),
                    humanness_score=humanness_score,
                )
            )
    return candidates
