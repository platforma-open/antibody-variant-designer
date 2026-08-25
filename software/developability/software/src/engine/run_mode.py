"""Resolves a run mode into the ordered objectives it runs, and each objective's design targets.

`liabilities` runs the liability objective alone, targeting every triaged liability, highest
risk level first. `humanization` runs the humanness objective alone, targeting every non-human
framework position its own selection finds; the liability objective does not run in this mode
and receives no targets. `liabilities + humanization` runs both, in that order.
"""

from collections.abc import Callable

from engine import design_objective, humanness_objective, liability_objective, liability_triage

LIABILITY = "liability"
HUMANNESS = "humanness"

LIABILITIES = "liabilities"
HUMANIZATION = "humanization"
LIABILITIES_AND_HUMANIZATION = "liabilities + humanization"

MODES = (LIABILITIES, HUMANIZATION, LIABILITIES_AND_HUMANIZATION)
DEFAULT_MODE = LIABILITIES

_RISK_LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _risk_level_order(triaged: liability_triage.Triaged) -> int:
    """Sort key from `_RISK_LEVEL_ORDER` — a higher-risk liability's target therefore reaches
    candidate generation, and ranking, before a lower-risk one's, within one parent."""
    return _RISK_LEVEL_ORDER[triaged.risk_level]


def _as_design_target(triaged: liability_triage.Triaged) -> design_objective.DesignTarget:
    """Converts one triaged liability to the one design-target shape every objective's
    selection returns. `triaged.site[0].region` is the same region a liability candidate is
    already labelled with — `variant_candidates.py`'s own `Candidate.region` reads it the
    same way."""
    return design_objective.DesignTarget(
        site=tuple(triaged.site),
        definition_id=triaged.definition_id,
        region=triaged.site[0].region,
        is_low_confidence=triaged.low_confidence,
        confidence_angstroms=triaged.confidence_angstroms,
    )


def objectives_for(
    mode: str, prior_path: str, non_human_prior_cutoff: float
) -> list[tuple[str, Callable[[list], design_objective.Objective]]]:
    """Returns the (name, builder) pairs `mode` runs, in run order.

    Each builder takes the parent's residue index and returns the built `Objective` —
    the humanization objective needs the residues to read the prior over and the cutoff to
    select against; the liability objective ignores them."""
    def humanness_builder(residues: list) -> design_objective.Objective:
        return humanness_objective.build(prior_path, residues, non_human_prior_cutoff)

    if mode == LIABILITIES:
        return [(LIABILITY, lambda _residues: liability_objective.OBJECTIVE)]
    if mode == HUMANIZATION:
        return [(HUMANNESS, humanness_builder)]
    if mode == LIABILITIES_AND_HUMANIZATION:
        return [
            (LIABILITY, lambda _residues: liability_objective.OBJECTIVE),
            (HUMANNESS, humanness_builder),
        ]
    raise ValueError(f"unknown run mode: {mode}")


def targets_for(
    name: str,
    mode: str,
    triaged_list: list[liability_triage.Triaged],
    objective: design_objective.Objective,
    residues: list,
) -> list[design_objective.DesignTarget]:
    """Returns which design targets `name` designs against, in `mode` — the one place that
    decides which targets an objective receives.

    The humanness objective's targets are its own selection over the parent's residues. The
    liability objective targets every triaged liability, highest risk level first, except in
    `humanization` mode, where it runs no design at all and gets none."""
    if name == HUMANNESS:
        return objective.select_target_positions(residues, [])
    if mode == HUMANIZATION:
        return []
    return [_as_design_target(t) for t in sorted(triaged_list, key=_risk_level_order)]
