"""Resolves a run mode into the ordered objectives it runs, and the one tagged target union
those objectives design against.

`liabilities` runs the liability objective alone, targeting every triaged liability, highest
risk level first. `humanization` runs the humanness objective alone, targeting every non-human
framework position its own selection finds; the liability objective does not run in this mode
and receives no targets. `liabilities + humanization` runs both, in that order — the liability
objective's targets there are cut to the triaged liabilities that lie entirely inside a CDR, so
the two objectives never propose against the same residue.
"""

from collections.abc import Callable

from engine import design_objective, humanness_objective, liability_objective, liability_triage

LIABILITY = "liability"
HUMANNESS = "humanness"

LIABILITIES = "liabilities"
HUMANIZATION = "humanization"
LIABILITIES_AND_HUMANIZATION = "liabilities + humanization"

CDR_PREFIX = "CDR"

MODES = (LIABILITIES, HUMANIZATION, LIABILITIES_AND_HUMANIZATION)
DEFAULT_MODE = LIABILITIES

_RISK_LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _risk_level_order(triaged: liability_triage.Triaged) -> int:
    """Sort key from `_RISK_LEVEL_ORDER` — a higher-risk liability's target therefore reaches
    candidate generation, and ranking, before a lower-risk one's, within one parent."""
    return _RISK_LEVEL_ORDER[triaged.risk_level]


def _as_design_target(triaged: liability_triage.Triaged) -> design_objective.DesignTarget:
    """Converts one triaged liability to the one design-target shape every objective's
    selection returns, tagged `LIABILITY`. `triaged.site[0].region` is the same region a
    liability candidate is already labelled with."""
    return design_objective.DesignTarget(
        site=tuple(triaged.site),
        definition_id=triaged.definition_id,
        region=triaged.site[0].region,
        is_low_confidence=triaged.low_confidence,
        confidence_angstroms=triaged.confidence_angstroms,
        objective=LIABILITY,
    )


def objectives_for(
    mode: str,
    prior_path: str,
    non_human_prior_margin: float,
    fr_confidence_threshold: float = liability_triage.DEFAULT_FR_CONFIDENCE_THRESHOLD,
    max_new_liabilities: int = humanness_objective.DEFAULT_MAX_NEW_LIABILITIES,
    ignored_liability_ids: frozenset[str] = humanness_objective.DEFAULT_IGNORED_LIABILITY_IDS,
) -> list[tuple[str, Callable[[list], design_objective.Objective]]]:
    """Returns the (name, builder) pairs `mode` runs, in run order.

    Each builder takes the parent's residue index and returns the built `Objective` —
    the humanization objective needs the residues to read the prior over, the cutoff to
    select against, the threshold to warn on, the liabilities its gate may accept and the
    ones it does not count at all; the liability objective ignores them, and reads its own
    confidence from the triage that already ran."""
    def humanness_builder(residues: list) -> design_objective.Objective:
        return humanness_objective.build(
            prior_path,
            residues,
            non_human_prior_margin,
            fr_confidence_threshold,
            max_new_liabilities,
            ignored_liability_ids,
        )

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


def _in_cdr(triaged: liability_triage.Triaged) -> bool:
    """A triaged liability's whole site lies inside a CDR — every one of its site residues
    carries a region starting with `CDR_PREFIX`."""
    return all(r.region is not None and r.region.startswith(CDR_PREFIX) for r in triaged.site)


def targets_for(
    mode: str,
    triaged_list: list[liability_triage.Triaged],
    objectives: dict[str, design_objective.Objective],
    residues: list,
) -> list[design_objective.DesignTarget]:
    """Returns the one tagged target union every objective `objectives` names designs
    against, liability targets first — the one place that decides which targets an objective
    receives.

    The liability objective's presence in `objectives` is what gates its targets, not `mode`
    directly: it is absent in `humanization` mode, where the liability objective does not run
    and gets none. In `liabilities + humanization` mode a liability target must also lie
    entirely inside a CDR — a framework liability is still scanned, triaged and reported, but
    never designed against while humanization runs beside it. The humanness objective's
    targets are its own selection over the parent's residues, tagged `HUMANNESS`."""
    liability_targets: list[design_objective.DesignTarget] = []
    if LIABILITY in objectives:
        triage_source = triaged_list
        if mode == LIABILITIES_AND_HUMANIZATION:
            triage_source = [t for t in triage_source if _in_cdr(t)]
        liability_targets = [
            _as_design_target(t) for t in sorted(triage_source, key=_risk_level_order)
        ]

    humanness_targets: list[design_objective.DesignTarget] = []
    if HUMANNESS in objectives:
        selected = objectives[HUMANNESS].select_target_positions(residues, [])
        humanness_targets = list(selected)

    return liability_targets + humanness_targets
