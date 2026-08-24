"""Resolves a run mode into the ordered objectives it runs, and each objective's targets.

`liabilities` runs the liability objective alone, unlocking every triaged liability as a
target. `liabilities + humanization` runs it together with the humanization objective, in
that order; the humanization objective has no target of its own until its target
selection lands, so it always contributes zero targets in this row.
"""

from collections.abc import Callable

from engine import design_objective, humanness_objective, liability_objective, liability_triage

LIABILITY = "liability"
HUMANNESS = "humanness"

LIABILITIES = "liabilities"
LIABILITIES_AND_HUMANIZATION = "liabilities + humanization"

MODES = (LIABILITIES, LIABILITIES_AND_HUMANIZATION)
DEFAULT_MODE = LIABILITIES


def objectives_for(
    mode: str, prior_path: str
) -> list[tuple[str, Callable[[list], design_objective.Objective]]]:
    """Returns the (name, builder) pairs `mode` runs, in run order.

    Each builder takes the parent's residue index and returns the built `Objective` —
    the humanization objective needs the residues to read the prior over; the liability
    objective ignores them."""
    if mode == LIABILITIES:
        return [(LIABILITY, lambda _residues: liability_objective.OBJECTIVE)]
    if mode == LIABILITIES_AND_HUMANIZATION:
        return [
            (LIABILITY, lambda _residues: liability_objective.OBJECTIVE),
            (HUMANNESS, lambda residues: humanness_objective.build(prior_path, residues)),
        ]
    raise ValueError(f"unknown run mode: {mode}")


def targets_for(name: str, triaged_list: list[liability_triage.Triaged]) -> list:
    """Returns which of a parent's triaged liabilities `name` targets.

    The liability objective targets every one of them, exactly as today. No other
    objective names a target of its own in this row."""
    if name == LIABILITY:
        return triaged_list
    return []
