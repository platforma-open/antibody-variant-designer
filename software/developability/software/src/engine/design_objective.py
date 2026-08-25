"""The seam between the design engine and what it is designing toward.

The engine owns exposure, triage, the confidence gate, binding-risk banding and ranking. Every
`Objective` plugs into that machinery rather than reimplementing it.

An objective owns exactly three things: which positions to target, an optional per-position
residue preference, and whether a candidate met the goal.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class GoalCheck:
    meets_goal: bool
    score: float


@dataclass(frozen=True)
class DesignTarget:
    """One objective's unit of work for one antibody: the residues a single candidate
    substitutes together.

    `definition_id` names the scanned liability the target addresses, and is `None` when the
    target addresses none — a humanization target has no taxonomy entry to point at. Every
    other field rides forward onto the candidate built here, and the generator computes none
    of them."""

    site: tuple
    definition_id: str | None
    region: str | None
    is_low_confidence: bool
    confidence_angstroms: float | None


@dataclass(frozen=True)
class Objective:
    select_target_positions: Callable[[list, list[dict]], list]
    position_prior: Callable[[list, list], dict[tuple[str, str], dict[str, float]]] | None
    score_candidate: Callable[[list, list[dict], dict], GoalCheck]


def mean_tolerance(mutated_site: list, tolerance_lookup: dict) -> float:
    """The mean AntiFold perplexity over `mutated_site`'s positions.

    Every `Candidate.tolerance` is this value, computed the same way regardless of which
    objective scored the candidate. It is never `GoalCheck.score`, which the humanization
    objective uses to carry a different quantity, the OASis identity."""
    perplexities = [
        tolerance_lookup[(residue.chain, residue.imgt)]["perplexity"] for residue in mutated_site
    ]
    return sum(perplexities) / len(perplexities)
