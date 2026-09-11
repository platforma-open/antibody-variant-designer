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
    # Which check turned the candidate away, and the measurement behind it. Both are `""` on
    # a pass and on an objective that names no reason. `build_variants` renders them onto the
    # rejection page, so a parent that produced nothing says why rather than only that it did.
    reason: str = ""
    detail: str = ""
    # The site's own `(chain, imgt)` positions that caused the failure. A builder may drop
    # these and score what is left. Empty means the whole site failed together, so dropping a
    # part of it would change nothing.
    blocking_positions: tuple = ()


@dataclass(frozen=True)
class EditCaps:
    """The two per-objective bounds a variant's edit set may not exceed. Neither cap
    bounds the other, and no single cap bounds the whole edit set."""

    liability: int
    framework: int


@dataclass(frozen=True)
class DesignTarget:
    """One objective's unit of work for one antibody: the residues a single candidate
    substitutes together.

    `definition_id` names the scanned liability the target addresses, and is `None` when the
    target addresses none — a humanization target has no taxonomy entry to point at. `objective`
    names which objective selected the target, and no consumer may re-derive it from
    `definition_id` being `None` or from a residue's region. Every other field rides
    forward onto the candidate built here, and the generator computes none of them."""

    site: tuple
    definition_id: str | None
    region: str | None
    is_low_confidence: bool
    confidence_angstroms: float | None
    objective: str


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
