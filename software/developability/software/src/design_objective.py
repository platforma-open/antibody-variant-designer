"""The seam between the design engine and what it is designing toward.

The engine owns exposure, triage, the confidence gate, binding-risk banding and ranking — every
`Objective` plugs into it rather than reimplementing any of that. An objective owns exactly three
things: which positions to target, an optional per-position residue preference, and whether a
candidate met the goal.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class GoalCheck:
    meets_goal: bool
    score: float


@dataclass(frozen=True)
class Objective:
    select_target_positions: Callable[[list, list[dict]], list]
    position_prior: Callable[[list, list], dict[tuple[str, str], dict[str, float]]] | None
    score_candidate: Callable[[list, list[dict], dict], GoalCheck]
