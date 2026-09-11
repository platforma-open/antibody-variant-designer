"""Builds candidates (edit sets) checking each against every objective that contributed to it —
the last filter between a parent's design targets and a shipped variant.

`build_variants.py` runs this module and then `variant_ranking.py` in one exec.
Ranking discards nothing and reads nothing this module did not just
produce, so nothing needs to cross a process boundary between the two.
"""

import heapq
import itertools
from dataclasses import dataclass, replace

from engine import design_objective, humanness_objective

DEFAULT_MAX_LIABILITY_EDITS = 10
DEFAULT_MAX_FRAMEWORK_EDITS = 20
DEFAULT_CANDIDATE_RESIDUES_PER_POSITION = 3
DEFAULT_STRUCTURAL_WEIGHT = 1.0
DEFAULT_OBJECTIVE_WEIGHT = 1.0

# The two objective tags `DesignTarget.objective` carries — the same spellings
# `run_mode.LIABILITY` and `humanness_objective.HUMANNESS_OBJECTIVE_TAG` use. This module reads
# no other shape of objective identity.
_LIABILITY = "liability"
_HUMANNESS = "humanness"


@dataclass(frozen=True)
class Edit:
    """One substitution: `wild_type` at `(chain, offset)` becomes `to`.

    Carries `imgt` alongside `offset` for the same reason `residue_store` does — the display
    label and the join key are different things. Carries its own `region` and `objective`
    because a candidate spanning both regions, or built from more than one objective's targets,
    has no single one of either to read off the candidate as a whole."""

    chain: str
    offset: int
    imgt: str
    wild_type: str
    to: str
    region: str | None
    objective: str


@dataclass(frozen=True)
class Candidate:
    """One re-scan-cleared edit set: every admitted design target's edits the walk in
    `build_candidates_and_declines` reached together. Handed straight to `variant_ranking.py`
    in the same exec, so it is never serialized."""

    # One id per addressed liability target, in target order. A humanization target
    # contributes no id — it has no taxonomy entry to point at.
    target_definition_ids: tuple[str, ...]
    edits: tuple[Edit, ...]
    # The mean AntiFold perplexity over the edited positions — the entropy, in bits, of
    # AntiFold's amino-acid distribution at each position, a property of the positions rather
    # than of the substituted amino acid, unlike the per-amino-acid log-probability
    # `_top_substitutions` ranks by. Computed by `design_objective.mean_tolerance`.
    tolerance: float
    # low_confidence and worst_confidence_angstroms fold every addressed target's own reading
    # together: low_confidence is true if any addressed target's own flag is, and
    # worst_confidence_angstroms is the worst of every addressed target's own value.
    low_confidence: bool
    worst_confidence_angstroms: float | None
    # addressed_target and changed_positions are this module's own output: a human-readable
    # label per addressed target, joined in target order, and the fixed-spelling rendering of
    # the edits.
    addressed_target: str
    changed_positions: str
    # How many design targets this edit set addresses — the primary ordering term
    # `variant_ranking` and `rank_variants` never see directly; ranking orders on `tolerance`.
    coverage: int


def _combined_scores(
    residue,
    log_probs: dict[str, float],
    prior: dict | None,
    structural_weight: float,
    objective_weight: float,
) -> dict[str, float]:
    """The structural log-probability row and the objective's position
    prior at this residue, each scaled by its own weight, then summed.

    An uncovered position, or an objective with no prior at all, takes
    only the `structural_weight` scaling. The weight then means the same thing on
    every path."""
    prior_row = None if prior is None else prior.get((residue.chain, residue.imgt))
    if prior_row is None:
        return {aa: structural_weight * value for aa, value in log_probs.items()}
    return {
        aa: structural_weight * value + objective_weight * prior_row.get(aa, 0.0)
        for aa, value in log_probs.items()
    }


def _ranked_substitutions(
    residue,
    tolerance_lookup: dict,
    k: int,
    prior: dict | None,
    structural_weight: float,
    objective_weight: float,
) -> list[tuple[str, float]]:
    """Up to `k` `(amino_acid, combined_score)` pairs at `residue`'s position, best first,
    wild type excluded. `_top_substitutions` is this with the score dropped; the walk in
    `build_candidates_and_declines` keeps the score to order edit sets by summed score loss."""
    row = tolerance_lookup.get((residue.chain, residue.imgt))
    if row is None:
        return []
    scores = _combined_scores(residue, row["logProbs"], prior, structural_weight, objective_weight)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(aa, score) for aa, score in ranked if aa != residue.wild_type][:k]


def _top_substitutions(
    residue,
    tolerance_lookup: dict,
    k: int,
    prior: dict | None,
    structural_weight: float,
    objective_weight: float,
) -> list[str]:
    """Up to `k` amino acids at `residue`'s position, ranked by the
    combined score, wild type excluded.

    Substituting a position to its own residue changes nothing, so that
    substitution never becomes a candidate."""
    return [
        aa
        for aa, _ in _ranked_substitutions(
            residue, tolerance_lookup, k, prior, structural_weight, objective_weight
        )
    ]


def unscored_positions(site: tuple, tolerance_lookup: dict) -> tuple:
    """The site's residues no tolerance row covers.

    An unscored position offers no substitution, so `admitted_targets` drops the whole
    target rather than a part of it. `build_variants` reads this same function to name why
    a parent produced nothing."""
    return tuple(r for r in site if (r.chain, r.imgt) not in tolerance_lookup)


def _changed_positions(edits: tuple) -> str:
    """The fixed CSV-contract spelling: `<chain>:<wt><imgtLabel><mut>`,
    comma-separated, one entry per edit in site order."""
    return ", ".join(f"{e.chain}:{e.wild_type}{e.imgt}{e.to}" for e in edits)


def _addressed_target(definition_id: str, site: tuple, taxonomy_by_id: dict) -> str:
    """A human-readable label for one liability this edit set addresses: the taxonomy's own
    name plus where it sits."""
    definition = taxonomy_by_id.get(definition_id, {})
    name = definition.get("name") or definition_id
    start = site[0]
    if start.region:
        return f"{name} @ {start.region} {start.chain}:{start.imgt}"
    return f"{name} @ {start.chain}:{start.imgt}"


def _humanization_addressed_target(site: tuple) -> str:
    """A human-readable label for the humanization contribution: the fixed label plus which
    chain it targets. A humanization target has no taxonomy entry to name instead."""
    return f"{humanness_objective.HUMANIZATION_LABEL} @ {site[0].chain}"


def _combined_addressed_target(targets_present: list, taxonomy_by_id: dict) -> str:
    """One label per addressed target, in target order — a liability target through
    `_addressed_target`, a humanization target through `_humanization_addressed_target`."""
    labels = [
        _addressed_target(target.definition_id, target.site, taxonomy_by_id)
        if target.definition_id is not None
        else _humanization_addressed_target(target.site)
        for target in targets_present
    ]
    return ", ".join(labels)


def _worst_confidence(targets_present: list) -> float | None:
    values = [t.confidence_angstroms for t in targets_present if t.confidence_angstroms is not None]
    return max(values) if values else None


@dataclass(frozen=True)
class Decline:
    """The one target set whose goal check turned away every edit set the walk tried.

    `reason` and `detail` come from the refusing objective's own `GoalCheck`, so a parent that
    produced nothing says which check stopped it, and `objective` names which one. A parent
    that produced at least one candidate raises no `Decline`."""

    chain: str
    reason: str
    detail: str
    objective: str


def admitted_targets(
    targets: list[design_objective.DesignTarget],
    tolerance_lookup: dict,
    caps: design_objective.EditCaps,
) -> list[design_objective.DesignTarget]:
    """Every target whose every position a tolerance row covers, each objective's targets
    truncated to that objective's own cap in `caps`. A target left with no position
    once its objective's cap is spent is dropped whole; a target only partly over the cap is
    truncated to what remains, keeping its earliest positions."""
    caps_by_objective = {_LIABILITY: caps.liability, _HUMANNESS: caps.framework}
    used: dict[str, int] = {}
    admitted: list[design_objective.DesignTarget] = []
    for target in targets:
        if unscored_positions(target.site, tolerance_lookup):
            continue
        cap = caps_by_objective.get(target.objective)
        if cap is None:
            admitted.append(target)
            continue
        remaining = cap - used.get(target.objective, 0)
        if remaining <= 0:
            continue
        site = target.site if len(target.site) <= remaining else target.site[:remaining]
        used[target.objective] = used.get(target.objective, 0) + len(site)
        admitted.append(target if site is target.site else replace(target, site=site))
    return admitted


def full_coverage_set(
    targets: list[design_objective.DesignTarget],
    per_position_options: dict,
) -> tuple:
    """The top-scoring residue at every admitted target's position, in target then site
    order — the seed `build_candidates_and_declines`'s walk starts from. `per_position_options`
    maps `(chain, imgt)` to that position's ranked `(amino_acid, score)` pairs, best first."""
    return tuple(
        per_position_options[(residue.chain, residue.imgt)][0][0]
        for target in targets
        for residue in target.site
    )


def _cleared_candidate(
    positions: list,
    combo: tuple,
    taxonomy: list[dict],
    tolerance_lookup: dict,
    objectives: dict[str, design_objective.Objective],
    taxonomy_by_id: dict,
):
    """Scores one substitution combo against every objective that contributed an edit to it,
    and, on a pass from all of them, builds the `Candidate`.

    Returns `(candidate, blocking_positions, refused_by, checks)`. `candidate` is `None` on a
    fail, `blocking_positions` the union of every failed check's own blocking positions,
    `refused_by` the name of the first objective that failed, and `checks` every objective's
    own `GoalCheck`, keyed by name, for a caller that needs to report why."""
    scored = [
        (target, residue, replace(residue, wild_type=to_aa))
        for (target, residue), to_aa in zip(positions, combo, strict=True)
    ]
    checks: dict[str, design_objective.GoalCheck] = {}
    blocking_positions: set = set()
    refused_by = None
    for name, objective in objectives.items():
        mine = [scored_residue for target, _, scored_residue in scored if target.objective == name]
        if not mine:
            continue  # this objective contributed no edit — not consulted
        check = objective.score_candidate(mine, taxonomy, tolerance_lookup)
        checks[name] = check
        if not check.meets_goal:
            if refused_by is None:
                refused_by = name
            blocking_positions.update(check.blocking_positions)

    if refused_by is not None:
        return None, blocking_positions, refused_by, checks

    targets_present = list(dict.fromkeys(target for target, _, _ in scored))
    edits = tuple(
        Edit(
            chain=residue.chain,
            offset=residue.offset,
            imgt=residue.imgt,
            wild_type=residue.wild_type,
            to=scored_residue.wild_type,
            region=residue.region,
            objective=target.objective,
        )
        for target, residue, scored_residue in scored
    )
    tolerance = design_objective.mean_tolerance(
        [scored_residue for _, _, scored_residue in scored], tolerance_lookup
    )
    candidate = Candidate(
        target_definition_ids=tuple(
            t.definition_id for t in targets_present if t.definition_id is not None
        ),
        edits=edits,
        tolerance=tolerance,
        low_confidence=any(t.is_low_confidence for t in targets_present),
        worst_confidence_angstroms=_worst_confidence(targets_present),
        addressed_target=_combined_addressed_target(targets_present, taxonomy_by_id),
        changed_positions=_changed_positions(edits),
        coverage=len(targets_present),
    )
    return candidate, None, None, checks


def _scored_dropping_blockers(
    positions: list,
    combo: tuple,
    taxonomy: list[dict],
    tolerance_lookup: dict,
    objectives: dict[str, design_objective.Objective],
    taxonomy_by_id: dict,
):
    """`_cleared_candidate`, retried without the positions each failed check blocks on.

    One position whose substitution fails an objective's check would otherwise sink every
    other edit beside it. Dropping that position and scoring the rest keeps the edits that are
    fine. A check that blocks on nothing ends the retry, and so does an empty position list.
    Returns `(candidate, refused_by, checks)`; `candidate` is `None` on a fail, with the last
    attempt's refusal and checks for a caller that still wants a reason."""
    refused_by, checks = None, {}
    while positions:
        candidate, blocking, refused_by, checks = _cleared_candidate(
            positions, combo, taxonomy, tolerance_lookup, objectives, taxonomy_by_id
        )
        if candidate is not None:
            return candidate, None, checks
        if not blocking:
            return None, refused_by, checks
        kept = [
            (position, to_aa)
            for position, to_aa in zip(positions, combo, strict=True)
            if (position[1].chain, position[1].imgt) not in blocking
        ]
        # Every position blocked means nothing is left to try, and a check that blocked none
        # of them already returned above — so the position list always shrinks here.
        positions = [position for position, _ in kept]
        combo = tuple(to_aa for _, to_aa in kept)
    return None, refused_by, checks


def build_candidates(
    targets: list[design_objective.DesignTarget],
    tolerance_lookup: dict,
    residues: list,
    taxonomy: list[dict],
    objectives: dict[str, design_objective.Objective],
    caps: design_objective.EditCaps,
    candidate_residues_per_position: int,
    structural_weight: float,
    objective_weight: float,
    set_limit: int,
) -> list[Candidate]:
    """The candidates of `build_candidates_and_declines`, for a caller with no use for the
    declines."""
    return build_candidates_and_declines(
        targets,
        tolerance_lookup,
        residues,
        taxonomy,
        objectives,
        caps,
        candidate_residues_per_position,
        structural_weight,
        objective_weight,
        set_limit,
    )[0]


def build_candidates_and_declines(
    targets: list[design_objective.DesignTarget],
    tolerance_lookup: dict,
    residues: list,
    taxonomy: list[dict],
    objectives: dict[str, design_objective.Objective],
    caps: design_objective.EditCaps,
    candidate_residues_per_position: int,
    structural_weight: float,
    objective_weight: float,
    set_limit: int,
) -> tuple[list[Candidate], list[Decline]]:
    """Walks edit sets best-first from full coverage and returns up to `set_limit`
    cleared candidates ordered by coverage then score, plus one `Decline` naming the objective
    that refused, when none clear at all.

    `admitted_targets` truncates each objective's targets to its own cap in `caps` and drops
    a target whose positions no tolerance row covers first. `full_coverage_set` is the walk's
    seed: every admitted position at its top-scoring residue. Each step swaps one position to
    its next-best residue, ordered by cumulative score loss against the seed ascending. A
    dropped position can lower a low-loss pick's coverage below a higher-loss pick's, so
    cleared sets are buffered — the walk keeps popping until it has confirmed a full-coverage
    set or exhausted the queue — then sorted by coverage before score and sliced to
    `set_limit`. A distinct pick that a blocked-position retry collapses onto an already-seen
    edit set is skipped rather than shipped twice. `score_candidate` sees only each
    objective's own contributed positions, never the residue index or the PDB."""
    admitted = admitted_targets(targets, tolerance_lookup, caps)
    if not admitted:
        return [], []

    taxonomy_by_id = {d["id"]: d for d in taxonomy}
    priors = {
        name: (
            None
            if objective.position_prior is None
            else objective.position_prior(residues, targets)
        )
        for name, objective in objectives.items()
    }

    positions = [(target, residue) for target in admitted for residue in target.site]
    ranked_by_position = {
        (residue.chain, residue.imgt): _ranked_substitutions(
            residue,
            tolerance_lookup,
            candidate_residues_per_position,
            priors.get(target.objective),
            structural_weight,
            objective_weight,
        )
        for target, residue in positions
    }
    if any(len(ranked) == 0 for ranked in ranked_by_position.values()):
        return [], []

    def ranked_of(index: int) -> list[tuple[str, float]]:
        return ranked_by_position[(positions[index][1].chain, positions[index][1].imgt)]

    def loss_of(picks: tuple) -> float:
        return sum(ranked_of(i)[0][1] - ranked_of(i)[choice][1] for i, choice in enumerate(picks))

    counter = itertools.count()
    seed_combo = full_coverage_set(admitted, ranked_by_position)
    seed = tuple(
        next(choice for choice, (aa, _) in enumerate(ranked_of(i)) if aa == seed_combo[i])
        for i in range(len(positions))
    )
    heap = [(0.0, next(counter), seed)]
    seen = {seed}

    full_coverage = len(admitted)
    buffered: list[tuple[int, float, Candidate]] = []
    seen_edit_sets: set = set()
    first_failure = None
    while heap and (
        len(buffered) < set_limit
        or not any(coverage == full_coverage for coverage, _, _ in buffered)
    ):
        loss, _, picks = heapq.heappop(heap)
        combo = tuple(ranked_of(i)[choice][0] for i, choice in enumerate(picks))
        candidate, refused_by, checks = _scored_dropping_blockers(
            positions, combo, taxonomy, tolerance_lookup, objectives, taxonomy_by_id
        )
        if candidate is not None:
            edit_set = tuple((edit.chain, edit.imgt, edit.to) for edit in candidate.edits)
            if edit_set not in seen_edit_sets:
                seen_edit_sets.add(edit_set)
                buffered.append((candidate.coverage, loss, candidate))
        elif first_failure is None and refused_by is not None:
            first_failure = (refused_by, checks[refused_by])

        for i in range(len(picks)):
            next_choice = picks[i] + 1
            if next_choice >= len(ranked_of(i)):
                continue
            neighbor = (*picks[:i], next_choice, *picks[i + 1 :])
            if neighbor in seen:
                continue
            seen.add(neighbor)
            heapq.heappush(heap, (loss_of(neighbor), next(counter), neighbor))

    buffered.sort(key=lambda entry: (-entry[0], entry[1]))
    candidates = [candidate for _, _, candidate in buffered[:set_limit]]

    declines: list[Decline] = []
    if not candidates and first_failure is not None:
        refused_by, check = first_failure
        declines.append(
            Decline(
                chain=positions[0][1].chain,
                reason=check.reason,
                detail=check.detail,
                objective=refused_by,
            )
        )
    return candidates, declines
