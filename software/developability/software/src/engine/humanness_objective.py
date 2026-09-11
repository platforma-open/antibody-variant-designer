"""The humanization objective."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

from engine import (
    design_objective,
    humanness_gate,
    liability_cysteines,
    liability_motifs,
    liability_triage,
    parent_summary,
    residue_index,
)

if TYPE_CHECKING:
    from engine import variant_candidates

FRAMEWORK_PREFIX = "FR"

# Separates one target's entry from the next in a parent's humanness summary line.
AMINO_SEP = ", "

DEFAULT_NON_HUMAN_PRIOR_MARGIN = 0.05

# How many liabilities a humanization candidate may introduce before its gate turns it away.
# Zero keeps a candidate that spells any new liability out of the run.
DEFAULT_MAX_NEW_LIABILITIES = 0

# Which liabilities the gate does not count, by taxonomy id. Empty counts every one of them,
# so a liability the taxonomy gains later blocks humanization until the operator says otherwise.
DEFAULT_IGNORED_LIABILITY_IDS: frozenset[str] = frozenset()

HUMANIZATION_LABEL = "Humanization"

# The tag `select_non_human_positions` stamps onto every `DesignTarget` it selects — the same
# spelling `run_mode.HUMANNESS` uses. A literal, not an import of `run_mode`, because `run_mode`
# imports this module to build the objective in the first place.
HUMANNESS_OBJECTIVE_TAG = "humanness"

# What `score_candidate` writes into `GoalCheck.reason`. `build_variants` renders these into
# one rejection row's detail, so the rejection page names the check a parent failed rather than only
# reporting that nothing survived.
SPANS_TWO_CHAINS_REASON = "site spans two chains"
OUTSIDE_FRAMEWORK_REASON = "position outside the framework"
TOO_MANY_NEW_LIABILITIES_REASON = "adds more liabilities than allowed"
UNSCOREABLE_REASON = "chain cannot be scored"
DID_NOT_RISE_REASON = "humanness did not rise"

# A VHH carries F, E, R and G at these IMGT positions where a human VH carries V, G, L and W.
# These residues hold the face a light chain would otherwise cover, so humanizing one costs
# the fold. `select_non_human_positions` never targets them on a nanobody.
VHH_FR2_HALLMARK_IMGT = frozenset({"42", "49", "50", "52"})


def load_prior(prior_path: str) -> dict[tuple[str, str], dict[str, float]]:
    """Read one Prior TSV into the `(chain, imgt) -> {aa: score}` shape the engine indexes by."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    with Path(prior_path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (row["chain"], row["imgt"])
            out[key] = {aa: float(v) for aa, v in row.items() if aa not in ("chain", "imgt")}
    return out


def worst_predicted_error(picked: tuple) -> float | None:
    """The worst predicted error over `picked`, in angstrom, or `None` when no residue in it
    was measured.

    The structure predictor writes each residue's predicted error into the PDB B-factor
    column, which `residue_store` carries through as `b_factor` — the same array
    `index_and_scan` reads to triage a liability, so a humanization target and a liability
    target report one quantity."""
    measured = [residue.b_factor for residue in picked if residue.b_factor is not None]
    return max(measured) if measured else None


def prefers_another_residue(prior_row: dict | None, wild_type: str, margin: float) -> bool:
    """Whether the prior's best residue beats `wild_type` by more than `margin`.

    A row says which residue human repertoires carry here, so reading it means comparing its
    best entry against the wild type. A floor under the wild type alone reads almost nothing:
    a residue human repertoires rarely carry still scores far above any usable floor.

    Entries are log-probabilities; `margin` is on the probability scale."""
    if not prior_row:
        return False
    wild_type_score = prior_row.get(wild_type)
    if wild_type_score is None:
        return False
    return math.exp(max(prior_row.values())) - math.exp(wild_type_score) > margin


def is_nanobody(chains: list) -> bool:
    """Whether `chains` is a single-domain antibody: a heavy chain with no light chain beside it."""
    return not any(chain.chain_role == "L" for chain in chains)


def liability_keys(chain_residues: tuple, taxonomy: list[dict]) -> set:
    """Every liability the two detectors find on one whole chain, keyed by what it is and
    where it sits.

    Both detectors read a chain, never a fragment: `liability_motifs` matches a regex over
    the joined sequence, and `liability_cysteines` counts against a region's own residue
    list."""
    whole_chain = list(chain_residues)
    hits = [
        *liability_motifs.detect_all(whole_chain, taxonomy),
        *liability_cysteines.detect_all(whole_chain, taxonomy),
    ]
    return {(hit.definition_id, tuple(r.imgt for r in hit.site)) for hit in hits}


def select_non_human_positions(
    residues: list,
    prior: dict[tuple[str, str], dict[str, float]],
    non_human_prior_margin: float,
    fr_confidence_threshold: float = liability_triage.DEFAULT_FR_CONFIDENCE_THRESHOLD,
) -> list[design_objective.DesignTarget]:
    """One target per in-scope chain holding every framework position where the prior prefers
    a residue over the wild type by more than `non_human_prior_margin` — the positions human
    repertoires read differently from this antibody.

    Each target reports the worst predicted error over its own positions and warns when any
    of them exceeds `fr_confidence_threshold`, so a humanization variant carries the same two
    structural readings a liability variant does.

    A CDR position is never selected, however the prior scores it. On a nanobody a
    `VHH_FR2_HALLMARK_IMGT` position is never selected either. A position absent from the
    prior is not selected: nothing scored it, so it is not evidence of non-humanness. Neither
    is a position whose wild type the prior file has no column for — a modified or unknown
    residue collapses to `X` in the residue index, and no prior column scores it. A chain with
    no such position contributes no target at all."""
    targets: list[design_objective.DesignTarget] = []
    chains = residue_index.in_scope_chains(residues)
    protected = VHH_FR2_HALLMARK_IMGT if is_nanobody(chains) else frozenset()
    for chain in chains:
        picked = tuple(
            residue
            for residue in chain.residues
            if residue.region is not None
            and residue.region.startswith(FRAMEWORK_PREFIX)
            and residue.imgt not in protected
            and prefers_another_residue(
                prior.get(residue.join_key), residue.wild_type, non_human_prior_margin
            )
        )
        if not picked:
            continue
        confidence_angstroms = worst_predicted_error(picked)
        targets.append(
            design_objective.DesignTarget(
                site=picked,
                definition_id=None,
                region=picked[0].region,
                # Every picked position is framework, so the framework threshold is the only
                # one that applies. An unmeasured position never reads as low-confidence,
                # matching `liability_triage._low_confidence_for`.
                is_low_confidence=any(
                    residue.b_factor is not None
                    and residue.b_factor > fr_confidence_threshold
                    for residue in picked
                ),
                confidence_angstroms=confidence_angstroms,
                objective=HUMANNESS_OBJECTIVE_TAG,
            )
        )
    return targets


def build(
    prior_path: str,
    residues: list,
    non_human_prior_margin: float,
    fr_confidence_threshold: float = liability_triage.DEFAULT_FR_CONFIDENCE_THRESHOLD,
    max_new_liabilities: int = DEFAULT_MAX_NEW_LIABILITIES,
    ignored_liability_ids: frozenset[str] = DEFAULT_IGNORED_LIABILITY_IDS,
) -> design_objective.Objective:
    """The objective for one antibody, closed over that antibody's Prior TSV, residue index,
    non-human-prior margin, framework confidence threshold, new-liability allowance and the
    liability ids its gate does not count."""

    parent_by_chain: dict[str, float | None] = {}
    prior_cache: dict[str, dict] = {}

    def prior() -> dict:
        if "prior" not in prior_cache:
            prior_cache["prior"] = load_prior(prior_path)
        return prior_cache["prior"]

    def position_prior(residues: list, targets: list) -> dict:
        """Loads the prior from `prior_path` instead of computing it. The model that produced
        these numbers needs torch and ran in an earlier step, in a different deployment unit.
        Only the file crosses this boundary."""
        del residues, targets  # the file already names its own positions
        return prior()

    def select_target_positions(residues: list, taxonomy: list[dict]) -> list:
        del taxonomy  # a humanization target has no taxonomy entry to match against
        return select_non_human_positions(
            residues, prior(), non_human_prior_margin, fr_confidence_threshold
        )

    def parent_identity(chain: str) -> float | None:
        if chain not in parent_by_chain:
            sequence = humanness_gate.chain_sequence(residues, chain, [])
            parent_by_chain[chain] = humanness_gate.identity(sequence)
        return parent_by_chain[chain]

    def score_candidate(
        mutated_site: list, taxonomy: list[dict], tolerance_lookup: dict
    ) -> design_objective.GoalCheck:
        """Accepts a candidate only when it raises humanness on the one chain its site touches,
        sits entirely inside a framework region, and adds no more liabilities than
        `max_new_liabilities` allows. `liability_motifs.py` and `liability_cysteines.py` say
        what counts as one, and an id in `ignored_liability_ids` counts as none.

        A liability the parent already carries never rejects the candidate: the objective
        gates on what the edits introduce, and the parent's own liabilities belong to the
        liability objective."""
        del tolerance_lookup  # this objective measures humanness, not structural tolerance
        chains = {residue.chain for residue in mutated_site}
        if len(chains) != 1:
            return design_objective.GoalCheck(
                meets_goal=False, score=0.0, reason=SPANS_TWO_CHAINS_REASON
            )

        outside = [
            residue
            for residue in mutated_site
            if residue.region is None or not residue.region.startswith(FRAMEWORK_PREFIX)
        ]
        if outside:
            return design_objective.GoalCheck(
                meets_goal=False,
                score=0.0,
                reason=OUTSIDE_FRAMEWORK_REASON,
                detail=", ".join(f"{r.chain}{r.imgt}" for r in outside),
            )

        chain = next(iter(chains))
        edited_chain = humanness_gate.chain_residues(residues, chain, mutated_site)
        parent_chain = humanness_gate.chain_residues(residues, chain, [])
        introduced = liability_keys(edited_chain, taxonomy) - liability_keys(
            parent_chain, taxonomy
        )
        added = {
            (definition_id, positions)
            for definition_id, positions in introduced
            if definition_id not in ignored_liability_ids
        }
        # Sorted so one candidate always spends its allowance on the same liabilities.
        # A re-run then reports the same excess.
        excess = sorted(added)[max_new_liabilities:]
        if excess:
            spoiled = {position for _definition_id, positions in excess for position in positions}
            return design_objective.GoalCheck(
                meets_goal=False,
                score=0.0,
                reason=TOO_MANY_NEW_LIABILITIES_REASON,
                detail=f"{len(added)} added, {max_new_liabilities} allowed: "
                + ", ".join(
                    f"{definition_id} at {'/'.join(positions)}"
                    for definition_id, positions in excess
                ),
                # Only the edited positions, never the unedited neighbours a motif also spans:
                # dropping a position the candidate never touched would change nothing.
                blocking_positions=tuple(
                    (residue.chain, residue.imgt)
                    for residue in mutated_site
                    if residue.imgt in spoiled
                ),
            )

        parent = parent_identity(chain)
        candidate = humanness_gate.identity("".join(r.wild_type for r in edited_chain))
        if parent is None or candidate is None:
            return design_objective.GoalCheck(
                meets_goal=False, score=0.0, reason=UNSCOREABLE_REASON
            )

        if candidate <= parent:
            return design_objective.GoalCheck(
                meets_goal=False,
                score=0.0,
                reason=DID_NOT_RISE_REASON,
                detail=f"{parent} to {candidate} over {len(mutated_site)} edit(s)",
            )

        return design_objective.GoalCheck(meets_goal=True, score=candidate)

    return design_objective.Objective(
        select_target_positions=select_target_positions,
        position_prior=position_prior,
        score_candidate=score_candidate,
    )


def summarize_humanness(
    targets: list[residue_index.Residue] | None,
    cleared: list[variant_candidates.Candidate],
) -> parent_summary.ParentSummary:
    """Build coarse verdict and summary for a parent's considered framework positions.

    targets is the humanization objective's selected positions for this parent, in its own
    order; `None` when the objective did not run for this parent at all. cleared is the
    candidates that passed the Humanness gate for that parent.

    ("",  "")            targets is None      — the objective did not look
    ("none", "None")     targets is empty     — it looked and left everything alone
    ("present", <line>)  otherwise            — one entry per target, in target order

    A target reads "humanised" when some cleared candidate edits its position, "declined"
    otherwise. The per-parent variant cap runs after the gate, so a candidate the cap later
    dropped still counts here — this reduction never sees the cap's decision."""
    if targets is None:
        return parent_summary.ParentSummary(verdict="", summary="")
    if not targets:
        return parent_summary.ParentSummary(verdict="none", summary="None")

    edited = {(edit.chain, edit.imgt) for candidate in cleared for edit in candidate.edits}
    entries = [
        f"{target.wild_type}@{target.chain}{target.imgt} "
        f"({'humanised' if target.join_key in edited else 'declined'})"
        for target in targets
    ]
    return parent_summary.ParentSummary(verdict="present", summary=AMINO_SEP.join(entries))
