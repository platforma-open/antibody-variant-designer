"""The humanization objective."""

import csv
import math
from pathlib import Path

from engine import (
    design_objective,
    humanness_gate,
    liability_cysteines,
    liability_motifs,
    residue_store,
)

NO_CANDIDATE_REASON = "no-candidate-raised-humanness"

FRAMEWORK_PREFIX = "FR"

DEFAULT_NON_HUMAN_PRIOR_CUTOFF = 0.05

HUMANIZATION_LABEL = "Humanization"


def load_prior(prior_path: str) -> dict[tuple[str, str], dict[str, float]]:
    """Read one Prior TSV into the `(chain, imgt) -> {aa: score}` shape the engine indexes by."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    with Path(prior_path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (row["chain"], row["imgt"])
            out[key] = {aa: float(v) for aa, v in row.items() if aa not in ("chain", "imgt")}
    return out


def select_non_human_positions(
    residues: list,
    prior: dict[tuple[str, str], dict[str, float]],
    non_human_prior_cutoff: float,
) -> list[design_objective.DesignTarget]:
    """One target per in-scope chain holding any framework position whose wild-type residue
    the prior scores strictly below `non_human_prior_cutoff` — the residue human repertoires
    rarely carry there. The prior file holds log-probabilities (`sapiens_prior._log_softmax`),
    so the comparison exponentiates a row's wild-type entry back to the `[0, 1]` probability
    scale the cutoff is stated in.

    A CDR position is never selected, however the prior scores it. A position absent from
    the prior is not selected either: nothing scored it, so it is not evidence of
    non-humanness. Neither is a position whose wild type the prior file has no column for — a
    modified or unknown residue collapses to `X` in the residue index, and no prior column
    scores it. A chain with no such position contributes no target at all."""
    targets: list[design_objective.DesignTarget] = []
    for chain in residue_store.in_scope_chains(residues):
        picked = tuple(
            residue
            for residue in chain.residues
            if residue.region is not None
            and residue.region.startswith(FRAMEWORK_PREFIX)
            and (residue.join_key in prior)
            and prior[residue.join_key].get(residue.wild_type) is not None
            and math.exp(prior[residue.join_key][residue.wild_type]) < non_human_prior_cutoff
        )
        if not picked:
            continue
        targets.append(
            design_objective.DesignTarget(
                site=picked,
                definition_id=None,
                region=picked[0].region,
                is_low_confidence=False,
                confidence_angstroms=None,
            )
        )
    return targets


def build(
    prior_path: str, residues: list, non_human_prior_cutoff: float
) -> design_objective.Objective:
    """The objective for one antibody, closed over that antibody's Prior TSV, residue index
    and non-human-prior cutoff."""

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
        return select_non_human_positions(residues, prior(), non_human_prior_cutoff)

    def parent_identity(chain: str) -> float | None:
        if chain not in parent_by_chain:
            sequence = humanness_gate.chain_sequence(residues, chain, [])
            parent_by_chain[chain] = humanness_gate.identity(sequence)
        return parent_by_chain[chain]

    def score_candidate(
        mutated_site: list, taxonomy: list[dict], tolerance_lookup: dict
    ) -> design_objective.GoalCheck:
        """Accepts a candidate only when it raises humanness on the one chain its site touches,
        sits entirely inside a framework region, and does not reintroduce a liability
        `liability_motifs.py` or `liability_cysteines.py` would flag."""
        del tolerance_lookup  # this objective measures humanness, not structural tolerance
        chains = {residue.chain for residue in mutated_site}
        if len(chains) != 1:
            return design_objective.GoalCheck(meets_goal=False, score=0.0)

        if not all(
            residue.region is not None and residue.region.startswith(FRAMEWORK_PREFIX)
            for residue in mutated_site
        ):
            return design_objective.GoalCheck(meets_goal=False, score=0.0)

        hits = liability_motifs.detect_all(mutated_site, taxonomy)
        hits += liability_cysteines.detect_all(mutated_site, taxonomy)
        if hits:
            return design_objective.GoalCheck(meets_goal=False, score=0.0)

        chain = next(iter(chains))
        parent = parent_identity(chain)
        candidate = humanness_gate.identity(
            humanness_gate.chain_sequence(residues, chain, mutated_site)
        )
        if parent is None or candidate is None:
            return design_objective.GoalCheck(meets_goal=False, score=0.0)

        return design_objective.GoalCheck(meets_goal=candidate > parent, score=candidate)

    return design_objective.Objective(
        select_target_positions=select_target_positions,
        position_prior=position_prior,
        score_candidate=score_candidate,
    )
