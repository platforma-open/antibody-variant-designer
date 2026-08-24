"""The humanization objective.

`position_prior` loads rather than computes: the model that produced these numbers needs torch and
ran in an earlier step, in a different deployment unit. What crosses the boundary is a file.

`score_candidate` accepts a candidate only when it raises humanness on the one chain its site
touches, without moving a binding-supporting position or reintroducing a liability motifs.py or
cysteine.py would flag.
"""

import csv
from pathlib import Path

import cysteine
import design_objective
import motifs
import oasis_gate

NO_CANDIDATE_REASON = "no-candidate-raised-humanness"

FRAMEWORK_PREFIX = "FR"


def load_prior(prior_path: str) -> dict[tuple[str, str], dict[str, float]]:
    """Read one Prior TSV into the `(chain, imgt) -> {aa: score}` shape the engine indexes by."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    with Path(prior_path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (row["chain"], row["imgt"])
            out[key] = {aa: float(v) for aa, v in row.items() if aa not in ("chain", "imgt")}
    return out


def build(prior_path: str, residues: list) -> design_objective.Objective:
    """The objective for one antibody, closed over that antibody's Prior TSV and residue index."""

    parent_by_chain: dict[str, float | None] = {}

    def position_prior(residues: list, triaged_list: list) -> dict:
        del residues, triaged_list  # the file already names its own positions
        return load_prior(prior_path)

    def parent_identity(chain: str) -> float | None:
        if chain not in parent_by_chain:
            sequence = oasis_gate.chain_sequence(residues, chain, [])
            parent_by_chain[chain] = oasis_gate.identity(sequence)
        return parent_by_chain[chain]

    def score_candidate(
        mutated_site: list, taxonomy: list[dict], tolerance_lookup: dict
    ) -> design_objective.GoalCheck:
        del tolerance_lookup  # this objective measures humanness, not structural tolerance
        chains = {residue.chain for residue in mutated_site}
        if len(chains) != 1:
            return design_objective.GoalCheck(meets_goal=False, score=0.0)

        if not all(
            residue.region is not None and residue.region.startswith(FRAMEWORK_PREFIX)
            for residue in mutated_site
        ):
            return design_objective.GoalCheck(meets_goal=False, score=0.0)

        hits = motifs.detect_all(mutated_site, taxonomy)
        hits += cysteine.detect_all(mutated_site, taxonomy)
        if hits:
            return design_objective.GoalCheck(meets_goal=False, score=0.0)

        chain = next(iter(chains))
        parent = parent_identity(chain)
        candidate = oasis_gate.identity(oasis_gate.chain_sequence(residues, chain, mutated_site))
        if parent is None or candidate is None:
            return design_objective.GoalCheck(meets_goal=False, score=0.0)

        return design_objective.GoalCheck(meets_goal=candidate > parent, score=candidate)

    return design_objective.Objective(
        select_target_positions=lambda _residues, _taxonomy: [],
        position_prior=position_prior,
        score_candidate=score_candidate,
    )
