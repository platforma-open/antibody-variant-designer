"""Liability-removal objective: accepts candidates only when no motif or cysteine defect
remains at the mutated site.

Re-scanning detects: (1) target motif still matches after edit, (2) a new liability
appeared at that span.

score_candidate sees only one site, not the full region. Re-scanning a cysteine
over its site is approximate, but that is how it works — the full region is unavailable.
"""

from engine import design_objective, liability_cysteines, liability_motifs


def select_target_positions(residues: list, taxonomy: list[dict]) -> list:
    return liability_motifs.detect_all(residues, taxonomy) + liability_cysteines.detect_all(
        residues, taxonomy
    )


def score_candidate(
    mutated_site: list, taxonomy: list[dict], tolerance_lookup: dict
) -> design_objective.GoalCheck:
    hits = liability_motifs.detect_all(mutated_site, taxonomy)
    hits += liability_cysteines.detect_all(mutated_site, taxonomy)
    perplexities = [
        tolerance_lookup[(residue.chain, residue.imgt)]["perplexity"] for residue in mutated_site
    ]
    return design_objective.GoalCheck(
        meets_goal=not hits, score=sum(perplexities) / len(perplexities)
    )


OBJECTIVE = design_objective.Objective(
    select_target_positions=select_target_positions,
    position_prior=None,
    score_candidate=score_candidate,
)
