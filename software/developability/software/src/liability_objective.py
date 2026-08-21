"""The liability-removal objective: target every detected motif and cysteine defect, and accept a
candidate only when neither detector matches its mutated site.

A hit reappearing on the re-scan means one of two things, and either discards the candidate: the
target motif still matches (not cleared), or a different taxonomy entry now matches inside that
same short span (a new liability). `score_candidate` only ever sees the site it is asked about, not
the whole region, so this is both the mechanism and its own boundary.

`cysteine.detect_all`'s expected-cysteine-position indexing is relative to a region's full residue
list, not to a bare site; re-scanning a cysteine candidate over its own (shorter) site is an
approximation this objective accepts because the full region is not something `score_candidate`
can see.
"""

import cysteine
import motifs
import objectives


def select_target_positions(residues: list, taxonomy: list[dict]) -> list:
    return motifs.detect_all(residues, taxonomy) + cysteine.detect_all(residues, taxonomy)


def score_candidate(
    mutated_site: list, taxonomy: list[dict], tolerance_lookup: dict
) -> objectives.GoalCheck:
    hits = motifs.detect_all(mutated_site, taxonomy) + cysteine.detect_all(mutated_site, taxonomy)
    perplexities = [
        tolerance_lookup[(residue.chain, residue.imgt)]["perplexity"] for residue in mutated_site
    ]
    return objectives.GoalCheck(meets_goal=not hits, score=sum(perplexities) / len(perplexities))


OBJECTIVE = objectives.Objective(
    select_target_positions=select_target_positions,
    position_prior=None,
    score_candidate=score_candidate,
)
