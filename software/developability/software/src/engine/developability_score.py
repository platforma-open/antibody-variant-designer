"""The engineering burden one antibody's liabilities add up to.

The same number the Sequence Liabilities block publishes as
`pl7.app/developabilityScore`: every liability costs its fixability weight times
the weight of the region it sits in, and the costs sum. A parent and each of its
variants are scored by this one function, so the two are read on one scale and a
repair that lowers the burden shows up as a smaller number.

`REGION_WEIGHTS` is a copy of the table Sequence Liabilities scores against. A
weight changed there and not here makes the two blocks disagree on one antibody.
"""

from dataclasses import replace

from engine import liability_objective, liability_triage

REGION_WEIGHTS: dict[str, float] = {
    "CDR3": 1.5,
    "CDR1": 1.2,
    "CDR2": 1.2,
    "FR1": 1.0,
    "FR2": 0.5,
    "FR3": 0.5,
    "FR4": 0.3,
}

UNKNOWN_REGION_WEIGHT = 0.5

DISQUALIFYING = "disqualifying"

PRECISION = 4


def score(hits, fixability_weights: dict[str, float]) -> float:
    """The summed burden of `hits` — motif hits, cysteine hits, or triaged ones alike.

    Charged at each hit's chemically relevant residue, not at its span start: a motif
    span can cross a region boundary while only one residue reacts.

    A disqualifying liability is left out rather than scored zero: it says the antibody
    cannot ship at all, which is a verdict the burden number does not carry.

    `fixability_weights` comes from the shared taxonomy package, so a class it does not
    name costs nothing here — the taxonomy, not this module, decides what a class is worth."""
    total = 0.0
    for hit in hits:
        if hit.fixability == DISQUALIFYING:
            continue
        region = liability_triage.relevant_residue(hit).region
        total += fixability_weights.get(hit.fixability, 0.0) * REGION_WEIGHTS.get(
            region, UNKNOWN_REGION_WEIGHT
        )
    # Both weight tables carry at most one decimal, so rounding here loses nothing a
    # reader wants and keeps binary summation noise (3.5999999999999996) out of the file.
    return round(total, PRECISION)


def score_after_edits(residues, edits, taxonomy: list[dict], fixability_weights) -> float:
    """The burden left on an antibody once `edits` are applied to `residues`.

    Re-scans the whole edited index rather than the edited span alone, so a motif the
    edit spelled across a span boundary — and one it silently destroyed elsewhere — both
    reach the number. The parent's own score is measured over the same total scan, which
    is what makes the two comparable."""
    substitution_by_key = {(edit.chain, edit.offset): edit.to for edit in edits}
    edited = [
        replace(residue, wild_type=substitution_by_key[(residue.chain, residue.offset)])
        if (residue.chain, residue.offset) in substitution_by_key
        else residue
        for residue in residues
    ]
    return score(
        liability_objective.select_target_positions(edited, taxonomy), fixability_weights
    )
