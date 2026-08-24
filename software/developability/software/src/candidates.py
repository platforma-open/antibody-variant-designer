"""Candidate substitutions and the objective's goal check — the block's only real
filter between a triaged liability and a variant.

`build_variants.py` runs this and then `ranking.py` in one exec, because
ranking discards nothing and reads nothing this module did not just produce.

This gate never sees the residue index or the PDB — only a triaged
liability and the tolerance table — so a candidate's own site (the exact
residues its originating motif or cysteine check matched over) is the only
window an objective's `score_candidate` can see. A liability whose site
runs longer than `max_edits_per_variant` is skipped outright, since
substituting every one of its positions would exceed the edit budget
before the objective is ever asked.

A surviving candidate's `tolerance` is the score its objective's
`score_candidate` returned — for the liability-removal objective, the
**mean** AntiFold perplexity over its edited positions, a property of the
positions themselves, independent of which amino acid was substituted
there, unlike the per-amino-acid log-probability `_top_substitutions`
ranks by. `region`, `low_confidence` and `worst_confidence_angstroms` ride
along unchanged from the triaged liability; `addressed_target` and
`changed_positions` are built here, from the taxonomy's own label and the
fixed `<chain>:<wt><imgtLabel><mut>` rendering, so `ranking.py` never has
to re-read the triaged liabilities or the taxonomy to report either one.
"""

import itertools
from dataclasses import dataclass, replace

import objectives

DEFAULT_MAX_EDITS_PER_VARIANT = 5
DEFAULT_CANDIDATE_RESIDUES_PER_POSITION = 3
DEFAULT_W_STRUCT = 1.0
DEFAULT_W_OBJ = 1.0


@dataclass(frozen=True)
class Edit:
    """One substitution: `wild_type` at `(chain, offset)` becomes `to`.

    Carries `imgt` alongside `offset` for the same reason `residue_store`
    does — the display label and the join key are different things, and a
    reader needing either finds it here without a second lookup."""

    chain: str
    offset: int
    imgt: str
    wild_type: str
    to: str


@dataclass(frozen=True)
class Candidate:
    """One re-scan-cleared substitution set, still keyed to the one
    liability it was built to address.

    `region`, `low_confidence` and `worst_confidence_angstroms` are carried
    forward unchanged from the `triage.Triaged` this candidate was built
    from — this module computes none of them itself. `addressed_target` and
    `changed_positions` are its own output: a human-readable label for the
    liability the edits target, and the fixed-spelling
    `<chain>:<wt><imgtLabel><mut>` rendering of the edits.

    Never serialized. Candidates are handed straight to `ranking.py` inside
    one exec, so this type crosses no boundary and needs no wire format."""

    target_definition_id: str
    edits: tuple[Edit, ...]
    tolerance: float
    region: str | None
    low_confidence: bool
    worst_confidence_angstroms: float | None
    addressed_target: str
    changed_positions: str


def _combined_scores(
    residue,
    log_probs: dict[str, float],
    prior: dict | None,
    w_struct: float,
    w_obj: float,
) -> dict[str, float]:
    """The structural log-probability row and the objective's position
    prior at this residue, each scaled by its own weight before they
    combine — an uncovered position, or an objective with no prior at
    all, still takes the `w_struct` scaling alone, so the weight means
    the same thing on every path."""
    prior_row = None if prior is None else prior.get((residue.chain, residue.imgt))
    if prior_row is None:
        return {aa: w_struct * value for aa, value in log_probs.items()}
    return {
        aa: w_struct * value + w_obj * prior_row.get(aa, 0.0)
        for aa, value in log_probs.items()
    }


def _top_substitutions(
    residue,
    tolerance_lookup: dict,
    k: int,
    prior: dict | None,
    w_struct: float,
    w_obj: float,
) -> list[str]:
    """Up to `k` amino acids at `residue`'s position, ranked by the
    combined score, wild type excluded — substituting a position to its
    own residue would neither change nor clear anything, so it is never a
    candidate substitution."""
    row = tolerance_lookup.get((residue.chain, residue.imgt))
    if row is None:
        return []
    scores = _combined_scores(residue, row["logProbs"], prior, w_struct, w_obj)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [aa for aa, _ in ranked if aa != residue.wild_type][:k]


def _changed_positions(edits: tuple) -> str:
    """The fixed CSV-contract spelling: `<chain>:<wt><imgtLabel><mut>`,
    comma-separated, one entry per edit in site order."""
    return ", ".join(f"{e.chain}:{e.wild_type}{e.imgt}{e.to}" for e in edits)


_RISK_LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _risk_level_order(triaged) -> int:
    """`High` before `Medium` before `Low` — the input order `build_candidates`
    iterates in, so a High-risk liability's candidates are built (and reach
    ranking) before a Low-risk one's, within one parent."""
    return _RISK_LEVEL_ORDER[triaged.risk_level]


def _addressed_target(definition_id: str, site: list, taxonomy_by_id: dict) -> str:
    """A human-readable label for the liability this candidate was built
    to clear — the taxonomy's own name plus where it sits, since neither
    `triage.Triaged` nor `Candidate` carries a display
    string on its own."""
    definition = taxonomy_by_id.get(definition_id, {})
    name = definition.get("name") or definition_id
    start = site[0]
    if start.region:
        return f"{name} @ {start.region} {start.chain}:{start.imgt}"
    return f"{name} @ {start.chain}:{start.imgt}"


def build_candidates(
    triaged_list: list,
    tolerance_lookup: dict,
    residues: list,
    taxonomy: list[dict],
    objective: objectives.Objective,
    max_edits_per_variant: int,
    candidate_residues_per_position: int,
    w_struct: float,
    w_obj: float,
) -> list[Candidate]:
    """Every candidate whose objective goal check passed, one liability at
    a time. Every position in a liability's site is substituted together,
    so each candidate's edit count equals that site's length — a site
    longer than `max_edits_per_variant` is skipped, and a site with no
    admissible substitution at any of its positions is skipped, both
    before the objective is asked at all."""
    taxonomy_by_id = {d["id"]: d for d in taxonomy}
    prior = (
        None
        if objective.position_prior is None
        else objective.position_prior(residues, triaged_list)
    )
    candidates: list[Candidate] = []
    for triaged in sorted(triaged_list, key=_risk_level_order):
        site = triaged.site
        if len(site) > max_edits_per_variant:
            continue
        per_position_options = [
            _top_substitutions(
                residue,
                tolerance_lookup,
                candidate_residues_per_position,
                prior,
                w_struct,
                w_obj,
            )
            for residue in site
        ]
        if any(len(options) == 0 for options in per_position_options):
            continue

        addressed_target = _addressed_target(triaged.definition_id, site, taxonomy_by_id)

        for combo in itertools.product(*per_position_options):
            mutated_site = [
                replace(residue, wild_type=to_aa)
                for residue, to_aa in zip(site, combo, strict=True)
            ]
            check = objective.score_candidate(mutated_site, taxonomy, tolerance_lookup)
            if not check.meets_goal:
                continue

            edits = tuple(
                Edit(
                    chain=residue.chain,
                    offset=residue.offset,
                    imgt=residue.imgt,
                    wild_type=residue.wild_type,
                    to=to_aa,
                )
                for residue, to_aa in zip(site, combo, strict=True)
            )
            candidates.append(
                Candidate(
                    target_definition_id=triaged.definition_id,
                    edits=edits,
                    tolerance=check.score,
                    region=site[0].region,
                    low_confidence=triaged.low_confidence,
                    worst_confidence_angstroms=triaged.confidence_angstroms,
                    addressed_target=addressed_target,
                    changed_positions=_changed_positions(edits),
                )
            )
    return candidates
