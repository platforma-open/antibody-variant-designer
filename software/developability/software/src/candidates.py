"""Candidate substitutions and the re-scan gate — the block's only real
filter between a triaged liability and a variant.

A module, not an entrypoint: `variants.py` runs this and then `ranking.py`
in one exec, because ranking discards nothing and reads nothing this module
did not just produce.

This gate never sees the residue index or the PDB — only a triaged
liability and the tolerance table — so a candidate's own site (the exact
residues its originating motif or cysteine check matched over) is the only
window the re-scan can see. Substituting every position in that site
together and re-running the same two detectors over just that window is
therefore both the mechanism and its own boundary: a hit reappearing in
the re-scan means either the target motif still matches (not cleared) or a
different taxonomy entry now matches inside that same short span (a new
liability), and either way the candidate is discarded. A liability whose
site runs longer than `max_edits_per_variant` is skipped outright, since
substituting every one of its positions would exceed the edit budget
before the re-scan even runs.

`cysteine.detect_all`'s expected-cysteine-position indexing is relative to
a region's full residue list, not to a bare site; re-scanning a cysteine
candidate over its own (shorter) site is an approximation this module
accepts because, again, the full region is not something it can see.

A surviving candidate's `tolerance` is the **mean** AntiFold perplexity
over its edited positions — a property of the positions themselves,
independent of which amino acid was substituted there, unlike the
per-amino-acid log-probability `_top_substitutions` ranks by. `region`,
`low_confidence` and `worst_confidence_angstroms` ride along unchanged from
the triaged liability; `addressed_target` and `changed_positions` are
built here, from the taxonomy's own label and the fixed
`<chain>:<wt><imgtLabel><mut>` rendering, so `ranking.py` never has to
re-read the triaged liabilities or the taxonomy to report either one.
"""

import itertools
from dataclasses import dataclass, replace

import cysteine
import motifs

DEFAULT_MAX_EDITS_PER_VARIANT = 5
DEFAULT_CANDIDATE_RESIDUES_PER_POSITION = 3


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


def _top_substitutions(residue, tolerance_lookup: dict, k: int) -> list[str]:
    """Up to `k` amino acids at `residue`'s position, ranked by AntiFold
    log-probability, wild type excluded — substituting a position to its
    own residue would neither change nor clear anything, so it is never a
    candidate substitution.

    The combined score a position's candidates should be ranked by is a
    log-space weighted sum of two terms: the structural log-probability
    row read here, and an objective prior over the same twenty amino
    acids. At V0.5 that second term does not exist — there is no
    conventional-fix table yet — so the sum has one live term with a
    positive weight, and a single positive-weighted term cannot reorder
    its own ranking. Sorting the log-probability row alone is therefore
    not an approximation of the full sum; it *is* the full sum, in the
    one-expert case this package ships. `perplexity` is one number for
    the whole position and could never order twenty amino acids at it."""
    row = tolerance_lookup.get((residue.chain, residue.imgt))
    if row is None:
        return []
    ranked = sorted(row["logProbs"].items(), key=lambda kv: (-kv[1], kv[0]))
    return [aa for aa, _ in ranked if aa != residue.wild_type][:k]


def _rescan_clears(mutated_site: list, taxonomy: list[dict]) -> bool:
    """True iff neither detector matches anywhere in the mutated site — the
    target's own motif no longer matches (cleared) and no other taxonomy
    entry now matches within the same short span (nothing new)."""
    hits = motifs.detect_all(mutated_site, taxonomy) + cysteine.detect_all(mutated_site, taxonomy)
    return not hits


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
    taxonomy: list[dict],
    max_edits_per_variant: int,
    candidate_residues_per_position: int,
) -> list[Candidate]:
    """Every re-scan-cleared substitution set, one liability at a time.
    Every position in a liability's site is substituted together, so each
    candidate's edit count equals that site's length — a site longer than
    `max_edits_per_variant` is skipped, and a site with no admissible
    substitution at any of its positions is skipped, both before the
    re-scan runs at all."""
    taxonomy_by_id = {d["id"]: d for d in taxonomy}
    candidates: list[Candidate] = []
    for triaged in sorted(triaged_list, key=_risk_level_order):
        site = triaged.site
        if len(site) > max_edits_per_variant:
            continue
        per_position_options = [
            _top_substitutions(residue, tolerance_lookup, candidate_residues_per_position)
            for residue in site
        ]
        if any(len(options) == 0 for options in per_position_options):
            continue

        # A property of the positions themselves, the same for every
        # combination substituted there — computed once per liability
        # rather than once per candidate.
        structural_tolerance = sum(
            tolerance_lookup[(residue.chain, residue.imgt)]["perplexity"] for residue in site
        ) / len(site)
        addressed_target = _addressed_target(triaged.definition_id, site, taxonomy_by_id)

        for combo in itertools.product(*per_position_options):
            mutated_site = [
                replace(residue, wild_type=to_aa)
                for residue, to_aa in zip(site, combo, strict=True)
            ]
            if not _rescan_clears(mutated_site, taxonomy):
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
                    tolerance=structural_tolerance,
                    region=site[0].region,
                    low_confidence=triaged.low_confidence,
                    worst_confidence_angstroms=triaged.confidence_angstroms,
                    addressed_target=addressed_target,
                    changed_positions=_changed_positions(edits),
                )
            )
    return candidates
