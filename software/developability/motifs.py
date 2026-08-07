"""Total, non-destructive motif detection.

Finds every regex match of every motif-bearing taxonomy entry over each
chain's own sequence. This module has no exposure or confidence input at
all, so it cannot suppress a buried match the way a scoring step might —
that separation is deliberate: whether a hit is worth acting on is a later
question (`triage.py`), never one this module answers by omission.

Only entries whose `motif` is a regex string qualify. The two cysteine
entries carry `motif: None` (`cysteine.py` evaluates those by counting, not
matching) and the two sequence-artifact entries (`contains_stop_codon`,
`out_of_frame`) are sequence-QC checks that a structural block never runs —
this module skips both by construction, not by an explicit id blocklist.
"""

import re
from dataclasses import dataclass

import residue_store

# Index within each regex match of the residue that actually undergoes the
# chemical change — the Asn in `N[GS]` deamidation sits at match position 0,
# but the Asn in `[STK]N` sits at position 1, since the motif's first
# character there is the residue ahead of it, not the reactive one itself.
CHEMICALLY_RELEVANT_INDEX = {
    "deamidation_ng": 0,
    "fragmentation_dp": 0,
    "isomerization_ddghst": 0,
    "n_linked_glycosylation": 0,
    "deamidation_nahnt": 0,
    "hydrolysis_np": 0,
    "fragmentation_ts": 0,
    "tryptophan_oxidation": 0,
    "methionine_oxidation": 0,
    "deamidation_stkn": 1,
    "integrin_binding": 0,
}


@dataclass
class DetectedMotif:
    """One regex match against one chain's sequence.

    `site` is the match's full span, offset order — this is the set of
    positions a later step is allowed to edit, since the residues around
    the reactive one are also part of what makes the motif a motif.
    `relevant` is the single residue within that span whose chemistry
    actually changes, the one whose rSASA a later triage step reads."""

    definition_id: str
    liability_type: str
    risk_level: str
    fixability: str
    site: list[residue_store.Residue]
    relevant: residue_store.Residue


def _qualifying_entries(taxonomy: list[dict]) -> list[dict]:
    return [
        entry
        for entry in taxonomy
        if entry.get("motif") is not None and entry.get("liabilityType") != "sequence-artifact"
    ]


def _by_chain(residues: list[residue_store.Residue]) -> dict:
    """Only in-scope residues — see `CLAUDE.md` § The scope rule. Dropping
    the rest before the sequence is joined is what keeps a constant domain,
    a second Fab arm and an antigen out of the scan, and what stops a motif
    matching across a V-domain / linker or V-domain / C-domain junction
    where no such motif exists."""
    chains: dict = {}
    for residue in residues:
        if not residue.in_scope:
            continue
        chains.setdefault(residue.chain, []).append(residue)
    for chain_residues in chains.values():
        chain_residues.sort(key=lambda r: r.offset)
    return chains


def detect_all(
    residues: list[residue_store.Residue], taxonomy: list[dict]
) -> list[DetectedMotif]:
    """Every motif match over every chain, taxonomy entry by taxonomy
    entry — total by construction, since nothing here reads exposure or
    confidence before deciding whether to keep a match."""
    hits: list[DetectedMotif] = []
    entries = _qualifying_entries(taxonomy)
    for chain_residues in _by_chain(residues).values():
        sequence = "".join(r.wild_type for r in chain_residues)
        for entry in entries:
            pattern = re.compile(entry["motif"])
            relevant_index = CHEMICALLY_RELEVANT_INDEX.get(entry["id"], 0)
            for match in pattern.finditer(sequence):
                site = chain_residues[match.start() : match.end()]
                hits.append(
                    DetectedMotif(
                        definition_id=entry["id"],
                        liability_type=entry["liabilityType"],
                        risk_level=entry["riskLevel"],
                        fixability=entry["fixability"],
                        site=site,
                        relevant=site[relevant_index],
                    )
                )
    return hits
