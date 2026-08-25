"""Total, non-destructive motif detection.

Finds every regex match of every motif-bearing taxonomy entry, over each
chain's own sequence. This module takes no exposure or confidence input.
It therefore never suppresses a buried match itself. Whether a hit is
worth acting on is `liability_triage.py`'s question, not this module's.
"""

import re
from dataclasses import dataclass

from engine import residue_store

# Index, within each regex match, of the residue whose chemistry
# actually changes. In the `N[GS]` deamidation motif, the reactive Asn
# sits at match position 0. In `[STK]N`, the reactive residue sits at
# position 1 instead. There, the motif's first character is the
# residue just ahead of the reactive one, not the reactive residue
# itself.
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

    `site` holds the match's full span, in offset order. A later step
    may edit any position in that span, because the residues flanking
    the reactive one help define the motif itself.

    `relevant` is the single residue within `site` whose chemistry
    actually changes. It is the residue whose rSASA a later triage step
    reads."""

    definition_id: str
    liability_type: str
    risk_level: str
    fixability: str
    site: list[residue_store.Residue]
    relevant: residue_store.Residue


def _qualifying_entries(taxonomy: list[dict]) -> list[dict]:
    """Only entries whose `motif` is a regex string qualify.

    The two cysteine entries carry `motif: None` — `liability_cysteines.py`
    evaluates them by counting, not matching. The two sequence-artifact
    entries, `contains_stop_codon` and `out_of_frame`, are sequence-QC
    checks a structural block never runs. Both kinds are skipped by
    construction here, never through an explicit id blocklist."""
    return [
        entry
        for entry in taxonomy
        if entry.get("motif") is not None and entry.get("liabilityType") != "sequence-artifact"
    ]


def detect_all(
    residues: list[residue_store.Residue], taxonomy: list[dict]
) -> list[DetectedMotif]:
    """Every motif match over every chain, taxonomy entry by taxonomy
    entry.

    This is total by construction: nothing here reads exposure or
    confidence before deciding whether to keep a match. Matching against
    one chain's own sequence, never a cross-chain join, keeps a
    V-domain/linker or V-domain/C-domain junction from spelling a motif
    that does not exist in either domain."""
    hits: list[DetectedMotif] = []
    entries = _qualifying_entries(taxonomy)
    for chain in residue_store.in_scope_chains(residues):
        sequence = chain.sequence
        for entry in entries:
            pattern = re.compile(entry["motif"])
            relevant_index = CHEMICALLY_RELEVANT_INDEX.get(entry["id"], 0)
            for match in pattern.finditer(sequence):
                site = list(chain.residues[match.start() : match.end()])
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
