"""Tests for `liability_motifs.py` — total, non-destructive motif detection."""

import inspect

from liability_motifs import detect_all
from residue_store import Residue


def _residue(chain, offset, wild_type, region="FR1"):
    """An in-scope residue by default — `detect_all` scans only residues on
    a role-bearing chain that carry a region (`CLAUDE.md` § The scope rule),
    so a role-less fixture would make every motif test vacuously pass."""
    return Residue(
        chain=chain,
        offset=offset,
        imgt=str(offset + 1),
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region=region,
        chain_role=chain if chain in ("H", "L") else "H",
    )


def _chain(chain_id, letters, region="FR1"):
    return [_residue(chain_id, i, letter, region) for i, letter in enumerate(letters)]


TAXONOMY = [
    {"id": "deamidation_ng", "name": "Deamidation (N[GS])",
     "liabilityType": "deamidation", "motif": r"N[GS]",
     "riskLevel": "High", "fixability": "fixable"},
    {"id": "deamidation_stkn", "name": "Deamidation ([STK]N)",
     "liabilityType": "deamidation", "motif": r"[STK]N",
     "riskLevel": "Low", "fixability": "easily_fixable"},
    {"id": "contains_stop_codon", "name": "Contains stop codon",
     "liabilityType": "sequence-artifact", "motif": r"\*",
     "riskLevel": "High", "fixability": "disqualifying"},
    {"id": "out_of_frame", "name": "Out of frame",
     "liabilityType": "sequence-artifact", "motif": r"_",
     "riskLevel": "High", "fixability": "disqualifying"},
    {"id": "missing_cysteines", "name": "Missing Cysteines",
     "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "structural"},
    {"id": "extra_cysteines", "name": "Extra Cysteines",
     "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "hard_to_fix"},
]


class TestNGDeamidation:
    def test_ng_hit_relevant_is_asn_site_is_two_residues(self):
        residues = _chain("H", "ANGA")  # N at offset 1, G at offset 2

        hits = detect_all(residues, TAXONOMY)

        ng_hits = [h for h in hits if h.definition_id == "deamidation_ng"]
        assert len(ng_hits) == 1
        hit = ng_hits[0]
        assert [r.wild_type for r in hit.site] == ["N", "G"]
        assert hit.relevant.wild_type == "N"
        assert hit.relevant is hit.site[0]
        assert hit.liability_type == "deamidation"
        assert hit.risk_level == "High"
        assert hit.fixability == "fixable"


class TestSTKNDeamidation:
    def test_stkn_hit_relevant_is_second_position_asn(self):
        residues = _chain("H", "ATN")  # T at offset 1, N at offset 2

        hits = detect_all(residues, TAXONOMY)

        stkn_hits = [h for h in hits if h.definition_id == "deamidation_stkn"]
        assert len(stkn_hits) == 1
        hit = stkn_hits[0]
        assert [r.wild_type for r in hit.site] == ["T", "N"]
        assert hit.relevant.wild_type == "N"
        assert hit.relevant is hit.site[1]


class TestDetectionIsTotal:
    def test_no_exposure_or_confidence_parameter_exists(self):
        # This module has no way to read exposure or confidence at all —
        # the strongest proof of "total" is that the function signature
        # cannot accept either signal.
        params = inspect.signature(detect_all).parameters
        assert "exposure" not in params
        assert "rsasa" not in params
        assert "confidence" not in params

    def test_hit_returned_regardless_of_any_burial(self):
        # No exposure physics here — a "buried" motif is simply one that
        # detect_all returns exactly like any other, since it takes no
        # signal that could have suppressed it.
        residues = _chain("H", "ANGA")
        hits = detect_all(residues, TAXONOMY)
        assert any(h.definition_id == "deamidation_ng" for h in hits)


class TestExcludedEntries:
    def test_sequence_artifact_and_cysteine_entries_never_hit(self):
        # A literal "*" would match contains_stop_codon's own motif if the
        # liabilityType filter didn't exclude it first.
        residues = _chain("H", "AN*_GA")

        hits = detect_all(residues, TAXONOMY)

        ids = {h.definition_id for h in hits}
        assert "contains_stop_codon" not in ids
        assert "out_of_frame" not in ids
        assert "missing_cysteines" not in ids
        assert "extra_cysteines" not in ids


class TestMultipleChains:
    def test_each_chain_produces_its_own_hit_with_correct_chain(self):
        residues = _chain("H", "ANGA") + _chain("L", "ANGA")

        hits = detect_all(residues, TAXONOMY)

        ng_hits = [h for h in hits if h.definition_id == "deamidation_ng"]
        assert len(ng_hits) == 2
        assert {h.site[0].chain for h in ng_hits} == {"H", "L"}
