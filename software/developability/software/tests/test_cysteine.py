"""Tests for `cysteine.py` — the two non-motif cysteine liabilities,
evaluated against real IMGT region tagging (built via the real parser,
never hand-tagged) so the FR1/FR3 boundaries used here are authentic."""

from cysteine import detect_all
from pdb_fixtures import make_chain, make_pdb, platforma_cdr_remark
from structure import index_residues, parse_pdb

TAXONOMY = [
    {"id": "missing_cysteines", "name": "Missing Cysteines",
     "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "structural"},
    {"id": "extra_cysteines", "name": "Extra Cysteines",
     "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "hard_to_fix"},
]

# Fixed IMGT ranges (no REMARK override): CDR1 27-38, CDR2 56-65, CDR3
# 105-117. FR1 is res_seq 1..26 (26 residues); FR3 is res_seq 66..104 (39
# residues). FR1's conserved cysteine legally sits at res_seq 23 or 24
# (indices -4/-3 of 26); FR3's sits at res_seq 104 (index -1 of 39).
_CDR_REMARKS = (
    platforma_cdr_remark("H", 1, "H", 27, 38)
    + "\n"
    + platforma_cdr_remark("H", 2, "H", 56, 65)
    + "\n"
    + platforma_cdr_remark("H", 3, "H", 105, 117)
)


def _residues(overrides: dict, length: int = 117):
    """A full single-chain V-domain, ALA everywhere except `overrides`
    (`res_seq -> res_name`)."""
    base = make_chain("H", length)
    rows = [
        (chain, res_seq, icode, overrides.get(res_seq, res_name), b)
        for chain, res_seq, icode, res_name, b in base
    ]
    parsed = parse_pdb(_CDR_REMARKS + "\n" + make_pdb(rows))
    return index_residues(parsed)


class TestBothConservedCysteinesPresent:
    def test_no_hits_at_all(self):
        residues = _residues({24: "CYS", 104: "CYS"})

        hits = detect_all(residues, TAXONOMY)

        assert hits == []


class TestMissingFR3Cysteine:
    def test_one_missing_hit_structural_fixability_fr1_clean(self):
        residues = _residues({24: "CYS"})  # FR3's 104 stays ALA

        hits = detect_all(residues, TAXONOMY)

        assert len(hits) == 1
        hit = hits[0]
        assert hit.definition_id == "missing_cysteines"
        assert hit.fixability == "structural"
        assert [r.imgt for r in hit.site] == ["104"]


class TestExtraCysteineInFR3:
    def test_one_extra_hit_hard_to_fix_alongside_correct_conserved(self):
        residues = _residues({24: "CYS", 104: "CYS", 70: "CYS"})

        hits = detect_all(residues, TAXONOMY)

        assert len(hits) == 1
        hit = hits[0]
        assert hit.definition_id == "extra_cysteines"
        assert hit.fixability == "hard_to_fix"
        assert {r.imgt for r in hit.site} == {"70", "104"}


class TestRegionTooShortForExpectedPositions:
    def test_out_of_range_position_is_silently_skipped_not_a_crash(self):
        # A chain whose FR1 is only 2 residues long (res_seq 25-26, both
        # before CDR1's res_seq-27 start): -4 and -3 are both out of range
        # for a 2-element list.
        rows = make_chain("H", 2, start=25)
        parsed = parse_pdb(_CDR_REMARKS + "\n" + make_pdb(rows))
        residues = index_residues(parsed)

        hits = detect_all(residues, TAXONOMY)

        assert hits == []
