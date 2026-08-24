"""Tests for `structure.py`: PDB parsing, the IMGT-numbered check and the
`(chain, offset)` residue index.

The module has no CLI — it is the index phase of the `index-and-scan`
entrypoint — so the end-to-end cases live in `test_index_and_scan.py`, and the format
cases in `test_formats.py`."""

import residue_store
from pdb_fixtures import make_chain, make_pdb, platforma_cdr_remark
from structure import (
    AA_THREE_TO_ONE,
    index_one,
    index_residues,
    is_imgt_numbered,
    parse_pdb,
    region_for,
)


class TestInsertionCodes:
    """`111`, `111A`..`111E`, `112` — the hazard a 1..N walk over the
    sequence cannot express, since insertion codes have no integer
    equivalent."""

    def test_six_insertion_code_variants_stay_distinct_and_ordered(self):
        residues = [
            ("H", 111, " ", "ALA", 20.0),
            ("H", 111, "A", "GLY", 20.0),
            ("H", 111, "B", "VAL", 20.0),
            ("H", 111, "C", "LEU", 20.0),
            ("H", 111, "D", "ILE", 20.0),
            ("H", 111, "E", "PRO", 20.0),
            ("H", 112, " ", "SER", 20.0),
        ]
        parsed = parse_pdb(make_pdb(residues))
        indexed = index_residues(parsed)

        assert [r.imgt for r in indexed] == ["111", "111A", "111B", "111C", "111D", "111E", "112"]
        # Six distinct residues share the base number 111; none collided
        # into a single entry.
        assert len({r.imgt for r in indexed if r.imgt.startswith("111")}) == 6
        # Offsets are a plain 0..N-1 walk over ATOM-record order, not over
        # `res_seq` — every imgt label is its own string, round-tripping
        # without becoming an int anywhere.
        assert [r.offset for r in indexed] == list(range(7))
        assert all(isinstance(r.imgt, str) for r in indexed)


class TestChainStartingAboveOne:
    def test_heavy_chain_starting_at_residue_two_has_no_off_by_one(self):
        parsed = parse_pdb(make_pdb(make_chain("H", 5, start=2)))
        indexed = index_residues(parsed)

        # offset 0 is the first ATOM record seen — res_seq 2 — not shifted
        # to pretend the chain started at 1.
        assert indexed[0].offset == 0
        assert indexed[0].imgt == "2"
        assert [r.imgt for r in indexed] == ["2", "3", "4", "5", "6"]


class TestMissingBackboneAtoms:
    def test_residue_missing_backbone_atoms_is_dropped(self):
        # A residue with only a CB atom — no N/CA/C — must not seat in the
        # index. Written directly (not via make_pdb, which always emits a
        # full backbone) so the rejection path is actually exercised.
        text = (
            make_pdb([("H", 1, " ", "ALA", 20.0)])
            + "ATOM      4  CB  ALA H   2      0.000   0.000   0.000  1.00 20.00           C\n"
            + make_pdb([("H", 3, " ", "GLY", 20.0)])
        )
        parsed = parse_pdb(text)
        indexed = index_residues(parsed)

        # Residue 2 (CB-only) is gone; residues 1 and 3 keep contiguous
        # offsets 0 and 1 — no silent gap where the dropped one was.
        assert [r.imgt for r in indexed] == ["1", "3"]
        assert [r.offset for r in indexed] == [0, 1]


class TestIsImgtNumbered:
    def test_remark_99_cdr_present_is_conclusive(self):
        text = platforma_cdr_remark("H", 1, "H", 27, 38) + "\n" + make_pdb(make_chain("H", 5))
        parsed = parse_pdb(text)
        assert is_imgt_numbered(parsed)

    def test_residue_at_imgt_ten_without_remark_is_accepted(self):
        parsed = parse_pdb(make_pdb(make_chain("H", 20)))  # residues 1..20, includes 10
        assert is_imgt_numbered(parsed)

    def test_no_remark_and_no_residue_at_imgt_ten_is_rejected(self):
        # A short chain, numbered 1..5, never reaches IMGT 10 and carries no
        # REMARK 99 CDR record — the un-numbered-structure case.
        parsed = parse_pdb(make_pdb(make_chain("H", 5)))
        assert not is_imgt_numbered(parsed)

    def test_empty_structure_is_rejected(self):
        parsed = parse_pdb("")
        assert not is_imgt_numbered(parsed)


class TestRegionFor:
    def test_imgt_cdr1_range_from_fixed_table(self):
        assert region_for("H", 27, None) == "CDR1"
        assert region_for("H", 38, None) == "CDR1"
        assert region_for("H", 39, None) == "FR2"

    def test_remark_99_overrides_fixed_table(self):
        platforma_cdrs = {"H": {"CDR1": (40, 50), "CDR2": (60, 70), "CDR3": (100, 115)}}
        assert region_for("H", 40, platforma_cdrs) == "CDR1"
        assert region_for("H", 27, platforma_cdrs) == "FR1"

    def test_unknown_chain_role_returns_none(self):
        assert region_for(None, 30, None) is None


class TestIndexResiduesRegionTagging:
    def test_region_assigned_via_remark_role_mapping(self):
        text = platforma_cdr_remark("H", 1, "B", 27, 38) + "\n" + make_pdb(make_chain("B", 40))
        parsed = parse_pdb(text)
        indexed = index_residues(parsed)

        by_imgt = {r.imgt: r for r in indexed}
        assert by_imgt["30"].region == "CDR1"
        assert by_imgt["1"].region == "FR1"

    def test_chain_with_no_role_gets_no_region(self):
        # An antigen chain has no REMARK 99 role mapping.
        parsed = parse_pdb(make_pdb(make_chain("A", 10)))
        indexed = index_residues(parsed)
        assert all(r.region is None for r in indexed)

    def test_wild_type_is_one_letter_code(self):
        parsed = parse_pdb(make_pdb([("H", 1, " ", "GLY", 20.0)]))
        indexed = index_residues(parsed)
        assert indexed[0].wild_type == AA_THREE_TO_ONE["GLY"] == "G"

    def test_unknown_residue_name_collapses_to_x(self):
        parsed = parse_pdb(make_pdb([("H", 1, " ", "MSE", 20.0)]))
        indexed = index_residues(parsed)
        assert indexed[0].wild_type == "X"


class TestArtifactsRoundTrip:
    def test_residues_round_trip_through_json(self, tmp_path):
        parsed = parse_pdb(make_pdb([("H", 111, "A", "ASN", 12.5)]))
        indexed = index_residues(parsed)

        path = tmp_path / "residues.json"
        residue_store.write_residues(str(path), indexed)
        read_back = residue_store.read_residues(str(path))

        assert read_back == indexed
        # The insertion-code label is exactly what came in, not "111.0" or
        # an int that lost the "A".
        assert read_back[0].imgt == "111A"


class TestIndexOneWritesOnlyOnSuccess:
    """Absence, not emptiness, is how a skipped antibody is signalled — an
    empty file would make every downstream `is_file()` guard a no-op and the
    antibody would be counted twice."""

    def test_a_passing_antibody_gets_its_residues_file(self, tmp_path):
        text = (
            platforma_cdr_remark("H", 1, "H", 27, 38)
            + "\n"
            + make_pdb(make_chain("H", 20))
        )
        pdb = tmp_path / "in.pdb"
        pdb.write_text(text)
        out = tmp_path / "residues.json"

        assert index_one(str(pdb), str(out)) == ""
        assert out.is_file()
        assert residue_store.read_residues(str(out))

    def test_a_skipped_antibody_leaves_no_file_at_all(self, tmp_path):
        pdb = tmp_path / "in.pdb"
        pdb.write_text(make_pdb(make_chain("H", 5)))  # never reaches IMGT 10
        out = tmp_path / "residues.json"

        assert index_one(str(pdb), str(out)) == "structure-not-imgt"
        assert not out.exists()

    def test_an_unparseable_structure_leaves_no_file_at_all(self, tmp_path):
        pdb = tmp_path / "in.pdb"
        pdb.write_text("")
        out = tmp_path / "residues.json"

        assert index_one(str(pdb), str(out)) == "no-structure"
        assert not out.exists()
