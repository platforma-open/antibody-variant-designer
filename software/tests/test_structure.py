"""Tests for `structure.py`: PDB parsing, the IMGT-numbered check, the
`(chain, offset)` residue index, and the CLI."""

import residue_store
import skip_store
from pdb_fixtures import make_chain, make_pdb, platforma_cdr_remark
from structure import (
    AA_THREE_TO_ONE,
    index_residues,
    is_imgt_numbered,
    main,
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


def _researchable_pdb(length: int = 20) -> str:
    """A chain H whose REMARK records give it a role — a role is half the
    scope rule, so without them every residue is out of scope and the CLI
    reaches `no-researchable-residue` instead of succeeding."""
    return (
        platforma_cdr_remark("H", 1, "H", 27, 38)
        + "\n"
        + platforma_cdr_remark("H", 2, "H", 56, 65)
        + "\n"
        + platforma_cdr_remark("H", 3, "H", 105, 117)
        + "\n"
        + make_pdb(make_chain("H", length))
    )


def _run(batch, out_dir="residues"):
    """Run the batch CLI over whatever `batch` holds. Returns
    `(skips, residues_dir)`."""
    residues_dir = batch.dir(out_dir)
    out_skip = batch.path("skip.tsv")

    rc = main([
        "--pdb-dir", str(batch.pdb_dir),
        "--pdb-index", batch.index,
        "--out-residues-dir", residues_dir,
        "--out-skip", out_skip,
    ])

    assert rc == 0
    return skip_store.read_skips(out_skip), residues_dir


class TestCli:
    def test_happy_path_writes_residues_and_empty_skip(self, batch):
        entry = batch.add("clonotype-1", _researchable_pdb())

        skips, residues_dir = _run(batch)

        assert skips == [("clonotype-1", "")]
        assert residue_store.read_residues(f"{residues_dir}/{entry.stem}.json")

    def test_no_atom_records_writes_no_structure_skip(self, batch):
        entry = batch.add("clonotype-1", "")

        skips, residues_dir = _run(batch)

        assert skips == [("clonotype-1", "no-structure")]
        assert residue_store.read_residues(f"{residues_dir}/{entry.stem}.json") == []

    def test_non_imgt_structure_writes_structure_not_imgt_skip(self, batch):
        entry = batch.add("clonotype-1", make_pdb(make_chain("H", 5)))

        skips, residues_dir = _run(batch)

        assert skips == [("clonotype-1", "structure-not-imgt")]
        assert residue_store.read_residues(f"{residues_dir}/{entry.stem}.json") == []


class TestBatchCli:
    """The failure modes one-exec-per-antibody could not have."""

    def test_one_bad_antibody_does_not_stop_the_others(self, batch):
        batch.add("good-1", _researchable_pdb())
        batch.add("bad", make_pdb(make_chain("H", 5)))  # not IMGT-numbered
        batch.add("good-2", _researchable_pdb())

        skips, residues_dir = _run(batch)

        assert skips == [
            ("good-1", ""),
            ("bad", "structure-not-imgt"),
            ("good-2", ""),
        ]
        assert residue_store.read_residues(f"{residues_dir}/good-1.json")
        assert residue_store.read_residues(f"{residues_dir}/good-2.json")

    def test_a_rostered_clonotype_with_no_staged_blob_is_no_structure(self, batch):
        # A clonotype the upstream block failed for has no ResourceMap entry
        # at all, so its blob is simply absent — never a null one.
        batch.add("present", _researchable_pdb())
        batch.add_without_blob("absent")

        skips, _ = _run(batch)

        assert skips == [("present", ""), ("absent", "no-structure")]

    def test_an_empty_roster_exits_zero_with_a_header_only_skip_tsv(self, batch):
        skips, _ = _run(batch)

        assert skips == []

    def test_a_clonotype_key_holding_json_punctuation_names_its_file_by_stem(self, batch):
        batch.add('["clone/1"]', _researchable_pdb(), stem="antibody-01")

        skips, residues_dir = _run(batch)

        assert skips == [('["clone/1"]', "")]
        assert residue_store.read_residues(f"{residues_dir}/antibody-01.json")

    def test_roster_order_does_not_change_the_per_antibody_result(self, batch):
        batch.add("a", _researchable_pdb())
        batch.add("b", make_pdb(make_chain("H", 5)))
        forward, _ = _run(batch, out_dir="residues-forward")

        batch.entries.reverse()
        reversed_skips, _ = _run(batch, out_dir="residues-reversed")

        assert dict(forward) == dict(reversed_skips)
