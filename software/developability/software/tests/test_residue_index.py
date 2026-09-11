"""Tests for `residue_index.py`: the residue model, the IMGT-numbered check and the
`(chain, offset)` residue index.

The module has no CLI — it is the index phase of the `index-and-scan`
entrypoint — so the end-to-end cases live in `test_index_and_scan.py`, and the format
cases in `test_formats.py`."""

from engine import residue_index
from engine.residue_index import (
    AA_THREE_TO_ONE,
    index_residues,
    is_imgt_numbered,
    region_for,
)
from engine.residue_store import parse_pdb
from pdb_fixtures import make_chain, make_pdb, platforma_cdr_remark


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


def _residue(chain, offset, imgt, wild_type="A", region="FR1", chain_role="H"):
    return residue_index.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt,
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region=region,
        chain_role=chain_role,
    )


class TestInScopeChains:
    def test_a_minted_chain_holds_every_in_scope_residue_of_that_chain(self):
        residues = [
            _residue("H", 2, "3", region="CDR1"),
            _residue("H", 0, "1"),
            _residue("H", 1, "2"),
            # Constant-domain tail: no region, so out of scope though the
            # chain carries a role.
            residue_index.Residue(
                chain="H", offset=3, imgt="129", wild_type="A", res_name="ALA",
                b_factor=20.0, region=None, chain_role="H",
            ),
            # A role-less second arm, in scope by neither test.
            _residue("X", 0, "1", chain_role=None, region="FR1"),
        ]

        [chain] = residue_index.in_scope_chains(residues)

        assert chain.chain == "H"
        assert [r.offset for r in chain.residues] == [0, 1, 2]

    def test_a_role_less_chain_mints_no_chain(self):
        residues = [
            _residue("H", 0, "1"),
            _residue("X", 0, "1", chain_role=None),
            residue_index.Residue(
                chain="Y", offset=0, imgt="1", wild_type="A", res_name="ALA",
                b_factor=20.0, region=None, chain_role=None,
            ),
        ]

        chains = residue_index.in_scope_chains(residues)

        assert len(chains) == 1
        assert {r.chain for c in chains for r in c.residues} == {"H"}

    def test_chains_are_ordered_heavy_before_light(self):
        # "A" (light) sorts before "H" (letter) alphabetically — the letter
        # order and the rendering order disagree, so this is the one input
        # where the fix actually shows.
        residues = [
            _residue("A", 0, "1", chain_role="L"),
            _residue("H", 0, "1", chain_role="H"),
        ]

        chains = residue_index.in_scope_chains(residues)

        assert [c.chain_role for c in chains] == ["H", "L"]


class TestSelectAndByRegion:
    def test_a_selection_carries_the_chain_it_came_from(self):
        residues = [_residue("H", 0, "1", region="FR1"), _residue("H", 1, "2", region="CDR1")]
        [chain] = residue_index.in_scope_chains(residues)

        selection = chain.select(lambda r: r.region == "FR1")

        assert [r.offset for r in selection.residues] == [0]
        assert selection.of_chain is chain

    def test_a_region_group_covers_every_region_present(self):
        residues = [
            _residue("H", 0, "1", region="FR1"),
            _residue("H", 1, "2", region="FR1"),
            _residue("H", 2, "3", region="CDR1"),
        ]
        [chain] = residue_index.in_scope_chains(residues)

        by_region = chain.by_region()

        assert set(by_region) == {"FR1", "CDR1"}
        assert [r.offset for r in by_region["FR1"].residues] == [0, 1]
        assert [r.offset for r in by_region["CDR1"].residues] == [2]


class TestJoinKey:
    def test_the_join_key_is_the_chain_and_the_imgt_label(self):
        residue = _residue("H", 0, "107A")

        assert residue.join_key == ("H", "107A")
