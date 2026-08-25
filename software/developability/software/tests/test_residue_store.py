"""Unit tests for `residue_store.py`'s value objects — the chain minter, the
selection a filter or a region group returns, and the join key every
cross-file join reads."""

from engine import residue_store


def _residue(chain, offset, imgt, wild_type="A", region="FR1", chain_role="H"):
    return residue_store.Residue(
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
            residue_store.Residue(
                chain="H", offset=3, imgt="129", wild_type="A", res_name="ALA",
                b_factor=20.0, region=None, chain_role="H",
            ),
            # A role-less second arm, in scope by neither test.
            _residue("X", 0, "1", chain_role=None, region="FR1"),
        ]

        [chain] = residue_store.in_scope_chains(residues)

        assert chain.chain == "H"
        assert [r.offset for r in chain.residues] == [0, 1, 2]

    def test_a_role_less_chain_mints_no_chain(self):
        residues = [
            _residue("H", 0, "1"),
            _residue("X", 0, "1", chain_role=None),
            residue_store.Residue(
                chain="Y", offset=0, imgt="1", wild_type="A", res_name="ALA",
                b_factor=20.0, region=None, chain_role=None,
            ),
        ]

        chains = residue_store.in_scope_chains(residues)

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

        chains = residue_store.in_scope_chains(residues)

        assert [c.chain_role for c in chains] == ["H", "L"]


class TestSelectAndByRegion:
    def test_a_selection_carries_the_chain_it_came_from(self):
        residues = [_residue("H", 0, "1", region="FR1"), _residue("H", 1, "2", region="CDR1")]
        [chain] = residue_store.in_scope_chains(residues)

        selection = chain.select(lambda r: r.region == "FR1")

        assert [r.offset for r in selection.residues] == [0]
        assert selection.of_chain is chain

    def test_a_region_group_covers_every_region_present(self):
        residues = [
            _residue("H", 0, "1", region="FR1"),
            _residue("H", 1, "2", region="FR1"),
            _residue("H", 2, "3", region="CDR1"),
        ]
        [chain] = residue_store.in_scope_chains(residues)

        by_region = chain.by_region()

        assert set(by_region) == {"FR1", "CDR1"}
        assert [r.offset for r in by_region["FR1"].residues] == [0, 1]
        assert [r.offset for r in by_region["CDR1"].residues] == [2]


class TestJoinKey:
    def test_the_join_key_is_the_chain_and_the_imgt_label(self):
        residue = _residue("H", 0, "107A")

        assert residue.join_key == ("H", "107A")
