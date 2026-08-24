"""Unit tests for `variant_ranking.py` — banding, sequence rendering and the
rank/truncate/epistasis-rescore pass, as a module. Its CLI lives in
`build_variants.py`, tested in `test_build_variants.py`."""


import pytest

from engine import residue_store, variant_candidates, variant_ranking


def _candidate(
    edits_count=1,
    tolerance=5.0,
    changed_positions="H:N107D",
    region="FR1",
    low_confidence=False,
    worst_confidence_angstroms=3.0,
    imgt_start=1,
):
    edits = tuple(
        variant_candidates.Edit(
            chain="H", offset=i, imgt=str(imgt_start + i), wild_type="N", to="D"
        )
        for i in range(edits_count)
    )
    return variant_candidates.Candidate(
        target_definition_id="deamidation_ng",
        edits=edits,
        tolerance=tolerance,
        region=region,
        low_confidence=low_confidence,
        worst_confidence_angstroms=worst_confidence_angstroms,
        addressed_target="Deamidation (N[GS]) @ FR1 H:107",
        changed_positions=changed_positions,
    )


def _residue(chain, offset, wild_type, imgt=None, role="H", region="CDR1"):
    return residue_store.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region=region,
        chain_role=role,
    )


def _tolerance_row(perplexity):
    return {"perplexity": perplexity, "logProbs": {}}


class TestLowTolerancePositions:
    def test_a_position_at_or_below_the_floor_is_low_tolerance(self):
        lookup = {("H", "1"): _tolerance_row(3.0), ("H", "2"): _tolerance_row(10.0)}

        positions = variant_ranking._low_tolerance_positions(lookup, floor=3.0)

        assert ("H", "1") in positions
        assert ("H", "2") not in positions

    def test_bottom_third_and_floor_can_disagree_neither_arm_alone_passes_both(self):
        # Nine positions, evenly spaced 1.0..9.0. The bottom third by count
        # is the three lowest (1.0, 2.0, 3.0); the floor is 2.5.
        lookup = {("H", str(i)): _tolerance_row(float(i)) for i in range(1, 10)}

        positions = variant_ranking._low_tolerance_positions(lookup, floor=2.5)

        # Position 3 (perplexity 3.0) is above the floor but inside the
        # bottom third — the bottom-third arm alone must catch it.
        assert ("H", "3") in positions
        # Position 2 (perplexity 2.0) is below the floor but also inside
        # the bottom third — both arms agree, unlike the case above.
        assert ("H", "2") in positions
        # Position 4 (perplexity 4.0) is above both the floor and the
        # bottom third — neither arm passes it.
        assert ("H", "4") not in positions


class TestBindingRisk:
    def test_cdr_edit_at_a_low_tolerance_position_is_high(self):
        candidate = _candidate(region="CDR3", changed_positions="H:N1D", imgt_start=1)

        risk = variant_ranking.binding_risk(candidate, low_tolerance_positions={("H", "1")})
        assert risk == "High"

    def test_cdr_edit_at_low_confidence_is_high_even_at_a_tolerant_position(self):
        candidate = _candidate(
            region="CDR3", changed_positions="H:N1D", imgt_start=1, low_confidence=True,
        )

        assert variant_ranking.binding_risk(candidate, low_tolerance_positions=set()) == "High"

    def test_cdr_edit_with_neither_problem_is_medium(self):
        candidate = _candidate(region="CDR3", changed_positions="H:N1D", imgt_start=1)

        assert variant_ranking.binding_risk(candidate, low_tolerance_positions=set()) == "Medium"

    def test_framework_edit_at_a_low_tolerance_position_and_confident_is_medium(self):
        # The one clause this TODO exists for: the shipped code required
        # low-tolerance AND low-confidence for a framework edit to reach
        # Medium — one clause, and a whole band of rows misreported.
        candidate = _candidate(region="FR2", changed_positions="H:N1D", imgt_start=1)

        risk = variant_ranking.binding_risk(candidate, low_tolerance_positions={("H", "1")})
        assert risk == "Medium"

    def test_framework_edit_at_a_low_tolerance_position_and_low_confidence_is_medium(self):
        # Same band as the case above — the clause is genuinely dropped,
        # not inverted.
        candidate = _candidate(
            region="FR2", changed_positions="H:N1D", imgt_start=1, low_confidence=True,
        )

        risk = variant_ranking.binding_risk(candidate, low_tolerance_positions={("H", "1")})
        assert risk == "Medium"

    def test_framework_edit_at_a_tolerant_position_is_low(self):
        candidate = _candidate(region="FR2", changed_positions="H:N1D", imgt_start=1)

        assert variant_ranking.binding_risk(candidate, low_tolerance_positions=set()) == "Low"


class TestBuildVariantSequence:
    def test_edited_position_is_replaced_unedited_ones_keep_their_wild_type(self):
        residues = [
            _residue("H", 0, "N", role="H"),
            _residue("H", 1, "G", role="H"),
            _residue("L", 0, "E", role="L"),
        ]
        edits = (variant_candidates.Edit(chain="H", offset=0, imgt="1", wild_type="N", to="D"),)

        assert variant_ranking.build_variant_sequence(residues, edits) == "DGE"

    def test_out_of_scope_residues_are_excluded(self):
        # A constant-domain residue: no role, no region — never in the
        # designed sequence, even though it is in the residue index.
        residues = [
            _residue("H", 0, "N", role="H"),
            residue_store.Residue(
                chain="H", offset=1, imgt="129", wild_type="A", res_name="ALA",
                b_factor=20.0, region=None, chain_role=None,
            ),
        ]

        assert variant_ranking.build_variant_sequence(residues, ()) == "N"

    def test_heavy_chain_precedes_light_chain(self):
        residues = [
            _residue("L", 0, "E", role="L"),
            _residue("H", 0, "N", role="H"),
        ]

        assert variant_ranking.build_variant_sequence(residues, ()) == "NE"


class TestRankVariantsOrderingAndTruncation:
    def test_ranks_restart_at_one_and_order_by_tolerance_not_band(self):
        # `high_risk` has the worse band (High, CDR + low-tolerance) but
        # the better tolerance — proves the band is no longer a sort key.
        low_risk = _candidate(
            region="FR1", tolerance=10.0, changed_positions="H:N107D", imgt_start=107,
        )
        high_risk = _candidate(
            region="CDR1", tolerance=20.0, changed_positions="H:N108D", imgt_start=108,
        )

        variants = variant_ranking.rank_variants(
            [high_risk, low_risk], residues=[],
            tolerance_lookup={("H", "108"): _tolerance_row(1.0)},
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=20,
        )

        assert [v.binding_risk for v in variants] == ["High", "Low"]
        assert [v.rank for v in variants] == [1, 2]

    def test_variants_per_parent_truncates_the_ranked_list(self):
        candidate_list = [
            _candidate(region="FR1", tolerance=float(i), changed_positions=f"H:N10{i}D")
            for i in range(5)
        ]

        variants = variant_ranking.rank_variants(
            candidate_list, residues=[], tolerance_lookup={},
            variants_per_parent=2, low_tolerance_floor=3.0, epistasis_rescore_top_k=20,
        )

        assert len(variants) == 2
        assert [v.rank for v in variants] == [1, 2]

    def test_epistasis_rescore_penalizes_extra_edits_within_the_window(self):
        # B starts ahead on raw tolerance (5.5 > 5.0), but carries three
        # edits against A's one — the window's edit-count penalty must
        # flip the order, with no band involved on either side.
        one_edit = _candidate(
            edits_count=1, tolerance=5.0, changed_positions="H:N107D", region="FR1",
        )
        three_edits = _candidate(
            edits_count=3, tolerance=5.5, changed_positions="H:N108D", region="FR1",
        )

        rescored = variant_ranking.rank_variants(
            [three_edits, one_edit], residues=[], tolerance_lookup={},
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=2,
        )
        assert [v.changed_positions for v in rescored] == ["H:N107D", "H:N108D"]

        not_rescored = variant_ranking.rank_variants(
            [three_edits, one_edit], residues=[], tolerance_lookup={},
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=0,
        )
        assert [v.changed_positions for v in not_rescored] == ["H:N108D", "H:N107D"]

    def test_epistasis_rescore_never_moves_a_candidate_outside_the_window(self):
        window = [
            _candidate(edits_count=5, tolerance=9.0, changed_positions="H:N107D", region="FR1"),
        ]
        outside = _candidate(
            edits_count=1, tolerance=8.0, changed_positions="H:N108D", region="FR1",
        )

        variants = variant_ranking.rank_variants(
            [*window, outside], residues=[], tolerance_lookup={},
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=1,
        )

        # `outside` ranks second regardless of the penalty the window's one
        # candidate absorbs, because the window never grows past top_k=1.
        assert [v.changed_positions for v in variants] == ["H:N107D", "H:N108D"]

    def test_reported_structural_tolerance_is_never_the_epistasis_adjusted_value(self):
        three_edits = _candidate(edits_count=3, tolerance=5.5, changed_positions="H:N108D")

        [variant] = variant_ranking.rank_variants(
            [three_edits], residues=[], tolerance_lookup={},
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=1,
        )

        assert variant.structural_tolerance == pytest.approx(5.5)

    def test_two_candidates_better_mean_tolerance_but_worse_band_ranks_first(self):
        better_tolerance_worse_band = _candidate(
            region="CDR1", tolerance=9.0, changed_positions="H:N107D", imgt_start=107,
        )
        worse_tolerance_better_band = _candidate(
            region="FR1", tolerance=1.0, changed_positions="H:N108D", imgt_start=108,
        )

        variants = variant_ranking.rank_variants(
            [worse_tolerance_better_band, better_tolerance_worse_band],
            residues=[],
            tolerance_lookup={("H", "107"): _tolerance_row(1.0), ("H", "108"): _tolerance_row(9.0)},
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=20,
        )

        assert variants[0].changed_positions == "H:N107D"
        assert variants[0].binding_risk == "High"


class TestRankVariantsVhhFlag:
    def test_a_residue_index_with_no_l_role_reports_chain_h(self):
        residues = [_residue("H", 0, "N", role="H")]
        candidate = _candidate()

        [variant] = variant_ranking.rank_variants(
            [candidate], residues=residues, tolerance_lookup={},
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=20,
        )

        assert variant.chain == "H"

    def test_a_residue_index_with_an_l_role_reports_chain_h_l(self):
        residues = [_residue("H", 0, "N", role="H"), _residue("L", 0, "E", role="L")]
        candidate = _candidate()

        [variant] = variant_ranking.rank_variants(
            [candidate], residues=residues, tolerance_lookup={},
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=20,
        )

        assert variant.chain == "H,L"
