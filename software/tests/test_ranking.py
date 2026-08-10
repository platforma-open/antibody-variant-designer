"""Unit tests for `ranking.py`: binding-risk banding, sequence rendering,
and the rank/truncate/epistasis-rescore pipeline."""

from pathlib import Path

import candidate_store
import pytest
import ranking
import residue_store
import skip_store
import variant_store


def _candidate(
    edits_count=1,
    tolerance=5.0,
    changed_positions="H:N107D",
    region="FR1",
    low_confidence=False,
    worst_confidence_angstroms=3.0,
):
    edits = tuple(
        candidate_store.Edit(chain="H", offset=i, imgt=str(i + 1), wild_type="N", to="D")
        for i in range(edits_count)
    )
    return candidate_store.Candidate(
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


class TestBindingRisk:
    def test_cdr_edit_below_the_tolerance_floor_is_high(self):
        candidate = _candidate(region="CDR3", tolerance=1.0, low_confidence=False)

        assert ranking.binding_risk(candidate, low_tolerance_floor=3.0) == "High"

    def test_cdr_edit_at_low_confidence_is_high_even_with_fine_tolerance(self):
        candidate = _candidate(region="CDR3", tolerance=10.0, low_confidence=True)

        assert ranking.binding_risk(candidate, low_tolerance_floor=3.0) == "High"

    def test_cdr_edit_with_neither_problem_is_medium(self):
        candidate = _candidate(region="CDR3", tolerance=10.0, low_confidence=False)

        assert ranking.binding_risk(candidate, low_tolerance_floor=3.0) == "Medium"

    def test_framework_edit_with_both_problems_is_medium(self):
        candidate = _candidate(region="FR2", tolerance=1.0, low_confidence=True)

        assert ranking.binding_risk(candidate, low_tolerance_floor=3.0) == "Medium"

    def test_framework_edit_with_one_problem_is_low(self):
        candidate = _candidate(region="FR2", tolerance=1.0, low_confidence=False)

        assert ranking.binding_risk(candidate, low_tolerance_floor=3.0) == "Low"

    def test_framework_edit_with_no_problem_is_low(self):
        candidate = _candidate(region="FR2", tolerance=10.0, low_confidence=False)

        assert ranking.binding_risk(candidate, low_tolerance_floor=3.0) == "Low"


class TestBuildVariantSequence:
    def test_edited_position_is_replaced_unedited_ones_keep_their_wild_type(self):
        residues = [
            _residue("H", 0, "N", role="H"),
            _residue("H", 1, "G", role="H"),
            _residue("L", 0, "E", role="L"),
        ]
        edits = (candidate_store.Edit(chain="H", offset=0, imgt="1", wild_type="N", to="D"),)

        assert ranking.build_variant_sequence(residues, edits) == "DGE"

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

        assert ranking.build_variant_sequence(residues, ()) == "N"

    def test_heavy_chain_precedes_light_chain(self):
        residues = [
            _residue("L", 0, "E", role="L"),
            _residue("H", 0, "N", role="H"),
        ]

        assert ranking.build_variant_sequence(residues, ()) == "NE"


class TestRankVariantsOrderingAndTruncation:
    def test_ranks_restart_at_one_and_order_by_band_then_tolerance(self):
        low_risk = _candidate(
            region="FR1", tolerance=10.0, low_confidence=False, changed_positions="H:N107D"
        )
        high_risk = _candidate(
            region="CDR1", tolerance=1.0, low_confidence=False, changed_positions="H:N108D"
        )

        variants = ranking.rank_variants(
            [high_risk, low_risk], residues=[],
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=20,
        )

        assert [v.binding_risk for v in variants] == ["Low", "High"]
        assert [v.rank for v in variants] == [1, 2]

    def test_variants_per_parent_truncates_the_ranked_list(self):
        candidates = [
            _candidate(region="FR1", tolerance=float(i), changed_positions=f"H:N10{i}D")
            for i in range(5)
        ]

        variants = ranking.rank_variants(
            candidates, residues=[],
            variants_per_parent=2, low_tolerance_floor=3.0, epistasis_rescore_top_k=20,
        )

        assert len(variants) == 2
        assert [v.rank for v in variants] == [1, 2]

    def test_epistasis_rescore_penalizes_extra_edits_within_the_window(self):
        # Same band (Low). B starts ahead on raw tolerance (5.5 > 5.0), but
        # carries three edits against A's one — the window's edit-count
        # penalty must flip the order.
        one_edit = _candidate(
            edits_count=1, tolerance=5.0, changed_positions="H:N107D", region="FR1",
        )
        three_edits = _candidate(
            edits_count=3, tolerance=5.5, changed_positions="H:N108D", region="FR1",
        )

        rescored = ranking.rank_variants(
            [three_edits, one_edit], residues=[],
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=2,
        )
        assert [v.changed_positions for v in rescored] == ["H:N107D", "H:N108D"]

        not_rescored = ranking.rank_variants(
            [three_edits, one_edit], residues=[],
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

        variants = ranking.rank_variants(
            [*window, outside], residues=[],
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=1,
        )

        # `outside` ranks second regardless of the penalty the window's one
        # candidate absorbs, because the window never grows past top_k=1.
        assert [v.changed_positions for v in variants] == ["H:N107D", "H:N108D"]

    def test_reported_structural_tolerance_is_never_the_epistasis_adjusted_value(self):
        three_edits = _candidate(edits_count=3, tolerance=5.5, changed_positions="H:N108D")

        [variant] = ranking.rank_variants(
            [three_edits], residues=[],
            variants_per_parent=10, low_tolerance_floor=3.0, epistasis_rescore_top_k=1,
        )

        assert variant.structural_tolerance == pytest.approx(5.5)


BLOCK_ID = "block-abc"


def _stage(batch, clonotype_key, candidate_list=None):
    """Stage one antibody's `candidates.json` and `residues.json`, the way
    candidates.py and structure.py would have left them."""
    entry = batch.add(clonotype_key)
    residue_store.write_residues(
        str(Path(batch.dir("residues"), f"{entry.stem}.json")),
        [_residue("H", 0, "N", role="H"), _residue("H", 1, "G", role="H")],
    )
    candidate_store.write_candidates(
        str(Path(batch.dir("candidates"), f"{entry.stem}.json")),
        candidate_list
        if candidate_list is not None
        else [
            _candidate(region="FR1", tolerance=10.0, changed_positions="H:N1D"),
            _candidate(region="CDR3", tolerance=1.0, changed_positions="H:N2D"),
        ],
    )
    return entry


def _run(batch, extra_args=None):
    out_variants = batch.path("variants.tsv")
    out_skip = batch.path("skip.tsv")

    rc = ranking.main(
        [
            "--candidates-dir", batch.dir("candidates"),
            "--residues-dir", batch.dir("residues"),
            "--pdb-index", batch.index,
            "--block-id", BLOCK_ID,
            "--out-variants", out_variants,
            "--out-skip", out_skip,
        ]
        + (extra_args or [])
    )

    assert rc == 0
    return skip_store.read_skips(out_skip), variant_store.read_variants_tsv(out_variants)


class TestMainWiresRankingAndWritesTheTsv:
    def test_a_successful_run_writes_one_row_per_candidate(self, batch):
        _stage(batch, "clone-1")

        skips, written = _run(batch)

        assert skips == [("clone-1", "")]
        assert len(written) == 2
        assert [v.status for _, _, v in written] == ["unvalidated-hypothesis"] * 2


class TestBatchCli:
    def test_one_variants_file_holds_every_parent_with_ranks_restarting(self, batch):
        _stage(batch, "clone-1")
        _stage(batch, "clone-2")

        skips, written = _run(batch)

        assert skips == [("clone-1", ""), ("clone-2", "")]
        assert [key for key, _, _ in written] == ["clone-1", "clone-1", "clone-2", "clone-2"]
        assert [v.rank for _, _, v in written] == [1, 2, 1, 2]

    def test_clonotype_key_and_variant_key_are_unique_across_the_file(self, batch):
        _stage(batch, "clone-1")
        _stage(batch, "clone-2")

        _, written = _run(batch)

        keys = [(clonotype, variant) for clonotype, variant, _ in written]
        assert len(set(keys)) == len(written) == 4

    def test_two_parents_sharing_an_edit_string_get_different_variant_keys(self, batch):
        one_candidate = [_candidate(region="FR1", tolerance=10.0, changed_positions="H:N1D")]
        _stage(batch, "clone-1", one_candidate)
        _stage(batch, "clone-2", one_candidate)

        _, written = _run(batch)

        assert [v.changed_positions for _, _, v in written] == ["H:N1D", "H:N1D"]
        assert len({variant for _, variant, _ in written}) == 2

    def test_variants_per_parent_truncates_within_each_parent(self, batch):
        _stage(batch, "clone-1")
        _stage(batch, "clone-2")

        _, written = _run(batch, ["--variants-per-parent", "1"])

        assert [key for key, _, _ in written] == ["clone-1", "clone-2"]
        assert [v.rank for _, _, v in written] == [1, 1]

    def test_an_antibody_missing_its_candidates_file_gets_no_row(self, batch):
        _stage(batch, "cleared")
        batch.add("no-candidates")

        skips, written = _run(batch)

        assert skips == [("cleared", "")]
        assert {key for key, _, _ in written} == {"cleared"}

    def test_an_empty_roster_leaves_a_header_only_variants_tsv(self, batch):
        skips, written = _run(batch)

        assert skips == []
        assert written == []
