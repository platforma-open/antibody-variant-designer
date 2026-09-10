"""Unit tests for `variant_store.py` — the `variants.tsv` round trip, the
per-parent-ordinal `variantKey` the variant axis is built from, and the
second pass that turns each parent's own local `rank` into one ordinal
across the whole run. The score that pass sorts on is tested in
`test_variant_ranking.py`."""

import csv
import io

from engine import variant_ranking, variant_store


def _variant(
    rank=1,
    parent_rank=None,
    chain="H,L",
    worst_confidence_angstroms=3.2,
    low_confidence_warning=False,
    changed_positions="H:N107D, H:G108S",
    structural_tolerance=2.0,
    humanness_score=None,
    addressed_target="Deamidation (N[GS]) @ CDR1 H:107",
    developability_score=0.0,
):
    # `parent_rank` defaults to `rank` — the shape every caller sees before
    # `rewrite_global_rank` ever runs, when the two are still identical.
    return variant_ranking.Variant(
        rank=rank,
        parent_rank=rank if parent_rank is None else parent_rank,
        chain=chain,
        addressed_target=addressed_target,
        changed_positions=changed_positions,
        variant_sequence="DSALA",
        structural_tolerance=structural_tolerance,
        humanness_score=humanness_score,
        worst_confidence_angstroms=worst_confidence_angstroms,
        binding_risk="Medium",
        low_confidence_warning=low_confidence_warning,
        status="unvalidated-hypothesis",
        developability_score=developability_score,
    )


def _rows_of(path):
    return list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))


class TestVariantsTsvRoundTrips:
    def test_write_then_read_returns_an_equal_variant(self, tmp_path):
        original = _variant()
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", [original])
        [(clonotype_key, _, rehydrated)] = variant_store.read_variants_tsv(str(path))

        assert clonotype_key == "clone-1"
        assert rehydrated == original

    def test_a_none_worst_confidence_round_trips_to_none_not_a_string(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(
            str(path), "clone-1", [_variant(worst_confidence_angstroms=None)]
        )
        [(_, _, rehydrated)] = variant_store.read_variants_tsv(str(path))

        assert rehydrated.worst_confidence_angstroms is None

    def test_low_confidence_warning_is_the_string_yes_or_no_never_a_bool(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(
            str(path),
            "clone-1",
            [
                _variant(rank=1, low_confidence_warning=True, changed_positions="H:N107D"),
                _variant(rank=2, low_confidence_warning=False, changed_positions="H:N108D"),
            ],
        )

        assert [r["lowConfidenceWarning"] for r in _rows_of(path)] == ["yes", "no"]

    def test_header_alone_is_a_valid_empty_file(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))

        assert variant_store.read_variants_tsv(str(path)) == []
        assert path.read_text().startswith("clonotypeKey\tvariantKey\t")

    def test_no_objective_column_reaches_the_file(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", [_variant()])

        rows = _rows_of(path)
        assert "objective" not in variant_store.TSV_COLUMNS
        assert "objective" not in rows[0]

    def test_a_variant_addressing_two_targets_holds_both_labels_joined(self, tmp_path):
        path = tmp_path / "variants.tsv"
        original = _variant(
            addressed_target="Deamidation (N[GS]) @ CDR1 H:107, Humanization @ H"
        )

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", [original])
        [(_, _, rehydrated)] = variant_store.read_variants_tsv(str(path))

        assert rehydrated.addressed_target == "Deamidation (N[GS]) @ CDR1 H:107, Humanization @ H"
        row = _rows_of(path)[0]
        assert row["addressedTarget"] == "Deamidation (N[GS]) @ CDR1 H:107, Humanization @ H"


class TestOneFileHoldsEveryParent:
    def test_each_appended_row_carries_its_own_clonotype_key(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", [_variant()])
        variant_store.append_variants_tsv(str(path), "clone-2", [_variant()])

        assert [r["clonotypeKey"] for r in _rows_of(path)] == ["clone-1", "clone-2"]

    def test_ranks_restart_per_parent_before_the_global_rewrite(self, tmp_path):
        # `append_variants_tsv` alone still writes the local, per-parent
        # rank `rank_variants` produced — only `rewrite_global_rank`, called
        # once after the whole batch loop, turns it into one ordinal.
        path = tmp_path / "variants.tsv"
        two = [_variant(rank=1, changed_positions="H:N107D"),
               _variant(rank=2, changed_positions="H:N108D")]

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", two)
        variant_store.append_variants_tsv(str(path), "clone-2", two)

        assert [r["rank"] for r in _rows_of(path)] == ["1", "2", "1", "2"]


class TestVariantKeyIsAPerParentOrdinal:
    def test_renders_zero_padded_to_two_digits_in_rank_order(self):
        assert variant_store.variant_key(1) == "v01"
        assert variant_store.variant_key(2) == "v02"
        assert variant_store.variant_key(10) == "v10"

    def test_one_parents_ranked_variants_get_v01_v02_in_rank_order(self, tmp_path):
        path = tmp_path / "variants.tsv"
        two = [_variant(rank=1, changed_positions="H:N107D"),
               _variant(rank=2, changed_positions="H:N108D")]

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", two)

        assert [r["variantKey"] for r in _rows_of(path)] == ["v01", "v02"]

    def test_no_run_identity_column_reaches_the_file(self, tmp_path):
        # Run identity is a spec domain the workflow applies after this
        # process, so nothing per-run enters the file or this process's
        # arguments — that is what makes two blocks over one dataset share
        # the exec instead of computing the same variants twice.
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", [_variant(rank=1)])

        rows = _rows_of(path)
        assert "blockId" not in variant_store.TSV_COLUMNS
        assert "blockId" not in rows[0]
        assert [r["variantKey"] for r in rows] == ["v01"]

    def test_clonotype_key_and_variant_key_are_unique_across_the_whole_file(self, tmp_path):
        path = tmp_path / "variants.tsv"
        two = [_variant(rank=1, changed_positions="H:N107D"),
               _variant(rank=2, changed_positions="H:N108D")]

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", two)
        variant_store.append_variants_tsv(str(path), "clone-2", two)

        rows = _rows_of(path)
        keys = [(r["clonotypeKey"], r["variantKey"]) for r in rows]
        assert len(set(keys)) == len(rows) == 4


class TestHumannessScoreRoundTrips:
    def test_a_none_humanness_score_writes_an_empty_cell_and_leaves_every_other_cell_alone(
        self, tmp_path
    ):
        path = tmp_path / "variants.tsv"
        original = _variant()

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", [original])
        [(_, _, rehydrated)] = variant_store.read_variants_tsv(str(path))

        row = _rows_of(path)[0]
        assert row["humannessScore"] == ""
        # Every other field round-trips to the same `Variant` the row before
        # this column existed would have produced.
        assert rehydrated == original

    def test_a_humanness_score_round_trips_as_a_float(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(
            str(path), "clone-1", [_variant(humanness_score=80.0)]
        )
        [(_, _, rehydrated)] = variant_store.read_variants_tsv(str(path))

        assert rehydrated.humanness_score == 80.0
        assert isinstance(rehydrated.humanness_score, float)


class TestRewriteGlobalRank:
    def test_renumbers_every_parents_survivors_by_tolerance_across_the_run(self, tmp_path):
        path = tmp_path / "variants.tsv"
        variant_store.write_variants_header(str(path))
        # clone-2's one variant tolerates edits better than either of
        # clone-1's — it must come out ranked ahead of both.
        variant_store.append_variants_tsv(
            str(path), "clone-1",
            [
                _variant(rank=1, structural_tolerance=9.0, changed_positions="H:N107D"),
                _variant(rank=2, structural_tolerance=5.0, changed_positions="H:N108D"),
            ],
        )
        variant_store.append_variants_tsv(
            str(path), "clone-2",
            [_variant(rank=1, structural_tolerance=12.0, changed_positions="H:N109D")],
        )

        variant_store.rewrite_global_rank(
            str(path),
            variant_ranking.DEFAULT_RERANK_STRUCTURAL_WEIGHT,
            variant_ranking.DEFAULT_RERANK_HUMANNESS_WEIGHT,
        )

        written = variant_store.read_variants_tsv(str(path))
        assert [(ck, v.rank) for ck, _, v in written] == [
            ("clone-2", 1), ("clone-1", 2), ("clone-1", 3),
        ]
        # `parent_rank` is untouched: clone-1's two rows still read 1, 2 —
        # their own local order — even though the global `rank` column now
        # interleaves them with clone-2's.
        assert [(ck, v.parent_rank) for ck, _, v in written] == [
            ("clone-2", 1), ("clone-1", 1), ("clone-1", 2),
        ]

    def test_variant_key_is_untouched_by_the_rewrite(self, tmp_path):
        path = tmp_path / "variants.tsv"
        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(
            str(path), "clone-1", [_variant(rank=1, structural_tolerance=9.0)]
        )
        variant_store.append_variants_tsv(
            str(path), "clone-2", [_variant(rank=1, structural_tolerance=5.0)]
        )

        variant_store.rewrite_global_rank(
            str(path),
            variant_ranking.DEFAULT_RERANK_STRUCTURAL_WEIGHT,
            variant_ranking.DEFAULT_RERANK_HUMANNESS_WEIGHT,
        )

        # Both parents' one survivor was locally rank 1, so both still
        # render `v01` — only the now-global `rank` column tells them apart.
        assert [variant for _, variant, _ in variant_store.read_variants_tsv(str(path))] == [
            "v01", "v01",
        ]

    def test_ties_break_by_clonotype_then_variant_key_for_determinism(self, tmp_path):
        path = tmp_path / "variants.tsv"
        variant_store.write_variants_header(str(path))
        tied = _variant(rank=1, structural_tolerance=5.0, changed_positions="H:N107D")
        variant_store.append_variants_tsv(str(path), "clone-2", [tied])
        variant_store.append_variants_tsv(str(path), "clone-1", [tied])

        variant_store.rewrite_global_rank(
            str(path),
            variant_ranking.DEFAULT_RERANK_STRUCTURAL_WEIGHT,
            variant_ranking.DEFAULT_RERANK_HUMANNESS_WEIGHT,
        )

        written = variant_store.read_variants_tsv(str(path))
        assert [(ck, v.rank) for ck, _, v in written] == [("clone-1", 1), ("clone-2", 2)]

    def test_an_empty_file_is_left_alone(self, tmp_path):
        path = tmp_path / "variants.tsv"
        variant_store.write_variants_header(str(path))

        variant_store.rewrite_global_rank(
            str(path),
            variant_ranking.DEFAULT_RERANK_STRUCTURAL_WEIGHT,
            variant_ranking.DEFAULT_RERANK_HUMANNESS_WEIGHT,
        )

        assert variant_store.read_variants_tsv(str(path)) == []

    def test_equal_tolerance_ranks_the_higher_humanness_score_first(self, tmp_path):
        path = tmp_path / "variants.tsv"
        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(
            str(path), "clone-1",
            [
                _variant(rank=1, structural_tolerance=5.0, humanness_score=40.0,
                         changed_positions="H:N107D"),
                _variant(rank=2, structural_tolerance=5.0, humanness_score=90.0,
                         changed_positions="H:N108D"),
            ],
        )

        variant_store.rewrite_global_rank(
            str(path),
            variant_ranking.DEFAULT_RERANK_STRUCTURAL_WEIGHT,
            variant_ranking.DEFAULT_RERANK_HUMANNESS_WEIGHT,
        )

        written = variant_store.read_variants_tsv(str(path))
        assert [v.changed_positions for _, _, v in written] == ["H:N108D", "H:N107D"]

    def test_no_humanness_number_in_the_file_orders_the_same_for_any_positive_alpha(self, tmp_path):
        path = tmp_path / "variants.tsv"
        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(
            str(path), "clone-1",
            [
                _variant(rank=1, structural_tolerance=9.0, changed_positions="H:N107D"),
                _variant(rank=2, structural_tolerance=5.0, changed_positions="H:N108D"),
            ],
        )

        one = tmp_path / "one.tsv"
        one.write_text(path.read_text())
        variant_store.rewrite_global_rank(
            str(one),
            rerank_structural_weight=1.0,
            rerank_humanness_weight=variant_ranking.DEFAULT_RERANK_HUMANNESS_WEIGHT,
        )

        seven = tmp_path / "seven.tsv"
        seven.write_text(path.read_text())
        variant_store.rewrite_global_rank(
            str(seven),
            rerank_structural_weight=7.0,
            rerank_humanness_weight=variant_ranking.DEFAULT_RERANK_HUMANNESS_WEIGHT,
        )

        order_at_one = [
            v.changed_positions for _, _, v in variant_store.read_variants_tsv(str(one))
        ]
        order_at_seven = [
            v.changed_positions for _, _, v in variant_store.read_variants_tsv(str(seven))
        ]
        assert order_at_one == order_at_seven == ["H:N107D", "H:N108D"]
