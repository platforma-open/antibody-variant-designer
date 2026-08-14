"""Unit tests for `variant_store.py` — the `variants.tsv` round trip, and
the per-parent-ordinal `variantKey` the variant axis is built from."""

import csv
import io

import variant_store

BLOCK_ID = "block-abc"


def _variant(
    rank=1,
    chain="H,L",
    worst_confidence_angstroms=3.2,
    low_confidence_warning=False,
    changed_positions="H:N107D, H:G108S",
):
    return variant_store.Variant(
        rank=rank,
        chain=chain,
        addressed_target="Deamidation (N[GS]) @ CDR1 H:107",
        changed_positions=changed_positions,
        variant_sequence="DSALA",
        structural_tolerance=2.0,
        worst_confidence_angstroms=worst_confidence_angstroms,
        binding_risk="Medium",
        low_confidence_warning=low_confidence_warning,
        status="unvalidated-hypothesis",
    )


def _rows_of(path):
    return list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))


class TestVariantsTsvRoundTrips:
    def test_write_then_read_returns_an_equal_variant(self, tmp_path):
        original = _variant()
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", BLOCK_ID, [original])
        [(clonotype_key, _, rehydrated)] = variant_store.read_variants_tsv(str(path))

        assert clonotype_key == "clone-1"
        assert rehydrated == original

    def test_a_none_worst_confidence_round_trips_to_none_not_a_string(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(
            str(path), "clone-1", BLOCK_ID, [_variant(worst_confidence_angstroms=None)]
        )
        [(_, _, rehydrated)] = variant_store.read_variants_tsv(str(path))

        assert rehydrated.worst_confidence_angstroms is None

    def test_low_confidence_warning_is_the_string_yes_or_no_never_a_bool(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(
            str(path),
            "clone-1",
            BLOCK_ID,
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


class TestOneFileHoldsEveryParent:
    def test_each_appended_row_carries_its_own_clonotype_key(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", BLOCK_ID, [_variant()])
        variant_store.append_variants_tsv(str(path), "clone-2", BLOCK_ID, [_variant()])

        assert [r["clonotypeKey"] for r in _rows_of(path)] == ["clone-1", "clone-2"]

    def test_ranks_restart_per_parent_inside_the_one_file(self, tmp_path):
        path = tmp_path / "variants.tsv"
        two = [_variant(rank=1, changed_positions="H:N107D"),
               _variant(rank=2, changed_positions="H:N108D")]

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", BLOCK_ID, two)
        variant_store.append_variants_tsv(str(path), "clone-2", BLOCK_ID, two)

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
        variant_store.append_variants_tsv(str(path), "clone-1", BLOCK_ID, two)

        assert [r["variantKey"] for r in _rows_of(path)] == ["v01", "v02"]

    def test_block_id_is_its_own_column_not_a_key_ingredient(self, tmp_path):
        # A different blockId changes the emitted `blockId` column, never
        # the ordinal `variantKey` — the third axis and the variant axis
        # are independent, per `073-decision-the-block-id-becomes-an-axis`.
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", "block-abc", [_variant(rank=1)])
        variant_store.append_variants_tsv(str(path), "clone-2", "block-xyz", [_variant(rank=1)])

        rows = _rows_of(path)
        assert [r["blockId"] for r in rows] == ["block-abc", "block-xyz"]
        assert [r["variantKey"] for r in rows] == ["v01", "v01"]

    def test_clonotype_key_and_variant_key_are_unique_across_the_whole_file(self, tmp_path):
        path = tmp_path / "variants.tsv"
        two = [_variant(rank=1, changed_positions="H:N107D"),
               _variant(rank=2, changed_positions="H:N108D")]

        variant_store.write_variants_header(str(path))
        variant_store.append_variants_tsv(str(path), "clone-1", BLOCK_ID, two)
        variant_store.append_variants_tsv(str(path), "clone-2", BLOCK_ID, two)

        rows = _rows_of(path)
        keys = [(r["clonotypeKey"], r["variantKey"]) for r in rows]
        assert len(set(keys)) == len(rows) == 4
