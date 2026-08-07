"""Unit tests for `variant_store.py` — the round trip for `variants.tsv`."""

import variant_store


def _variant(rank=1, worst_confidence_angstroms=3.2, low_confidence_warning=False):
    return variant_store.Variant(
        rank=rank,
        addressed_target="Deamidation (N[GS]) @ CDR1 H:107",
        changed_positions="H:N107D, H:G108S",
        variant_sequence="DSALA",
        structural_tolerance=2.0,
        worst_confidence_angstroms=worst_confidence_angstroms,
        binding_risk="Medium",
        low_confidence_warning=low_confidence_warning,
        status="unvalidated-hypothesis",
    )


class TestVariantsTsvRoundTrips:
    def test_write_then_read_returns_an_equal_variant(self, tmp_path):
        original = _variant()
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_tsv(str(path), [original])
        [rehydrated] = variant_store.read_variants_tsv(str(path))

        assert rehydrated == original

    def test_a_none_worst_confidence_round_trips_to_none_not_a_string(self, tmp_path):
        original = _variant(worst_confidence_angstroms=None)
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_tsv(str(path), [original])
        [rehydrated] = variant_store.read_variants_tsv(str(path))

        assert rehydrated.worst_confidence_angstroms is None

    def test_low_confidence_warning_is_the_string_yes_or_no_never_a_bool(self, tmp_path):
        low = _variant(rank=1, low_confidence_warning=True)
        high = _variant(rank=2, low_confidence_warning=False)
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_tsv(str(path), [low, high])

        text = path.read_text()
        rows = [line.split("\t") for line in text.strip("\n").split("\n")[1:]]
        header = variant_store.TSV_COLUMNS
        column = header.index("lowConfidenceWarning")
        assert [row[column] for row in rows] == ["yes", "no"]

    def test_empty_list_round_trips_to_empty(self, tmp_path):
        path = tmp_path / "variants.tsv"

        variant_store.write_variants_tsv(str(path), [])

        assert variant_store.read_variants_tsv(str(path)) == []
