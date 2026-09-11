"""Unit tests for `tolerance_store.py` — the keyed tolerance TSV round trip."""

from engine import tolerance_store

AMINO_ACIDS = tolerance_store.AMINO_ACIDS


def _row(imgt, chain="H", perplexity=3.0, value=-1.0):
    row = {"chain": chain, "imgt": imgt, "perplexity": perplexity}
    row.update(dict.fromkeys(AMINO_ACIDS, value))
    return row


class TestToleranceArtifactRoundTrips:
    def test_two_parents_rows_never_merge(self, tmp_path):
        path = tmp_path / "tolerance.tsv"

        with tolerance_store.ToleranceWriter(str(path)) as writer:
            writer.write("clone-1", [_row("1"), _row("2")])
            writer.write("clone-2", [_row("1")])

        reader = tolerance_store.open_tolerance(str(path))
        clone_1 = tolerance_store.lookup_from_payload(reader.take("clone-1"))
        clone_2 = tolerance_store.lookup_from_payload(reader.take("clone-2"))

        assert set(clone_1) == {("H", "1"), ("H", "2")}
        assert set(clone_2) == {("H", "1")}

    def test_every_field_round_trips_intact(self, tmp_path):
        path = tmp_path / "tolerance.tsv"

        with tolerance_store.ToleranceWriter(str(path)) as writer:
            writer.write("clone-1", [_row("1", perplexity=4.2, value=-2.5)])

        payload = tolerance_store.open_tolerance(str(path)).take("clone-1")
        lookup = tolerance_store.lookup_from_payload(payload)

        row = lookup[("H", "1")]
        assert row["perplexity"] == 4.2
        assert all(row["logProbs"][aa] == -2.5 for aa in AMINO_ACIDS)

    def test_a_key_the_artifact_does_not_carry_returns_none(self, tmp_path):
        path = tmp_path / "tolerance.tsv"

        with tolerance_store.ToleranceWriter(str(path)) as writer:
            writer.write("clone-1", [_row("1")])
            writer.write("clone-3", [_row("1")])

        reader = tolerance_store.open_tolerance(str(path))
        assert reader.take("clone-2") is None
        assert reader.take("clone-3") is not None
