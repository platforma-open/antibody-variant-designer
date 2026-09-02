"""Unit tests for `rejection_store.py` — the per-clonotype rejection TSV that carries
attribution now that one exec covers the whole dataset."""

from engine import rejection_store


class TestRejectionTsvRoundTrips:
    def test_write_then_read_returns_equal_rows(self, tmp_path):
        rows = [("clone-1", "", "", "parent"), ("clone-2", "no-structure", "", "parent")]
        path = tmp_path / "rejected.tsv"

        rejection_store.write_rejections(str(path), rows)

        assert rejection_store.read_rejections(str(path)) == rows

    def test_a_passing_clonotype_is_an_empty_reason_not_an_absent_row(self, tmp_path):
        # Present-with-empty and absent mean different things: empty is
        # "this step attempted it and it passed", absent is "an earlier
        # step already named its reason".
        path = tmp_path / "rejected.tsv"

        rejection_store.write_rejections(str(path), [("clone-1", "", "", "parent")])

        assert rejection_store.read_rejections(str(path)) == [("clone-1", "", "", "parent")]

    def test_a_reason_can_carry_free_text_detail(self, tmp_path):
        # `backend-failed` is the one reason a caught exception's message
        # fills in; every other reason writes an empty detail.
        path = tmp_path / "rejected.tsv"

        rejection_store.write_rejections(
            str(path), [("clone-1", "backend-failed", "CUDA out of memory", "parent")]
        )

        assert rejection_store.read_rejections(str(path)) == [
            ("clone-1", "backend-failed", "CUDA out of memory", "parent")
        ]

    def test_empty_list_leaves_a_header_only_file(self, tmp_path):
        path = tmp_path / "rejected.tsv"

        rejection_store.write_rejections(str(path), [])

        assert path.read_text() == "clonotypeKey\treason\tdetail\trejectedType\n"
        assert rejection_store.read_rejections(str(path)) == []
