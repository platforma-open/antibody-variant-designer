"""Unit tests for `rejection_store.py` — the per-clonotype rejection TSV that carries
attribution now that one exec covers the whole dataset."""

from engine import rejection_store


class TestRejectionTsvRoundTrips:
    def test_write_then_read_returns_equal_rows(self, tmp_path):
        rows = [
            ("clone-1", "", "", "parent", "liability"),
            ("clone-2", "no-structure", "", "parent", "liability"),
        ]
        path = tmp_path / "rejected.tsv"

        rejection_store.write_rejections(str(path), rows)

        assert rejection_store.read_rejections(str(path)) == rows

    def test_empty_list_leaves_a_header_only_file(self, tmp_path):
        path = tmp_path / "rejected.tsv"

        rejection_store.write_rejections(str(path), [])

        assert path.read_text() == "clonotypeKey\treason\tdetail\trejectedType\tobjective\n"
        assert rejection_store.read_rejections(str(path)) == []
