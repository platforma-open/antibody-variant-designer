"""Unit tests for `pdb_index.py` — the `pdb_index.tsv` round trip and the
clonotype-key-to-stem resolution every batch entrypoint depends on."""

import pdb_index


class TestPdbIndexRoundTrips:
    def test_write_then_read_returns_equal_entries(self, tmp_path):
        entries = [
            pdb_index.Entry(clonotype_key="clone-1", filename="clone-1.pdb"),
            pdb_index.Entry(clonotype_key="clone-2", filename="clone-2.pdb"),
        ]
        path = tmp_path / "pdb_index.tsv"

        pdb_index.write_index(str(path), entries)

        assert pdb_index.read_index(str(path)) == entries

    def test_file_order_is_preserved_not_re_sorted(self, tmp_path):
        # The workflow writes the pdb_index in sorted-key order; every step
        # must iterate in that same order so the exec input stays canonical.
        entries = [
            pdb_index.Entry(clonotype_key="b", filename="b.pdb"),
            pdb_index.Entry(clonotype_key="a", filename="a.pdb"),
        ]
        path = tmp_path / "pdb_index.tsv"

        pdb_index.write_index(str(path), entries)

        assert [e.clonotype_key for e in pdb_index.read_index(str(path))] == ["b", "a"]

    def test_empty_index_round_trips_to_empty(self, tmp_path):
        path = tmp_path / "pdb_index.tsv"

        pdb_index.write_index(str(path), [])

        assert pdb_index.read_index(str(path)) == []


class TestStemIsTheOnlyFilenameSource:
    def test_stem_drops_the_extension(self):
        entry = pdb_index.Entry(clonotype_key="clone-1", filename="clone-1.pdb")

        assert entry.stem == "clone-1"

    def test_a_clonotype_key_holding_json_punctuation_still_resolves_to_a_safe_stem(
        self, tmp_path
    ):
        # A clonotype key is a JSON-encoded array and is not a legal
        # filename. This is the whole reason `stem` exists: no artifact
        # path is ever derived from the key.
        entry = pdb_index.Entry(clonotype_key='["odd/key"]', filename="antibody-07.pdb")
        path = tmp_path / "pdb_index.tsv"

        pdb_index.write_index(str(path), [entry])
        [rehydrated] = pdb_index.read_index(str(path))

        assert rehydrated.clonotype_key == '["odd/key"]'
        assert rehydrated.stem == "antibody-07"
