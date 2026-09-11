"""Unit tests for `keyed_artifact.py` — the writer/reader pair every boundary artifact sits on."""

import pytest

from engine import keyed_artifact


def _write_jsonl(path, rows):
    with keyed_artifact.KeyedWriter(str(path)) as writer:
        for key, payload in rows:
            writer.write(key, payload)


class TestJsonlRoundTrip:
    def test_three_parents_read_back_in_key_order_unchanged(self, tmp_path):
        path = tmp_path / "artifact.jsonl"
        _write_jsonl(path, [("a", [1]), ("b", [2, 3]), ("c", [])])

        reader = keyed_artifact.read_jsonl(str(path))

        assert reader.take("a") == [1]
        assert reader.take("b") == [2, 3]
        assert reader.take("c") == []

    def test_a_key_between_two_carried_keys_returns_none(self, tmp_path):
        path = tmp_path / "artifact.jsonl"
        _write_jsonl(path, [("a", [1]), ("c", [3])])

        reader = keyed_artifact.read_jsonl(str(path))

        assert reader.take("a") == [1]
        assert reader.take("b") is None
        assert reader.take("c") == [3]

    def test_a_key_below_the_readers_position_raises(self, tmp_path):
        path = tmp_path / "artifact.jsonl"
        _write_jsonl(path, [("a", [1]), ("b", [2])])

        reader = keyed_artifact.read_jsonl(str(path))
        reader.take("b")

        with pytest.raises(ValueError, match="clonotype-key order"):
            reader.take("a")


class TestKeyedTsvRoundTrip:
    def test_consecutive_rows_sharing_a_key_group_into_one_payload(self, tmp_path):
        path = tmp_path / "artifact.tsv"
        path.write_text(
            "clonotypeKey\tchain\timgt\n"
            "a\tH\t1\n"
            "a\tH\t2\n"
            "b\tH\t1\n"
        )

        reader = keyed_artifact.read_keyed_tsv(str(path), ["chain", "imgt"])

        assert reader.take("a") == [{"chain": "H", "imgt": "1"}, {"chain": "H", "imgt": "2"}]
        assert reader.take("b") == [{"chain": "H", "imgt": "1"}]


class TestKeysOf:
    def test_returns_every_key_in_file_order(self, tmp_path):
        path = tmp_path / "artifact.jsonl"
        _write_jsonl(path, [("a", []), ("b", []), ("c", [])])

        assert keyed_artifact.keys_of(str(path)) == ["a", "b", "c"]
