"""Unit tests for `batch.py` — the loop the five entrypoints share, and the
attribution rules that replace one-exec-per-antibody."""

import pytest

import batch
import pdb_index
import skip_store


def _entries(*keys):
    return [pdb_index.Entry(clonotype_key=k, filename=f"{k}.pdb") for k in keys]


class TestOneRowPerAttemptedClonotype:
    def test_a_passing_clonotype_gets_an_empty_reason_row(self, tmp_path):
        out_skip = tmp_path / "skip.tsv"

        batch.run(_entries("a", "b"), lambda _e: "", str(out_skip))

        assert skip_store.read_skips(str(out_skip)) == [("a", "", ""), ("b", "", "")]

    def test_a_reason_is_recorded_against_its_own_clonotype(self, tmp_path):
        out_skip = tmp_path / "skip.tsv"

        def one(entry):
            return "no-structure" if entry.clonotype_key == "b" else ""

        batch.run(_entries("a", "b", "c"), one, str(out_skip))

        assert skip_store.read_skips(str(out_skip)) == [
            ("a", "", ""),
            ("b", "no-structure", ""),
            ("c", "", ""),
        ]

    def test_returning_none_writes_no_row_so_a_prior_skip_is_not_double_counted(self, tmp_path):
        out_skip = tmp_path / "skip.tsv"

        def one(entry):
            return None if entry.clonotype_key == "a" else ""

        batch.run(_entries("a", "b"), one, str(out_skip))

        assert skip_store.read_skips(str(out_skip)) == [("b", "", "")]

    def test_an_empty_index_leaves_a_header_only_skip_tsv_and_exits_zero(self, tmp_path):
        out_skip = tmp_path / "skip.tsv"

        rc = batch.run([], lambda _e: "", str(out_skip))

        assert rc == 0
        assert skip_store.read_skips(str(out_skip)) == []


class TestExceptionsPropagateUnlessTheStepOptsIn:
    def test_without_an_error_reason_an_exception_fails_the_whole_exec(self, tmp_path):
        out_skip = tmp_path / "skip.tsv"

        def one(_entry):
            raise ValueError("a bug, not a bad antibody")

        with pytest.raises(ValueError, match="a bug"):
            batch.run(_entries("a"), one, str(out_skip))

    def test_with_an_error_reason_the_failing_antibody_is_named_and_the_rest_continue(
        self, tmp_path, capsys
    ):
        out_skip = tmp_path / "skip.tsv"
        seen = []

        def one(entry):
            seen.append(entry.clonotype_key)
            if entry.clonotype_key == "middle":
                raise RuntimeError("torch exploded")
            return ""

        rc = batch.run(
            _entries("first", "middle", "last"), one, str(out_skip), error_reason="backend-failed"
        )

        assert rc == 0
        assert seen == ["first", "middle", "last"]
        assert skip_store.read_skips(str(out_skip)) == [
            ("first", "", ""),
            ("middle", "backend-failed", "torch exploded"),
            ("last", "", ""),
        ]
        # The clonotype key must reach stderr — an exec log naming only the
        # exception cannot be traced back to one antibody.
        assert "middle" in capsys.readouterr().err
