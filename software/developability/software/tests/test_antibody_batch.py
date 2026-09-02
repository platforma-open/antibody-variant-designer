"""Unit tests for `antibody_batch.py` — the loop the three entrypoints
share, and the attribution rules that replace one-exec-per-antibody."""

import pytest

from engine import antibody_batch, parent_clonotypes, rejection_store


def _parents(*keys):
    return [parent_clonotypes.ParentClonotype(clonotype_key=k) for k in keys]


class TestOneRowPerAttemptedClonotype:
    def test_a_passing_clonotype_gets_an_empty_reason_row(self, tmp_path):
        out_rejected = tmp_path / "rejected.tsv"

        antibody_batch.process_every_parent(
            _parents("a", "b"), lambda _e: ("", "", "parent"), str(out_rejected)
        )

        assert rejection_store.read_rejections(str(out_rejected)) == [
            ("a", "", "", "parent"),
            ("b", "", "", "parent"),
        ]

    def test_a_reason_is_recorded_against_its_own_clonotype(self, tmp_path):
        out_rejected = tmp_path / "rejected.tsv"

        def one(entry):
            if entry.clonotype_key == "b":
                return ("no-structure", "", "parent")
            return ("", "", "parent")

        antibody_batch.process_every_parent(_parents("a", "b", "c"), one, str(out_rejected))

        assert rejection_store.read_rejections(str(out_rejected)) == [
            ("a", "", "", "parent"),
            ("b", "no-structure", "", "parent"),
            ("c", "", "", "parent"),
        ]

    def test_returning_none_writes_no_row_so_a_prior_rejection_is_not_double_counted(
        self, tmp_path
    ):
        out_rejected = tmp_path / "rejected.tsv"

        def one(entry):
            return None if entry.clonotype_key == "a" else ("", "", "parent")

        antibody_batch.process_every_parent(_parents("a", "b"), one, str(out_rejected))

        assert rejection_store.read_rejections(str(out_rejected)) == [("b", "", "", "parent")]

    def test_an_empty_index_leaves_a_header_only_rejection_tsv_and_exits_zero(self, tmp_path):
        out_rejected = tmp_path / "rejected.tsv"

        rc = antibody_batch.process_every_parent(
            [], lambda _e: ("", "", "parent"), str(out_rejected)
        )

        assert rc == 0
        assert rejection_store.read_rejections(str(out_rejected)) == []


class TestExceptionsPropagateUnlessTheStepOptsIn:
    def test_without_an_error_reason_an_exception_fails_the_whole_exec(self, tmp_path):
        out_rejected = tmp_path / "rejected.tsv"

        def one(_entry):
            raise ValueError("a bug, not a bad antibody")

        with pytest.raises(ValueError, match="a bug"):
            antibody_batch.process_every_parent(_parents("a"), one, str(out_rejected))

    def test_with_an_error_reason_the_failing_antibody_is_named_and_the_rest_continue(
        self, tmp_path, capsys
    ):
        out_rejected = tmp_path / "rejected.tsv"
        seen = []

        def one(entry):
            seen.append(entry.clonotype_key)
            if entry.clonotype_key == "middle":
                raise RuntimeError("torch exploded")
            return ("", "", "parent")

        rc = antibody_batch.process_every_parent(
            _parents("first", "middle", "last"),
            one,
            str(out_rejected),
            error_reason="backend-failed",
        )

        assert rc == 0
        assert seen == ["first", "middle", "last"]
        assert rejection_store.read_rejections(str(out_rejected)) == [
            ("first", "", "", "parent"),
            ("middle", "backend-failed", "torch exploded", "parent"),
            ("last", "", "", "parent"),
        ]
        # The clonotype key must reach stderr — an exec log naming only the
        # exception cannot be traced back to one antibody.
        assert "middle" in capsys.readouterr().err
