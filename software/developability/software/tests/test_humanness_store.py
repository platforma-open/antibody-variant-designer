"""Unit tests for `humanness_store.py` — the write-only path for `humanness.tsv`."""

import csv
import io

from engine import (
    humanness_store,
    residue_index,
    variant_candidates,
)


def _residue(chain, offset, imgt, wild_type):
    return residue_index.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt,
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region="FR1",
    )


def _candidate(edits):
    return variant_candidates.Candidate(
        target_definition_id="framework_liability",
        edits=tuple(edits),
        tolerance=3.0,
        region="FR1",
        low_confidence=False,
        worst_confidence_angstroms=3.0,
        addressed_target="Framework liability @ FR1 H:107",
        changed_positions="H:N107D",
    )


def _edit(chain, imgt, wild_type, to="D"):
    return variant_candidates.Edit(
        chain=chain, offset=0, imgt=imgt, wild_type=wild_type, to=to
    )


def _rows_of(path):
    return list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))


class TestHumannessTsv:
    def test_header_is_written_before_any_row(self, tmp_path):
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))

        assert _rows_of(path) == []
        assert path.read_text() == (
            "clonotypeKey\thumannessVerdict\thumannessSummary\theavyHumannessScore\n"
        )

    def test_a_summary_holding_a_comma_round_trips_as_one_field(self, tmp_path):
        targets = [_residue("H", 0, "107", "N"), _residue("H", 1, "108", "G")]
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))
        humanness_store.append_humanness_tsv(str(path), "clone-1", targets, [], {})

        [row] = _rows_of(path)
        assert row["humannessSummary"] == "N@H107 (declined), G@H108 (declined)"

    def test_targets_none_writes_both_cells_empty(self, tmp_path):
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))
        humanness_store.append_humanness_tsv(str(path), "clone-1", None, [], {})

        [row] = _rows_of(path)
        assert row["humannessVerdict"] == ""
        assert row["humannessSummary"] == ""

    def test_each_appended_row_carries_its_own_clonotype_key(self, tmp_path):
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))
        humanness_store.append_humanness_tsv(str(path), "clone-1", [], [], {})
        humanness_store.append_humanness_tsv(str(path), "clone-2", [], [], {})

        assert [r["clonotypeKey"] for r in _rows_of(path)] == ["clone-1", "clone-2"]

    def test_the_heavy_chain_score_is_the_one_emitted(self, tmp_path):
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))
        humanness_store.append_humanness_tsv(
            str(path), "clone-1", [], [], {"H": 91.0, "L": 84.0}
        )

        [row] = _rows_of(path)
        assert row["heavyHumannessScore"] == "91.0"
        assert "lightHumannessScore" not in row

    def test_a_parent_with_no_heavy_score_leaves_the_column_empty_not_zero(self, tmp_path):
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))
        humanness_store.append_humanness_tsv(str(path), "clone-1", [], [], {"L": 84.0})

        [row] = _rows_of(path)
        assert row["heavyHumannessScore"] == ""

    def test_a_parent_the_objective_never_ran_for_leaves_the_score_column_empty(
        self, tmp_path
    ):
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))
        humanness_store.append_humanness_tsv(str(path), "clone-1", None, [], {})

        [row] = _rows_of(path)
        assert row["humannessVerdict"] == ""
        assert row["humannessSummary"] == ""
        assert row["heavyHumannessScore"] == ""
