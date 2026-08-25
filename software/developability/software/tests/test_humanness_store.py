"""Unit tests for `humanness_store.py` — the reduction from a parent's considered
framework positions to a coarse verdict, and the write-only path for `humanness.tsv`."""

import csv
import io

from engine import humanness_store, residue_store, variant_candidates


def _residue(chain, offset, imgt, wild_type):
    return residue_store.Residue(
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


class TestSummarizeHumanness:
    def test_targets_none_makes_no_claim(self):
        assert humanness_store.summarize_humanness(None, []) == ("", "")

    def test_targets_empty_reads_none_and_none(self):
        assert humanness_store.summarize_humanness([], []) == ("none", "None")

    def test_every_target_edited_by_some_cleared_candidate_reads_humanised(self):
        targets = [_residue("H", 0, "107", "N"), _residue("H", 1, "108", "G")]
        cleared = [_candidate([_edit("H", "107", "N")]), _candidate([_edit("H", "108", "G")])]

        verdict, summary = humanness_store.summarize_humanness(targets, cleared)

        assert verdict == "present"
        assert summary == "N@H107 (humanised), G@H108 (humanised)"

    def test_a_target_no_cleared_candidate_edits_reads_declined(self):
        targets = [_residue("H", 0, "107", "N"), _residue("H", 1, "108", "G")]
        cleared = [_candidate([_edit("H", "107", "N")])]

        _, summary = humanness_store.summarize_humanness(targets, cleared)

        assert summary == "N@H107 (humanised), G@H108 (declined)"

    def test_entries_follow_target_order_joined_by_comma_space(self):
        targets = [_residue("H", 1, "108", "G"), _residue("H", 0, "107", "N")]
        cleared = [_candidate([_edit("H", "107", "N")]), _candidate([_edit("H", "108", "G")])]

        _, summary = humanness_store.summarize_humanness(targets, cleared)

        assert summary == "G@H108 (humanised), N@H107 (humanised)"

    def test_entry_spelling_is_wild_type_at_chain_imgt_verdict(self):
        targets = [_residue("H", 0, "107", "N")]

        _, summary = humanness_store.summarize_humanness(targets, [])

        assert summary == "N@H107 (declined)"

    def test_a_target_edited_by_a_candidate_the_rank_cap_dropped_still_reads_humanised(self):
        # summarize_humanness sees only the gate-cleared candidates, never the
        # per-parent cap's later decision — the cap is not a decline.
        targets = [_residue("H", 0, "107", "N")]
        cleared = [_candidate([_edit("H", "107", "N")])]

        _, summary = humanness_store.summarize_humanness(targets, cleared)

        assert summary == "N@H107 (humanised)"


class TestHumannessTsv:
    def test_header_is_written_before_any_row(self, tmp_path):
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))

        assert _rows_of(path) == []
        assert path.read_text() == "clonotypeKey\thumannessVerdict\thumannessSummary\n"

    def test_a_summary_holding_a_comma_round_trips_as_one_field(self, tmp_path):
        targets = [_residue("H", 0, "107", "N"), _residue("H", 1, "108", "G")]
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))
        humanness_store.append_humanness_tsv(str(path), "clone-1", targets, [])

        [row] = _rows_of(path)
        assert row["humannessSummary"] == "N@H107 (declined), G@H108 (declined)"

    def test_targets_none_writes_both_cells_empty(self, tmp_path):
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))
        humanness_store.append_humanness_tsv(str(path), "clone-1", None, [])

        [row] = _rows_of(path)
        assert row["humannessVerdict"] == ""
        assert row["humannessSummary"] == ""

    def test_each_appended_row_carries_its_own_clonotype_key(self, tmp_path):
        path = tmp_path / "humanness.tsv"

        humanness_store.write_humanness_header(str(path))
        humanness_store.append_humanness_tsv(str(path), "clone-1", [], [])
        humanness_store.append_humanness_tsv(str(path), "clone-2", [], [])

        assert [r["clonotypeKey"] for r in _rows_of(path)] == ["clone-1", "clone-2"]
