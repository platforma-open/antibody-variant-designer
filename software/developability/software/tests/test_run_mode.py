"""Unit tests for `run_mode.py` — resolving a run mode into its ordered
objectives, and each objective's targets."""

import pytest

from engine import liability_objective, run_mode


class TestObjectivesFor:
    def test_liabilities_selects_one_objective_and_no_more(self):
        pairs = run_mode.objectives_for(run_mode.LIABILITIES, "unused-prior-path")

        assert [name for name, _ in pairs] == [run_mode.LIABILITY]

    def test_liabilities_and_humanization_selects_both_liability_first(self):
        pairs = run_mode.objectives_for(run_mode.LIABILITIES_AND_HUMANIZATION, "unused-prior-path")

        assert [name for name, _ in pairs] == [run_mode.LIABILITY, run_mode.HUMANNESS]

    def test_an_unrecognised_mode_raises_instead_of_falling_back(self):
        with pytest.raises(ValueError, match="misspelled-mode"):
            run_mode.objectives_for("misspelled-mode", "unused-prior-path")

    def test_the_default_mode_is_liabilities(self):
        assert run_mode.DEFAULT_MODE == run_mode.LIABILITIES

    def test_the_liability_builder_returns_the_liability_objective(self):
        [(_, build_liability)] = run_mode.objectives_for(run_mode.LIABILITIES, "unused-prior-path")

        assert build_liability([]) is liability_objective.OBJECTIVE


class TestTargetsFor:
    def test_the_liability_objectives_targets_are_every_triaged_liability_unchanged(self):
        triaged_list = ["stand-in-for-a-triaged-liability"]

        assert run_mode.targets_for(run_mode.LIABILITY, triaged_list) is triaged_list

    def test_the_humanization_objectives_targets_are_empty(self):
        triaged_list = ["stand-in-for-a-triaged-liability"]

        assert run_mode.targets_for(run_mode.HUMANNESS, triaged_list) == []
