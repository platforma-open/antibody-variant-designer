"""Unit tests for `run_mode.py` — resolving a run mode into its ordered
objectives, and each objective's design targets."""

import pytest

from engine import (
    design_objective,
    liability_objective,
    liability_triage,
    residue_index,
    run_mode,
)

_UNUSED_PRIOR_PATH = "unused-prior-path"
_UNUSED_CUTOFF = 0.05


def _residue(chain, offset, region="CDR1", wild_type="A"):
    return residue_index.Residue(
        chain=chain,
        offset=offset,
        imgt=str(offset + 1),
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region=region,
        chain_role="H",
    )


def _triaged(definition_id="deamidation_ng", risk_level="High", low_confidence=False,
             confidence_angstroms=3.0, site=None):
    return liability_triage.Triaged(
        definition_id=definition_id,
        liability_type="deamidation",
        risk_level=risk_level,
        fixability="fixable",
        site=site if site is not None else [_residue("H", 6, region="FR1")],
        verdict="exposed",
        low_confidence=low_confidence,
        confidence_angstroms=confidence_angstroms,
        rsasa=0.5,
    )


def _stub_objective(selection):
    return design_objective.Objective(
        select_target_positions=lambda residues, taxonomy: selection,
        position_prior=None,
        score_candidate=lambda mutated_site, taxonomy, tolerance_lookup: (
            design_objective.GoalCheck(meets_goal=True, score=0.0)
        ),
    )


class TestObjectivesFor:
    def test_liabilities_selects_one_objective_and_no_more(self):
        pairs = run_mode.objectives_for(run_mode.LIABILITIES, _UNUSED_PRIOR_PATH, _UNUSED_CUTOFF)

        assert [name for name, _ in pairs] == [run_mode.LIABILITY]

    def test_humanization_selects_exactly_one_objective_named_humanness(self):
        pairs = run_mode.objectives_for(run_mode.HUMANIZATION, _UNUSED_PRIOR_PATH, _UNUSED_CUTOFF)

        assert [name for name, _ in pairs] == [run_mode.HUMANNESS]

    def test_liabilities_and_humanization_selects_both_liability_first(self):
        pairs = run_mode.objectives_for(
            run_mode.LIABILITIES_AND_HUMANIZATION, _UNUSED_PRIOR_PATH, _UNUSED_CUTOFF
        )

        assert [name for name, _ in pairs] == [run_mode.LIABILITY, run_mode.HUMANNESS]

    def test_an_unrecognised_mode_raises_instead_of_falling_back(self):
        with pytest.raises(ValueError, match="misspelled-mode"):
            run_mode.objectives_for("misspelled-mode", _UNUSED_PRIOR_PATH, _UNUSED_CUTOFF)

    def test_the_default_mode_is_liabilities(self):
        assert run_mode.DEFAULT_MODE == run_mode.LIABILITIES

    def test_the_three_mode_strings_are_exactly_modes(self):
        assert run_mode.MODES == (
            run_mode.LIABILITIES, run_mode.HUMANIZATION, run_mode.LIABILITIES_AND_HUMANIZATION,
        )

    def test_every_mode_string_is_accepted(self):
        for mode in run_mode.MODES:
            pairs = run_mode.objectives_for(mode, _UNUSED_PRIOR_PATH, _UNUSED_CUTOFF)
            assert len(pairs) > 0

    def test_the_liability_builder_returns_the_liability_objective(self):
        [(_, build_liability)] = run_mode.objectives_for(
            run_mode.LIABILITIES, _UNUSED_PRIOR_PATH, _UNUSED_CUTOFF
        )

        assert build_liability([]) is liability_objective.OBJECTIVE


class TestTargetsFor:
    def test_the_liability_objectives_targets_are_every_triaged_liability_converted(self):
        triaged_list = [_triaged()]

        targets = run_mode.targets_for(
            run_mode.LIABILITY, run_mode.LIABILITIES, triaged_list, liability_objective.OBJECTIVE,
            [],
        )

        assert [t.definition_id for t in targets] == ["deamidation_ng"]

    def test_a_liability_target_keeps_definition_id_region_low_confidence_and_confidence_value(
        self,
    ):
        triaged = _triaged(low_confidence=True, confidence_angstroms=7.5)

        [target] = run_mode.targets_for(
            run_mode.LIABILITY, run_mode.LIABILITIES, [triaged], liability_objective.OBJECTIVE, []
        )

        assert target.definition_id == "deamidation_ng"
        assert target.region == "FR1"
        assert target.is_low_confidence is True
        assert target.confidence_angstroms == 7.5
        assert target.site == tuple(triaged.site)

    def test_liability_targets_are_ordered_by_risk_level_highest_first(self):
        low = _triaged(definition_id="low_risk_lib", risk_level="Low")
        high = _triaged(definition_id="high_risk_lib", risk_level="High")
        medium = _triaged(definition_id="medium_risk_lib", risk_level="Medium")

        targets = run_mode.targets_for(
            run_mode.LIABILITY, run_mode.LIABILITIES, [low, high, medium],
            liability_objective.OBJECTIVE, [],
        )

        assert [t.definition_id for t in targets] == [
            "high_risk_lib", "medium_risk_lib", "low_risk_lib",
        ]

    def test_the_liability_objectives_targets_are_empty_in_humanization_mode(self):
        triaged_list = [_triaged()]

        targets = run_mode.targets_for(
            run_mode.LIABILITY, run_mode.HUMANIZATION, triaged_list, liability_objective.OBJECTIVE,
            [],
        )

        assert targets == []

    def test_the_humanness_objectives_targets_come_from_its_own_selection(self):
        selection = ["stand-in-for-a-design-target"]
        objective = _stub_objective(selection)

        targets = run_mode.targets_for(
            run_mode.HUMANNESS, run_mode.HUMANIZATION, [_triaged()], objective,
            ["stand-in-residues"],
        )

        assert targets is selection

    def test_liabilities_mode_keeps_a_framework_site_as_a_target(self):
        framework = _triaged(definition_id="framework_lib", site=[_residue("H", 6, region="FR1")])

        targets = run_mode.targets_for(
            run_mode.LIABILITY, run_mode.LIABILITIES, [framework], liability_objective.OBJECTIVE,
            [],
        )

        assert [t.definition_id for t in targets] == ["framework_lib"]

    def test_mode_2_keeps_a_site_whose_every_residue_carries_a_cdr_region(self):
        cdr_site = _triaged(
            definition_id="cdr_lib",
            site=[_residue("H", 6, region="CDR1"), _residue("H", 7, region="CDR1")],
        )

        targets = run_mode.targets_for(
            run_mode.LIABILITY, run_mode.LIABILITIES_AND_HUMANIZATION, [cdr_site],
            liability_objective.OBJECTIVE, [],
        )

        assert [t.definition_id for t in targets] == ["cdr_lib"]

    def test_mode_2_drops_a_site_whose_every_residue_carries_a_framework_region(self):
        framework_site = _triaged(
            definition_id="framework_lib", site=[_residue("H", 6, region="FR1")]
        )

        targets = run_mode.targets_for(
            run_mode.LIABILITY, run_mode.LIABILITIES_AND_HUMANIZATION, [framework_site],
            liability_objective.OBJECTIVE, [],
        )

        assert targets == []

    def test_mode_2_drops_a_site_straddling_a_cdr_and_a_framework_position(self):
        straddling_site = _triaged(
            definition_id="straddling_lib",
            site=[_residue("H", 6, region="CDR1"), _residue("H", 7, region="FR2")],
        )

        targets = run_mode.targets_for(
            run_mode.LIABILITY, run_mode.LIABILITIES_AND_HUMANIZATION, [straddling_site],
            liability_objective.OBJECTIVE, [],
        )

        assert targets == []

    def test_mode_2_over_an_all_framework_parent_yields_no_targets_at_all(self):
        framework_a = _triaged(
            definition_id="framework_a", site=[_residue("H", 6, region="FR1")]
        )
        framework_b = _triaged(
            definition_id="framework_b", site=[_residue("H", 20, region="FR2")]
        )

        targets = run_mode.targets_for(
            run_mode.LIABILITY, run_mode.LIABILITIES_AND_HUMANIZATION,
            [framework_a, framework_b], liability_objective.OBJECTIVE, [],
        )

        assert targets == []

    @pytest.mark.parametrize("mode", run_mode.MODES)
    def test_the_humanness_objectives_targets_are_unchanged_by_the_mode(self, mode):
        selection = ["stand-in-for-a-design-target"]
        objective = _stub_objective(selection)

        targets = run_mode.targets_for(
            run_mode.HUMANNESS, mode, [_triaged()], objective, ["stand-in-residues"],
        )

        assert targets is selection
