"""Unit tests for `liability_objective.py` — the liability-removal objective's three parts."""

import dataclasses

import pytest

from engine import (
    design_objective,
    liability_cysteines,
    liability_motifs,
    liability_objective,
    residue_index,
)

TAXONOMY = [
    {"id": "deamidation_ng", "name": "Deamidation (N[GS])", "liabilityType": "deamidation",
     "motif": r"N[GS]", "riskLevel": "High", "fixability": "fixable"},
    {"id": "fragmentation_dp", "name": "Fragmentation (DP)", "liabilityType": "fragmentation",
     "motif": r"DP", "riskLevel": "High", "fixability": "fixable"},
    {"id": "missing_cysteines", "name": "Missing Cysteines", "liabilityType": "cysteine",
     "motif": None, "riskLevel": "High", "fixability": "structural"},
    {"id": "extra_cysteines", "name": "Extra Cysteines", "liabilityType": "cysteine",
     "motif": None, "riskLevel": "High", "fixability": "hard_to_fix"},
]


def _residue(chain, offset, wild_type, imgt=None, region="CDR1"):
    return residue_index.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region=region,
        chain_role="H",
    )


def _ng_site():
    # "N" then "G" — the exact span `deamidation_ng`'s `N[GS]` matches.
    return [_residue("H", 6, "N", imgt="107"), _residue("H", 7, "G", imgt="108")]


class TestObjectiveShape:
    def test_the_objective_carries_exactly_three_named_parts_in_order(self):
        assert [f.name for f in dataclasses.fields(design_objective.Objective)] == [
            "select_target_positions", "position_prior", "score_candidate",
        ]

    def test_the_liability_objective_has_no_position_prior(self):
        assert liability_objective.OBJECTIVE.position_prior is None


class TestSelectTargetPositions:
    def test_matches_both_detectors_run_directly_in_the_same_order(self):
        site = _ng_site()

        got = liability_objective.select_target_positions(site, TAXONOMY)
        want = liability_motifs.detect_all(site, TAXONOMY)
        want += liability_cysteines.detect_all(site, TAXONOMY)

        assert got == want


class TestScoreCandidate:
    def _tolerance_lookup(self, perplexities):
        return {
            ("H", "107"): {"perplexity": perplexities[0]},
            ("H", "108"): {"perplexity": perplexities[1]},
        }

    def test_a_site_cleared_of_every_hit_meets_the_goal(self):
        mutated_site = [
            _residue("H", 6, "D", imgt="107"), _residue("H", 7, "S", imgt="108"),
        ]

        check = liability_objective.score_candidate(
            mutated_site, TAXONOMY, self._tolerance_lookup([5.0, 2.0]),
        )

        assert check.meets_goal is True

    def test_a_site_that_spells_a_different_motif_does_not_meet_the_goal(self):
        # D then P spells `fragmentation_dp`'s own motif — clearing the NG
        # target this way must not read as cleared.
        mutated_site = [
            _residue("H", 6, "D", imgt="107"), _residue("H", 7, "P", imgt="108"),
        ]

        check = liability_objective.score_candidate(
            mutated_site, TAXONOMY, self._tolerance_lookup([5.0, 2.0]),
        )

        assert check.meets_goal is False

    def test_score_is_the_mean_perplexity_over_the_site_not_either_positions_own_value(self):
        mutated_site = [
            _residue("H", 6, "D", imgt="107"), _residue("H", 7, "S", imgt="108"),
        ]

        check = liability_objective.score_candidate(
            mutated_site, TAXONOMY, self._tolerance_lookup([5.0, 2.0]),
        )

        assert check.score == pytest.approx(3.5)

    def test_a_site_position_absent_from_the_tolerance_table_raises_key_error(self):
        mutated_site = [
            _residue("H", 6, "D", imgt="107"), _residue("H", 7, "S", imgt="108"),
        ]
        lookup = {("H", "107"): {"perplexity": 5.0}}  # "108" absent

        with pytest.raises(KeyError):
            liability_objective.score_candidate(mutated_site, TAXONOMY, lookup)
