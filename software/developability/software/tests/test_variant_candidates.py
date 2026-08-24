"""Unit tests for `variant_candidates.py` — the objective-driven gate, as a module. Its CLI
lives in `build_variants.py`, tested in `test_build_variants.py`."""


import pytest

from engine import (
    liability_objective,
    liability_triage,
    residue_store,
    tolerance_store,
    variant_candidates,
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

_OBJECTIVE = liability_objective.OBJECTIVE


def _residue(chain, offset, wild_type, imgt=None):
    return residue_store.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region="CDR1",
        chain_role="H",
    )


def _ng_site():
    # "N" then "G" — the exact span `deamidation_ng`'s `N[GS]` matches.
    return [_residue("H", 6, "N", imgt="107"), _residue("H", 7, "G", imgt="108")]


def _triaged(site, definition_id="deamidation_ng", low_confidence=False, confidence_angstroms=3.0):
    return liability_triage.Triaged(
        definition_id=definition_id,
        liability_type="deamidation",
        risk_level="High",
        fixability="fixable",
        site=site,
        verdict="exposed",
        low_confidence=low_confidence,
        confidence_angstroms=confidence_angstroms,
        rsasa=0.5,
    )


def _row(ranked_log_probs, wild_type, perplexity=3.0):
    """`ranked_log_probs` names the amino acids that must outrank every
    other letter, best first; every other letter of the alphabet gets a
    log-probability far below all of them, so the ranking is unambiguous."""
    log_probs = dict.fromkeys(tolerance_store.AMINO_ACIDS, -5.0)
    for rank, aa in enumerate(ranked_log_probs):
        log_probs[aa] = -0.1 * (rank + 1)
    log_probs[wild_type] = -5.0
    return {"perplexity": perplexity, "logProbs": log_probs}


def _ng_tolerance_lookup():
    # Different perplexities per position, so a test can tell "the worst
    # position's perplexity" apart from "the first position's".
    return {
        ("H", "107"): _row(["D", "Q", "A"], wild_type="N", perplexity=5.0),
        ("H", "108"): _row(["P", "S", "A"], wild_type="G", perplexity=2.0),
    }


class TestBuildCandidatesRescanGate:
    def test_a_combination_that_clears_the_target_and_creates_nothing_survives(self):
        candidates_out = variant_candidates.build_candidates(
            [_triaged(_ng_site())], _ng_tolerance_lookup(), _ng_site(), TAXONOMY,
            objective=_OBJECTIVE,
            max_edits_per_variant=5, candidate_residues_per_position=3,
            w_struct=1.0, w_obj=1.0,
        )

        survivors = {tuple(e.to for e in c.edits) for c in candidates_out}
        assert ("D", "S") in survivors

    def test_a_combination_that_clears_the_target_but_creates_a_new_liability_is_discarded(self):
        # D then P spells `fragmentation_dp`'s own motif — clearing the
        # NG target this way must not survive the re-scan.
        candidates_out = variant_candidates.build_candidates(
            [_triaged(_ng_site())], _ng_tolerance_lookup(), _ng_site(), TAXONOMY,
            objective=_OBJECTIVE,
            max_edits_per_variant=5, candidate_residues_per_position=3,
            w_struct=1.0, w_obj=1.0,
        )

        survivors = {tuple(e.to for e in c.edits) for c in candidates_out}
        assert ("D", "P") not in survivors

    def test_every_surviving_candidate_targets_the_triaged_liability(self):
        [candidate] = [
            c
            for c in variant_candidates.build_candidates(
                [_triaged(_ng_site())], _ng_tolerance_lookup(), _ng_site(), TAXONOMY,
                objective=_OBJECTIVE,
                max_edits_per_variant=5, candidate_residues_per_position=3,
                w_struct=1.0, w_obj=1.0,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.target_definition_id == "deamidation_ng"

    def test_tolerance_is_the_mean_perplexity_over_the_edited_positions(self):
        [candidate] = [
            c
            for c in variant_candidates.build_candidates(
                [_triaged(_ng_site())], _ng_tolerance_lookup(), _ng_site(), TAXONOMY,
                objective=_OBJECTIVE,
                max_edits_per_variant=5, candidate_residues_per_position=3,
                w_struct=1.0, w_obj=1.0,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        # Position 107's perplexity is 5.0, position 108's is 2.0 — the
        # candidate's tolerance is their mean, not either position's own
        # perplexity, and not their min or their sum.
        assert candidate.tolerance == pytest.approx(3.5)

    def test_region_low_confidence_and_worst_confidence_carry_forward_from_triage(self):
        [candidate] = [
            c
            for c in variant_candidates.build_candidates(
                [_triaged(_ng_site(), low_confidence=True, confidence_angstroms=7.5)],
                _ng_tolerance_lookup(), _ng_site(), TAXONOMY,
                objective=_OBJECTIVE,
                max_edits_per_variant=5, candidate_residues_per_position=3,
                w_struct=1.0, w_obj=1.0,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.region == "CDR1"
        assert candidate.low_confidence is True
        assert candidate.worst_confidence_angstroms == 7.5

    def test_changed_positions_is_the_fixed_csv_contract_spelling(self):
        [candidate] = [
            c
            for c in variant_candidates.build_candidates(
                [_triaged(_ng_site())], _ng_tolerance_lookup(), _ng_site(), TAXONOMY,
                objective=_OBJECTIVE,
                max_edits_per_variant=5, candidate_residues_per_position=3,
                w_struct=1.0, w_obj=1.0,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.changed_positions == "H:N107D, H:G108S"

    def test_addressed_target_names_the_taxonomy_entry_and_where_it_sits(self):
        [candidate] = [
            c
            for c in variant_candidates.build_candidates(
                [_triaged(_ng_site())], _ng_tolerance_lookup(), _ng_site(), TAXONOMY,
                objective=_OBJECTIVE,
                max_edits_per_variant=5, candidate_residues_per_position=3,
                w_struct=1.0, w_obj=1.0,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.addressed_target == "Deamidation (N[GS]) @ CDR1 H:107"


class TestTopSubstitutionsRanksByLogProbability:
    def test_offered_substitutions_are_the_top_k_of_the_log_probability_row_wild_type_excluded(
        self,
    ):
        # This position's `perplexity` (3.0, one number for the whole
        # position) disagrees with which amino acids the per-amino-acid
        # log-probability row favors — pinning that `_top_substitutions`
        # reads the row, not the scalar, so a future reader cannot
        # re-open which signal ranks a position's candidate residues
        # against a test suite that never asserted it.
        residue = _residue("H", 6, "N", imgt="107")
        lookup = {("H", "107"): _row(["D", "Q", "A"], wild_type="N", perplexity=3.0)}

        assert variant_candidates._top_substitutions(
            residue, lookup, k=2, prior=None, w_struct=1.0, w_obj=1.0
        ) == ["D", "Q"]

    def test_a_prior_favourite_that_disagrees_with_the_log_row_wins_first(self):
        # The prior adds an elementwise term over the amino-acid order
        # ACDEFGHIKLMNPQRSTVWY; a large enough prior weight on "E" — which
        # the log-probability row does not favour at all — must still push
        # it to the front once combined.
        residue = _residue("H", 6, "N", imgt="107")
        lookup = {("H", "107"): _row(["D", "Q", "A"], wild_type="N", perplexity=3.0)}
        prior = {("H", "107"): {"E": 10.0}}

        assert variant_candidates._top_substitutions(
            residue, lookup, k=2, prior=prior, w_struct=1.0, w_obj=1.0
        ) == ["E", "D"]

    def test_a_prior_with_no_row_for_this_position_falls_back_to_prior_none(self):
        residue = _residue("H", 6, "N", imgt="107")
        lookup = {("H", "107"): _row(["D", "Q", "A"], wild_type="N", perplexity=3.0)}
        prior = {("H", "999"): {"E": 10.0}}  # a different position — uncovered here

        assert variant_candidates._top_substitutions(
            residue, lookup, k=2, prior=prior, w_struct=1.0, w_obj=1.0
        ) == (
            variant_candidates._top_substitutions(
                residue, lookup, k=2, prior=None, w_struct=1.0, w_obj=1.0
            )
        )


class TestCombinedScoresWeighting:
    def test_default_weights_equal_the_plain_elementwise_sum(self):
        residue = _residue("H", 6, "N", imgt="107")
        log_probs = _row(["D", "Q", "A"], wild_type="N")["logProbs"]
        prior = {("H", "107"): {"D": 0.4, "Q": 0.2}}

        weighted = variant_candidates._combined_scores(
            residue, log_probs, prior, w_struct=1.0, w_obj=1.0
        )

        prior_row = prior[("H", "107")]
        unweighted = {aa: value + prior_row.get(aa, 0.0) for aa, value in log_probs.items()}
        assert weighted == unweighted

    def test_a_high_objective_weight_lets_the_prior_favourite_win_where_default_does_not(self):
        # "E" scores far below every log-probability favourite on its own;
        # a modest prior term is not enough to overcome that gap at
        # `w_obj = 1.0`, but tripling it is.
        residue = _residue("H", 6, "N", imgt="107")
        log_probs = _row(["D", "Q", "A"], wild_type="N")["logProbs"]
        prior = {("H", "107"): {"E": 2.0}}

        at_default = variant_candidates._combined_scores(
            residue, log_probs, prior, w_struct=1.0, w_obj=1.0
        )
        at_triple = variant_candidates._combined_scores(
            residue, log_probs, prior, w_struct=1.0, w_obj=3.0
        )

        assert max(at_default, key=lambda aa: (at_default[aa], aa)) != "E"
        assert max(at_triple, key=lambda aa: (at_triple[aa], aa)) == "E"

    def test_a_zero_objective_weight_switches_the_prior_off_cleanly(self):
        residue = _residue("H", 6, "N", imgt="107")
        log_probs = _row(["D", "Q", "A"], wild_type="N")["logProbs"]
        prior = {("H", "107"): {"D": 5.0, "Q": -5.0}}

        with_prior_zeroed = variant_candidates._combined_scores(
            residue, log_probs, prior, w_struct=1.0, w_obj=0.0
        )
        with_no_prior_at_all = variant_candidates._combined_scores(
            residue, log_probs, None, w_struct=1.0, w_obj=0.0
        )
        assert with_prior_zeroed == with_no_prior_at_all

    def test_an_uncovered_position_still_takes_the_structural_weight_scaling(self):
        residue = _residue("H", 6, "N", imgt="107")
        log_probs = _row(["D", "Q", "A"], wild_type="N")["logProbs"]
        prior = {("H", "999"): {"E": 10.0}}  # a different position — uncovered here

        scores = variant_candidates._combined_scores(
            residue, log_probs, prior, w_struct=2.5, w_obj=1.0
        )

        assert scores == {aa: 2.5 * value for aa, value in log_probs.items()}

    def test_zero_structural_weight_with_no_prior_collapses_to_the_shared_alphabetical_order(self):
        residue = _residue("H", 6, "N", imgt="107")
        lookup = {("H", "107"): _row(["D", "Q", "A"], wild_type="N")}

        substitutions = variant_candidates._top_substitutions(
            residue, lookup, k=5, prior=None, w_struct=0.0, w_obj=1.0
        )

        assert substitutions == [aa for aa in tolerance_store.AMINO_ACIDS if aa != "N"][:5]


class TestBuildCandidatesEditBudget:
    def test_a_site_longer_than_the_edit_budget_produces_no_candidate(self):
        candidates_out = variant_candidates.build_candidates(
            [_triaged(_ng_site())], _ng_tolerance_lookup(), _ng_site(), TAXONOMY,
            objective=_OBJECTIVE,
            max_edits_per_variant=1, candidate_residues_per_position=3,
            w_struct=1.0, w_obj=1.0,
        )

        assert candidates_out == []

    def test_a_position_missing_from_the_tolerance_table_produces_no_candidate(self):
        lookup = {("H", "107"): _ng_tolerance_lookup()[("H", "107")]}  # "108" absent

        candidates_out = variant_candidates.build_candidates(
            [_triaged(_ng_site())], lookup, _ng_site(), TAXONOMY,
            objective=_OBJECTIVE,
            max_edits_per_variant=5, candidate_residues_per_position=3,
            w_struct=1.0, w_obj=1.0,
        )

        assert candidates_out == []


class TestBuildCandidatesRiskLevelOrder:
    def test_triaged_liabilities_in_low_high_medium_order_are_built_high_medium_low(self):
        # Three isolated, non-interacting single-residue liabilities, one per
        # risk level, submitted in an order that matches none of the three
        # possible sorted orders — only `_risk_level_order` explains the
        # output order.
        low = _triaged(
            [_residue("H", 20, "N", imgt="200")], definition_id="low_risk_lib",
        )
        low.risk_level = "Low"
        high = _triaged(
            [_residue("H", 21, "M", imgt="201")], definition_id="high_risk_lib",
        )
        high.risk_level = "High"
        medium = _triaged(
            [_residue("H", 22, "P", imgt="202")], definition_id="medium_risk_lib",
        )
        medium.risk_level = "Medium"
        lookup = {
            ("H", "200"): _row(["A"], wild_type="N"),
            ("H", "201"): _row(["A"], wild_type="M"),
            ("H", "202"): _row(["A"], wild_type="P"),
        }

        candidates_out = variant_candidates.build_candidates(
            [low, high, medium], lookup, [], TAXONOMY,
            objective=_OBJECTIVE,
            max_edits_per_variant=5, candidate_residues_per_position=1,
            w_struct=1.0, w_obj=1.0,
        )

        assert [c.target_definition_id for c in candidates_out] == [
            "high_risk_lib", "medium_risk_lib", "low_risk_lib",
        ]
