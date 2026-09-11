"""Unit tests for `variant_candidates.py` — the objective-driven gate, as a module. Its CLI
lives in `build_variants.py`, tested in `test_build_variants.py`."""


import pytest

from engine import (
    design_objective,
    liability_objective,
    residue_index,
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

_LIABILITY = "liability"
_HUMANNESS = "humanness"

_OBJECTIVE = liability_objective.OBJECTIVE
_CAPS = design_objective.EditCaps(liability=10, framework=20)


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


def _wide_site():
    return [
        _residue("H", 10, "N", imgt="150"),
        _residue("H", 11, "N", imgt="151"),
        _residue("H", 12, "N", imgt="152"),
    ]


def _liability_target(site, definition_id="deamidation_ng", low_confidence=False,
                       confidence_angstroms=3.0):
    return design_objective.DesignTarget(
        site=tuple(site),
        definition_id=definition_id,
        region=site[0].region,
        is_low_confidence=low_confidence,
        confidence_angstroms=confidence_angstroms,
        objective=_LIABILITY,
    )


def _humanization_target(site):
    return design_objective.DesignTarget(
        site=tuple(site),
        definition_id=None,
        region=site[0].region,
        is_low_confidence=False,
        confidence_angstroms=None,
        objective=_HUMANNESS,
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


def _wide_tolerance_lookup():
    return {
        ("H", "150"): _row(["D", "Q", "A"], wild_type="N"),
        ("H", "151"): _row(["E", "Q", "A"], wild_type="N"),
        ("H", "152"): _row(["F", "Q", "A"], wild_type="N"),
    }


def _humanness_objective(identity):
    """A stub whose goal check reports `identity` as its score — an OASis
    identity, not a perplexity, so a test can tell the two numbers apart
    at the same call site a real humanization objective would use."""
    return design_objective.Objective(
        select_target_positions=lambda residues, taxonomy: [],
        position_prior=None,
        score_candidate=lambda mutated_site, taxonomy, tolerance_lookup: (
            design_objective.GoalCheck(meets_goal=True, score=identity)
        ),
    )


def _build(targets, tolerance_lookup, residues, objectives, caps=_CAPS, set_limit=10,
           candidate_residues_per_position=3):
    return variant_candidates.build_candidates(
        targets, tolerance_lookup, residues, TAXONOMY, objectives, caps,
        candidate_residues_per_position, structural_weight=1.0, objective_weight=1.0,
        set_limit=set_limit,
    )


def _build_and_declines(targets, tolerance_lookup, residues, objectives, caps=_CAPS, set_limit=10,
                         candidate_residues_per_position=3):
    return variant_candidates.build_candidates_and_declines(
        targets, tolerance_lookup, residues, TAXONOMY, objectives, caps,
        candidate_residues_per_position, structural_weight=1.0, objective_weight=1.0,
        set_limit=set_limit,
    )


class TestOneCandidateCarriesEveryAdmittedTarget:
    """A variant carries every design target the caps admit, best-covering set first."""

    def test_three_clearable_liability_targets_ship_as_one_candidate_covering_all_three(self):
        targets = [
            _liability_target(_ng_site(), definition_id="deamidation_ng"),
            _liability_target(
                [_residue("H", 20, "N", imgt="200"), _residue("H", 21, "G", imgt="201")],
                definition_id="deamidation_ng",
            ),
            _liability_target(
                [_residue("H", 30, "N", imgt="300"), _residue("H", 31, "G", imgt="301")],
                definition_id="deamidation_ng",
            ),
        ]
        lookup = {
            **_ng_tolerance_lookup(),
            ("H", "200"): _row(["D", "Q", "A"], wild_type="N"),
            ("H", "201"): _row(["P", "S", "A"], wild_type="G"),
            ("H", "300"): _row(["D", "Q", "A"], wild_type="N"),
            ("H", "301"): _row(["P", "S", "A"], wild_type="G"),
        }
        residues = _ng_site()

        [candidate] = _build(targets, lookup, residues, {_LIABILITY: _OBJECTIVE}, set_limit=1)

        assert candidate.coverage == 3
        assert len(candidate.edits) == 6

    def test_more_liability_targets_than_the_cap_admits_carries_exactly_the_cap(self):
        targets = [
            _liability_target(
                [_residue("H", i * 10, "N", imgt=str(i * 10)),
                 _residue("H", i * 10 + 1, "G", imgt=str(i * 10 + 1))],
                definition_id="deamidation_ng",
            )
            for i in range(6)  # 6 targets * 2 residues each = 12 positions
        ]
        lookup = {
            (r.chain, r.imgt): _row(["D" if r.wild_type == "N" else "P", "Q", "A"],
                                     wild_type=r.wild_type)
            for target in targets
            for r in target.site
        }
        caps = design_objective.EditCaps(liability=5, framework=20)

        [candidate] = _build(targets, lookup, [], {_LIABILITY: _OBJECTIVE}, caps=caps, set_limit=1)

        assert len(candidate.edits) == 5

    def test_a_framework_target_set_wider_than_its_cap_is_truncated_liability_edits_untouched(
        self,
    ):
        liability = _liability_target(_ng_site())
        framework = _humanization_target(_wide_site())
        lookup = {**_ng_tolerance_lookup(), **_wide_tolerance_lookup()}
        caps = design_objective.EditCaps(liability=10, framework=2)

        [candidate] = _build(
            [liability, framework], lookup, _wide_site(),
            {_LIABILITY: _OBJECTIVE, _HUMANNESS: _humanness_objective(80.0)}, caps=caps,
            set_limit=1,
        )

        framework_edits = [e for e in candidate.edits if e.objective == _HUMANNESS]
        liability_edits = [e for e in candidate.edits if e.objective == _LIABILITY]
        assert len(framework_edits) == 2
        assert len(liability_edits) == 2

    def test_combined_framework_and_cdr_liability_ship_as_one_candidate_tagged_both(self):
        liability = _liability_target(_ng_site())
        framework = _humanization_target(_wide_site())
        lookup = {**_ng_tolerance_lookup(), **_wide_tolerance_lookup()}

        [candidate] = _build(
            [liability, framework], lookup, _wide_site(),
            {_LIABILITY: _OBJECTIVE, _HUMANNESS: _humanness_objective(80.0)}, set_limit=1,
        )

        assert candidate.coverage == 2
        assert {e.objective for e in candidate.edits} == {_LIABILITY, _HUMANNESS}

    def test_a_goal_check_blocking_one_framework_position_drops_only_that_position(self):
        blocked_imgt = {"151"}

        def score_candidate(mutated_site, taxonomy, tolerance_lookup):
            del taxonomy, tolerance_lookup
            blocking = tuple(
                (r.chain, r.imgt) for r in mutated_site if r.imgt in blocked_imgt
            )
            if blocking:
                return design_objective.GoalCheck(
                    meets_goal=False, score=0.0, reason="adds a liability",
                    blocking_positions=blocking,
                )
            return design_objective.GoalCheck(meets_goal=True, score=80.0)

        objective = design_objective.Objective(
            select_target_positions=lambda residues, taxonomy: [],
            position_prior=None,
            score_candidate=score_candidate,
        )
        framework = _humanization_target(_wide_site())

        [candidate] = _build(
            [framework], _wide_tolerance_lookup(), _wide_site(), {_HUMANNESS: objective},
            set_limit=1,
        )

        assert tuple(e.imgt for e in candidate.edits) == ("150", "152")

    def test_the_humanness_gate_refusing_the_whole_framework_contribution_ships_nothing(self):
        objective = design_objective.Objective(
            select_target_positions=lambda residues, taxonomy: [],
            position_prior=None,
            score_candidate=lambda mutated_site, taxonomy, tolerance_lookup: (
                design_objective.GoalCheck(
                    meets_goal=False, score=0.0, reason="humanness did not rise"
                )
            ),
        )
        framework = _humanization_target(_wide_site())

        candidates, declines = _build_and_declines(
            [framework], _wide_tolerance_lookup(), _wide_site(), {_HUMANNESS: objective},
        )

        assert candidates == []
        [decline] = declines
        assert decline.objective == _HUMANNESS

    def test_two_candidates_off_the_same_target_set_differ_at_exactly_one_position(self):
        framework = _humanization_target(_wide_site())

        candidates = _build(
            [framework], _wide_tolerance_lookup(), _wide_site(),
            {_HUMANNESS: _humanness_objective(80.0)}, set_limit=2,
        )

        assert len(candidates) == 2
        first, second = candidates
        assert first.edits[0].to == "D"
        assert first.edits[1].to == "E"
        assert first.edits[2].to == "F"
        differences = [
            i for i in range(3) if first.edits[i].to != second.edits[i].to
        ]
        assert len(differences) == 1  # the least-lossy single swap off the seed

    def test_set_limit_of_one_yields_exactly_one_candidate(self):
        framework = _humanization_target(_wide_site())

        candidates = _build(
            [framework], _wide_tolerance_lookup(), _wide_site(),
            {_HUMANNESS: _humanness_objective(80.0)}, set_limit=1,
        )

        assert len(candidates) == 1


class TestAdmittedTargets:
    def test_a_target_no_tolerance_row_covers_is_dropped_whole(self):
        site = _ng_site()
        lookup = {("H", "107"): _ng_tolerance_lookup()[("H", "107")]}  # "108" absent

        admitted = variant_candidates.admitted_targets(
            [_liability_target(site)], lookup, _CAPS
        )

        assert admitted == []


class TestBuildCandidatesRescanGate:
    def test_a_combination_that_clears_the_target_and_creates_nothing_survives(self):
        candidates_out = _build(
            [_liability_target(_ng_site())], _ng_tolerance_lookup(), _ng_site(),
            {_LIABILITY: _OBJECTIVE},
        )

        survivors = {tuple(e.to for e in c.edits) for c in candidates_out}
        assert ("D", "S") in survivors

    def test_a_combination_that_clears_the_target_but_creates_a_new_liability_is_discarded(self):
        # D then P spells `fragmentation_dp`'s own motif — clearing the
        # NG target this way must not survive the re-scan.
        candidates_out = _build(
            [_liability_target(_ng_site())], _ng_tolerance_lookup(), _ng_site(),
            {_LIABILITY: _OBJECTIVE}, set_limit=6,
        )

        survivors = {tuple(e.to for e in c.edits) for c in candidates_out}
        assert ("D", "P") not in survivors

    def test_every_surviving_candidate_targets_the_triaged_liability(self):
        [candidate] = [
            c
            for c in _build(
                [_liability_target(_ng_site())], _ng_tolerance_lookup(), _ng_site(),
                {_LIABILITY: _OBJECTIVE}, set_limit=6,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.target_definition_ids == ("deamidation_ng",)

    def test_tolerance_is_the_mean_perplexity_over_the_edited_positions(self):
        [candidate] = [
            c
            for c in _build(
                [_liability_target(_ng_site())], _ng_tolerance_lookup(), _ng_site(),
                {_LIABILITY: _OBJECTIVE}, set_limit=6,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        # Position 107's perplexity is 5.0, position 108's is 2.0 — the
        # candidate's tolerance is their mean, not either position's own
        # perplexity, and not their min or their sum.
        assert candidate.tolerance == pytest.approx(3.5)


class TestHumanizationCandidateKeepsToleranceAndHumannessDistinct:
    def test_tolerance_comes_from_the_table_humanness_comes_from_the_goal_check(self):
        lookup = {
            ("H", "107"): _row(["D", "Q", "A"], wild_type="N", perplexity=4.0),
            ("H", "108"): _row(["P", "S", "A"], wild_type="G", perplexity=6.0),
        }
        [candidate] = _build(
            [_liability_target(_ng_site())], lookup, _ng_site(),
            {_LIABILITY: _humanness_objective(identity=80.0)}, set_limit=1,
        )

        # Tolerance is the mean perplexity the table gives, never the goal check's OASis
        # identity. This module carries no humanness number at all: `variant_ranking`
        # measures it on the variants it keeps.
        assert candidate.tolerance == pytest.approx(5.0)

    def test_low_confidence_and_worst_confidence_carry_forward_from_the_target(self):
        [candidate] = _build(
            [_liability_target(_ng_site(), low_confidence=True, confidence_angstroms=7.5)],
            _ng_tolerance_lookup(), _ng_site(), {_LIABILITY: _OBJECTIVE}, set_limit=1,
        )

        assert candidate.low_confidence is True
        assert candidate.worst_confidence_angstroms == 7.5

    def test_changed_positions_is_the_fixed_csv_contract_spelling(self):
        [candidate] = [
            c
            for c in _build(
                [_liability_target(_ng_site())], _ng_tolerance_lookup(), _ng_site(),
                {_LIABILITY: _OBJECTIVE}, set_limit=6,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.changed_positions == "H:N107D, H:G108S"

    def test_addressed_target_names_the_taxonomy_entry_and_where_it_sits(self):
        [candidate] = [
            c
            for c in _build(
                [_liability_target(_ng_site())], _ng_tolerance_lookup(), _ng_site(),
                {_LIABILITY: _OBJECTIVE}, set_limit=6,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.addressed_target == "Deamidation (N[GS]) @ CDR1 H:107"


class TestBuildCandidatesHumanizationTarget:
    """A target with no `definition_id`: one full-set candidate seeded first."""

    def test_the_best_candidate_carries_one_edit_per_selected_position(self):
        site = _wide_site()
        [candidate] = _build(
            [_humanization_target(site)], _wide_tolerance_lookup(), site,
            {_HUMANNESS: _humanness_objective(identity=80.0)}, set_limit=1,
        )

        assert len(candidate.edits) == 3

    def test_the_best_candidate_substitutes_each_position_to_its_own_top_scoring_residue(self):
        site = _wide_site()
        [candidate] = _build(
            [_humanization_target(site)], _wide_tolerance_lookup(), site,
            {_HUMANNESS: _humanness_objective(identity=80.0)}, set_limit=1,
        )

        assert tuple(e.to for e in candidate.edits) == ("D", "E", "F")

    def test_the_addressed_target_names_the_fixed_label_and_the_chain(self):
        site = _wide_site()
        [candidate] = _build(
            [_humanization_target(site)], _wide_tolerance_lookup(), site,
            {_HUMANNESS: _humanness_objective(identity=80.0)}, set_limit=1,
        )

        assert candidate.target_definition_ids == ()
        assert candidate.addressed_target == "Humanization @ H"

    def test_a_position_with_no_admissible_substitution_skips_the_whole_target(self):
        site = _wide_site()
        lookup = {("H", "150"): _wide_tolerance_lookup()[("H", "150")]}  # "151", "152" absent

        candidates_out = _build(
            [_humanization_target(site)], lookup, site,
            {_HUMANNESS: _humanness_objective(identity=80.0)},
        )

        assert candidates_out == []

    def test_the_positions_that_forced_the_rejection_are_the_ones_no_tolerance_row_covers(self):
        site = _wide_site()
        lookup = {("H", "150"): _wide_tolerance_lookup()[("H", "150")]}

        unscored = variant_candidates.unscored_positions(tuple(site), lookup)

        # The same two positions the drop above turned on, named rather than
        # left for a caller to re-derive.
        assert tuple(r.imgt for r in unscored) == ("151", "152")

    def test_a_fully_covered_site_names_no_unscored_position(self):
        site = _wide_site()

        assert variant_candidates.unscored_positions(
            tuple(site), _wide_tolerance_lookup()
        ) == ()


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
            residue, lookup, k=2, prior=None, structural_weight=1.0, objective_weight=1.0
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
            residue, lookup, k=2, prior=prior, structural_weight=1.0, objective_weight=1.0
        ) == ["E", "D"]

    def test_a_prior_with_no_row_for_this_position_falls_back_to_prior_none(self):
        residue = _residue("H", 6, "N", imgt="107")
        lookup = {("H", "107"): _row(["D", "Q", "A"], wild_type="N", perplexity=3.0)}
        prior = {("H", "999"): {"E": 10.0}}  # a different position — uncovered here

        assert variant_candidates._top_substitutions(
            residue, lookup, k=2, prior=prior, structural_weight=1.0, objective_weight=1.0
        ) == (
            variant_candidates._top_substitutions(
                residue, lookup, k=2, prior=None, structural_weight=1.0, objective_weight=1.0
            )
        )


class TestCombinedScoresWeighting:
    def test_default_weights_equal_the_plain_elementwise_sum(self):
        residue = _residue("H", 6, "N", imgt="107")
        log_probs = _row(["D", "Q", "A"], wild_type="N")["logProbs"]
        prior = {("H", "107"): {"D": 0.4, "Q": 0.2}}

        weighted = variant_candidates._combined_scores(
            residue, log_probs, prior, structural_weight=1.0, objective_weight=1.0
        )

        prior_row = prior[("H", "107")]
        unweighted = {aa: value + prior_row.get(aa, 0.0) for aa, value in log_probs.items()}
        assert weighted == unweighted

    def test_a_high_objective_weight_lets_the_prior_favourite_win_where_default_does_not(self):
        # "E" scores far below every log-probability favourite on its own;
        # a modest prior term is not enough to overcome that gap at
        # `objective_weight = 1.0`, but tripling it is.
        residue = _residue("H", 6, "N", imgt="107")
        log_probs = _row(["D", "Q", "A"], wild_type="N")["logProbs"]
        prior = {("H", "107"): {"E": 2.0}}

        at_default = variant_candidates._combined_scores(
            residue, log_probs, prior, structural_weight=1.0, objective_weight=1.0
        )
        at_triple = variant_candidates._combined_scores(
            residue, log_probs, prior, structural_weight=1.0, objective_weight=3.0
        )

        assert max(at_default, key=lambda aa: (at_default[aa], aa)) != "E"
        assert max(at_triple, key=lambda aa: (at_triple[aa], aa)) == "E"

    def test_a_zero_objective_weight_switches_the_prior_off_cleanly(self):
        residue = _residue("H", 6, "N", imgt="107")
        log_probs = _row(["D", "Q", "A"], wild_type="N")["logProbs"]
        prior = {("H", "107"): {"D": 5.0, "Q": -5.0}}

        with_prior_zeroed = variant_candidates._combined_scores(
            residue, log_probs, prior, structural_weight=1.0, objective_weight=0.0
        )
        with_no_prior_at_all = variant_candidates._combined_scores(
            residue, log_probs, None, structural_weight=1.0, objective_weight=0.0
        )
        assert with_prior_zeroed == with_no_prior_at_all

    def test_an_uncovered_position_still_takes_the_structural_weight_scaling(self):
        residue = _residue("H", 6, "N", imgt="107")
        log_probs = _row(["D", "Q", "A"], wild_type="N")["logProbs"]
        prior = {("H", "999"): {"E": 10.0}}  # a different position — uncovered here

        scores = variant_candidates._combined_scores(
            residue, log_probs, prior, structural_weight=2.5, objective_weight=1.0
        )

        assert scores == {aa: 2.5 * value for aa, value in log_probs.items()}

    def test_zero_structural_weight_with_no_prior_collapses_to_the_shared_alphabetical_order(self):
        residue = _residue("H", 6, "N", imgt="107")
        lookup = {("H", "107"): _row(["D", "Q", "A"], wild_type="N")}

        substitutions = variant_candidates._top_substitutions(
            residue, lookup, k=5, prior=None, structural_weight=0.0, objective_weight=1.0
        )

        assert substitutions == [aa for aa in tolerance_store.AMINO_ACIDS if aa != "N"][:5]


class TestBuildCandidatesEditBudget:
    def test_a_target_no_tolerance_row_fully_covers_produces_no_candidate(self):
        lookup = {("H", "107"): _ng_tolerance_lookup()[("H", "107")]}  # "108" absent

        candidates_out = _build(
            [_liability_target(_ng_site())], lookup, _ng_site(), {_LIABILITY: _OBJECTIVE},
        )

        assert candidates_out == []


def _objective_blocking(blocked_imgt):
    """A stub goal check that rejects while any position in `blocked_imgt` is still edited,
    naming those positions as the blockers — the shape `humanness_objective` reports when a
    substitution spells a liability the parent did not carry."""

    def score_candidate(mutated_site, taxonomy, tolerance_lookup):
        del taxonomy, tolerance_lookup
        blocking = tuple(
            (residue.chain, residue.imgt)
            for residue in mutated_site
            if residue.imgt in blocked_imgt
        )
        if blocking:
            return design_objective.GoalCheck(
                meets_goal=False, score=0.0, reason="adds a liability",
                detail="stub", blocking_positions=blocking,
            )
        return design_objective.GoalCheck(meets_goal=True, score=80.0)

    return design_objective.Objective(
        select_target_positions=lambda residues, taxonomy: [],
        position_prior=None,
        score_candidate=score_candidate,
    )


class TestABlockedPositionIsDroppedRatherThanSinkingTheSet:
    """One position whose substitution the goal check rejects must not take the other edits
    down with it. The humanization objective edits its full non-human framework set, and an
    all-or-nothing drop over a single spoiled position ships nothing at all."""

    def _build(self, objective, set_limit=1):
        return _build_and_declines(
            [_humanization_target(_wide_site())], _wide_tolerance_lookup(), _wide_site(),
            {_HUMANNESS: objective}, set_limit=set_limit,
        )

    def test_the_other_positions_still_ship(self):
        candidates, declines = self._build(_objective_blocking({"151"}))

        [candidate] = candidates
        assert tuple(e.imgt for e in candidate.edits) == ("150", "152")
        assert declines == []

    def test_every_position_blocked_ships_nothing_and_names_the_reason(self):
        candidates, declines = self._build(_objective_blocking({"150", "151", "152"}))

        assert candidates == []
        [decline] = declines
        assert (decline.chain, decline.reason, decline.objective) == (
            "H", "adds a liability", _HUMANNESS,
        )

    def test_a_check_that_blocks_nothing_still_drops_the_whole_target(self):
        # No blocking position means dropping a part changes nothing, so the retry stops
        # rather than shrinking the site one residue at a time to no purpose. Exercised
        # directly against the retry helper: the walk itself keeps trying nearby residue
        # choices once the seed fails, which is its whole point.
        target = _humanization_target(_wide_site())
        positions = [(target, residue) for residue in target.site]
        combo = ("D", "E", "F")  # each position's top-scoring residue

        def score_candidate(mutated_site, taxonomy, tolerance_lookup):
            del taxonomy, tolerance_lookup
            return design_objective.GoalCheck(
                meets_goal=False, score=0.0, reason="humanness did not rise"
            )

        objective = design_objective.Objective(
            select_target_positions=lambda residues, taxonomy: [],
            position_prior=None,
            score_candidate=score_candidate,
        )

        candidate, refused_by, checks = variant_candidates._scored_dropping_blockers(
            positions, combo, TAXONOMY, _wide_tolerance_lookup(), {_HUMANNESS: objective}, {},
        )

        assert candidate is None
        assert refused_by == _HUMANNESS
        assert checks[_HUMANNESS].reason == "humanness did not rise"
