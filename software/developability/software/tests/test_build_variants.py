"""Unit tests for `build_variants.py` — the fused gate-then-rank entrypoint, and
the dataset-wide `variants.tsv` it writes."""

import csv
import math
from pathlib import Path

import pytest

import build_variants
from engine import (
    design_objective,
    humanness_gate,
    liability_store,
    liability_triage,
    rejection_store,
    residue_index,
    residue_store,
    run_mode,
    tolerance_store,
    variant_candidates,
    variant_ranking,
    variant_store,
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

AMINO_ACIDS = tolerance_store.AMINO_ACIDS


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


def _triaged(site):
    return liability_triage.Triaged(
        definition_id="deamidation_ng",
        liability_type="deamidation",
        risk_level="High",
        fixability="fixable",
        site=site,
        verdict="exposed",
        low_confidence=False,
        confidence_angstroms=3.0,
        rsasa=0.5,
    )


def _target_from_triaged(triaged, objective=run_mode.LIABILITY):
    """The same conversion `run_mode._as_design_target` does — used here to stand a stub
    `targets_for` in for the real one without reaching into that private helper."""
    return design_objective.DesignTarget(
        site=tuple(triaged.site),
        definition_id=triaged.definition_id,
        region=triaged.site[0].region,
        is_low_confidence=triaged.low_confidence,
        confidence_angstroms=triaged.confidence_angstroms,
        objective=objective,
    )


def _row(ranked, wild_type, perplexity):
    log_probs = dict.fromkeys(AMINO_ACIDS, -5.0)
    for rank, aa in enumerate(ranked):
        log_probs[aa] = -0.1 * (rank + 1)
    log_probs[wild_type] = -5.0
    return {"chain": "H", "perplexity": perplexity, **log_probs}


def _tolerance_rows():
    return [
        {"imgt": "107", **_row(["D", "Q", "A"], "N", 5.0)},
        {"imgt": "108", **_row(["P", "S", "A"], "G", 2.0)},
    ]


def _stage(batch, clonotype_key, with_prior=False):
    """Stage one antibody's triaged liabilities, tolerance table and residue
    index — the three predecessors this step joins. `with_prior` also
    stages the (real, if uninformative) prior TSV mode 2 requires."""
    entry = batch.add(clonotype_key)
    liability_store.write_triaged(
        str(Path(batch.dir("triaged"), f"{entry.stem}.json")), [_triaged(_ng_site())]
    )
    tolerance_store.write_tolerance_tsv(
        str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv")), _tolerance_rows()
    )
    residue_store.write_residues(
        str(Path(batch.dir("residues"), f"{entry.stem}.json")), _ng_site()
    )
    if with_prior:
        Path(batch.dir("tolerance"), f"{entry.stem}{build_variants.PRIOR_SUFFIX}").write_text(
            "chain\timgt\n"
        )
    return entry


def _framework_residue():
    # Wild type "N", chosen so a substitution to the tolerance table's
    # top-ranked "D" is content the parent chain otherwise never carries —
    # `_stub_rising_on_d` below reads a "D" count off the assembled sequence.
    return residue_index.Residue(
        chain="H", offset=0, imgt="1", wild_type="N", res_name="N",
        b_factor=20.0, region="FR1", chain_role="H",
    )


def _write_prior_tsv(path, wild_type_probability):
    """One position's prior row, in the log-probability scale
    `sapiens_prior._log_softmax` writes: every amino acid scores `log(0.9)` except the
    wild type `N`, which scores `log(wild_type_probability)` — the one cell every
    selection case in this class turns the margin against."""
    scores = dict.fromkeys(AMINO_ACIDS, math.log(0.9))
    scores["N"] = math.log(wild_type_probability)
    with Path(path).open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["chain", "imgt", *AMINO_ACIDS])
        writer.writerow(["H", "1", *(scores[aa] for aa in AMINO_ACIDS)])


def _stub_rising_on_d(monkeypatch):
    """Stands in for the real OASis measurement: a sequence's score is how
    many "D"s it holds. The fixture's parent chain carries none, so a
    candidate substituting the framework "N" to the tolerance table's
    top-ranked "D" always raises the score — deterministic, no `promb`
    install or reference database needed."""
    monkeypatch.setattr(humanness_gate, "identity", lambda seq: float(seq.count("D")))


def _light_framework_residue():
    # The paired fixture's one light-chain residue — enough for
    # `residue_index.in_scope_chains` to yield an `L` chain beside the `H`
    # one `_framework_residue`/`_ng_site` already carry.
    return residue_index.Residue(
        chain="L", offset=0, imgt="1", wild_type="Q", res_name="Q",
        b_factor=20.0, region="FR1", chain_role="L",
    )


def _stage_paired(batch, clonotype_key, non_human_prior_score):
    """`_stage_mixed`'s parent with one light-chain residue added — a paired
    antibody, for the two humanness-score columns to tell apart."""
    entry = batch.add(clonotype_key)
    residues = [_framework_residue(), *_ng_site(), _light_framework_residue()]
    liability_store.write_triaged(
        str(Path(batch.dir("triaged"), f"{entry.stem}.json")), [_triaged(_ng_site())]
    )
    tolerance_store.write_tolerance_tsv(
        str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv")),
        [*_tolerance_rows(), {"imgt": "1", **_row(["D", "Q", "A"], "N", 3.0)}],
    )
    residue_store.write_residues(
        str(Path(batch.dir("residues"), f"{entry.stem}.json")), residues
    )
    _write_prior_tsv(
        Path(batch.dir("tolerance"), f"{entry.stem}{build_variants.PRIOR_SUFFIX}"),
        non_human_prior_score,
    )
    return entry


def _stage_mixed(batch, clonotype_key, non_human_prior_score, score_framework_position=True):
    """One parent carrying both a liability (the `_ng_site` deamidation
    motif, on CDR1) and a non-human framework position (`_framework_residue`,
    on FR1) — the fixture `TestHumanizationModeEndToEnd` runs against.

    `score_framework_position=False` withholds that position's tolerance row, leaving the
    humanization objective a target AntiFold never scored."""
    entry = batch.add(clonotype_key)
    residues = [_framework_residue(), *_ng_site()]
    liability_store.write_triaged(
        str(Path(batch.dir("triaged"), f"{entry.stem}.json")), [_triaged(_ng_site())]
    )
    framework_rows = (
        [{"imgt": "1", **_row(["D", "Q", "A"], "N", 3.0)}] if score_framework_position else []
    )
    tolerance_store.write_tolerance_tsv(
        str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv")),
        [*_tolerance_rows(), *framework_rows],
    )
    residue_store.write_residues(
        str(Path(batch.dir("residues"), f"{entry.stem}.json")), residues
    )
    _write_prior_tsv(
        Path(batch.dir("tolerance"), f"{entry.stem}{build_variants.PRIOR_SUFFIX}"),
        non_human_prior_score,
    )
    return entry


def _run(batch, extra_args=None):
    definitions = batch.definitions(TAXONOMY)
    out_variants = batch.path("variants.tsv")
    out_rejected = batch.path("rejected.tsv")

    rc = build_variants.main(
        [
            "--triaged-dir", batch.dir("triaged"),
            "--tolerance-dir", batch.dir("tolerance"),
            "--residues-dir", batch.dir("residues"),
            "--definitions", definitions,
            "--out-variants", out_variants,
            "--out-rejected", out_rejected,
            "--out-humanness", batch.path("humanness.tsv"),
        ]
        + (extra_args or [])
    )

    assert rc == 0
    return (
        rejection_store.read_rejections(out_rejected),
        variant_store.read_variants_tsv(out_variants),
    )


class TestGateAndRankInOnePass:
    def test_a_cleared_liability_becomes_ranked_variants(self, batch):
        _stage(batch, "clone-1")

        rejections, written = _run(batch)

        assert rejections == [("clone-1", "", "", "parent")]
        assert len(written) > 0
        assert [v.status for _, _, v in written] == ["unvalidated-hypothesis"] * len(written)

    def test_the_gate_still_discards_a_candidate_that_creates_a_new_liability(self, batch):
        # D then P spells `fragmentation_dp` — it clears the NG target but
        # must not survive, and fusing the steps must not lose that.
        _stage(batch, "clone-1")

        _, written = _run(batch)

        assert "H:N107D, H:G108P" not in [v.changed_positions for _, _, v in written]
        assert "H:N107D, H:G108S" in [v.changed_positions for _, _, v in written]

    def test_no_surviving_candidate_writes_the_named_rejection_and_no_rows(
        self, batch, monkeypatch
    ):
        # A liability cap of 0 admits nothing at all — the whole target is dropped and the
        # gate clears no candidate.
        monkeypatch.setattr(build_variants.variant_candidates, "DEFAULT_MAX_LIABILITY_EDITS", 0)
        _stage(batch, "clone-1")

        rejections, written = _run(batch)

        assert rejections == [("clone-1", "no-candidate-cleared-the-gate", "", "parent")]
        assert written == []

    def test_the_default_mode_still_names_the_same_rejection_reason(self, batch, monkeypatch):
        # `--run-mode` omitted and `--run-mode liabilities` given explicitly
        # must resolve to the identical rejection reason — proving the mode
        # selector's own wiring left the default path's rejection-reason string
        # untouched.
        # A liability cap of 0 admits nothing at all — the whole target is dropped and the
        # gate clears no candidate.
        monkeypatch.setattr(build_variants.variant_candidates, "DEFAULT_MAX_LIABILITY_EDITS", 0)
        _stage(batch, "clone-1")

        default_rejections, default_written = _run(batch)
        explicit_rejections, explicit_written = _run(batch, ["--run-mode", "liabilities"])

        assert (
            default_rejections
            == explicit_rejections
            == [("clone-1", "no-candidate-cleared-the-gate", "", "parent")]
        )
        assert default_written == explicit_written == []

    def test_a_gate_threshold_and_a_ranking_threshold_both_reach_this_one_command(self, batch):
        # The fused step owns both families of flag; neither is silently
        # ignored now that one command carries them.
        _stage(batch, "clone-1")

        _, written = _run(
            batch, ["--candidate-residues-per-position", "3", "--variants-per-parent", "1"]
        )

        assert len(written) == 1
        assert [v.rank for _, _, v in written] == [1]

    def test_the_weight_flags_default_to_one(self, batch):
        # The entrypoint's own `default=` carries the same `1.0` the block
        # arg does — passing it explicitly must not move a single byte.
        _stage(batch, "clone-1")

        _, omitted = _run(batch)
        _, explicit = _run(batch, ["--structural-weight", "1.0", "--objective-weight", "1.0"])

        assert explicit == omitted


class TestTheTwoEditCapFlags:
    def test_both_cap_flags_reach_process_one_in_their_own_fields(self, batch, monkeypatch):
        _stage(batch, "clone-1")
        calls = []
        real_process_one = build_variants.process_one

        def capture(*args, **kwargs):
            calls.append((args[6], args[7]))
            return real_process_one(*args, **kwargs)

        monkeypatch.setattr(build_variants, "process_one", capture)

        _run(batch, ["--max-liability-edits", "3", "--max-framework-edits", "7"])

        assert calls == [(3, 7)]

    def test_neither_cap_flag_given_carries_the_module_defaults(self, batch, monkeypatch):
        _stage(batch, "clone-1")
        calls = []
        real_process_one = build_variants.process_one

        def capture(*args, **kwargs):
            calls.append((args[6], args[7]))
            return real_process_one(*args, **kwargs)

        monkeypatch.setattr(build_variants, "process_one", capture)

        _run(batch)

        assert calls == [
            (
                variant_candidates.DEFAULT_MAX_LIABILITY_EDITS,
                variant_candidates.DEFAULT_MAX_FRAMEWORK_EDITS,
            )
        ]

    def _stage_capped(self, batch, clonotype_key):
        """Three clearable liability targets and three non-human framework targets, each its
        own single residue, over one parent — enough positions for a cap of 1 to visibly
        truncate one kind and leave the other whole."""
        entry = batch.add(clonotype_key)
        liability_residues = [
            _residue("H", 10, "N", imgt="10"),
            _residue("H", 11, "S", imgt="11"),
            _residue("H", 12, "T", imgt="12"),
        ]
        framework_residues = [
            _residue("H", 21, "K", imgt="21", region="FR1"),
            _residue("H", 22, "R", imgt="22", region="FR1"),
            _residue("H", 23, "H", imgt="23", region="FR1"),
        ]
        all_residues = [*liability_residues, *framework_residues]
        liability_store.write_triaged(str(Path(batch.dir("triaged"), f"{entry.stem}.json")), [])
        tolerance_store.write_tolerance_tsv(
            str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv")),
            [{"imgt": r.imgt, **_row(["D", "Q", "A"], r.wild_type, 3.0)} for r in all_residues],
        )
        residue_store.write_residues(
            str(Path(batch.dir("residues"), f"{entry.stem}.json")), all_residues
        )
        _write_prior_tsv(
            Path(batch.dir("tolerance"), f"{entry.stem}{build_variants.PRIOR_SUFFIX}"), 0.9
        )
        return liability_residues, framework_residues

    def _stub_clearing_targets(self, monkeypatch, liability_residues, framework_residues):
        """Every target clears at once — the walk's own seed always passes — so what survives
        onto the emitted variant is exactly what `admitted_targets` let through each cap."""
        clearing = design_objective.Objective(
            select_target_positions=lambda residues, taxonomy: [],
            position_prior=None,
            score_candidate=lambda *_args: design_objective.GoalCheck(
                meets_goal=True, score=0.0
            ),
        )
        monkeypatch.setattr(
            build_variants.run_mode,
            "objectives_for",
            lambda *_args: [
                (run_mode.LIABILITY, lambda _residues: clearing),
                (run_mode.HUMANNESS, lambda _residues: clearing),
            ],
        )
        targets = [
            design_objective.DesignTarget(
                site=(r,), definition_id="stub-liability", region="CDR1",
                is_low_confidence=False, confidence_angstroms=3.0, objective=run_mode.LIABILITY,
            )
            for r in liability_residues
        ] + [
            design_objective.DesignTarget(
                site=(r,), definition_id=None, region="FR1",
                is_low_confidence=False, confidence_angstroms=3.0, objective=run_mode.HUMANNESS,
            )
            for r in framework_residues
        ]
        monkeypatch.setattr(
            build_variants.run_mode,
            "targets_for",
            lambda mode, triaged_list, objectives, residues: targets,
        )

    def test_the_liability_cap_truncates_only_the_liability_edits(self, batch, monkeypatch):
        liability_residues, framework_residues = self._stage_capped(batch, "clone-1")
        self._stub_clearing_targets(monkeypatch, liability_residues, framework_residues)

        _, written = _run(
            batch, ["--run-mode", "liabilities + humanization", "--max-liability-edits", "1"]
        )

        assert len(written) > 0
        best = written[0][2]
        liability_hits = [f"H:{r.wild_type}{r.imgt}" for r in liability_residues]
        framework_hits = [f"H:{r.wild_type}{r.imgt}" for r in framework_residues]
        assert sum(best.changed_positions.count(p) for p in liability_hits) == 1
        assert sum(best.changed_positions.count(p) for p in framework_hits) == len(
            framework_residues
        )

    def test_the_framework_cap_truncates_only_the_framework_edits(self, batch, monkeypatch):
        liability_residues, framework_residues = self._stage_capped(batch, "clone-1")
        self._stub_clearing_targets(monkeypatch, liability_residues, framework_residues)

        _, written = _run(
            batch, ["--run-mode", "liabilities + humanization", "--max-framework-edits", "1"]
        )

        assert len(written) > 0
        best = written[0][2]
        liability_hits = [f"H:{r.wild_type}{r.imgt}" for r in liability_residues]
        framework_hits = [f"H:{r.wild_type}{r.imgt}" for r in framework_residues]
        assert sum(best.changed_positions.count(p) for p in liability_hits) == len(
            liability_residues
        )
        assert sum(best.changed_positions.count(p) for p in framework_hits) == 1


class TestDatasetWideVariantsTsv:
    def test_one_file_holds_every_parent_with_rank_global_across_the_run(self, batch):
        # Both parents are staged from the identical fixture, so clone-1's
        # and clone-2's rank-1 candidates tie on tolerance and changed
        # positions, and likewise for their rank-2 candidates — genuinely
        # global rank interleaves the two parents rather than keeping each
        # parent's pair together, with `(clonotypeKey, variantKey)` breaking
        # the tie deterministically within each interleaved pair.
        _stage(batch, "clone-1")
        _stage(batch, "clone-2")

        rejections, written = _run(batch, ["--variants-per-parent", "2"])

        assert rejections == [("clone-1", "", "", "parent"), ("clone-2", "", "", "parent")]
        assert [(key, v.rank) for key, _, v in written] == [
            ("clone-1", 1), ("clone-2", 2), ("clone-1", 3), ("clone-2", 4),
        ]

    def test_clonotype_key_and_variant_key_are_unique_across_the_file(self, batch):
        _stage(batch, "clone-1")
        _stage(batch, "clone-2")

        _, written = _run(batch, ["--variants-per-parent", "2"])

        keys = [(clonotype, variant) for clonotype, variant, _ in written]
        assert len(set(keys)) == len(written) == 4

    def test_two_parents_rank_one_share_the_same_ordinal_variant_key(self, batch):
        # `variantKey` is a per-parent ordinal, not content-addressed —
        # two parents' rank-1 variant both render `v01`; only the
        # `(clonotypeKey, variantKey)` pair, asserted unique above, tells
        # them apart.
        _stage(batch, "clone-1")
        _stage(batch, "clone-2")

        _, written = _run(batch, ["--variants-per-parent", "1"])

        assert {variant for _, variant, _ in written} == {"v01"}

    def test_an_empty_index_leaves_a_header_only_variants_tsv(self, batch):
        rejections, written = _run(batch)

        assert rejections == []
        assert written == []


class TestNoReReportingAcrossPredecessors:
    def test_an_antibody_missing_any_predecessors_file_gets_no_row(self, batch):
        _stage(batch, "complete")
        partial = batch.add("no-tolerance")
        liability_store.write_triaged(
            str(Path(batch.dir("triaged"), f"{partial.stem}.json")), [_triaged(_ng_site())]
        )
        residue_store.write_residues(
            str(Path(batch.dir("residues"), f"{partial.stem}.json")), _ng_site()
        )

        rejections, written = _run(batch)

        assert rejections == [("complete", "", "", "parent")]
        assert {key for key, _, _ in written} == {"complete"}


class TestMainRequiresTheTaxonomy:
    def test_a_missing_definitions_file_is_a_readable_non_zero_exit(self, batch):
        missing = batch.path("definitions.json")

        with pytest.raises(SystemExit, match=missing):
            build_variants.main(
                [
                    "--triaged-dir", batch.dir("triaged"),
                    "--tolerance-dir", batch.dir("tolerance"),
                    "--residues-dir", batch.dir("residues"),
                    "--definitions", missing,
                    "--out-variants", batch.path("variants.tsv"),
                    "--out-rejected", batch.path("rejected.tsv"),
                    "--out-humanness", batch.path("humanness.tsv"),
                ]
            )


class TestRunModeSelectsObjectives:
    def test_mode_2_over_the_same_fixture_matches_mode_1_since_humanization_proposes_nothing(
        self, batch
    ):
        # Selecting the second objective changes nothing while it proposes
        # nothing: no target of its own means no candidate, no variant.
        _stage(batch, "clone-1", with_prior=True)

        mode_1_skips, mode_1_written = _run(batch, ["--run-mode", "liabilities"])
        mode_2_skips, mode_2_written = _run(
            batch, ["--run-mode", "liabilities + humanization"]
        )

        assert mode_1_skips == mode_2_skips
        assert [(ck, v.structural_tolerance, v.changed_positions)
                for ck, _, v in mode_1_written] == [
            (ck, v.structural_tolerance, v.changed_positions)
            for ck, _, v in mode_2_written
        ]

    def test_an_unrecognised_mode_exits_non_zero_and_writes_no_rows(self, batch):
        _stage(batch, "clone-1")
        out_variants = batch.path("variants.tsv")

        with pytest.raises(SystemExit):
            build_variants.main(
                [
                    "--triaged-dir", batch.dir("triaged"),
                    "--tolerance-dir", batch.dir("tolerance"),
                    "--residues-dir", batch.dir("residues"),
                    "--definitions", batch.definitions(TAXONOMY),
                    "--out-variants", out_variants,
                    "--out-rejected", batch.path("rejected.tsv"),
                    "--out-humanness", batch.path("humanness.tsv"),
                    "--run-mode", "misspelled-mode",
                ]
            )

    def test_a_stub_humanization_objective_that_yields_a_candidate_keeps_variant_keys_unique(
        self, batch, monkeypatch
    ):
        # A stub standing in for humanization, over the exact same site the
        # liability objective already cleared, proves key uniqueness holds
        # the moment the second objective becomes productive.
        entry = _stage(batch, "clone-1")
        stub_objective = design_objective.Objective(
            select_target_positions=lambda residues, taxonomy: [],
            position_prior=None,
            score_candidate=lambda mutated_site, taxonomy, tolerance_lookup: (
                design_objective.GoalCheck(meets_goal=True, score=0.0)
            ),
        )
        monkeypatch.setattr(
            build_variants.run_mode,
            "objectives_for",
            lambda *_args: [
                (run_mode.LIABILITY, lambda _residues: stub_objective),
                (run_mode.HUMANNESS, lambda _residues: stub_objective),
            ],
        )
        monkeypatch.setattr(
            build_variants.run_mode,
            "targets_for",
            lambda mode, triaged_list, objectives, residues: [
                _target_from_triaged(t) for t in triaged_list
            ],
        )

        triaged_path = str(Path(batch.dir("triaged"), f"{entry.stem}.json"))
        tolerance_path = str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv"))
        residues_path = str(Path(batch.dir("residues"), f"{entry.stem}.json"))
        result = build_variants.process_one(
            triaged_path, tolerance_path, residues_path, TAXONOMY,
            "liabilities + humanization", "unused-prior-path",
            10, 20, 3, 1.0, 1.0, 0.05, 10, 3.0, 20,
        )

        keys = [v.parent_rank for v in result.variants]
        assert len(keys) == len(set(keys))
        assert len(result.variants) > 0

    def test_one_variant_carries_a_cdr_liability_edit_and_a_framework_edit_together(
        self, batch, monkeypatch
    ):
        # The golden fixture holds no non-human framework position, so its captures
        # cannot show the two edit kinds on one molecule. A framework position with a
        # tolerance row of its own, targeted beside the CDR liability, can.
        entry = batch.add("clone-1")
        cdr_site = _ng_site()
        framework = _residue("H", 20, "K", imgt="21", region="FR1")
        liability_store.write_triaged(
            str(Path(batch.dir("triaged"), f"{entry.stem}.json")), [_triaged(cdr_site)]
        )
        tolerance_store.write_tolerance_tsv(
            str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv")),
            [*_tolerance_rows(), {"imgt": "21", **_row(["R", "T", "V"], "K", 4.0)}],
        )
        residue_store.write_residues(
            str(Path(batch.dir("residues"), f"{entry.stem}.json")), [*cdr_site, framework]
        )

        clearing = design_objective.Objective(
            select_target_positions=lambda residues, taxonomy: [],
            position_prior=None,
            score_candidate=lambda *_args: design_objective.GoalCheck(
                meets_goal=True, score=0.0
            ),
        )
        monkeypatch.setattr(
            build_variants.run_mode,
            "objectives_for",
            lambda *_args: [
                (run_mode.LIABILITY, lambda _residues: clearing),
                (run_mode.HUMANNESS, lambda _residues: clearing),
            ],
        )
        monkeypatch.setattr(
            build_variants.run_mode,
            "targets_for",
            lambda mode, triaged_list, objectives, residues: [
                *[_target_from_triaged(t) for t in triaged_list],
                design_objective.DesignTarget(
                    site=(framework,),
                    definition_id=None,
                    region="FR1",
                    is_low_confidence=False,
                    confidence_angstroms=3.0,
                    objective=run_mode.HUMANNESS,
                ),
            ],
        )

        result = build_variants.process_one(
            str(Path(batch.dir("triaged"), f"{entry.stem}.json")),
            str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv")),
            str(Path(batch.dir("residues"), f"{entry.stem}.json")),
            TAXONOMY,
            "liabilities + humanization",
            "unused-prior-path",
            10, 20, 3, 1.0, 1.0, 0.05, 10, 3.0, 20,
        )

        assert result.variants
        best = result.variants[0]
        # Both edit kinds on the one molecule, and both targets named beside each other.
        assert "H:K21" in best.changed_positions
        assert "H:N107" in best.changed_positions
        assert "Deamidation" in best.addressed_target
        assert "Humanization" in best.addressed_target

    def test_every_objectives_variants_carry_a_measured_humanness_score(
        self, batch, monkeypatch
    ):
        # The goal check's own score (80.0) and every fold tolerance in the fixture are
        # distinguishable from the measurement's, so a landing on the wrong field would be a
        # visible number match, not a silent one.
        _stub_rising_on_d(monkeypatch)
        entry = _stage(batch, "clone-1")
        stub_objective = design_objective.Objective(
            select_target_positions=lambda residues, taxonomy: [],
            position_prior=None,
            score_candidate=lambda mutated_site, taxonomy, tolerance_lookup: (
                design_objective.GoalCheck(meets_goal=True, score=80.0)
            ),
        )
        monkeypatch.setattr(
            build_variants.run_mode,
            "objectives_for",
            lambda *_args: [
                (run_mode.LIABILITY, lambda _residues: stub_objective),
                (run_mode.HUMANNESS, lambda _residues: stub_objective),
            ],
        )
        monkeypatch.setattr(
            build_variants.run_mode,
            "targets_for",
            lambda mode, triaged_list, objectives, residues: [
                _target_from_triaged(t) for t in triaged_list
            ],
        )

        triaged_path = str(Path(batch.dir("triaged"), f"{entry.stem}.json"))
        tolerance_path = str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv"))
        residues_path = str(Path(batch.dir("residues"), f"{entry.stem}.json"))
        result = build_variants.process_one(
            triaged_path, tolerance_path, residues_path, TAXONOMY,
            "liabilities + humanization", "unused-prior-path",
            10, 20, 3, 1.0, 1.0, 0.05, 10, 3.0, 20,
        )

        # Measured on the variant's own heavy chain — never the goal check's score, which is
        # what the humanization gate compares and this column is not.
        scored = result.variants
        assert len(scored) > 0
        assert all(v.humanness_score is not None for v in scored)
        assert all(v.humanness_score != pytest.approx(80.0) for v in scored)
        # And the tolerance the stub's own score never reaches: this fixture's
        # two-position mean, never the stub's score.
        assert all(v.structural_tolerance == pytest.approx(3.5) for v in scored)

    def test_the_design_result_names_its_two_lists(self, batch, monkeypatch):
        entry = _stage(batch, "clone-1", with_prior=True)
        stub_objective = design_objective.Objective(
            select_target_positions=lambda residues, taxonomy: [],
            position_prior=None,
            score_candidate=lambda mutated_site, taxonomy, tolerance_lookup: (
                design_objective.GoalCheck(meets_goal=True, score=80.0)
            ),
        )
        fixed_site = _ng_site()[:1]
        fixed_target = _target_from_triaged(_triaged(fixed_site), objective=run_mode.HUMANNESS)
        monkeypatch.setattr(
            build_variants.run_mode,
            "objectives_for",
            lambda *_args: [
                (run_mode.LIABILITY, lambda _residues: stub_objective),
                (run_mode.HUMANNESS, lambda _residues: stub_objective),
            ],
        )
        monkeypatch.setattr(
            build_variants.run_mode,
            "targets_for",
            lambda mode, triaged_list, objectives, residues: (
                [_target_from_triaged(t) for t in triaged_list] + [fixed_target]
            ),
        )

        triaged_path = str(Path(batch.dir("triaged"), f"{entry.stem}.json"))
        tolerance_path = str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv"))
        residues_path = str(Path(batch.dir("residues"), f"{entry.stem}.json"))
        result = build_variants.process_one(
            triaged_path, tolerance_path, residues_path, TAXONOMY,
            "liabilities + humanization", "unused-prior-path",
            10, 20, 3, 1.0, 1.0, 0.05, 10, 3.0, 20,
        )

        # `humanness_targets` is the objective's own selected framework
        # positions, distinct from `humanness_cleared`'s gate survivors —
        # neither list's contents leak into the other's field.
        assert result.humanness_targets == list(fixed_target.site)
        assert all(isinstance(c, variant_candidates.Candidate) for c in result.humanness_cleared)

    def test_a_mode_that_never_runs_humanization_leaves_the_targets_absent(self, batch):
        entry = _stage(batch, "clone-1")

        triaged_path = str(Path(batch.dir("triaged"), f"{entry.stem}.json"))
        tolerance_path = str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv"))
        residues_path = str(Path(batch.dir("residues"), f"{entry.stem}.json"))
        result = build_variants.process_one(
            triaged_path, tolerance_path, residues_path, TAXONOMY,
            "liabilities", "unused-prior-path",
            10, 20, 3, 1.0, 1.0, 0.05, 10, 3.0, 20,
        )

        # Mode 1 never runs the humanization objective at all — its targets
        # stay `None`, distinct from the empty list an objective that ran
        # and selected nothing would leave.
        assert result.humanness_targets is None
        assert result.humanness_cleared == []


def _run_mixed(batch, extra_args=None):
    out_humanness = batch.path("humanness.tsv")
    rejections, written = _run(batch, extra_args)
    with Path(out_humanness).open(newline="") as fh:
        humanness_rows = list(csv.DictReader(fh, delimiter="\t"))
    return rejections, written, humanness_rows


class TestHumanizationModeEndToEnd:
    """`--run-mode humanization` over one parent carrying both a liability and a
    non-human framework position — Outcome's motivating case, and R16/R12's own targets."""

    def test_variants_hold_only_humanness_rows_each_with_a_humanness_score(self, batch,
                                                                             monkeypatch):
        _stub_rising_on_d(monkeypatch)
        _stage_mixed(batch, "clone-1", non_human_prior_score=0.01)

        _, written, _ = _run_mixed(batch, ["--run-mode", "humanization"])

        assert len(written) > 0
        assert all(v.humanness_score is not None for _, _, v in written)

    def test_the_humanness_row_reads_present_and_names_the_selected_position(self, batch,
                                                                                monkeypatch):
        _stub_rising_on_d(monkeypatch)
        _stage_mixed(batch, "clone-1", non_human_prior_score=0.01)

        _, _, humanness_rows = _run_mixed(batch, ["--run-mode", "humanization"])

        [row] = humanness_rows
        assert row["humannessVerdict"] == "present"
        assert row["humannessSummary"] == "N@H1 (humanised)"

    def test_a_prior_leading_with_the_wild_type_names_the_no_target_rejection_reason(self, batch):
        # The prior's best residue is the one the antibody already carries, so no gap exists
        # to clear and even a margin of 0 — the loosest setting there is — selects nothing.
        _stage_mixed(batch, "clone-1", non_human_prior_score=0.95)

        rejections, written, humanness_rows = _run_mixed(
            batch, ["--run-mode", "humanization", "--non-human-prior-margin", "0.0"]
        )

        assert rejections == [("clone-1", "no-nonhuman-framework-position", "", "parent")]
        assert written == []
        [row] = humanness_rows
        assert row["humannessVerdict"] == "none"

    def test_an_unscored_selected_position_names_its_own_rejection_reason(self, batch, monkeypatch):
        # The whole chain's set is dropped together, so the parent emits nothing.
        # The reason must say which quantity was missing: no motif was involved,
        # and the humanness gate never ran to reject anything.
        _stub_rising_on_d(monkeypatch)
        _stage_mixed(batch, "clone-1", non_human_prior_score=0.01,
                     score_framework_position=False)

        rejections, written, humanness_rows = _run_mixed(batch, ["--run-mode", "humanization"])

        assert rejections == [("clone-1", "no-tolerance-at-humanization-position", "", "parent")]
        assert written == []
        # The position was selected, so the parent's own row still reports it —
        # `declined`, because no candidate was ever built at it.
        [row] = humanness_rows
        assert row["humannessVerdict"] == "present"
        assert row["humannessSummary"] == "N@H1 (declined)"

    def test_liabilities_mode_over_the_same_fixture_is_unaffected_by_the_extra_residue(
        self, batch
    ):
        # R17: the run mode changes what the design step acts on, never what
        # the scan detects. The framework residue `_stage_mixed` adds is
        # never a liability target, so a liabilities-mode run reads the same
        # for that parent as it does for the plain `_stage` fixture that
        # carries no such residue at all — both staged in the one batch, so
        # one `_run` call reports both.
        _stage_mixed(batch, "mixed", non_human_prior_score=0.01)
        _stage(batch, "plain")

        rejections, written = _run(batch, ["--run-mode", "liabilities"])

        by_key = {"mixed": [], "plain": []}
        for clonotype_key, _, variant in written:
            by_key[clonotype_key].append(
                (variant.structural_tolerance, variant.changed_positions)
            )
        assert {reason for _, reason, _detail, _type in rejections} == {""}
        assert by_key["mixed"] == by_key["plain"]

    def test_combined_mode_writes_both_targets_on_one_row(self, batch, monkeypatch):
        # The written TSV row, not just the returned `Variant`, so the addressed-target
        # column survives the full write path in the mode the Variants page actually offers.
        _stub_rising_on_d(monkeypatch)
        _stage_mixed(batch, "clone-1", non_human_prior_score=0.01)

        _, written, _ = _run_mixed(batch, ["--run-mode", "liabilities + humanization"])

        assert any(
            "Deamidation" in v.addressed_target and "Humanization" in v.addressed_target
            for _, _, v in written
        )


def _stage_liability_mix(batch, clonotype_key, triaged):
    """One parent whose triaged list `_stage_liability_mix`'s caller supplies, with the
    framework residue and the `_ng_site` CDR residues both in its residue index and its
    tolerance table — the fixture `TestModeTwoLiabilityTargetsCutToCdrs` runs against. The
    prior scores every position above the default cutoff, so the humanization objective
    selects nothing and every emitted variant is the liability objective's own."""
    entry = batch.add(clonotype_key)
    residues = [_framework_residue(), *_ng_site()]
    liability_store.write_triaged(
        str(Path(batch.dir("triaged"), f"{entry.stem}.json")), triaged
    )
    tolerance_store.write_tolerance_tsv(
        str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv")),
        [*_tolerance_rows(), {"imgt": "1", **_row(["D", "Q", "A"], "N", 3.0)}],
    )
    residue_store.write_residues(
        str(Path(batch.dir("residues"), f"{entry.stem}.json")), residues
    )
    _write_prior_tsv(
        Path(batch.dir("tolerance"), f"{entry.stem}{build_variants.PRIOR_SUFFIX}"), 0.9
    )
    return entry


class TestModeTwoLiabilityTargetsCutToCdrs:
    """Mode 2 designs the liability objective only against a triaged site whose whole site
    lies inside a CDR — TODO-20's Outcome."""

    def test_a_cdr_and_a_framework_liability_yield_variants_for_the_cdr_liability_only(
        self, batch
    ):
        _stage_liability_mix(
            batch, "clone-1", [_triaged(_ng_site()), _triaged([_framework_residue()])]
        )

        _, written = _run(batch, ["--run-mode", "liabilities + humanization"])

        assert len(written) > 0
        assert all("H:N1D" not in v.changed_positions for _, _, v in written)

    def test_liabilities_mode_over_the_same_parent_still_designs_the_framework_liability(
        self, batch
    ):
        _stage_liability_mix(
            batch, "clone-1", [_triaged(_ng_site()), _triaged([_framework_residue()])]
        )

        _, written = _run(batch, ["--run-mode", "liabilities"])

        assert any("H:N1D" in v.changed_positions for _, _, v in written)

    def test_a_framework_only_parent_in_mode_2_gets_no_variant_and_a_named_skip(self, batch):
        _stage_liability_mix(batch, "clone-1", [_triaged([_framework_residue()])])

        rejections, written = _run(batch, ["--run-mode", "liabilities + humanization"])

        assert written == []
        [(clonotype_key, reason, _detail, _type)] = rejections
        assert clonotype_key == "clone-1"
        assert reason != ""


class TestParentHumannessScoresEndToEnd:
    """`humanness.tsv`'s score column, across the run modes and antibody shapes the
    Outcome names."""

    def test_a_paired_parent_emits_its_heavy_chain_score(self, batch, monkeypatch):
        # A stub distinguishable by chain: the fixture's H sequence is three
        # residues, its L sequence one, so the light chain's score landing in
        # the column would be a visible number mismatch, not a silent one.
        monkeypatch.setattr(humanness_gate, "identity", lambda seq: float(len(seq)))
        _stage_paired(batch, "clone-1", non_human_prior_score=0.01)

        _, _, humanness_rows = _run_mixed(batch, ["--run-mode", "humanization"])

        [row] = humanness_rows
        assert row["heavyHumannessScore"] == "3.0"
        assert "lightHumannessScore" not in row

    def test_a_single_chain_parent_still_carries_its_heavy_score(self, batch, monkeypatch):
        _stub_rising_on_d(monkeypatch)
        _stage_mixed(batch, "clone-1", non_human_prior_score=0.01)

        _, _, humanness_rows = _run_mixed(batch, ["--run-mode", "humanization"])

        [row] = humanness_rows
        assert row["heavyHumannessScore"] != ""

    def test_liabilities_mode_over_the_same_input_leaves_the_column_empty(self, batch):
        _stage_mixed(batch, "clone-1", non_human_prior_score=0.01)

        _, _, humanness_rows = _run_mixed(batch, ["--run-mode", "liabilities"])

        [row] = humanness_rows
        assert row["heavyHumannessScore"] == ""

    def test_a_parent_whose_candidates_were_all_gate_discarded_still_gets_its_scores(
        self, batch, monkeypatch
    ):
        # Every candidate ties the parent's own score rather than raising
        # it, so the gate's strict-rise rule discards them all — the row's
        # own scores must not depend on any candidate surviving.
        monkeypatch.setattr(humanness_gate, "identity", lambda seq: 5.0)
        _stage_mixed(batch, "clone-1", non_human_prior_score=0.01)

        _, written, humanness_rows = _run_mixed(batch, ["--run-mode", "humanization"])

        assert written == []
        [row] = humanness_rows
        assert row["heavyHumannessScore"] == "5.0"


class TestReRankWeightsReachTheGlobalRewrite:
    def test_alpha_and_beta_flags_reach_rewrite_global_rank(self, batch, monkeypatch):
        _stage(batch, "clone-1")
        calls = []
        monkeypatch.setattr(
            variant_store,
            "rewrite_global_rank",
            lambda path, structural, humanness: calls.append((structural, humanness)),
        )

        _run(batch, ["--rerank-structural-weight", "2.5", "--rerank-humanness-weight", "0.5"])

        assert calls == [(2.5, 0.5)]

    def test_the_weights_default_to_the_module_defaults(self, batch, monkeypatch):
        _stage(batch, "clone-1")
        calls = []
        monkeypatch.setattr(
            variant_store,
            "rewrite_global_rank",
            lambda path, structural, humanness: calls.append((structural, humanness)),
        )

        _run(batch)

        assert calls == [
            (
                variant_ranking.DEFAULT_RERANK_STRUCTURAL_WEIGHT,
                variant_ranking.DEFAULT_RERANK_HUMANNESS_WEIGHT,
            )
        ]
